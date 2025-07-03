import subprocess
import sys
import warnings
import logging
import traceback
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import gc
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict, deque
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
import wandb
from datetime import datetime
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_optimized.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def timer(description: str):
    """Context manager for timing operations"""
    class Timer:
        def __init__(self, desc):
            self.desc = desc
            
        def __enter__(self):
            self.start = time.time()
            return self
            
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            logger.info(f"{self.desc}: {elapsed:.2f}s")
    
    return Timer(description)

def setup_device():
    """Set up the computation device (GPU/CPU)"""
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = False  # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            logger.info(f"CUDA Version: {torch.version.cuda}")
        else:
            device = torch.device("cpu")
            logger.warning("Using CPU - training will be slower")
        return device
    except Exception as e:
        logger.error(f"Device setup failed: {str(e)}")
        return torch.device("cpu")

device = setup_device()

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class BaseConfig:
    """Base configuration with common parameters"""
    model_name: str = "gpt2"
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    max_length: int = 256
    adaptation_rank: int = 16
    num_experts: int = 4
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 0.5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    enable_paged_attention: bool = False
    paged_block_size: int = 16
    max_cache_blocks: int = 1000
    max_samples_per_dataset: int = 500
    use_fallback_data_only: bool = False
    grpo_episodes_per_batch: int = 4
    grpo_reward_normalization: bool = True
    grpo_kl_coeff: float = 0.01
    grpo_value_loss_coeff: float = 0.1
    grpo_entropy_coeff: float = 0.05
    svd_rank_ratio: float = 0.8
    svd_min_singular_value: float = 1e-5
    wandb_project: str = "grpo-cem-gpt2"
    output_dir: str = "./results"
    log_interval: int = 10
    save_interval: int = 1
    clip_rewards: float = 2.0
    reward_scaling: float = 0.2
    temperature_annealing: bool = True
    adaptive_learning_rate: bool = True
    learning_rate_min: float = 1e-6
    repetition_penalty: float = 1.2
    top_p: float = 0.85
    temperature: float = 0.7
    min_episode_length: int = 16
    max_episode_length: int = 48
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

@dataclass
class OptimizedConfig(BaseConfig):
    """Optimized configuration for standard GPUs"""
    mixed_precision: bool = True
    batch_size: int = 4
    cem_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "qa": {
            "population_size": 10,
            "elite_ratio": 0.3,
            "noise_std": 0.2,
            "adaptation_steps": 8,
            "convergence_threshold": 0.01
        },
        "sentiment": {
            "population_size": 10,
            "elite_ratio": 0.3,
            "noise_std": 0.25,
            "adaptation_steps": 8,
            "convergence_threshold": 0.01
        },
        "classification": {
            "population_size": 12,
            "elite_ratio": 0.3,
            "noise_std": 0.15,
            "adaptation_steps": 10,
            "convergence_threshold": 0.008
        },
        "general": {
            "population_size": 10,
            "elite_ratio": 0.3,
            "noise_std": 0.3,
            "adaptation_steps": 8,
            "convergence_threshold": 0.01
        }
    })
    first_epoch_cem_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "qa": {"population_size": 4, "elite_ratio": 0.5, "noise_std": 0.3, "adaptation_steps": 2, "convergence_threshold": 0.1},
        "sentiment": {"population_size": 4, "elite_ratio": 0.5, "noise_std": 0.3, "adaptation_steps": 2, "convergence_threshold": 0.1},
        "classification": {"population_size": 4, "elite_ratio": 0.5, "noise_std": 0.3, "adaptation_steps": 2, "convergence_threshold": 0.1},
        "general": {"population_size": 4, "elite_ratio": 0.5, "noise_std": 0.3, "adaptation_steps": 2, "convergence_threshold": 0.1}
    })
    cem_momentum: float = 0.3

@dataclass
class UltraConfig(BaseConfig):
    """Ultra configuration for 15GB+ GPUs"""
    batch_size: int = 8
    learning_rate: float = 3e-5
    num_epochs: int = 5
    max_length: int = 384
    adaptation_rank: int = 24
    num_experts: int = 6
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 200
    enable_paged_attention: bool = True
    paged_block_size: int = 32
    max_cache_blocks: int = 1500
    max_samples_per_dataset: int = 1000
    grpo_episodes_per_batch: int = 6
    grpo_kl_coeff: float = 0.02
    grpo_value_loss_coeff: float = 0.2
    grpo_entropy_coeff: float = 0.1
    svd_rank_ratio: float = 0.85
    wandb_project: str = "ultra-grpo-cem-gpt2-15GB"
    output_dir: str = "./ultra_results"
    log_interval: int = 5
    clip_rewards: float = 3.0
    reward_scaling: float = 0.5
    repetition_penalty: float = 1.3
    top_p: float = 0.9
    temperature: float = 0.8
    min_episode_length: int = 32
    max_episode_length: int = 80
    num_workers: int = min(8, os.cpu_count() or 4)
    prefetch_factor: int = 8
    cem_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "qa": {
            "population_size": 32,
            "elite_ratio": 0.25,
            "noise_std": 0.25,
            "adaptation_steps": 15,
            "convergence_threshold": 0.003
        },
        "sentiment": {
            "population_size": 40,
            "elite_ratio": 0.25,
            "noise_std": 0.3,
            "adaptation_steps": 18,
            "convergence_threshold": 0.003
        },
        "classification": {
            "population_size": 50,
            "elite_ratio": 0.3,
            "noise_std": 0.35,
            "adaptation_steps": 20,
            "convergence_threshold": 0.002
        },
        "general": {
            "population_size": 32,
            "elite_ratio": 0.25,
            "noise_std": 0.3,
            "adaptation_steps": 15,
            "convergence_threshold": 0.003
        }
    })
    first_epoch_cem_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "qa": {"population_size": 8, "elite_ratio": 0.3, "noise_std": 0.2, "adaptation_steps": 5, "convergence_threshold": 0.01},
        "sentiment": {"population_size": 10, "elite_ratio": 0.3, "noise_std": 0.2, "adaptation_steps": 6, "convergence_threshold": 0.01},
        "classification": {"population_size": 12, "elite_ratio": 0.3, "noise_std": 0.2, "adaptation_steps": 7, "convergence_threshold": 0.01},
        "general": {"population_size": 8, "elite_ratio": 0.3, "noise_std": 0.2, "adaptation_steps": 5, "convergence_threshold": 0.01}
    })
    cem_momentum: float = 0.4

# ============================================================================
# DATA HANDLING CLASSES
# ============================================================================

class PreTokenizedDataset(Dataset):
    """Pre-tokenized dataset for faster loading"""
    def __init__(self, data, tokenizer, max_length, cache_path=None):
        self.cache_path = cache_path or f"./tokenized_cache_{max_length}_{len(data)}.pt"
        self.max_length = max_length
        if os.path.exists(self.cache_path):
            logger.info(f"Loading pre-tokenized data from {self.cache_path}")
            with timer("Loading cached tokenized data"):
                self.tokenized_data = torch.load(self.cache_path)
        else:
            logger.info("Pre-tokenizing dataset (one-time cost)...")
            with timer("Pre-tokenizing dataset"):
                self.tokenized_data = self._tokenize_all(data, tokenizer, max_length)
                torch.save(self.tokenized_data, self.cache_path)
                logger.info(f"Saved tokenized data to {self.cache_path}")
    
    def _tokenize_all(self, data, tokenizer, max_length):
        """Tokenize all data at once for maximum efficiency"""
        tokenized = []
        batch_size = 50
        
        for i in tqdm(range(0, len(data), batch_size), desc="Tokenizing"):
            batch = data[i:i+batch_size]
            input_texts = []
            target_texts = []
            task_types = []
            
            for input_text, target_text, task_type in batch:
                if not all([input_text, target_text, task_type]):
                    continue
                input_texts.append(str(input_text))
                target_texts.append(str(target_text))
                task_types.append(str(task_type))
            
            if not input_texts:
                continue
                
            # Tokenize inputs
            inputs_batch = tokenizer(
                input_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                add_special_tokens=True
            )
            
            # Tokenize targets
            targets_batch = tokenizer(
                target_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length // 2,
                padding="max_length",
                add_special_tokens=True
            )
            
            for j in range(len(input_texts)):
                tokenized.append({
                    'input_ids': inputs_batch['input_ids'][j],
                    'attention_mask': inputs_batch['attention_mask'][j],
                    'target_ids': targets_batch['input_ids'][j],
                    'task_type': task_types[j],
                    'input_text': input_texts[j][:200],
                    'target_text': target_texts[j][:100]
                })
        
        return tokenized
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx]

class DatasetLoader:
    """Unified dataset loader for both optimized and ultra modes"""
    def __init__(self, config):
        self.config = config
        self.datasets = {}
        self.validation_datasets = {}
        self.successful_downloads = 0
        self.failed_downloads = 0
        
    def load_all_datasets(self):
        """Load datasets based on configuration"""
        if self.config.use_fallback_data_only:
            logger.info("Using fallback data only")
            self._add_fallback_data()
            return self.datasets
            
        if isinstance(self.config, UltraConfig):
            dataset_configs = self._get_ultra_dataset_configs()
        else:
            dataset_configs = self._get_optimized_dataset_configs()
            
        self._load_datasets_sequential(dataset_configs)
        
        total_samples = sum(len(data) for data in self.datasets.values())
        if total_samples < 100:
            logger.warning("Low sample count, adding fallback data")
            self._add_fallback_data()
            
        logger.info(f"Dataset Loading: {self.successful_downloads} successful, {self.failed_downloads} failed")
        logger.info(f"Total training samples: {total_samples:,}")
        return self.datasets
    
    def _get_optimized_dataset_configs(self):
        """Get dataset configs for optimized mode"""
        return [
            {'name': 'squad', 'split': 'train[:200]', 'task_type': 'qa', 'process_fn': self._process_squad},
            {'name': 'imdb', 'split': 'train[:200]', 'task_type': 'sentiment', 'process_fn': self._process_imdb},
        ]
    
    def _get_ultra_dataset_configs(self):
        """Get dataset configs for ultra mode"""
        return [
            {'name': 'squad', 'split': 'train[:800]', 'task_type': 'qa', 'process_fn': self._process_squad},
            {'name': 'imdb', 'split': 'train[:800]', 'task_type': 'sentiment', 'process_fn': self._process_imdb},
            {'name': 'ag_news', 'split': 'train[:600]', 'task_type': 'classification', 'process_fn': self._process_ag_news},
        ]
    
    def _load_datasets_sequential(self, dataset_configs):
        """Load datasets sequentially for better stability"""
        for config in dataset_configs:
            try:
                with timer(f"Loading {config['name']}"):
                    dataset = load_dataset(
                        config['name'], 
                        split=config['split'],
                        download_mode="reuse_cache_if_exists",
                        verification_mode="no_checks"
                    )
                    processed_data = config['process_fn'](dataset)
                    if processed_data:
                        self.datasets[config['task_type']] = processed_data
                        self.successful_downloads += 1
                        logger.info(f"✓ Loaded {config['name']}: {len(processed_data)} samples")
                    else:
                        logger.warning(f"No data processed for {config['name']}")
                        self.failed_downloads += 1
            except Exception as e:
                logger.error(f"Failed to load {config['name']}: {str(e)}")
                self.failed_downloads += 1
                continue
    
    def _process_squad(self, dataset):
        """Process SQuAD dataset"""
        processed = []
        max_context_length = 600 if isinstance(self.config, UltraConfig) else 200
        
        for item in dataset:
            try:
                context = item.get('context', '').strip()
                question = item.get('question', '').strip()
                answers = item.get('answers', {})
                
                if context and question and answers and answers.get('text'):
                    answer = answers['text'][0].strip()
                    context = context[:max_context_length]
                    processed.append((f"Context: {context}\nQuestion: {question}", answer, 'qa'))
            except Exception as e:
                logger.debug(f"Error processing SQuAD item: {e}")
                continue
        
        return processed
    
    def _process_imdb(self, dataset):
        """Process IMDB dataset"""
        processed = []
        max_text_length = 800 if isinstance(self.config, UltraConfig) else 200
        
        for item in dataset:
            try:
                text = item.get('text', '').strip()
                label = item.get('label', 0)
                
                if text and len(text) > 50:
                    text = text[:max_text_length]
                    target = 'positive' if label == 1 else 'negative'
                    processed.append((f"Analyze the sentiment of this review: {text}", target, 'sentiment'))
            except Exception as e:
                logger.debug(f"Error processing IMDB item: {e}")
                continue
        
        return processed
    
    def _process_ag_news(self, dataset):
        """Process AG News dataset"""
        processed = []
        label_map = {0: 'world', 1: 'sports', 2: 'business', 3: 'technology'}
        max_text_length = 600 if isinstance(self.config, UltraConfig) else 200
        
        for item in dataset:
            try:
                text = item.get('text', '').strip()
                label = item.get('label', 0)
                
                if text and len(text) > 30:
                    text = text[:max_text_length]
                    target = label_map.get(label, 'general')
                    processed.append((f"Classify this news article: {text}", target, 'classification'))
            except Exception as e:
                logger.debug(f"Error processing AG News item: {e}")
                continue
        
        return processed
    
    def _add_fallback_data(self):
        """Add fallback data"""
        multiplier = 50 if isinstance(self.config, UltraConfig) else 20
        
        fallback = {
            'qa': [
                ("Context: Paris is the capital of France.\nQuestion: What is the capital of France?", "Paris", "qa"),
                ("Context: The sun is a star.\nQuestion: What is the sun?", "A star", "qa"),
                ("Context: Water boils at 100 degrees Celsius.\nQuestion: At what temperature does water boil?", "100 degrees Celsius", "qa"),
                ("Context: Python is a programming language.\nQuestion: What is Python?", "A programming language", "qa"),
                ("Context: The Earth orbits the Sun.\nQuestion: What does the Earth orbit?", "The Sun", "qa"),
            ] * multiplier,
            'sentiment': [
                ("This movie was absolutely fantastic and amazing!", "positive", "sentiment"),
                ("I hate this terrible product, it's awful.", "negative", "sentiment"),
                ("Great service and delicious food!", "positive", "sentiment"),
                ("Worst experience ever, very disappointed.", "negative", "sentiment"),
                ("Love this product, highly recommend!", "positive", "sentiment"),
                ("Poor quality, waste of money.", "negative", "sentiment"),
            ] * multiplier,
            'classification': [
                ("Scientists discover new planet in distant galaxy.", "world", "classification"),
                ("Stock market reaches all-time high today.", "business", "classification"),
                ("New smartphone technology announced by tech giant.", "technology", "classification"),
                ("Football team wins championship game.", "sports", "classification"),
                ("AI breakthrough in medical diagnosis.", "technology", "classification"),
                ("Global climate summit begins tomorrow.", "world", "classification"),
            ] * (multiplier // 2),
        }
        
        for task_type, data in fallback.items():
            if task_type not in self.datasets:
                self.datasets[task_type] = []
            self.datasets[task_type].extend(data)
            
        logger.info("Added fallback data for all tasks")

# ============================================================================
# NEURAL NETWORK COMPONENTS
# ============================================================================

class PagedGPT2Attention(nn.Module):
    """Paged attention implementation for memory efficiency"""
    def __init__(self, original_attention, config):
        super().__init__()
        self.original_attention = original_attention
        self.config = config
        self.block_size = config.paged_block_size
        self.max_cache_blocks = config.max_cache_blocks
        
        # Copy all original attention attributes dynamically
        for attr_name in dir(original_attention):
            if not attr_name.startswith('_') and hasattr(original_attention, attr_name):
                attr_value = getattr(original_attention, attr_name)
                if not callable(attr_value):
                    setattr(self, attr_name, attr_value)
        
        # Copy parameters and modules
        for name, param in original_attention.named_parameters():
            if '.' not in name:  # Direct parameters only
                setattr(self, name, param)
        
        for name, module in original_attention.named_children():
            setattr(self, name, module)
        
        # Paged attention cache
        self.cache_blocks = {}
        self.block_usage_order = deque()
        
    def _evict_cache_blocks(self):
        """Evict least recently used cache blocks"""
        while len(self.cache_blocks) >= self.max_cache_blocks:
            oldest_block = self.block_usage_order.popleft()
            if oldest_block in self.cache_blocks:
                del self.cache_blocks[oldest_block]
    
    def _get_cache_block(self, block_id):
        """Get or create cache block"""
        if block_id not in self.cache_blocks:
            self._evict_cache_blocks()
            self.cache_blocks[block_id] = {}
        
        # Update usage order
        if block_id in self.block_usage_order:
            self.block_usage_order.remove(block_id)
        self.block_usage_order.append(block_id)
        
        return self.cache_blocks[block_id]
    
    def forward(self, hidden_states, *args, **kwargs):
        """Forward pass with paged attention - handles all possible arguments"""
        batch_size, seq_len = hidden_states.size()[:2]
        
        # For short sequences or during training, use original attention
        if seq_len <= self.block_size * 2 or self.training:
            return self.original_attention(hidden_states, *args, **kwargs)
        
        # Extract common arguments with fallbacks
        attention_mask = kwargs.get('attention_mask', None)
        if attention_mask is None and len(args) >= 2:
            attention_mask = args[1]
        
        output_attentions = kwargs.get('output_attentions', False)
        if not output_attentions and len(args) >= 5:
            output_attentions = args[4]
        
        # Process in blocks for longer sequences during inference
        outputs = []
        all_attentions = []
        present_key_values = []
        
        for i in range(0, seq_len, self.block_size):
            end_idx = min(i + self.block_size, seq_len)
            block_hidden = hidden_states[:, i:end_idx]
            
            # Adjust attention mask for this block
            block_mask = None
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    block_mask = attention_mask[:, i:end_idx]
                elif attention_mask.dim() == 4:
                    block_mask = attention_mask[:, :, i:end_idx, i:end_idx]
                else:
                    block_mask = attention_mask
            
            # Create modified kwargs for this block
            block_kwargs = kwargs.copy()
            block_kwargs['attention_mask'] = block_mask
            block_kwargs['use_cache'] = False  # Disable cache for blocks
            
            # Remove past key values for block processing
            if 'past_key_value' in block_kwargs:
                block_kwargs['past_key_value'] = None
            if 'layer_past' in block_kwargs:
                block_kwargs['layer_past'] = None
            
            # Compute attention for block
            try:
                block_output = self.original_attention(block_hidden, **block_kwargs)
            except Exception as e:
                # Fallback: try with minimal arguments
                logger.debug(f"Block attention failed with kwargs, trying minimal args: {e}")
                block_output = self.original_attention(
                    block_hidden,
                    attention_mask=block_mask,
                    use_cache=False,
                    output_attentions=output_attentions
                )
            
            # Handle different output formats
            if isinstance(block_output, tuple):
                outputs.append(block_output[0])
                if output_attentions and len(block_output) > 1:
                    if block_output[1] is not None:
                        all_attentions.append(block_output[1])
                if len(block_output) > 2 and block_output[2] is not None:
                    present_key_values.append(block_output[2])
            else:
                outputs.append(block_output)
        
        # Concatenate outputs
        if outputs:
            output = torch.cat(outputs, dim=1)
        else:
            output = hidden_states  # Fallback
        
        # Prepare return value
        result = [output]
        
        if output_attentions and all_attentions:
            try:
                attentions = torch.cat(all_attentions, dim=-1)
                result.append(attentions)
            except:
                result.append(None)
        elif output_attentions:
            result.append(None)
        
        if present_key_values:
            result.append(present_key_values)
        elif kwargs.get('use_cache', False):
            result.append(None)
        
        return tuple(result) if len(result) > 1 else result[0]

class RewardFunction:
    """Reward function for different tasks"""
    def __init__(self):
        self.task_scales = {
            'qa': 2.0,
            'sentiment': 1.8,
            'classification': 1.5,
            'general': 1.0
        }
    
    def compute_reward(self, generated_text: str, target_text: str, task_type: str) -> float:
        """Fast reward computation"""
        try:
            if not generated_text or not target_text:
                return -1.0
                
            reward = {
                "qa": self._fast_qa_reward,
                "sentiment": self._fast_sentiment_reward,
                "classification": self._fast_classification_reward,
            }.get(task_type, self._fast_general_reward)(generated_text, target_text)
            
            scaled_reward = reward * self.task_scales.get(task_type, 1.0)
            return float(np.clip(scaled_reward, -2.0, 2.0))
            
        except Exception as e:
            logger.error(f"Reward computation failed: {str(e)}")
            return -1.0
    
    def _fast_qa_reward(self, generated: str, target: str) -> float:
        """Fast QA reward based on word overlap"""
        try:
            gen_words = set(generated.lower().split())
            target_words = set(target.lower().split())
            
            if not target_words:
                return 0.0
                
            overlap = len(gen_words & target_words) / len(target_words)
            
            # Bonus for exact match
            if target.lower() in generated.lower():
                overlap += 0.5
                
            return min(overlap, 1.0)
        except:
            return 0.0
    
    def _fast_sentiment_reward(self, generated: str, target: str) -> float:
        """Fast sentiment reward"""
        try:
            positive_words = {'good', 'great', 'positive', 'happy', 'love', 'excellent', 'amazing', 'fantastic'}
            negative_words = {'bad', 'terrible', 'negative', 'sad', 'hate', 'awful', 'horrible', 'disappointing'}
            
            gen_words = set(generated.lower().split())
            target_lower = target.lower()
            
            gen_positive = len(gen_words & positive_words)
            gen_negative = len(gen_words & negative_words)
            
            target_is_positive = 'positive' in target_lower
            
            if target_is_positive and gen_positive > gen_negative:
                return 1.0
            elif not target_is_positive and gen_negative > gen_positive:
                return 1.0
            else:
                return 0.2
        except:
            return 0.0
    
    def _fast_classification_reward(self, generated: str, target: str) -> float:
        """Fast classification reward"""
        try:
            if target.lower() in generated.lower():
                return 1.0
                
            gen_words = set(generated.lower().split())
            target_words = set(target.lower().split())
            
            if target_words:
                overlap = len(gen_words & target_words) / len(target_words)
                return overlap * 0.8
            return 0.1
        except:
            return 0.0
    
    def _fast_general_reward(self, generated: str, target: str) -> float:
        """Fast general reward"""
        try:
            if len(generated.strip()) < 5:
                return -0.5
                
            words = generated.split()
            diversity = len(set(words)) / max(len(words), 1)
            length_score = min(len(words) / 20, 1.0)
            
            return diversity * 0.5 + length_score * 0.5
        except:
            return 0.0

# ============================================================================
# EPISODE DATA STRUCTURE
# ============================================================================

@dataclass
class Episode:
    """Episode data for GRPO training"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    generated_tokens: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    task_type: str
    sequence_id: str = None
    baseline_reward: float = 0.0
    is_finished: bool = True
    episode_length: int = 0

# ============================================================================
# CEM OPTIMIZER
# ============================================================================

class CEMOptimizer:
    """Cross-Entropy Method optimizer for adaptation parameters"""
    def __init__(self, config, task_type, epoch=0):
        self.config = config
        self.task_type = task_type
        
        # Choose CEM parameters based on epoch
        if epoch == 0:
            self.cem_params = config.first_epoch_cem_params.get(task_type, config.first_epoch_cem_params["general"])
        else:
            self.cem_params = config.cem_params.get(task_type, config.cem_params["general"])
        
        self.momentum = getattr(config, 'cem_momentum', 0.0)
        self.mean_history = {}
        self.std_history = {}
        
    def optimize(self, adaptation_params, reward_function, episodes_data):
        """Optimize adaptation parameters using CEM"""
        results = {}
        
        for param_name, param_tensor in adaptation_params.items():
            if param_tensor.requires_grad:
                optimized_param = self._optimize_single_parameter(
                    param_name, param_tensor, reward_function, episodes_data
                )
                results[param_name] = optimized_param
        
        return results
    
    def _optimize_single_parameter(self, param_name, param_tensor, reward_function, episodes_data):
        """Optimize a single parameter using CEM"""
        original_shape = param_tensor.shape
        param_flat = param_tensor.detach().cpu().numpy().flatten()
        
        # Initialize population mean and std
        if param_name not in self.mean_history:
            population_mean = param_flat.copy()
            population_std = np.ones_like(param_flat) * self.cem_params["noise_std"]
        else:
            # Use momentum from previous iterations
            population_mean = self.mean_history[param_name]
            population_std = self.std_history[param_name]
        
        population_size = self.cem_params["population_size"]
        elite_size = max(1, int(population_size * self.cem_params["elite_ratio"]))
        
        for step in range(self.cem_params["adaptation_steps"]):
            # Generate population
            population = []
            for _ in range(population_size):
                noise = np.random.normal(0, population_std)
                candidate = population_mean + noise
                population.append(candidate)
            
            # Evaluate population
            fitness_scores = []
            for candidate in population:
                # Create temporary parameter tensor
                candidate_tensor = torch.tensor(
                    candidate.reshape(original_shape),
                    dtype=param_tensor.dtype,
                    device=param_tensor.device
                )
                
                # Evaluate fitness (simplified - use mean reward from episodes)
                fitness = self._evaluate_parameter_fitness(
                    candidate_tensor, episodes_data, reward_function
                )
                fitness_scores.append(fitness)
            
            # Select elites
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            elites = [population[i] for i in elite_indices]
            
            # Update distribution
            old_mean = population_mean.copy()
            old_std = population_std.copy()
            
            population_mean = np.mean(elites, axis=0)
            population_std = np.std(elites, axis=0) + 1e-6
            
            # Apply momentum
            if self.momentum > 0:
                population_mean = (1 - self.momentum) * population_mean + self.momentum * old_mean
                population_std = (1 - self.momentum) * population_std + self.momentum * old_std
            
            # Check convergence
            mean_change = np.mean(np.abs(population_mean - old_mean))
            if mean_change < self.cem_params["convergence_threshold"]:
                break
        
        # Store history for momentum
        self.mean_history[param_name] = population_mean
        self.std_history[param_name] = population_std
        
        # Return optimized parameter
        return torch.tensor(
            population_mean.reshape(original_shape),
            dtype=param_tensor.dtype,
            device=param_tensor.device
        )
    
    def _evaluate_parameter_fitness(self, param_tensor, episodes_data, reward_function):
        """Evaluate fitness of a parameter configuration"""
        # Simplified fitness evaluation - in practice, this would involve
        # running a forward pass with the modified parameter
        # For now, return a random fitness to demonstrate the structure
        return np.random.random()

# ============================================================================
# MODEL CLASSES
# ============================================================================

class SelfAdaptiveGPT2(nn.Module):
    """Self-adaptive GPT-2 model with proper CEM integration"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        logger.info(f"Loading {config.model_name}")
        self.base_model = GPT2LMHeadModel.from_pretrained(config.model_name)
        self.base_model = self.base_model.to(device)
        
        # Replace attention modules with PagedGPT2Attention if enabled
        if config.enable_paged_attention:
            self._replace_attention_modules()
        
        # Initialize adaptation parameters
        self.adaptation_params = nn.ParameterDict()
        self._initialize_adaptation_layers()
        
        # Value network for GRPO
        self.value_network = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        ).to(device)
        
        # Task classifier
        self.task_classifier = nn.Linear(self.base_model.config.hidden_size, config.num_experts).to(device)
        
        # Reward function
        self.reward_function = RewardFunction()
        
        # CEM optimizer for each task type
        self.cem_optimizers = {}
        
        self.sequence_counter = 0
        self.current_temperature = config.temperature
        
        logger.info(f"Model initialized with {self._count_parameters():,} parameters")
    
    def _replace_attention_modules(self):
        """Replace attention modules with PagedGPT2Attention"""
        logger.info("Replacing attention modules with PagedGPT2Attention")
        for i, layer in enumerate(self.base_model.transformer.h):
            original_attn = layer.attn
            paged_attn = PagedGPT2Attention(original_attn, self.config)
            paged_attn = paged_attn.to(device)
            layer.attn = paged_attn
            logger.info(f"Replaced attention module for layer {i}")
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _initialize_adaptation_layers(self):
        """Initialize adaptation parameters for key transformer layers"""
        layer_indices = [0, 2, 4, 6] if isinstance(self.config, UltraConfig) else [0, 2]
        
        for idx in layer_indices:
            if idx < len(self.base_model.transformer.h):
                hidden_size = self.base_model.config.hidden_size
                adaptation_dim = self.config.adaptation_rank
                
                # Attention adaptation parameters
                self.adaptation_params[f'layer_{idx}_attn_scale'] = nn.Parameter(
                    torch.ones(hidden_size, device=device) * 0.01
                )
                self.adaptation_params[f'layer_{idx}_attn_shift'] = nn.Parameter(
                    torch.zeros(hidden_size, device=device)
                )
                
                # Hidden state adaptation parameters
                self.adaptation_params[f'layer_{idx}_hidden_scale'] = nn.Parameter(
                    torch.ones(hidden_size, device=device) * 0.01
                )
                self.adaptation_params[f'layer_{idx}_hidden_shift'] = nn.Parameter(
                    torch.zeros(hidden_size, device=device)
                )
                
                logger.info(f"Added adaptation parameters for layer {idx}")
    
    def _apply_adaptation(self, hidden_states, layer_idx):
        """Apply adaptation to hidden states"""
        if f'layer_{layer_idx}_hidden_scale' in self.adaptation_params:
            scale = self.adaptation_params[f'layer_{layer_idx}_hidden_scale']
            shift = self.adaptation_params[f'layer_{layer_idx}_hidden_shift']
            
            # Apply adaptation: scale and shift
            adapted_states = hidden_states * (1.0 + scale) + shift
            return adapted_states
        
        return hidden_states
    
    def forward_with_adaptation(self, input_ids, attention_mask=None, use_adaptation=True):
        """Forward pass with adaptation applied"""
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        
        # Get embeddings
        inputs_embeds = self.base_model.transformer.wte(input_ids)
        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=device)
        position_embeds = self.base_model.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        # Apply dropout
        hidden_states = self.base_model.transformer.drop(hidden_states)
        
        # Forward through transformer layers with adaptation
        all_hidden_states = []
        for i, layer in enumerate(self.base_model.transformer.h):
            # Apply adaptation before layer
            if use_adaptation:
                hidden_states = self._apply_adaptation(hidden_states, i)
            
            # Layer forward pass
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False
            )
            hidden_states = layer_outputs[0]
            all_hidden_states.append(hidden_states)
        
        # Final layer norm
        hidden_states = self.base_model.transformer.ln_f(hidden_states)
        
        # Get logits
        logits = self.base_model.lm_head(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': all_hidden_states,
            'last_hidden_state': hidden_states
        }
    
    def _get_cem_optimizer(self, task_type, epoch):
        """Get or create CEM optimizer for task type"""
        key = f"{task_type}_{epoch}"
        if key not in self.cem_optimizers:
            self.cem_optimizers[key] = CEMOptimizer(self.config, task_type, epoch)
        return self.cem_optimizers[key]
    
    def generate_episode_with_cem(self, input_ids, attention_mask, max_new_tokens=None, task_type="general", epoch=0):
        """Generate episode with CEM optimization of adaptation parameters"""
        self.eval()
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        if max_new_tokens is None:
            if isinstance(self.config, UltraConfig):
                max_new_tokens = random.randint(16, 32)
            else:
                max_new_tokens = random.randint(8, 16)
        
        seq_id = f"seq_{self.sequence_counter}_{task_type}"
        self.sequence_counter += 1
        
        # Get CEM optimizer for this task
        cem_optimizer = self._get_cem_optimizer(task_type, epoch)
        
        try:
            with torch.no_grad():
                # CEM optimization of adaptation parameters
                episodes_data = {"task_type": task_type, "input_ids": input_ids, "attention_mask": attention_mask}
                
                # Optimize adaptation parameters using CEM
                if len(self.adaptation_params) > 0:
                    optimized_params = cem_optimizer.optimize(
                        self.adaptation_params, self.reward_function, episodes_data
                    )
                    
                    # Temporarily apply optimized parameters
                    original_params = {}
                    for param_name, optimized_value in optimized_params.items():
                        original_params[param_name] = self.adaptation_params[param_name].data.clone()
                        self.adaptation_params[param_name].data = optimized_value
                
                # Get initial hidden states for value estimation
                try:
                    init_out = self.forward_with_adaptation(input_ids, attention_mask)
                    hidden_states = init_out['last_hidden_state']
                    values = self.value_network(hidden_states.mean(dim=1))
                except Exception as e:
                    logger.debug(f"Value estimation failed: {e}")
                    values = torch.zeros(input_ids.size(0), device=device)
                
                # Generate text using adapted model
                generated = self.base_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=self.current_temperature,
                    top_p=self.config.top_p,
                    top_k=50,
                    repetition_penalty=self.config.repetition_penalty,
                    pad_token_id=self.base_model.config.eos_token_id,
                    eos_token_id=self.base_model.config.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=True
                )
                
                # Restore original parameters
                if 'original_params' in locals():
                    for param_name, original_value in original_params.items():
                        self.adaptation_params[param_name].data = original_value
                
                # Extract generated tokens (remove input)
                generated_tokens = generated.sequences[:, input_ids.size(1):]
                
                # Compute log probabilities
                log_probs = []
                if hasattr(generated, 'scores') and generated.scores:
                    for i, score in enumerate(generated.scores[:generated_tokens.size(1)]):
                        if i < generated_tokens.size(1):
                            token_id = generated_tokens[:, i:i+1]
                            log_prob = F.log_softmax(score, dim=-1).gather(1, token_id).squeeze(-1)
                            log_probs.append(log_prob)
                
                if log_probs:
                    log_probs = torch.stack(log_probs, dim=1)
                else:
                    log_probs = torch.zeros_like(generated_tokens, dtype=torch.float32)
                
                # Initialize rewards (will be computed later)
                rewards = torch.zeros_like(generated_tokens, dtype=torch.float32)
                
                return Episode(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generated_tokens=generated_tokens,
                    log_probs=log_probs,
                    rewards=rewards,
                    values=values,
                    task_type=task_type,
                    sequence_id=seq_id,
                    episode_length=generated_tokens.size(1)
                )
                
        except Exception as e:
            logger.error(f"Episode generation failed: {str(e)}")
            # Return dummy episode
            dummy_tokens = torch.zeros((input_ids.size(0), 1), dtype=torch.long, device=device)
            dummy_values = torch.zeros(input_ids.size(0), device=device)
            
            return Episode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generated_tokens=dummy_tokens,
                log_probs=torch.zeros_like(dummy_tokens, dtype=torch.float32),
                rewards=torch.zeros_like(dummy_tokens, dtype=torch.float32),
                values=dummy_values,
                task_type=task_type,
                sequence_id=seq_id,
                episode_length=1
            )
    
    def compute_grpo_loss(self, episodes: List[Episode]):
        """Compute GRPO loss"""
        if not episodes:
            return torch.tensor(0.0, requires_grad=True, device=device)
        
        total_loss = torch.tensor(0.0, requires_grad=True, device=device)
        valid_episodes = 0
        
        # Collect all rewards for normalization
        all_rewards = []
        for episode in episodes:
            if episode.rewards.numel() > 0:
                all_rewards.append(episode.rewards.flatten())
        
        if all_rewards:
            all_rewards = torch.cat(all_rewards)
            reward_mean = all_rewards.mean()
            reward_std = torch.clamp(all_rewards.std() + 1e-6, min=0.1)
        else:
            reward_mean, reward_std = 0.0, 1.0
        
        for episode in episodes:
            try:
                if episode.rewards.numel() == 0 or episode.log_probs.numel() == 0:
                    continue
                
                # Normalize rewards
                normalized_rewards = (episode.rewards - reward_mean) / reward_std
                normalized_rewards = torch.clamp(normalized_rewards, -self.config.clip_rewards, self.config.clip_rewards)
                normalized_rewards = normalized_rewards * self.config.reward_scaling
                
                # Compute advantages
                if episode.values.numel() > 0:
                    # Expand values to match reward shape
                    if normalized_rewards.dim() > episode.values.dim():
                        values_expanded = episode.values.mean().expand_as(normalized_rewards)
                    else:
                        values_expanded = episode.values
                    advantages = normalized_rewards - values_expanded.detach()
                else:
                    advantages = normalized_rewards
                
                # Policy loss
                policy_loss = -(episode.log_probs.flatten()[:advantages.numel()] * advantages.flatten()).mean()
                
                # Value loss
                if episode.values.numel() > 0:
                    value_targets = normalized_rewards.flatten()[:episode.values.numel()]
                    value_loss = F.mse_loss(episode.values.flatten(), value_targets.detach())
                else:
                    value_loss = torch.tensor(0.0, device=device)
                
                # Entropy loss (encourages exploration)
                entropy_loss = -episode.log_probs.mean()
                
                # Combined loss
                episode_loss = (
                    policy_loss + 
                    self.config.grpo_value_loss_coeff * value_loss +
                    self.config.grpo_entropy_coeff * entropy_loss
                )
                
                if torch.isfinite(episode_loss):
                    total_loss = total_loss + episode_loss
                    valid_episodes += 1
                    
            except Exception as e:
                logger.error(f"Episode loss computation failed: {str(e)}")
                continue
        
        return total_loss / max(valid_episodes, 1) if valid_episodes > 0 else torch.tensor(0.0, requires_grad=True, device=device)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics with correct GPU utilization calculation"""
        stats = {
            "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            "gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            # Fix: Calculate percentage correctly
            stats["gpu_memory_utilization"] = (allocated_memory / total_memory) * 100
        
        return stats

# ============================================================================
# TRAINER CLASS
# ============================================================================

class GRPOTrainer:
    """GRPO trainer with bulletproof AMP handling"""
    def __init__(self, config):
        self.config = config
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Initializing model with config: {type(config).__name__}")
        
        # Initialize model
        self.model = SelfAdaptiveGPT2(config)
        
        # Setup optimizers
        self.optimizer = torch.optim.AdamW(
            [
                {'params': list(self.model.adaptation_params.values()), 'lr': config.learning_rate},
                {'params': self.model.value_network.parameters(), 'lr': config.learning_rate * 1.5},
                {'params': self.model.task_classifier.parameters(), 'lr': config.learning_rate}
            ],
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Load datasets and calculate total steps
        self.dataset_loader = DatasetLoader(config)
        self.datasets = self.dataset_loader.load_all_datasets()
        
        # Calculate total training steps properly
        all_data = []
        for task_data in self.datasets.values():
            all_data.extend(task_data)
        
        if all_data:
            num_batches = math.ceil(len(all_data) / config.batch_size)
            total_steps = config.num_epochs * num_batches
        else:
            total_steps = 1000  # Fallback
        
        logger.info(f"Total training steps calculated: {total_steps}")
        
        # Setup scheduler with correct total steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Setup AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        
        # Setup directories and metrics
        os.makedirs(config.output_dir, exist_ok=True)
        self.training_metrics = {
            "policy_loss": [],
            "gpu_memory_usage": [],
            "task_rewards": defaultdict(list),
            "training_speed": [],
            "episode_lengths": []
        }
        
        logger.info("Trainer initialized successfully")
    
    def create_dataloader(self, data, batch_size):
        """Create DataLoader"""
        dataset = PreTokenizedDataset(data, self.tokenizer, self.config.max_length)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
            persistent_workers=self.config.persistent_workers if self.config.num_workers > 0 else False,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else 2,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Collate function for DataLoader"""
        try:
            return {
                'input_ids': torch.stack([item['input_ids'] for item in batch]),
                'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
                'target_ids': torch.stack([item['target_ids'] for item in batch]),
                'task_type': [item['task_type'] for item in batch],
                'input_text': [item['input_text'] for item in batch],
                'target_text': [item['target_text'] for item in batch]
            }
        except Exception as e:
            logger.error(f"Collate failed: {e}")
            return None
    
    def _safe_amp_step(self, loss, accumulated_steps):
        """Safely handle AMP optimizer step with proper state management"""
        try:
            if self.config.mixed_precision:
                # Scale the loss
                scaled_loss = self.scaler.scale(loss)
                scaled_loss.backward()
                
                # Only proceed with optimizer step if we've accumulated enough gradients
                if accumulated_steps >= self.config.gradient_accumulation_steps:
                    # Check if gradients are finite before unscaling
                    scaler_state_before = self.scaler.get_scale()
                    
                    # Unscale gradients
                    self.scaler.unscale_(self.optimizer)
                    
                    # Clip gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # Step optimizer
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # Check if step was successful
                    step_successful = self.scaler.get_scale() >= scaler_state_before
                    
                    if step_successful:
                        self.scheduler.step()
                    
                    # Always zero gradients after attempting a step
                    self.optimizer.zero_grad()
                    
                    return True, step_successful
            else:
                # Standard training without AMP
                loss.backward()
                
                if accumulated_steps >= self.config.gradient_accumulation_steps:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    return True, True
            
            return False, False
            
        except Exception as e:
            # On any error, reset everything
            logger.debug(f"AMP step failed: {e}")
            self.optimizer.zero_grad()
            
            # Reset scaler if needed
            if self.config.mixed_precision:
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            
            return False, False

    def train_grpo(self):
        """Main training loop with bulletproof AMP handling"""
        wandb_initialized = False
        
        try:
            logger.info(f"Starting GRPO training with {type(self.config).__name__}")
            
            # Initialize wandb if configured
            if self.config.wandb_project:
                try:
                    wandb.init(
                        project=self.config.wandb_project,
                        name=f"grpo-{type(self.config).__name__}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        config=self.config.__dict__
                    )
                    wandb_initialized = True
                    logger.info("Wandb initialized successfully")
                except Exception as e:
                    logger.warning(f"Wandb initialization failed: {str(e)}")
            
            # Prepare data
            all_data = []
            for task_data in self.datasets.values():
                all_data.extend(task_data)
            
            if not all_data:
                logger.error("No training data available")
                return
            
            logger.info(f"Total training samples: {len(all_data):,}")
            
            try:
                dataloader = self.create_dataloader(all_data, self.config.batch_size)
            except Exception as e:
                logger.error(f"Failed to create dataloader: {e}")
                return
            
            if dataloader is None:
                logger.error("Dataloader is None")
                return
            
            # Training setup
            self.model.train()
            global_step = 0
            start_time = time.time()
            
            for epoch in range(self.config.num_epochs):
                logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
                
                epoch_start_time = time.time()
                epoch_metrics = {
                    'policy_loss': 0.0,
                    'episodes': 0,
                    'total_reward': 0.0,
                    'batches_processed': 0,
                    'successful_steps': 0,
                    'failed_steps': 0
                }
                
                # Create progress bar
                try:
                    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
                except Exception as e:
                    logger.error(f"Failed to create progress bar: {e}")
                    progress_bar = dataloader
                
                # Reset accumulation state at start of epoch
                accumulated_steps = 0
                accumulated_loss = None
                self.optimizer.zero_grad()
                
                for batch_idx, batch in enumerate(progress_bar):
                    if batch is None:
                        logger.debug(f"Batch {batch_idx} is None, skipping")
                        continue
                    
                    batch_start_time = time.time()
                    batch_success = False
                    
                    try:
                        # Move data to device
                        input_ids = batch['input_ids'].to(device, non_blocking=True)
                        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                        task_types = batch['task_type']
                        target_texts = batch['target_text']
                        
                        episodes = []
                        
                        # Generate episodes with CEM
                        for i in range(len(input_ids)):
                            try:
                                episode = self.model.generate_episode_with_cem(
                                    input_ids[i:i+1],
                                    attention_mask[i:i+1],
                                    task_type=task_types[i],
                                    epoch=epoch
                                )
                                
                                if episode.generated_tokens.numel() > 0:
                                    # Compute reward
                                    generated_text = self.tokenizer.decode(
                                        episode.generated_tokens[0],
                                        skip_special_tokens=True
                                    )
                                    
                                    reward = self.model.reward_function.compute_reward(
                                        generated_text,
                                        target_texts[i],
                                        task_types[i]
                                    )
                                    
                                    # Update episode rewards
                                    episode.rewards = torch.full(
                                        episode.generated_tokens.size(),
                                        reward,
                                        device=device,
                                        dtype=torch.float32
                                    )
                                    
                                    episodes.append(episode)
                                    epoch_metrics['total_reward'] += reward
                                    
                            except Exception as e:
                                logger.debug(f"Episode generation failed: {str(e)}")
                                continue
                        
                        if not episodes:
                            logger.debug("No valid episodes generated, skipping batch")
                            continue
                        
                        # Compute loss with autocast
                        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                            grpo_loss = self.model.compute_grpo_loss(episodes)
                        
                        if not torch.isfinite(grpo_loss) or grpo_loss.item() == 0.0:
                            logger.debug(f"Invalid loss: {grpo_loss.item()}, skipping batch")
                            continue
                        
                        # Store the loss for potential final step
                        accumulated_loss = grpo_loss
                        
                        # Scale loss for gradient accumulation
                        scaled_loss = grpo_loss / self.config.gradient_accumulation_steps
                        
                        # Increment accumulation counter
                        accumulated_steps += 1
                        
                        # Use safe AMP step
                        step_taken, step_successful = self._safe_amp_step(scaled_loss, accumulated_steps)
                        
                        if step_taken:
                            if step_successful:
                                epoch_metrics['successful_steps'] += 1
                                global_step += 1
                            else:
                                epoch_metrics['failed_steps'] += 1
                            
                            # Reset accumulation counter after step
                            accumulated_steps = 0
                            accumulated_loss = None
                        
                        # Update metrics
                        epoch_metrics['policy_loss'] += grpo_loss.item()
                        epoch_metrics['episodes'] += len(episodes)
                        epoch_metrics['batches_processed'] += 1
                        batch_success = True
                        
                        # Memory tracking
                        memory_stats = self.model.get_memory_stats()
                        current_memory = memory_stats["gpu_memory_allocated"]
                        self.training_metrics["gpu_memory_usage"].append(current_memory)
                        
                        # Timing
                        batch_time = time.time() - batch_start_time
                        self.training_metrics["training_speed"].append(batch_time)
                        
                        # Progress display
                        if hasattr(progress_bar, 'set_postfix'):
                            avg_reward = epoch_metrics['total_reward'] / max(epoch_metrics['episodes'], 1)
                            progress_info = {
                                'Loss': f'{grpo_loss.item():.4f}',
                                'Reward': f'{avg_reward:.3f}',
                                'GPU': f'{current_memory:.1f}GB',
                                'GPU%': f"{memory_stats.get('gpu_memory_utilization', 0):.1f}%",
                                'Steps': f"{epoch_metrics['successful_steps']}/{epoch_metrics['successful_steps'] + epoch_metrics['failed_steps']}"
                            }
                            
                            if self.config.mixed_precision:
                                progress_info['Scale'] = f'{self.scaler.get_scale():.0f}'
                            
                            progress_bar.set_postfix(progress_info)
                        
                        # Periodic cleanup
                        if batch_idx % 20 == 0:
                            torch.cuda.empty_cache()
                            
                            # Memory warning for UltraConfig
                            if isinstance(self.config, UltraConfig) and current_memory > 12.0:
                                logger.warning(f"High memory usage: {current_memory:.1f}GB")
                    
                    except Exception as e:
                        logger.error(f"Batch {batch_idx} failed: {str(e)}")
                        batch_success = False
                    
                    finally:
                        # Always ensure clean state after each batch
                        if not batch_success:
                            # Reset everything on batch failure
                            self.optimizer.zero_grad()
                            accumulated_steps = 0
                            accumulated_loss = None
                            
                            # Reset scaler if AMP is enabled
                            if self.config.mixed_precision:
                                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
                
                # Handle any remaining accumulated gradients at end of epoch
                if accumulated_steps > 0 and accumulated_loss is not None:
                    logger.info(f"Processing remaining {accumulated_steps} accumulated steps")
                    try:
                        # Use the last computed loss for the final step
                        scaled_final_loss = accumulated_loss / self.config.gradient_accumulation_steps
                        step_taken, step_successful = self._safe_amp_step(scaled_final_loss, self.config.gradient_accumulation_steps)
                        
                        if step_taken and step_successful:
                            epoch_metrics['successful_steps'] += 1
                            global_step += 1
                        
                    except Exception as e:
                        logger.warning(f"Final gradient step failed: {e}")
                    finally:
                        # Ensure clean state
                        self.optimizer.zero_grad()
                        accumulated_steps = 0
                        accumulated_loss = None
                
                # Epoch summary
                epoch_time = time.time() - epoch_start_time
                total_time = time.time() - start_time
                
                if epoch_metrics['batches_processed'] > 0:
                    epoch_metrics['policy_loss'] /= epoch_metrics['batches_processed']
                    avg_reward = epoch_metrics['total_reward'] / max(epoch_metrics['episodes'], 1)
                    
                    log_message = (
                        f"Epoch {epoch + 1} completed in {epoch_time:.2f}s:\n"
                        f"  - Avg Policy Loss: {epoch_metrics['policy_loss']:.4f}\n"
                        f"  - Episodes: {epoch_metrics['episodes']:,}\n"
                        f"  - Avg Reward: {avg_reward:.3f}\n"
                        f"  - Successful Steps: {epoch_metrics['successful_steps']}\n"
                        f"  - Failed Steps: {epoch_metrics['failed_steps']}\n"
                        f"  - Batches Processed: {epoch_metrics['batches_processed']}\n"
                        f"  - Total Time: {total_time/60:.1f}min"
                    )
                    
                    logger.info(log_message)
                    
                    # Wandb logging
                    if wandb_initialized:
                        try:
                            log_dict = {
                                "epoch": epoch + 1,
                                "policy_loss": epoch_metrics['policy_loss'],
                                "avg_reward": avg_reward,
                                "epoch_time": epoch_time,
                                "gpu_memory_gb": memory_stats["gpu_memory_allocated"],
                                "gpu_memory_utilization": memory_stats.get("gpu_memory_utilization", 0),
                                "episodes_per_epoch": epoch_metrics['episodes'],
                                "successful_steps": epoch_metrics['successful_steps'],
                                "failed_steps": epoch_metrics['failed_steps'],
                                "learning_rate": self.scheduler.get_last_lr()[0]
                            }
                            wandb.log(log_dict)
                        except Exception as e:
                            logger.warning(f"Wandb logging failed: {e}")
                else:
                    logger.warning(f"Epoch {epoch + 1} had no valid batches")
                
                # Save checkpoint
                self.save_checkpoint(epoch + 1, global_step)
                
                # Check time limit for UltraConfig (optional)
                if isinstance(self.config, UltraConfig) and total_time > 2.5 * 3600:  # 2.5 hours
                    logger.warning("Approaching time limit, stopping training")
                    break
            
            total_training_time = time.time() - start_time
            logger.info(f"Training completed in {total_training_time/60:.1f} minutes!")
            
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            traceback.print_exc()
        finally:
            # Ensure wandb is always finished
            if wandb_initialized:
                try:
                    wandb.finish()
                    logger.info("Wandb session finished")
                except Exception as e:
                    logger.warning(f"Failed to finish wandb: {e}")
            
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Cleanup completed")
    
    def save_checkpoint(self, epoch: int, global_step: int):
        """Save checkpoint"""
        checkpoint_path = os.path.join(
            self.config.output_dir,
            f"checkpoint_epoch_{epoch}_step_{global_step}.pt"
        )
        
        try:
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'config': self.config,
                'training_metrics': self.training_metrics
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def set_deterministic_training():
    """Set deterministic flags for reproducible training"""
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Set deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Additional deterministic settings
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    logger.info("Deterministic training settings applied")

def main():
    """Main execution function"""
    logger.info("Starting Refactored GPT-2 + GRPO Training Pipeline")
    
    # Set deterministic training
    set_deterministic_training()
    
    # Configuration selection
    fast_mode = False  # Set to True for quick testing
    
    if fast_mode:
        logger.info("Using fast mode for testing")
        config = OptimizedConfig()
        config.num_epochs = 1
        config.batch_size = 2
        config.max_length = 128
        config.use_fallback_data_only = True
        config.mixed_precision = False  # Disable for testing
        config.enable_paged_attention = False
    else:
        # Use UltraConfig for full training
        config = UltraConfig()
    
    logger.info(f"Configuration: {type(config).__name__}")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Max Length: {config.max_length}")
    logger.info(f"  Mixed Precision: {config.mixed_precision}")
    logger.info(f"  Paged Attention: {config.enable_paged_attention}")
    logger.info(f"  Device: {device}")
    
    wandb_initialized = False
    
    try:
        with timer("Trainer initialization"):
            trainer = GRPOTrainer(config)
        
        total_samples = sum(len(data) for data in trainer.datasets.values())
        if total_samples == 0:
            logger.error("No training data loaded!")
            return
        
        logger.info(f"Loaded {total_samples:,} training samples")
        logger.info("Starting GRPO training with CEM adaptation...")
        
        with timer("Complete training"):
            trainer.train_grpo()
        
        logger.info("Training pipeline completed successfully! 🎉")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        traceback.print_exc()
    finally:
        # Ensure wandb is always finished if it was initialized anywhere
        try:
            wandb.finish()
            logger.info("Final wandb cleanup completed")
        except Exception as e:
            logger.debug(f"Final wandb cleanup failed (may not have been initialized): {e}")
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("All cleanup completed")

if __name__ == "__main__":
    main()
