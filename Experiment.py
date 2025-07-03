import subprocess
import sys
import warnings
import logging
import traceback
import time
import requests
from contextlib import contextmanager
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
import threading
import concurrent.futures
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

@contextmanager
def timer(description: str):
    """Context manager for timing operations"""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{description}: {elapsed:.2f}s")

def setup_device():
    """Set up the computation device (GPU/CPU)"""
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
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
    mixed_precision: bool = True  # Enable for better performance
    batch_size: int = 4  # Reduced for stability
    cem_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "qa": {
            "population_size": 10,  # Reduced for faster convergence
            "elite_ratio": 0.3,
            "noise_std": 0.2,
            "adaptation_steps": 8,  # Reduced
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
    batch_size: int = 8  # Reduced for stability
    learning_rate: float = 3e-5  # Reduced for stability
    num_epochs: int = 5
    max_length: int = 384  # Reduced from 512
    adaptation_rank: int = 24  # Reduced from 32
    num_experts: int = 6  # Reduced from 8
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 200
    enable_paged_attention: bool = True
    paged_block_size: int = 32
    max_cache_blocks: int = 1500  # Reduced from 2000
    max_samples_per_dataset: int = 1000  # Reduced from 2000
    grpo_episodes_per_batch: int = 6  # Reduced from 8
    grpo_kl_coeff: float = 0.02
    grpo_value_loss_coeff: float = 0.2
    grpo_entropy_coeff: float = 0.1
    svd_rank_ratio: float = 0.85  # Reduced from 0.9
    wandb_project: str = "ultra-grpo-cem-gpt2-15GB"
    output_dir: str = "./ultra_results"
    log_interval: int = 5
    clip_rewards: float = 3.0
    reward_scaling: float = 0.5
    repetition_penalty: float = 1.3
    top_p: float = 0.9
    temperature: float = 0.8
    min_episode_length: int = 32
    max_episode_length: int = 80  # Reduced from 96
    num_workers: int = min(8, os.cpu_count() or 4)  # Reduced from 16
    prefetch_factor: int = 8
    cem_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "qa": {
            "population_size": 32,  # Reduced from 64
            "elite_ratio": 0.25,
            "noise_std": 0.25,
            "adaptation_steps": 15,  # Reduced from 25
            "convergence_threshold": 0.003
        },
        "sentiment": {
            "population_size": 40,  # Reduced from 80
            "elite_ratio": 0.25,
            "noise_std": 0.3,
            "adaptation_steps": 18,  # Reduced from 30
            "convergence_threshold": 0.003
        },
        "classification": {
            "population_size": 50,  # Reduced from 100
            "elite_ratio": 0.3,
            "noise_std": 0.35,
            "adaptation_steps": 20,  # Reduced from 35
            "convergence_threshold": 0.002
        },
        "general": {
            "population_size": 32,  # Reduced from 64
            "elite_ratio": 0.25,
            "noise_std": 0.3,
            "adaptation_steps": 15,  # Reduced from 25
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
        batch_size = 50  # Reduced batch size for memory efficiency
        
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
# MODEL CLASSES
# ============================================================================

class SelfAdaptiveGPT2(nn.Module):
    """Simplified Self-adaptive GPT-2 model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        logger.info(f"Loading {config.model_name}")
        self.base_model = GPT2LMHeadModel.from_pretrained(config.model_name)
        self.base_model = self.base_model.to(device)
        
        # Simplified adaptation parameters
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
        
        self.sequence_counter = 0
        self.current_temperature = config.temperature
        
        logger.info(f"Model initialized with {self._count_parameters():,} parameters")
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _initialize_adaptation_layers(self):
        """Initialize simple adaptation layers"""
        # Add adaptation parameters for key transformer layers
        layer_indices = [0, 2, 4, 6] if isinstance(self.config, UltraConfig) else [0, 2]
        
        for idx in layer_indices:
            if idx < len(self.base_model.transformer.h):
                # Create adaptation parameters for attention
                hidden_size = self.base_model.config.hidden_size
                adaptation_dim = self.config.adaptation_rank
                
                self.adaptation_params[f'layer_{idx}_attn'] = nn.Parameter(
                    torch.randn(hidden_size, adaptation_dim, device=device) * 0.01
                )
                
                logger.info(f"Added adaptation for layer {idx}")
    
    def forward_with_adaptation(self, input_ids, attention_mask=None, use_adaptation=True):
        """Forward pass with optional adaptation"""
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        
        # For simplicity, just use the base model
        # In a full implementation, you would apply adaptation here
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        return outputs
    
    def generate_episode(self, input_ids, attention_mask, max_new_tokens=None, task_type="general", epoch=0):
        """Generate an episode for GRPO training"""
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
        
        try:
            with torch.no_grad():
                # Get initial hidden states for value estimation
                try:
                    init_out = self.forward_with_adaptation(input_ids, attention_mask)
                    hidden_states = init_out.hidden_states[-1]
                    values = self.value_network(hidden_states.mean(dim=1))
                except Exception as e:
                    logger.debug(f"Value estimation failed: {e}")
                    values = torch.zeros(input_ids.size(0), device=device)
                
                # Generate text
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
        """Get memory statistics"""
        stats = {
            "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            "gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
        }
        
        if isinstance(self.config, UltraConfig) and torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            stats["gpu_memory_utilization"] = (torch.cuda.memory_allocated() / (total_memory * 1e9)) * 100
        
        return stats

# ============================================================================
# TRAINER CLASS
# ============================================================================

class GRPOTrainer:
    """GRPO trainer with proper AMP handling"""
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
        
        # Setup scheduler
        total_steps = config.num_epochs * 100  # Rough estimate
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Setup AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        
        # Load datasets
        self.dataset_loader = DatasetLoader(config)
        self.datasets = self.dataset_loader.load_all_datasets()
        
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
    
    def train_grpo(self):
        """Main training loop with proper AMP handling"""
        logger.info(f"Starting GRPO training with {type(self.config).__name__}")
        
        # Initialize wandb if configured
        if self.config.wandb_project:
            try:
                wandb.init(
                    project=self.config.wandb_project,
                    name=f"grpo-{type(self.config).__name__}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    config=self.config.__dict__
                )
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
                'batches_processed': 0
            }
            
            # Create progress bar
            try:
                progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            except Exception as e:
                logger.error(f"Failed to create progress bar: {e}")
                progress_bar = dataloader
            
            accumulated_steps = 0
            
            for batch_idx, batch in enumerate(progress_bar):
                if batch is None:
                    logger.debug(f"Batch {batch_idx} is None, skipping")
                    continue
                
                batch_start_time = time.time()
                
                try:
                    # Move data to device
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                    task_types = batch['task_type']
                    target_texts = batch['target_text']
                    
                    episodes = []
                    
                    # Generate episodes
                    for i in range(len(input_ids)):
                        try:
                            episode = self.model.generate_episode(
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
                    
                    # Scale loss for gradient accumulation
                    scaled_loss = grpo_loss / self.config.gradient_accumulation_steps
                    
                    # Backward pass
                    if self.config.mixed_precision:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    
                    accumulated_steps += 1
                    
                    # Optimizer step when accumulation is complete
                    if accumulated_steps >= self.config.gradient_accumulation_steps:
                        if self.config.mixed_precision:
                            # Unscale gradients and clip
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm
                            )
                            
                            # Optimizer step
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            # Standard training without AMP
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm
                            )
                            self.optimizer.step()
                        
                        # Update learning rate
                        self.scheduler.step()
                        
                        # Zero gradients
                        self.optimizer.zero_grad()
                        
                        accumulated_steps = 0
                        global_step += 1
                    
                    # Update metrics
                    epoch_metrics['policy_loss'] += grpo_loss.item()
                    epoch_metrics['episodes'] += len(episodes)
                    epoch_metrics['batches_processed'] += 1
                    
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
                            'Time': f'{batch_time:.2f}s'
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
                    # Reset gradients on error
                    self.optimizer.zero_grad()
                    accumulated_steps = 0
                    continue
            
            # Handle remaining accumulated gradients
            if accumulated_steps > 0:
                try:
                    if self.config.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    
                except Exception as e:
                    logger.error(f"Final optimizer step failed: {e}")
                finally:
                    self.optimizer.zero_grad()
            
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
                    f"  - Batches Processed: {epoch_metrics['batches_processed']}\n"
                    f"  - Total Time: {total_time/60:.1f}min"
                )
                
                logger.info(log_message)
                
                # Wandb logging
                if self.config.wandb_project:
                    try:
                        log_dict = {
                            "epoch": epoch + 1,
                            "policy_loss": epoch_metrics['policy_loss'],
                            "avg_reward": avg_reward,
                            "epoch_time": epoch_time,
                            "gpu_memory_gb": memory_stats["gpu_memory_allocated"],
                            "episodes_per_epoch": epoch_metrics['episodes'],
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

def main():
    """Main execution function"""
    logger.info("Starting Fixed GPT-2 Training Pipeline")
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Configuration selection
    # Set to True for quick testing
    fast_mode = False
    
    if fast_mode:
        logger.info("Using fast mode for testing")
        config = OptimizedConfig()
        config.num_epochs = 1
        config.batch_size = 2
        config.max_length = 128
        config.use_fallback_data_only = True
        config.mixed_precision = False  # Disable for testing
    else:
        # Use UltraConfig for full training
        config = UltraConfig()
    
    logger.info(f"Configuration: {type(config).__name__}")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Max Length: {config.max_length}")
    logger.info(f"  Mixed Precision: {config.mixed_precision}")
    logger.info(f"  Device: {device}")
    
    try:
        with timer("Trainer initialization"):
            trainer = GRPOTrainer(config)
        
        total_samples = sum(len(data) for data in trainer.datasets.values())
        if total_samples == 0:
            logger.error("No training data loaded!")
            return
        
        logger.info(f"Loaded {total_samples:,} training samples")
        logger.info("Starting GRPO training...")
        
        with timer("Complete training"):
            trainer.train_grpo()
        
        logger.info("Training pipeline completed successfully! 🎉")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        traceback.print_exc()
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Close wandb
        try:
            wandb.finish()
        except:
            pass
        
        logger.info("Cleanup completed")

if __name__ == "__main__":
    main()
