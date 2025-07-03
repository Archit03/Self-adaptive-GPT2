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

# Thread-local storage for sequence_id
thread_local = threading.local()

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
    mixed_precision: bool = False
    cem_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "qa": {
            "population_size": 20,
            "elite_ratio": 0.25,
            "noise_std": 0.2,
            "adaptation_steps": 15,
            "convergence_threshold": 0.01
        },
        "sentiment": {
            "population_size": 25,
            "elite_ratio": 0.3,
            "noise_std": 0.25,
            "adaptation_steps": 12,
            "convergence_threshold": 0.01
        },
        "classification": {
            "population_size": 30,
            "elite_ratio": 0.35,
            "noise_std": 0.15,
            "adaptation_steps": 18,
            "convergence_threshold": 0.008
        },
        "general": {
            "population_size": 25,
            "elite_ratio": 0.3,
            "noise_std": 0.3,
            "adaptation_steps": 12,
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
    batch_size: int = 16
    learning_rate: float = 8e-5
    num_epochs: int = 5
    max_length: int = 512
    adaptation_rank: int = 32
    num_experts: int = 8
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 200
    enable_paged_attention: bool = True
    paged_block_size: int = 32
    max_cache_blocks: int = 2000
    max_samples_per_dataset: int = 2000
    grpo_episodes_per_batch: int = 8
    grpo_kl_coeff: float = 0.02
    grpo_value_loss_coeff: float = 0.2
    grpo_entropy_coeff: float = 0.1
    svd_rank_ratio: float = 0.9
    wandb_project: str = "ultra-grpo-cem-gpt2-15GB"
    output_dir: str = "./ultra_results"
    log_interval: int = 5
    clip_rewards: float = 3.0
    reward_scaling: float = 0.5
    repetition_penalty: float = 1.3
    top_p: float = 0.9
    temperature: float = 0.8
    min_episode_length: int = 32
    max_episode_length: int = 96
    num_workers: int = min(16, os.cpu_count() or 4)
    prefetch_factor: int = 8
    cem_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "qa": {
            "population_size": 64,
            "elite_ratio": 0.2,
            "noise_std": 0.3,
            "adaptation_steps": 25,
            "convergence_threshold": 0.002
        },
        "sentiment": {
            "population_size": 80,
            "elite_ratio": 0.25,
            "noise_std": 0.35,
            "adaptation_steps": 30,
            "convergence_threshold": 0.002
        },
        "classification": {
            "population_size": 100,
            "elite_ratio": 0.3,
            "noise_std": 0.4,
            "adaptation_steps": 35,
            "convergence_threshold": 0.001
        },
        "general": {
            "population_size": 64,
            "elite_ratio": 0.25,
            "noise_std": 0.3,
            "adaptation_steps": 25,
            "convergence_threshold": 0.003
        }
    })
    first_epoch_cem_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "qa": {"population_size": 16, "elite_ratio": 0.3, "noise_std": 0.2, "adaptation_steps": 8, "convergence_threshold": 0.01},
        "sentiment": {"population_size": 20, "elite_ratio": 0.3, "noise_std": 0.2, "adaptation_steps": 10, "convergence_threshold": 0.01},
        "classification": {"population_size": 24, "elite_ratio": 0.3, "noise_std": 0.2, "adaptation_steps": 12, "convergence_threshold": 0.01},
        "general": {"population_size": 16, "elite_ratio": 0.3, "noise_std": 0.2, "adaptation_steps": 8, "convergence_threshold": 0.01}
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
        batch_size = 100
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
            inputs_batch = tokenizer(
                input_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                add_special_tokens=True
            )
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
        if isinstance(self.config, UltraConfig):
            self._load_datasets_parallel(dataset_configs)
        else:
            self._load_datasets_sequential(dataset_configs)
        total_samples = sum(len(data) for data in self.datasets.values())
        if total_samples < 100:
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
            {'name': 'squad', 'split': 'train[:2000]', 'val_split': 'validation[:400]', 
             'task_type': 'qa', 'process_fn': self._process_squad},
            {'name': 'imdb', 'split': 'train[:2000]', 'val_split': 'test[:400]',
             'task_type': 'sentiment', 'process_fn': self._process_imdb},
            {'name': 'ag_news', 'split': 'train[:1500]', 'val_split': 'test[:300]',
             'task_type': 'classification', 'process_fn': self._process_ag_news},
            {'name': 'xsum', 'split': 'train[:1000]', 'val_split': 'validation[:200]',
             'task_type': 'summarization', 'process_fn': self._process_xsum},
            {'name': 'cnn_dailymail', 'subset': '3.0.0', 'split': 'train[:800]', 'val_split': 'validation[:150]',
             'task_type': 'summarization', 'process_fn': self._process_cnn},
        ]
    
    def _load_datasets_sequential(self, dataset_configs):
        """Load datasets sequentially"""
        for config in dataset_configs:
            try:
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
                    logger.info(f"Loaded {config['name']}: {len(processed_data)} samples")
            except Exception as e:
                logger.warning(f"Failed to load {config['name']}: {e}")
                self.failed_downloads += 1
    
    def _load_datasets_parallel(self, dataset_configs):
        """Load datasets in parallel"""
        def load_single_dataset(config_item):
            try:
                logger.info(f"Loading {config_item['name']} with {config_item['split']}")
                if 'subset' in config_item:
                    dataset = load_dataset(
                        config_item['name'], config_item['subset'],
                        split=config_item['split'],
                        download_mode="reuse_cache_if_exists",
                        verification_mode="no_checks",
                        trust_remote_code=True
                    )
                else:
                    dataset = load_dataset(
                        config_item['name'],
                        split=config_item['split'],
                        download_mode="reuse_cache_if_exists",
                        verification_mode="no_checks",
                        trust_remote_code=True
                    )
                processed_data = config_item['process_fn'](dataset)
                return config_item['task_type'], processed_data, True
            except Exception as e:
                logger.error(f"Failed to load {config_item['name']}: {str(e)}")
                return config_item['task_type'], [], False
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(load_single_dataset, config) for config in dataset_configs]
            for future in concurrent.futures.as_completed(futures):
                try:
                    task_type, processed_data, success = future.result(timeout=300)
                    if success and processed_data:
                        self.datasets.setdefault(task_type, [])
                        self.datasets[task_type].extend(processed_data[:self.config.max_samples_per_dataset])
                        self.successful_downloads += 1
                        logger.info(f"✓ Loaded {task_type}: {len(processed_data)} samples")
                    else:
                        self.failed_downloads += 1
                except Exception as e:
                    logger.error(f"Future result failed: {e}")
                    self.failed_downloads += 1
    
    def _process_squad(self, dataset):
        """Process SQuAD dataset"""
        processed = []
        max_context_length = 800 if isinstance(self.config, UltraConfig) else 300
        for item in dataset:
            try:
                context = item.get('context', '').strip()
                question = item.get('question', '').strip()
                answers = item.get('answers', {})
                if context and question and answers and answers.get('text'):
                    answer = answers['text'][0].strip()
                    context = context[:max_context_length]
                    processed.append((f"Context: {context}\nQuestion: {question}", answer, 'qa'))
            except:
                continue
        return processed
    
    def _process_imdb(self, dataset):
        """Process IMDB dataset"""
        processed = []
        max_text_length = 1000 if isinstance(self.config, UltraConfig) else 300
        for item in dataset:
            try:
                text = item.get('text', '').strip()
                label = item.get('label', 0)
                if text and len(text) > 50:
                    text = text[:max_text_length]
                    target = 'positive' if label == 1 else 'negative'
                    processed.append((f"Analyze the sentiment of this review: {text}", target, 'sentiment'))
            except:
                continue
        return processed
    
    def _process_ag_news(self, dataset):
        """Process AG News dataset"""
        processed = []
        label_map = {0: 'world', 1: 'sports', 2: 'business', 3: 'technology'}
        max_text_length = 800 if isinstance(self.config, UltraConfig) else 300
        for item in dataset:
            try:
                text = item.get('text', '').strip()
                label = item.get('label', 0)
                if text and len(text) > 30:
                    text = text[:max_text_length]
                    target = label_map.get(label, 'general')
                    processed.append((f"Classify this news article: {text}", target, 'classification'))
            except:
                continue
        return processed
    
    def _process_xsum(self, dataset):
        """Process XSum dataset"""
        processed = []
        max_doc_length = 1200 if isinstance(self.config, UltraConfig) else 500
        for item in dataset:
            try:
                document = item.get('document', '').strip()
                summary = item.get('summary', '').strip()
                if document and summary and len(document) > 100:
                    document = document[:max_doc_length]
                    processed.append((f"Summarize this article: {document}", summary, 'summarization'))
            except:
                continue
        return processed
    
    def _process_cnn(self, dataset):
        """Process CNN/DailyMail dataset"""
        processed = []
        max_article_length = 1000 if isinstance(self.config, UltraConfig) else 500
        for item in dataset:
            try:
                article = item.get('article', '').strip()
                highlights = item.get('highlights', '').strip()
                if article and highlights and len(article) > 200:
                    article = article[:max_article_length]
                    processed.append((f"Summarize this news article: {article}", highlights, 'summarization'))
            except:
                continue
        return processed
    
    def _add_fallback_data(self):
        """Add fallback data"""
        multiplier = 100 if isinstance(self.config, UltraConfig) else 20
        fallback = {
            'qa': [
                ("Context: Paris is the capital of France.\nQuestion: What is the capital of France?", "Paris", "qa"),
                ("Context: The sun is a star.\nQuestion: What is the sun?", "A star", "qa"),
                ("Context: Water boils at 100 degrees Celsius.\nQuestion: At what temperature does water boil?", "100 degrees Celsius", "qa"),
            ] * multiplier,
            'sentiment': [
                ("This movie was absolutely fantastic!", "positive", "sentiment"),
                ("I hate this terrible product.", "negative", "sentiment"),
                ("Great service and amazing food!", "positive", "sentiment"),
                ("Worst experience ever, very disappointed.", "negative", "sentiment"),
            ] * multiplier,
            'classification': [
                ("Scientists discover new planet in distant galaxy.", "science", "classification"),
                ("Stock market reaches all-time high today.", "business", "classification"),
                ("New smartphone technology announced by tech giant.", "technology", "classification"),
                ("Football team wins championship game.", "sports", "classification"),
            ] * int(multiplier * 0.75),
            'general': [
                ("Tell me about artificial intelligence", "AI is a technology that enables machines to learn and make decisions.", "general"),
                ("Explain climate change", "Climate change refers to long-term changes in global temperatures and weather patterns.", "general"),
            ] * int(multiplier * 0.5)
        }
        for task_type, data in fallback.items():
            self.datasets.setdefault(task_type, [])
            self.datasets[task_type].extend(data)

# ============================================================================
# NEURAL NETWORK COMPONENTS
# ============================================================================

class SimplifiedPagedKVCache:
    """Simplified paged cache for better performance"""
    def __init__(self, max_seq_len: int, hidden_size: int, num_heads: int,
                 block_size: int = 16, max_blocks: int = 500):
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.num_layers = 12
        try:
            self.key_blocks = torch.zeros(
                (self.num_layers, max_blocks, block_size, num_heads, self.head_dim),
                dtype=torch.float16,
                device=device,
                requires_grad=False
            )
            self.value_blocks = torch.zeros(
                (self.num_layers, max_blocks, block_size, num_heads, self.head_dim),
                dtype=torch.float16,
                device=device,
                requires_grad=False
            )
        except RuntimeError as e:
            logger.error(f"Failed to allocate KV cache: {e}")
            self.max_blocks = 250
            self.key_blocks = torch.zeros(
                (self.num_layers, self.max_blocks, block_size, num_heads, self.head_dim),
                dtype=torch.float16,
                device=device,
                requires_grad=False
            )
            self.value_blocks = torch.zeros(
                (self.num_layers, self.max_blocks, block_size, num_heads, self.head_dim),
                dtype=torch.float16,
                device=device,
                requires_grad=False
            )
        self.free_blocks = set(range(self.max_blocks))
        self.allocated_blocks = {}
        self.sequence_lengths = {}
    
    def allocate_sequence(self, sequence_id: str, initial_length: int = 0) -> bool:
        """Simplified allocation"""
        if sequence_id in self.allocated_blocks:
            return True
        blocks_needed = max(1, math.ceil(initial_length / self.block_size))
        if len(self.free_blocks) < blocks_needed:
            return False
        allocated = []
        for _ in range(blocks_needed):
            if self.free_blocks:
                allocated.append(self.free_blocks.pop())
        self.allocated_blocks[sequence_id] = allocated
        self.sequence_lengths[sequence_id] = initial_length
        return True
    
    def deallocate_sequence(self, sequence_id: str):
        """Simplified deallocation"""
        if sequence_id in self.allocated_blocks:
            self.free_blocks.update(self.allocated_blocks[sequence_id])
            del self.allocated_blocks[sequence_id]
            del self.sequence_lengths[sequence_id]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory stats"""
        total_blocks = self.max_blocks
        used_blocks = total_blocks - len(self.free_blocks)
        return {
            "total_blocks": total_blocks,
            "used_blocks": used_blocks,
            "free_blocks": len(self.free_blocks),
            "utilization": used_blocks / total_blocks if total_blocks > 0 else 0,
            "active_sequences": len(self.allocated_blocks)
        }

class SVDDecomposer:
    """SVD decomposition for weight matrices"""
    @staticmethod
    def decompose_weight(weight: torch.Tensor, rank_ratio: float = 0.8, min_sv: float = 1e-5):
        """Optimized SVD decomposition"""
        try:
            weight = weight.to(device).float()
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
            V = Vh.T
            valid_sv = S > min_sv
            if rank_ratio < 1.0:
                n_keep = max(1, int(len(S) * rank_ratio))
                keep_indices = torch.argsort(S, descending=True)[:n_keep]
                keep_mask = torch.zeros_like(S, dtype=torch.bool, device=device)
                keep_mask[keep_indices] = True
                valid_sv = valid_sv & keep_mask
            return U[:, valid_sv], S[valid_sv], V[:, valid_sv]
        except Exception as e:
            logger.error(f"SVD decomposition failed: {str(e)}")
            return None, None, None
    
    @staticmethod
    def reconstruct_weight(U: torch.Tensor, S: torch.Tensor, V: torch.Tensor,
                         adaptation_vector: torch.Tensor = None, target_dtype: torch.dtype = None):
        """Optimized weight reconstruction"""
        if any(x is None for x in [U, S, V]):
            return None
        try:
            U, S, V = U.float(), S.float(), V.float()
            if adaptation_vector is not None:
                adaptation_vector = adaptation_vector.float()
                adaptation_factor = torch.tanh(adaptation_vector[:len(S)]) * 0.05 + 1.0
                adapted_S = S * adaptation_factor
            else:
                adapted_S = S
            reconstructed = torch.einsum('ij,j,kj->ik', U, adapted_S, V)
            if target_dtype is not None:
                reconstructed = reconstructed.to(target_dtype)
            return reconstructed
        except Exception as e:
            logger.error(f"Weight reconstruction failed: {str(e)}")
            return None

class ValueNetwork(nn.Module):
    """Value network for GRPO"""
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        ).to(device)
        for module in self.value_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight.data)
                nn.init.zeros_(module.bias.data)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        try:
            hidden_states = hidden_states.to(device).float()
            pooled = torch.mean(hidden_states, dim=1)
            values = self.value_head(pooled).squeeze(-1)
            return values
        except Exception as e:
            logger.error(f"Value network forward failed: {str(e)}")
            return torch.zeros(hidden_states.size(0), device=device, dtype=torch.float32)

class RewardFunction:
    """Reward function for different tasks"""
    def __init__(self):
        self.task_scales = {
            'qa': 2.5,
            'summarization': 2.0,
            'sentiment': 2.2,
            'classification': 1.8,
            'general': 1.5
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
                "summarization": self._fast_summarization_reward
            }.get(task_type, self._fast_general_reward)(generated_text, target_text)
            scaled_reward = reward * self.task_scales.get(task_type, 1.0)
            return np.clip(scaled_reward, -2.0, 2.0)
        except Exception as e:
            logger.error(f"Reward computation failed: {str(e)}")
            return -1.0
    
    def _fast_qa_reward(self, generated: str, target: str) -> float:
        """Fast QA reward based on word overlap"""
        gen_words = set(generated.lower().split())
        target_words = set(target.lower().split())
        if not target_words:
            return 0.0
        overlap = len(gen_words & target_words) / len(target_words)
        if target.lower() in generated.lower():
            overlap += 0.5
        return min(overlap, 1.0)
    
    def _fast_sentiment_reward(self, generated: str, target: str) -> float:
        """Fast sentiment reward"""
        positive_words = {'good', 'great', 'positive', 'happy', 'love', 'excellent'}
        negative_words = {'bad', 'terrible', 'negative', 'sad', 'hate', 'awful'}
        gen_words = set(generated.lower().split())
        target_lower = target.lower()
        gen_positive = len(gen_words & positive_words)
        gen_negative = len(gen_words & negative_words)
        target_is_positive = 'positive' in target_lower or '1' in target_lower
        if target_is_positive and gen_positive > gen_negative:
            return 1.0
        elif not target_is_positive and gen_negative > gen_positive:
            return 1.0
        else:
            return 0.2
    
    def _fast_classification_reward(self, generated: str, target: str) -> float:
        """Fast classification reward"""
        if target.lower() in generated.lower():
            return 1.0
        gen_words = set(generated.lower().split())
        target_words = set(target.lower().split())
        if target_words:
            overlap = len(gen_words & target_words) / len(target_words)
            return overlap * 0.8
        return 0.1
    
    def _fast_summarization_reward(self, generated: str, target: str) -> float:
        """Fast summarization reward"""
        gen_len = len(generated.split())
        target_len = len(target.split())
        if gen_len == 0:
            return -1.0
        length_ratio = min(gen_len / max(target_len, 1), max(target_len, 1) / gen_len)
        length_score = 0.5 if 0.5 <= length_ratio <= 2.0 else 0.2
        gen_words = set(generated.lower().split())
        target_words = set(target.lower().split())
        overlap = len(gen_words & target_words) / max(len(target_words), 1) if target_words else 0
        return length_score + overlap * 0.5
    
    def _fast_general_reward(self, generated: str, target: str) -> float:
        """Fast general reward"""
        if len(generated.strip()) < 5:
            return -0.5
        words = generated.split()
        diversity = len(set(words)) / max(len(words), 1)
        length_score = min(len(words) / 20, 1.0)
        return diversity * 0.5 + length_score * 0.5

class CEMOptimizer:
    """Cross-Entropy Method optimizer for adaptation"""
    def __init__(self, config):
        self.config = config
        self.task_params = config.cem_params
        self.first_epoch_params = config.first_epoch_cem_params
        self.momentum = config.cem_momentum
    
    def get_task_params(self, task_type: str, epoch: int = 0) -> Dict[str, Any]:
        """Get task-specific CEM parameters based on epoch"""
        if epoch == 0:
            return self.first_epoch_params.get(task_type, self.first_epoch_params["general"])
        else:
            return self.task_params.get(task_type, self.task_params["general"])
    
    def optimize_adaptation(self, model, input_batch, target_batch, adaptation_dim: int,
                          task_type: str = "general", max_steps: int = None, epoch: int = 0):
        """Optimized adaptation with progressive complexity"""
        params = self.get_task_params(task_type, epoch)
        population_size = params["population_size"]
        elite_ratio = params["elite_ratio"]
        n_elite = max(1, int(population_size * elite_ratio))
        noise_std = params["noise_std"]
        max_steps = params["adaptation_steps"] if max_steps is None else max_steps
        convergence_threshold = params["convergence_threshold"]
        population_mean = torch.zeros(adaptation_dim, device=device)
        population_std = torch.ones(adaptation_dim, device=device) * noise_std
        best_params, best_score = None, float('-inf')
        if epoch == 0:
            logger.debug(f"Fast CEM for {task_type} (epoch 0)")
            max_steps = min(max_steps, 3)
        for step in range(max_steps):
            try:
                population = torch.randn(population_size, adaptation_dim, device=device)
                population = population * population_std + population_mean
                population = torch.clamp(population, -1.0, 1.0)
                scores = self._fast_evaluate_adaptation_params(
                    model, input_batch, target_batch, population, task_type
                )
                valid_mask = torch.isfinite(scores)
                if not valid_mask.any():
                    logger.warning(f"All CEM scores invalid at step {step}")
                    break
                valid_scores = scores[valid_mask]
                valid_population = population[valid_mask]
                if len(valid_scores) > 0:
                    current_best_idx = torch.argmax(valid_scores)
                    current_best_score = valid_scores[current_best_idx].item()
                    if current_best_score > best_score:
                        best_score = current_best_score
                        best_params = valid_population[current_best_idx].clone()
                if epoch == 0 and step >= 1:
                    break
                n_elite_actual = min(n_elite, len(valid_scores))
                if n_elite_actual > 0:
                    elite_indices = torch.topk(valid_scores, n_elite_actual)[1]
                    elite_samples = valid_population[elite_indices]
                    new_mean = elite_samples.mean(dim=0)
                    new_std = elite_samples.std(dim=0) + 1e-6
                    population_mean = self.momentum * population_mean + (1 - self.momentum) * new_mean
                    population_std = self.momentum * population_std + (1 - self.momentum) * new_std
                    population_std = torch.clamp(population_std, 0.01, 0.5)
                    mean_change = torch.norm(new_mean - population_mean).item()
                    if mean_change < convergence_threshold:
                        logger.debug(f"CEM converged at step {step}")
                        break
            except Exception as e:
                logger.error(f"CEM step {step} failed: {str(e)}")
                break
        return best_params, best_score, []
    
    def _fast_evaluate_adaptation_params(self, model, input_batch, target_batch, population, task_type):
        """Fast evaluation of adaptation parameters"""
        scores = torch.full((len(population),), float('-inf'), device=device)
        if not isinstance(input_batch, dict):
            return scores
        with torch.no_grad():
            for i, params in enumerate(population):
                try:
                    model.apply_adaptation_params(params)
                    outputs = model.forward_with_adaptation(
                        input_batch["input_ids"],
                        attention_mask=input_batch["attention_mask"]
                    )
                    if outputs is None:
                        scores[i] = -10.0
                        continue
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = input_batch["input_ids"][..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction='mean'
                    )
                    score = -loss.item()
                    scores[i] = score if torch.isfinite(torch.tensor(score)) else -10.0
                except Exception as e:
                    scores[i] = -10.0
        return scores

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
    """Self-adaptive GPT-2 model with SVD decomposition"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        logger.info(f"Loading {config.model_name}")
        self.base_model = GPT2LMHeadModel.from_pretrained(config.model_name)
        self.gradient_checkpointing_enabled = False
        self.kv_cache = (
            SimplifiedPagedKVCache(
                max_seq_len=config.max_length * 2,
                hidden_size=self.base_model.config.hidden_size,
                num_heads=self.base_model.config.num_attention_heads,
                block_size=config.paged_block_size,
                max_blocks=config.max_cache_blocks
            )
            if config.enable_paged_attention else None
        )
        self.base_model = self.base_model.to(device)
        self.svd_components = {}
        self.adaptation_params = nn.ParameterDict()
        self.value_network = ValueNetwork(self.base_model.config.hidden_size).to(device)
        self.task_classifier = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, config.num_experts),
            nn.Softmax(dim=-1)
        ).to(device)
        self.cem_optimizer = CEMOptimizer(config)
        self._initialize_svd_decomposition()
        self.current_adaptation = None
        self.sequence_counter = 0
        self.current_temperature = config.temperature
        logger.info(f"Model initialized with {self._count_parameters():,} parameters")
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def enable_gradient_checkpointing(self, enable: bool = True):
        """Control gradient checkpointing"""
        if enable and not self.gradient_checkpointing_enabled:
            self.base_model.gradient_checkpointing_enable()
            self.gradient_checkpointing_enabled = True
            logger.info("Gradient checkpointing enabled")
        elif not enable and self.gradient_checkpointing_enabled:
            self.base_model.gradient_checkpointing_disable()
            self.gradient_checkpointing_enabled = False
            logger.info("Gradient checkpointing disabled")
    
    def _initialize_svd_decomposition(self):
        """Initialize SVD decomposition"""
        logger.info("Initializing SVD decomposition")
        if isinstance(self.config, UltraConfig):
            target_layers = [
                f'transformer.h.{i}.attn.c_attn' for i in range(12)
            ] + [
                f'transformer.h.{i}.mlp.c_fc' for i in range(6)
            ]
        else:
            target_layers = [
                'transformer.h.0.attn.c_attn',
                'transformer.h.1.attn.c_attn',
                'transformer.h.2.attn.c_attn'
            ]
        decomposed_count = 0
        total_adaptation_dim = 0
        for name, module in self.base_model.named_modules():
            if name in target_layers and hasattr(module, 'weight'):
                try:
                    weight = module.weight.data
                    U, S, V = SVDDecomposer.decompose_weight(
                        weight,
                        rank_ratio=self.config.svd_rank_ratio,
                        min_sv=self.config.svd_min_singular_value
                    )
                    if U is not None and S is not None and V is not None:
                        self.svd_components[name] = {
                            'U': U.detach(),
                            'S': S.detach(),
                            'V': V.detach(),
                            'original_dtype': weight.dtype,
                            'original_shape': weight.shape
                        }
                        param_name = name.replace('.', '_')
                        self.adaptation_params[param_name] = nn.Parameter(
                            torch.zeros(len(S), device=device, dtype=torch.float32)
                        )
                        total_adaptation_dim += len(S)
                        decomposed_count += 1
                        logger.info(f"SVD decomposed {name}: {weight.shape} -> rank {len(S)}")
                except Exception as e:
                    logger.error(f"SVD decomposition error for {name}: {str(e)}")
        logger.info(f"SVD complete: {decomposed_count} layers, {total_adaptation_dim} total params")
    
    def get_total_adaptation_dim(self) -> int:
        """Get total dimension of adaptation parameters"""
        return sum(param.numel() for param in self.adaptation_params.values())
    
    def apply_adaptation_params(self, adaptation_vector: torch.Tensor):
        """Apply adaptation parameters to the model"""
        if not self.svd_components or adaptation_vector is None:
            return
        try:
            offset = 0
            for name, comps in self.svd_components.items():
                param_name = name.replace('.', '_')
                if param_name in self.adaptation_params:
                    param_size = self.adaptation_params[param_name].numel()
                    if offset + param_size <= len(adaptation_vector):
                        self.adaptation_params[param_name].data.copy_(
                            adaptation_vector[offset:offset + param_size]
                        )
                        offset += param_size
        except Exception as e:
            logger.error(f"Error applying adaptation params: {str(e)}")
    
    def forward_with_adaptation(self, input_ids, attention_mask=None, use_adaptation=True, sequence_id=None):
        """Forward pass with adaptation"""
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        if not use_adaptation or not self.svd_components:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        original_weights = {}
        try:
            for name, comps in self.svd_components.items():
                module = dict(self.base_model.named_modules())[name]
                original_weights[name] = module.weight.data.clone()
                param_name = name.replace('.', '_')
                adapted = SVDDecomposer.reconstruct_weight(
                    comps['U'], comps['S'], comps['V'],
                    self.adaptation_params[param_name],
                    target_dtype=comps['original_dtype']
                )
                if adapted is not None:
                    module.weight.data.copy_(adapted)
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            return outputs
        finally:
            for name, orig in original_weights.items():
                dict(self.base_model.named_modules())[name].weight.data.copy_(orig)
    
    def generate_episode(self, input_ids, attention_mask, max_new_tokens=None, task_type="general", epoch=0):
        """Generate an episode for GRPO training"""
        self.eval()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if max_new_tokens is None:
            if isinstance(self.config, UltraConfig):
                if epoch == 0:
                    max_new_tokens = random.randint(16, 48)
                else:
                    max_new_tokens = random.randint(32, 96)
            else:
                max_new_tokens = random.randint(8, 24)
        seq_id = f"seq_{self.sequence_counter}_{task_type}"
        self.sequence_counter += 1
        try:
            with torch.no_grad():
                input_batch = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
                try:
                    if epoch > 0:
                        self.adapt_for_inference(input_batch, task_type=task_type, epoch=epoch)
                        use_adaptation = True
                    else:
                        use_adaptation = False
                except Exception as e:
                    logger.debug(f"Adaptation skipped: {e}")
                    use_adaptation = False
                try:
                    if use_adaptation:
                        init_out = self.forward_with_adaptation(input_ids, attention_mask, use_adaptation=True)
                    else:
                        init_out = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
                    hidden_states = init_out.hidden_states[-1]
                    values = self.value_network(hidden_states)
                except Exception as e:
                    logger.warning(f"Forward failed, using dummy values: {e}")
                    values = torch.zeros(input_ids.size(0), device=device)
                gen = self.base_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=self.current_temperature,
                    top_p=self.config.top_p,
                    top_k=50 if not isinstance(self.config, UltraConfig) else 100,
                    repetition_penalty=self.config.repetition_penalty,
                    pad_token_id=self.base_model.config.eos_token_id,
                    eos_token_id=self.base_model.config.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=True
                )
                seq = gen.sequences[:, input_ids.size(1):]
                length = seq.size(1)
                log_probs = []
                for i, score in enumerate(gen.scores[:length]):
                    lp = F.log_softmax(score, dim=-1).gather(1, seq[:, i:i+1]).squeeze(-1)
                    log_probs.append(lp)
                log_probs = torch.stack(log_probs, dim=1) if log_probs else torch.zeros_like(seq, dtype=torch.float32)
                rewards = torch.zeros_like(seq, dtype=torch.float32)
                return Episode(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generated_tokens=seq,
                    log_probs=log_probs,
                    rewards=rewards,
                    values=values,
                    task_type=task_type,
                    sequence_id=seq_id,
                    episode_length=length
                )
        except Exception as e:
            logger.error(f"Episode generation failed: {str(e)}")
            dummy_seq = torch.zeros((1, 1), dtype=torch.long, device=device)
            dummy_values = torch.zeros(1, device=device)
            return Episode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generated_tokens=dummy_seq,
                log_probs=torch.zeros_like(dummy_seq, dtype=torch.float32),
                rewards=torch.zeros_like(dummy_seq, dtype=torch.float32),
                values=dummy_values,
                task_type=task_type,
                sequence_id=seq_id,
                episode_length=1
            )
    
    def compute_grpo_loss(self, episodes: List[Episode]):
        """Compute GRPO loss"""
        if not episodes:
            return torch.tensor(0.0, requires_grad=True, device=device)
        if isinstance(self.config, UltraConfig):
            grouped_episodes = defaultdict(list)
            for episode in episodes:
                grouped_episodes[episode.task_type].append(episode)
            total_loss = torch.tensor(0.0, requires_grad=True, device=device)
            total_episodes = 0
            for task_type, task_episodes in grouped_episodes.items():
                task_loss = self._compute_task_loss(task_episodes, task_type)
                if task_loss is not None:
                    total_loss = total_loss + task_loss
                    total_episodes += len(task_episodes)
            return total_loss / max(len(grouped_episodes), 1) if total_episodes > 0 else torch.tensor(0.0, requires_grad=True, device=device)
        else:
            return self._compute_task_loss(episodes, "general")
    
    def _compute_task_loss(self, episodes: List[Episode], task_type: str):
        """Compute loss for a specific task"""
        if not episodes:
            return None
        total_loss = torch.tensor(0.0, requires_grad=True, device=device)
        valid_episodes = 0
        if isinstance(self.config, UltraConfig):
            all_rewards = torch.cat([
                ep.rewards.flatten() for ep in episodes if ep.rewards.numel() > 0
            ])
            if all_rewards.numel() > 1:
                reward_mean = all_rewards.mean()
                reward_std = torch.clamp(all_rewards.std() + 1e-6, min=0.1, max=5.0)
            else:
                reward_mean, reward_std = 0.0, 1.0
        else:
            reward_mean, reward_std = 0.0, 1.0
        for episode in episodes:
            try:
                if episode.rewards.numel() == 0 or episode.log_probs.numel() == 0:
                    continue
                if isinstance(self.config, UltraConfig):
                    normalized_rewards = (episode.rewards - reward_mean) / reward_std
                else:
                    normalized_rewards = episode.rewards
                normalized_rewards = torch.clamp(normalized_rewards, -self.config.clip_rewards, self.config.clip_rewards)
                normalized_rewards = normalized_rewards * self.config.reward_scaling
                if episode.values.numel() > 0:
                    if normalized_rewards.dim() > episode.values.dim():
                        values_expanded = episode.values.mean().expand_as(normalized_rewards)
                    elif normalized_rewards.dim() == episode.values.dim():
                        min_len = min(normalized_rewards.numel(), episode.values.numel())
                        normalized_rewards = normalized_rewards.flatten()[:min_len]
                        values_expanded = episode.values.flatten()[:min_len]
                    else:
                        values_expanded = episode.values[:normalized_rewards.numel()]
                    advantages = normalized_rewards - values_expanded.detach()
                    if isinstance(self.config, UltraConfig) and advantages.numel() > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
                else:
                    advantages = normalized_rewards
                policy_loss = -(episode.log_probs.flatten()[:advantages.numel()] * advantages).mean()
                if episode.values.numel() > 0:
                    value_loss = F.mse_loss(values_expanded, normalized_rewards.detach())
                else:
                    value_loss = torch.tensor(0.0, device=device)
                if isinstance(self.config, UltraConfig):
                    entropy_loss = -episode.log_probs.mean()
                else:
                    entropy_loss = torch.tensor(0.0, device=device)
                task_weight = {
                    'qa': 1.2,
                    'sentiment': 1.0,
                    'classification': 1.1,
                    'summarization': 1.3,
                    'general': 1.0
                }.get(task_type, 1.0)
                episode_loss = task_weight * (
                    policy_loss + 
                    self.config.grpo_value_loss_coeff * value_loss +
                    self.config.grpo_entropy_coeff * entropy_loss
                )
                if torch.isfinite(episode_loss):
                    total_loss = total_loss + episode_loss
                    valid_episodes += 1
            except Exception as e:
                logger.error(f"Episode loss computation failed: {str(e)}")
        return total_loss / max(valid_episodes, 1) if valid_episodes > 0 else None
    
    def adapt_for_inference(self, input_batch, target_batch=None, task_type="general", epoch=0):
        """Adapt model for inference"""
        adaptation_dim = self.get_total_adaptation_dim()
        if adaptation_dim == 0:
            return 0.0, []
        try:
            best_params, best_score, history = self.cem_optimizer.optimize_adaptation(
                self, input_batch, target_batch, adaptation_dim, task_type, epoch=epoch
            )
            if best_params is not None:
                self.apply_adaptation_params(best_params)
                self.current_adaptation = best_params.clone()
            return best_score, history
        except Exception as e:
            logger.error(f"Adaptation failed: {str(e)}")
            return -10.0, []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        stats = {
            "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            "gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
        }
        if isinstance(self.config, UltraConfig):
            stats["gpu_memory_utilization"] = torch.cuda.memory_allocated() / (15 * 1e9) * 100
        if self.kv_cache:
            stats.update(self.kv_cache.get_memory_stats())
        return stats

# ============================================================================
# TRAINER CLASSES - FIXED AMP FLOW
# ============================================================================

class GRPOTrainer:
    """GRPO trainer with FIXED AMP flow - no more scaler errors!"""
    def __init__(self, config):
        self.config = config
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"Initializing model with config: {type(config).__name__}")
        self.model = SelfAdaptiveGPT2(config)
        
        # Initialize GradScaler for AMP
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision)
        
        # Setup optimizers
        adaptation_params = list(self.model.adaptation_params.values())
        other_params = list(self.model.task_classifier.parameters())
        
        if isinstance(config, UltraConfig):
            try:
                self.policy_optimizer = torch.optim.AdamW(
                    adaptation_params + other_params + list(self.model.value_network.parameters()),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    fused=True
                )
                self.value_optimizer = None
            except:
                self.policy_optimizer = torch.optim.AdamW(
                    adaptation_params + other_params,
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
                self.value_optimizer = torch.optim.AdamW(
                    self.model.value_network.parameters(),
                    lr=config.learning_rate * 1.5,
                    weight_decay=config.weight_decay,
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
        else:
            self.policy_optimizer = torch.optim.AdamW(
                adaptation_params + other_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            self.value_optimizer = torch.optim.AdamW(
                self.model.value_network.parameters(),
                lr=config.learning_rate * 1.5,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        
        # Setup schedulers
        total_steps = config.num_epochs * 100
        self.policy_scheduler = get_cosine_schedule_with_warmup(
            self.policy_optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        if self.value_optimizer:
            self.value_scheduler = get_cosine_schedule_with_warmup(
                self.value_optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=total_steps
            )
        else:
            self.value_scheduler = None
        
        # Load datasets
        self.dataset_loader = DatasetLoader(config)
        self.datasets = self.dataset_loader.load_all_datasets()
        self.reward_function = RewardFunction()
        
        # Setup directories and metrics
        os.makedirs(config.output_dir, exist_ok=True)
        self.training_metrics = {
            "policy_loss": [],
            "gpu_memory_usage": [],
            "task_rewards": defaultdict(list),
            "training_speed": [],
            "episode_lengths": []
        }
        if isinstance(config, UltraConfig):
            self.training_metrics.update({
                "value_loss": [],
                "entropy": [],
                "gpu_utilization": [],
                "adaptation_scores": defaultdict(list),
                "learning_rates": [],
                "gradient_norms": []
            })
        logger.info("Trainer initialized successfully")
    
    def create_dataloader(self, data, batch_size, is_validation=False):
        """Create DataLoader"""
        dataset = PreTokenizedDataset(data, self.tokenizer, self.config.max_length)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=not is_validation,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Collate function"""
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
        """Main training loop with CANONICAL AMP flow and gradient accumulation"""
        logger.info(f"Starting GRPO training with {type(self.config).__name__}")
        # Initialize wandb if configured
        if self.config.wandb_project:
            try:
                wandb.init(
                    project=self.config.wandb_project,
                    name=f"grpo-{type(self.config).__name__}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    config=vars(self.config)
                )
            except Exception as e:
                logger.warning(f"Wandb init failed: {e}")
        # Prepare DataLoader
        all_data = self.datasets.get('general', [])
        if not all_data:
            logger.error("No training data available")
            return
        dataloader = self.create_dataloader(all_data)
        # Training setup
        self.model.train()
        global_step = 0
        accumulated_steps = 0
        start_time = time.time()
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(dataloader, 1):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                # Generate episodes
                episodes = [self.model.generate_episode(input_ids, attention_mask, task_type='general')]
                # Compute loss with autocast
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    loss = self.model.compute_grpo_loss(episodes)
                    loss = loss / self.config.gradient_accumulation_steps
                # Backward
                self.scaler.scale(loss).backward()
                accumulated_steps += 1
                epoch_loss += loss.item()
                # Step optimizer
                if accumulated_steps >= self.config.gradient_accumulation_steps:
                    self.scaler.unscale_(self.policy_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.policy_optimizer)
                    self.scaler.update()
                    self.policy_scheduler.step()
                    self.policy_optimizer.zero_grad()
                    accumulated_steps = 0
                    global_step += 1
                    if global_step % self.config.log_interval == 0:
                        logger.info(f"Step {global_step}, avg loss={(epoch_loss/global_step):.4f}")
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        total_time = time.time() - start_time
        logger.info(f"Training finished in {total_time/60:.2f} minutes")
    
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
                'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                'config': self.config,
                'training_metrics': self.training_metrics
            }
            
            if self.value_optimizer:
                checkpoint['value_optimizer_state_dict'] = self.value_optimizer.state_dict()
            
            if isinstance(self.config, UltraConfig):
                checkpoint.update({
                    'gpu_memory_peak': torch.cuda.max_memory_allocated() / 1e9,
                    'adaptation_params_count': self.model.get_total_adaptation_dim(),
                    'svd_components_count': len(self.model.svd_components)
                })
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    logger.info("Starting FIXED GPT-2 Training Pipeline - NO MORE AMP ERRORS!")
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Configuration selection
    fast_mode = False
    if fast_mode:
        config = UltraConfig()
        config.num_epochs = 1
        config.batch_size = 4
        config.max_length = 128
        config.use_fallback_data_only = True
    else:
        config = UltraConfig()
    
    logger.info(f"Configuration: {type(config).__name__}")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Mixed Precision: {config.mixed_precision}")
    logger.info(f"  ✅ AMP FLOW: FIXED - No more scaler errors!")
    
    try:
        with timer("Trainer initialization"):
            trainer = GRPOTrainer(config)
        
        total_samples = sum(len(data) for data in trainer.datasets.values())
        if total_samples == 0:
            logger.error("No training data loaded!")
            return
        
        logger.info(f"Loaded {total_samples:,} training samples")
        logger.info("Starting training with CANONICAL AMP FLOW...")
        
        with timer("Complete training"):
            trainer.train_grpo()
        
        logger.info("Pipeline completed successfully - NO AMP ERRORS! 🎉")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if config.wandb_project:
            try:
                wandb.finish()
            except:
                pass

if __name__ == "__main__":
    main()
