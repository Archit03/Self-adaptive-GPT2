pip install evaluate

!pip install --upgrade datasets

!pip install rouge_score

# -*- coding: utf-8 -*-
"""
Fixed Enhanced GPU-Optimized Self-Adaptive GPT2 with GRPO Training + CEM Inference + PagedAttention
Fixes:
- Mixed precision scaler usage
- Cache allocation issues
- Memory management improvements
"""

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
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    get_cosine_schedule_with_warmup,
    AutoTokenizer, AutoModel
)
from datasets import load_dataset, concatenate_datasets, Dataset
import wandb
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import evaluate
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_fixed.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ### Utility Functions
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

# ### Fixed Enhanced Paged KV Cache
class FixedEnhancedPagedKVCache:
    """
    Fixed PagedAttention-inspired KV cache with better memory management
    """
    def __init__(self, max_seq_len: int, hidden_size: int, num_heads: int,
                 block_size: int = 16, max_blocks: int = 1000, enable_prefix_caching: bool = True):
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.enable_prefix_caching = enable_prefix_caching

        # Allocate cache blocks
        try:
            self.key_blocks = torch.zeros(
                max_blocks, block_size, num_heads, self.head_dim,
                dtype=torch.float32, device=device, requires_grad=False
            )
            self.value_blocks = torch.zeros(
                max_blocks, block_size, num_heads, self.head_dim,
                dtype=torch.float32, device=device, requires_grad=False
            )
        except RuntimeError as e:
            logger.error(f"Failed to allocate KV cache blocks: {e}")
            # Fallback to smaller cache
            self.max_blocks = min(500, max_blocks)
            self.key_blocks = torch.zeros(
                self.max_blocks, block_size, num_heads, self.head_dim,
                dtype=torch.float32, device=device, requires_grad=False
            )
            self.value_blocks = torch.zeros(
                self.max_blocks, block_size, num_heads, self.head_dim,
                dtype=torch.float32, device=device, requires_grad=False
            )

        self.free_blocks = set(range(self.max_blocks))
        self.allocated_blocks = {}
        self.block_tables = {}
        self.sequence_lengths = {}
        self.sequence_last_access = {}
        self.sequence_prefixes = {}
        self.prefix_to_blocks = {}

        self.allocation_count = 0
        self.deallocation_count = 0
        self.cleanup_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.reuse_count = 0

        logger.info(f"FixedEnhancedPagedKVCache initialized: {self.max_blocks} blocks of size {block_size}")
        logger.info(f"Total KV cache memory: {self._calculate_memory_usage():.2f} MB")

    def _calculate_memory_usage(self) -> float:
        """Calculate total memory usage in MB"""
        bytes_per_block = 2 * self.block_size * self.num_heads * self.head_dim * 4
        total_bytes = self.max_blocks * bytes_per_block
        return total_bytes / (1024 * 1024)

    def _compute_prefix_hash(self, input_ids: torch.Tensor, length: int) -> str:
        """Compute hash of sequence prefix for reuse detection"""
        if not self.enable_prefix_caching or length < self.block_size:
            return ""
        prefix_length = min((length // self.block_size) * self.block_size, 64)  # Limit prefix length
        prefix = input_ids.flatten()[:prefix_length].cpu().numpy().tobytes()
        return str(hash(prefix))

    def _cleanup_lru_sequences(self, needed_blocks: int = 1, aggressive: bool = False):
        """Clean up least recently used sequences to free up blocks"""
        if len(self.free_blocks) >= needed_blocks:
            return True

        # If aggressive cleanup requested, free more space
        if aggressive:
            target_free = max(needed_blocks * 2, self.max_blocks // 4)
        else:
            target_free = needed_blocks

        sorted_sequences = sorted(self.sequence_last_access.items(), key=lambda x: x[1])
        freed_blocks = 0
        sequences_to_remove = []

        for seq_id, _ in sorted_sequences:
            if len(self.free_blocks) >= target_free:
                break
            if seq_id in self.allocated_blocks:
                freed_blocks += len(self.allocated_blocks[seq_id])
                sequences_to_remove.append(seq_id)

        for seq_id in sequences_to_remove:
            self.deallocate_sequence(seq_id)
            self.cleanup_count += 1

        logger.debug(f"Cleaned up {len(sequences_to_remove)} sequences, freed {freed_blocks} blocks")
        return len(self.free_blocks) >= needed_blocks

    def allocate_sequence(self, sequence_id: str, initial_length: int = 0, input_ids: torch.Tensor = None) -> bool:
        """Allocate blocks for a new sequence with improved error handling"""
        self.sequence_last_access[sequence_id] = time.time()

        if sequence_id in self.allocated_blocks:
            logger.debug(f"Sequence {sequence_id} already allocated")
            return True

        blocks_needed = max(1, math.ceil(initial_length / self.block_size))

        # First try normal cleanup
        if len(self.free_blocks) < blocks_needed:
            if not self._cleanup_lru_sequences(blocks_needed):
                # Try aggressive cleanup
                logger.debug("Normal cleanup insufficient, trying aggressive cleanup")
                if not self._cleanup_lru_sequences(blocks_needed, aggressive=True):
                    logger.warning(f"Insufficient blocks for {sequence_id} after aggressive cleanup")
                    return False

        allocated = []
        for _ in range(blocks_needed):
            if self.free_blocks:
                block_idx = self.free_blocks.pop()
                allocated.append(block_idx)
            else:
                break

        if len(allocated) < blocks_needed:
            # Return allocated blocks to free pool
            self.free_blocks.update(allocated)
            logger.warning(f"Could only allocate {len(allocated)}/{blocks_needed} blocks")
            return False

        self.allocated_blocks[sequence_id] = allocated
        self.block_tables[sequence_id] = {i: allocated[i] for i in range(len(allocated))}
        self.sequence_lengths[sequence_id] = initial_length
        self.allocation_count += 1

        logger.debug(f"Allocated {len(allocated)} blocks for sequence {sequence_id}")
        return True

    def deallocate_sequence(self, sequence_id: str):
        """Deallocate blocks for a sequence"""
        if sequence_id not in self.allocated_blocks:
            return

        try:
            # Free all blocks
            for block_idx in self.allocated_blocks[sequence_id]:
                self.free_blocks.add(block_idx)
                # Clear the blocks
                self.key_blocks[block_idx].zero_()
                self.value_blocks[block_idx].zero_()

            # Remove from tracking
            del self.allocated_blocks[sequence_id]
            del self.block_tables[sequence_id]
            del self.sequence_lengths[sequence_id]
            if sequence_id in self.sequence_last_access:
                del self.sequence_last_access[sequence_id]
            if sequence_id in self.sequence_prefixes:
                del self.sequence_prefixes[sequence_id]

            self.deallocation_count += 1
            logger.debug(f"Deallocated sequence {sequence_id}")
        except Exception as e:
            logger.error(f"Error deallocating sequence {sequence_id}: {e}")

    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache hit/miss statistics"""
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_accesses, 1)
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "reuse_count": self.reuse_count,
            "total_accesses": total_accesses
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics"""
        total_blocks = self.max_blocks
        used_blocks = total_blocks - len(self.free_blocks)
        cache_stats = self.get_cache_stats()

        return {
            "total_blocks": total_blocks,
            "used_blocks": used_blocks,
            "free_blocks": len(self.free_blocks),
            "utilization": used_blocks / total_blocks if total_blocks > 0 else 0,
            "memory_mb": self._calculate_memory_usage(),
            "active_sequences": len(self.allocated_blocks),
            "avg_blocks_per_seq": used_blocks / max(len(self.allocated_blocks), 1),
            "allocation_count": self.allocation_count,
            "deallocation_count": self.deallocation_count,
            "cleanup_count": self.cleanup_count,
            "cache_hit_rate": cache_stats["hit_rate"],
            "cache_hits": cache_stats["cache_hits"],
            "cache_misses": cache_stats["cache_misses"],
            "reuse_count": cache_stats["reuse_count"]
        }

    def force_cleanup(self, keep_ratio: float = 0.5):
        """Force cleanup of cache to free up memory"""
        target_sequences = int(len(self.allocated_blocks) * keep_ratio)
        if target_sequences >= len(self.allocated_blocks):
            return

        sorted_sequences = sorted(self.sequence_last_access.items(), key=lambda x: x[1])
        sequences_to_remove = sorted_sequences[:-target_sequences] if target_sequences > 0 else sorted_sequences

        for seq_id, _ in sequences_to_remove:
            self.deallocate_sequence(seq_id)
        logger.info(f"Force cleanup: removed {len(sequences_to_remove)} sequences")

# ### Configuration with Fixes
@dataclass
class FixedEnhancedConfig:
    # Model configuration
    model_name: str = "gpt2"
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    max_length: int = 256
    adaptation_rank: int = 16
    num_experts: int = 4

    # Training optimization
    mixed_precision: bool = False  # Disabled due to scaler issues
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 0.5
    warmup_steps: int = 100
    weight_decay: float = 0.01

    # Fixed PagedAttention configuration
    enable_paged_attention: bool = True
    paged_block_size: int = 16  # Smaller blocks for better utilization
    max_cache_blocks: int = 1000  # More blocks
    cache_sequence_parallel: bool = True
    memory_efficient_attention: bool = True
    cache_cleanup_interval: int = 5  # More frequent cleanup
    max_cache_age_seconds: float = 300.0
    enable_prefix_caching: bool = False  # Disabled to reduce complexity

    # Dataset configuration
    max_samples_per_dataset: int = 500
    use_fallback_data_only: bool = False
    enable_internet_check: bool = True
    dataset_download_timeout: int = 300
    max_download_retries: int = 3

    # Dataset variety controls
    enable_qa_datasets: bool = True
    enable_sentiment_datasets: bool = True
    enable_summarization_datasets: bool = True
    enable_classification_datasets: bool = True
    enable_generation_datasets: bool = True

    # GRPO parameters
    grpo_episodes_per_batch: int = 4
    grpo_reward_normalization: bool = True
    grpo_kl_coeff: float = 0.01
    grpo_value_loss_coeff: float = 0.1
    grpo_entropy_coeff: float = 0.05

    # Task-specific CEM parameters
    cem_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "qa": {
            "population_size": 40,
            "elite_ratio": 0.25,
            "noise_std": 0.2,
            "adaptation_steps": 30,
            "convergence_threshold": 0.003
        },
        "sentiment": {
            "population_size": 50,
            "elite_ratio": 0.3,
            "noise_std": 0.25,
            "adaptation_steps": 25,
            "convergence_threshold": 0.005
        },
        "classification": {
            "population_size": 60,
            "elite_ratio": 0.35,
            "noise_std": 0.15,
            "adaptation_steps": 35,
            "convergence_threshold": 0.002
        },
        "general": {
            "population_size": 50,
            "elite_ratio": 0.3,
            "noise_std": 0.3,
            "adaptation_steps": 25,
            "convergence_threshold": 0.005
        }
    })
    cem_momentum: float = 0.3

    # SVD parameters
    svd_rank_ratio: float = 0.8
    svd_min_singular_value: float = 1e-5

    # Logging and saving
    wandb_project: str = "enhanced-grpo-cem-gpt2-fixed-final"
    output_dir: str = "./enhanced_results_fixed_final"
    log_interval: int = 10
    save_interval: int = 1

    # Stability improvements
    clip_rewards: float = 2.0
    reward_scaling: float = 0.2
    temperature_annealing: bool = True
    adaptive_learning_rate: bool = True
    learning_rate_min: float = 1e-6

    # Generation parameters
    repetition_penalty: float = 1.2
    top_p: float = 0.85
    temperature: float = 0.7
    min_episode_length: int = 16
    max_episode_length: int = 48

# ### SVD Decomposition
class StabilizedSVDDecomposer:
    @staticmethod
    def decompose_weight(weight: torch.Tensor, rank_ratio: float = 0.8, min_sv: float = 1e-5):
        """Decompose weight matrix using SVD"""
        try:
            weight = weight.to(device).float()
            reg_weight = weight + torch.randn_like(weight) * 1e-8
            U, S, Vh = torch.linalg.svd(reg_weight, full_matrices=False)
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
        """Reconstruct weight matrix from SVD components"""
        if any(x is None for x in [U, S, V]):
            return None
        try:
            U, S, V = U.float(), S.float(), V.float()

            if adaptation_vector is not None:
                adaptation_vector = adaptation_vector.float()
                adaptation_factor = torch.tanh(adaptation_vector[:len(S)]) * 0.1 + 1.0
                adapted_S = S * adaptation_factor
                scale_factor = S.sum() / (adapted_S.sum() + 1e-8)
                adapted_S = adapted_S * scale_factor
            else:
                adapted_S = S

            S_diag = torch.diag(adapted_S)
            reconstructed = torch.chain_matmul(U, S_diag, V.T)

            if target_dtype is not None:
                reconstructed = reconstructed.to(target_dtype)

            return reconstructed
        except Exception as e:
            logger.error(f"Weight reconstruction failed: {str(e)}")
            return None

# ### Value Network
class EnhancedValueNetwork(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        ).to(device)

        # Initialize with xavier_normal
        for module in self.value_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight.data)
                nn.init.zeros_(module.bias.data)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for value prediction"""
        try:
            hidden_states = hidden_states.to(device).float()
            attention_weights = F.softmax(torch.mean(hidden_states, dim=-1), dim=-1).unsqueeze(-1)
            pooled = torch.sum(hidden_states * attention_weights, dim=1)
            values = self.value_head(pooled).squeeze(-1)
            return values
        except Exception as e:
            logger.error(f"Value network forward failed: {str(e)}")
            return torch.zeros(hidden_states.size(0), device=device, dtype=torch.float32)

# ### Reward Function
class TaskSpecificRewardFunction:
    def __init__(self):
        try:
            self.rouge = evaluate.load("rouge")
            self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2').to(device)
        except Exception as e:
            logger.warning(f"Could not load evaluation tools: {str(e)}")
            self.rouge = None
            self.sentence_encoder = None

        self.task_scales = {
            'qa': 2.5,
            'summarization': 2.0,
            'sentiment': 2.2,
            'classification': 1.8,
            'general': 1.5
        }

    def compute_reward(self, generated_text: str, target_text: str, task_type: str) -> float:
        """Compute task-specific reward"""
        try:
            if not generated_text or not target_text:
                return -1.5

            reward = {
                "qa": self._qa_reward,
                "summarization": self._summarization_reward,
                "sentiment": self._sentiment_reward,
                "classification": self._classification_reward
            }.get(task_type, self._general_reward)(generated_text, target_text)

            scaled_reward = reward * self.task_scales.get(task_type, 1.0)

            if task_type == "classification":
                return np.clip(scaled_reward, -1.5, 1.5)
            else:
                return np.clip(scaled_reward, -2.0, 2.0)
        except Exception as e:
            logger.error(f"Reward computation failed for {task_type}: {str(e)}")
            return -1.5

    def _qa_reward(self, generated: str, target: str) -> float:
        generated_lower, target_lower = generated.lower().strip(), target.lower().strip()

        if generated_lower == target_lower:
            return 2.0

        if target_lower in generated_lower:
            position_penalty = 1.0 - (generated_lower.index(target_lower) / max(len(generated_lower), 1))
            return 1.5 + 0.3 * position_penalty

        gen_words = set(generated_lower.split())
        target_words = set(target_lower.split())
        if not target_words:
            return 0.0

        overlap = len(gen_words & target_words) / len(target_words)

        similarity = 0.5
        if self.sentence_encoder:
            try:
                embeddings = self.sentence_encoder.encode([generated, target], convert_to_tensor=True)
                similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            except:
                pass

        length_ratio = min(len(generated.split()) / max(len(target.split()) * 2, 1), 1.0)

        return overlap * 0.3 + similarity * 0.5 + length_ratio * 0.2

    def _summarization_reward(self, generated: str, target: str) -> float:
        gen_len, target_len = len(generated.split()), len(target.split())
        if gen_len == 0 or target_len == 0:
            return -1.0

        length_ratio = min(gen_len / target_len, target_len / gen_len)
        length_score = 1.0 if 0.7 <= length_ratio <= 1.3 else 0.5 * length_ratio

        rouge_avg = 0.5
        if self.rouge:
            try:
                rouge_scores = self.rouge.compute(predictions=[generated], references=[target])
                rouge_avg = (rouge_scores['rouge1'] + rouge_scores['rouge2'] + rouge_scores['rougeL']) / 3
            except:
                pass

        semantic_similarity = 0.5
        if self.sentence_encoder:
            try:
                embeddings = self.sentence_encoder.encode([generated, target], convert_to_tensor=True)
                semantic_similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            except:
                pass

        return rouge_avg * 0.4 + semantic_similarity * 0.4 + length_score * 0.2

    def _sentiment_reward(self, generated: str, target: str) -> float:
        positive_words = {'good', 'great', 'excellent', 'positive', 'happy', 'love',
                         'amazing', 'wonderful', 'fantastic', 'awesome', 'brilliant', 'perfect'}
        negative_words = {'bad', 'terrible', 'awful', 'negative', 'sad', 'hate',
                         'horrible', 'disgusting', 'worst', 'disappointing', 'annoying', 'frustrating'}

        gen_words = set(generated.lower().split())
        target_lower = target.lower()

        gen_positive = len(gen_words & positive_words)
        gen_negative = len(gen_words & negative_words)

        target_is_positive = any(word in target_lower for word in ['positive', '1', 'good', 'great'])

        total_sentiment_words = gen_positive + gen_negative
        if total_sentiment_words == 0:
            return 0.1

        confidence = abs(gen_positive - gen_negative) / total_sentiment_words

        if target_is_positive and gen_positive > gen_negative:
            return 1.0 + 0.5 * confidence
        elif not target_is_positive and gen_negative > gen_positive:
            return 1.0 + 0.5 * confidence
        else:
            return -0.5 * confidence

    def _classification_reward(self, generated: str, target: str) -> float:
        generated_lower, target_lower = generated.lower().strip(), target.lower().strip()

        if generated_lower == target_lower:
            return 1.5

        if target_lower in generated_lower:
            return 1.2

        category_synonyms = {
            'world': ['global', 'international', 'politics', 'nation', 'country'],
            'sports': ['athletics', 'games', 'competition', 'team', 'player'],
            'business': ['finance', 'economy', 'market', 'company', 'trade'],
            'technology': ['tech', 'computer', 'digital', 'software', 'innovation'],
            'science': ['research', 'study', 'discovery', 'experiment', 'scientific']
        }

        for category, synonyms in category_synonyms.items():
            if category in target_lower:
                if any(syn in generated_lower for syn in synonyms):
                    return 0.8

        similarity = 0.3
        if self.sentence_encoder:
            try:
                embeddings = self.sentence_encoder.encode([generated, target], convert_to_tensor=True)
                similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            except:
                pass

        return similarity * 0.6

    def _general_reward(self, generated: str, target: str) -> float:
        if len(generated.strip()) < 10:
            return -1.0

        words = generated.split()

        diversity = len(set(words)) / len(words) if words else 0

        length_score = min(len(words) / 30, 1.0) * (1.0 - max(0, len(words) - 100) / 100)

        sentences = [s.strip() for s in generated.split('.') if s.strip()]
        fluency_score = min(len(sentences) / 3, 1.0)

        coherence_score = 0.3
        if self.sentence_encoder:
            try:
                embeddings = self.sentence_encoder.encode([generated, target], convert_to_tensor=True)
                coherence_score = util.cos_sim(embeddings[0], embeddings[1]).item()
            except:
                pass

        return diversity * 0.2 + length_score * 0.3 + fluency_score * 0.2 + coherence_score * 0.3

# ### Dataset Loader
class RobustDatasetLoader:
    def __init__(self, config: FixedEnhancedConfig):
        self.config = config
        self.datasets = {}
        self.validation_datasets = {}
        self.successful_downloads = 0
        self.failed_downloads = 0

    def check_internet_connection(self):
        """Check internet connectivity"""
        if not self.config.enable_internet_check:
            return True
        try:
            response = requests.get("https://huggingface.co", timeout=10)
            return response.status_code == 200
        except:
            return False

    def download_with_retry(self, dataset_name, subset=None, split='train', max_retries=None):
        """Download dataset with retry logic"""
        max_retries = self.config.max_download_retries if max_retries is None else max_retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {dataset_name} (attempt {attempt + 1}/{max_retries})")
                dataset = load_dataset(
                    dataset_name, subset, split=split,
                    download_mode="reuse_cache_if_exists",
                    verification_mode="no_checks",
                    trust_remote_code=True  # For xsum
                ) if subset else load_dataset(
                    dataset_name, split=split,
                    download_mode="reuse_cache_if_exists",
                    verification_mode="no_checks",
                    trust_remote_code=True
                )
                logger.info(f"Successfully loaded {dataset_name} with {len(dataset)} samples")
                return dataset
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {dataset_name}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        logger.error(f"Failed to load {dataset_name} after {max_retries} attempts")
        return None

    def load_all_datasets(self):
        """Load datasets with improved variety and longer contexts"""
        if self.config.use_fallback_data_only:
            logger.info("Using fallback data only")
            self._add_comprehensive_fallback_data()
            return self.datasets

        if not self.check_internet_connection():
            logger.warning("No internet connection, using fallback data")
            self._add_comprehensive_fallback_data()
            return self.datasets

        # Dataset configurations
        dataset_configs = [
            {'name': 'squad', 'subset': None, 'split': 'train[:1000]', 'val_split': 'validation[:200]',
             'task_type': 'qa', 'process_fn': self._process_squad_extended, 'max_samples': 500, 'priority': 1}
            if self.config.enable_qa_datasets else None,

            {'name': 'imdb', 'subset': None, 'split': 'train[:1000]', 'val_split': 'test[:200]',
             'task_type': 'sentiment', 'process_fn': self._process_imdb_extended, 'max_samples': 500, 'priority': 1}
            if self.config.enable_sentiment_datasets else None,

            {'name': 'ag_news', 'subset': None, 'split': 'train[:1000]', 'val_split': 'test[:200]',
             'task_type': 'classification', 'process_fn': self._process_ag_news_extended, 'max_samples': 500, 'priority': 2}
            if self.config.enable_classification_datasets else None,

            {'name': 'xsum', 'subset': None, 'split': 'train[:500]', 'val_split': 'validation[:100]',
             'task_type': 'summarization', 'process_fn': self._process_xsum, 'max_samples': 300, 'priority': 2}
            if self.config.enable_summarization_datasets else None,
        ]

        dataset_configs = [cfg for cfg in dataset_configs if cfg is not None]
        dataset_configs.sort(key=lambda x: x['priority'])

        logger.info(f"Attempting to load {len(dataset_configs)} HuggingFace datasets")

        for config in dataset_configs:
            try:
                logger.info(f"Loading {config['name']} for {config['task_type']} task")
                dataset = self.download_with_retry(config['name'], config['subset'], config['split'])
                if dataset is None:
                    self.failed_downloads += 1
                    logger.warning(f"Skipping {config['name']} - download failed")
                    continue

                val_dataset = self.download_with_retry(
                    config['name'], config['subset'], config['val_split']
                ) if config['val_split'] else None

                processed_data = config['process_fn'](dataset)
                val_processed = config['process_fn'](val_dataset) if val_dataset else []

                if processed_data and len(processed_data) > 0:
                    task_type = config['task_type']
                    self.datasets.setdefault(task_type, [])
                    self.validation_datasets.setdefault(task_type, [])

                    max_samples = min(len(processed_data), config['max_samples'])
                    self.datasets[task_type].extend(processed_data[:max_samples])
                    self.validation_datasets[task_type].extend(val_processed[:50])

                    self.successful_downloads += 1
                    logger.info(f"✓ Added {len(processed_data[:max_samples])} training samples from {config['name']}")
                else:
                    logger.warning(f"No valid samples extracted from {config['name']}")
            except Exception as e:
                self.failed_downloads += 1
                logger.error(f"Failed to process {config['name']}: {str(e)}")

        total_samples = sum(len(data) for data in self.datasets.values())
        logger.info(f"Dataset Loading Summary: {self.successful_downloads} successful, {self.failed_downloads} failed")
        logger.info(f"Total training samples: {total_samples:,}")

        if self.successful_downloads < 2 or total_samples < 500:
            logger.warning("Insufficient real data, adding supplementary fallback data")
            self._add_supplementary_fallback_data()

        for task, data in self.datasets.items():
            val_count = len(self.validation_datasets.get(task, []))
            logger.info(f"  {task}: {len(data)} training + {val_count} validation samples")

        return self.datasets

    def _process_squad_extended(self, dataset):
        """Process SQuAD dataset with longer contexts"""
        processed = []
        for item in dataset:
            try:
                context = item.get('context', '').strip()
                question = item.get('question', '').strip()
                answers = item.get('answers', {})

                if not all([context, question, answers, answers.get('text')]):
                    continue

                answer = answers['text'][0].strip()
                if len(answer) > 0 and len(context) > 50:
                    context_truncated = context[:400] if len(context) > 400 else context
                    processed.append((f"Context: {context_truncated}\nQuestion: {question}", answer, 'qa'))
            except:
                continue
        return processed

    def _process_imdb_extended(self, dataset):
        """Process IMDB dataset with longer reviews"""
        processed = []
        for item in dataset:
            try:
                text = item.get('text', '').strip()
                label = item.get('label', 0)

                if len(text) > 100:
                    text = text[:500] if len(text) > 500 else text
                    target = 'positive' if label == 1 else 'negative'
                    processed.append((f"Analyze the sentiment of this review: {text}", target, 'sentiment'))
            except:
                continue
        return processed

    def _process_ag_news_extended(self, dataset):
        """Process AG News dataset with full articles"""
        processed = []
        label_map = {0: 'world', 1: 'sports', 2: 'business', 3: 'technology'}

        for item in dataset:
            try:
                text = item.get('text', '').strip()
                label = item.get('label', 0)

                if len(text) > 50:
                    text = text[:400] if len(text) > 400 else text
                    target = label_map.get(label, 'general')
                    processed.append((f"Classify this news article: {text}", target, 'classification'))
            except:
                continue
        return processed

    def _process_xsum(self, dataset):
        """Process XSum dataset for summarization"""
        processed = []
        for item in dataset:
            try:
                document = item.get('document', '').strip()
                summary = item.get('summary', '').strip()

                if len(document) > 100 and len(summary) > 10:
                    document = document[:600] if len(document) > 600 else document
                    processed.append((f"Summarize this text: {document}", summary, 'summarization'))
            except:
                continue
        return processed

    def _add_supplementary_fallback_data(self):
        """Add supplementary fallback data"""
        logger.info("Adding supplementary fallback data")

        supplementary_data = {
            'qa': [
                ("Context: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.\nQuestion: Who designed the Eiffel Tower?", "Gustave Eiffel", "qa"),
                ("Context: Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.\nQuestion: What is machine learning?", "A subset of artificial intelligence that enables systems to learn from experience", "qa"),
            ],
            'sentiment': [
                ("This movie exceeded all my expectations! The cinematography was breathtaking, the acting was superb, and the storyline kept me engaged from start to finish.", "positive", "sentiment"),
                ("I'm extremely disappointed with this product. The quality is far below what was advertised, it broke after just one week of use.", "negative", "sentiment"),
            ],
            'classification': [
                ("Scientists at MIT have developed a new type of battery that could revolutionize electric vehicle technology.", "technology", "classification"),
                ("The Federal Reserve announced today that it will maintain current interest rates through the end of the quarter.", "business", "classification"),
            ],
            'summarization': [
                ("Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures are leading to melting ice caps and rising sea levels.",
                 "Climate change is causing severe environmental impacts requiring urgent action.", "summarization"),
            ],
            'general': [
                ("The future of renewable energy", "Renewable energy sources like solar and wind are becoming increasingly efficient.", "general"),
            ]
        }

        for task_type, data in supplementary_data.items():
            self.datasets.setdefault(task_type, [])
            self.validation_datasets.setdefault(task_type, [])
            self.datasets[task_type].extend(data)
            self.validation_datasets[task_type].extend(data[:1])
            logger.info(f"Added {len(data)} supplementary samples for {task_type}")

    def _add_comprehensive_fallback_data(self):
        """Add comprehensive fallback data"""
        logger.info("Adding comprehensive fallback dataset")

        self.datasets = {
            'qa': [
                ("Context: Paris is the capital city of France.\nQuestion: What is the capital of France?", "Paris", "qa"),
            ] * 3,
            'sentiment': [
                ("This movie was absolutely phenomenal!", "positive", "sentiment"),
                ("I'm thoroughly disappointed with this purchase.", "negative", "sentiment"),
            ] * 3,
            'classification': [
                ("Breaking: Scientists announce discovery of new particle.", "science", "classification"),
                ("Championship game goes into overtime.", "sports", "classification"),
            ] * 3,
            'summarization': [
                ("Artificial intelligence is transforming industries. From healthcare to transportation, AI systems process data to identify patterns.",
                 "AI revolutionizes industries through advanced data processing.", "summarization"),
            ] * 2,
            'general': [
                ("The evolution of technology", "Technology continues to advance at an unprecedented rate.", "general"),
            ] * 2
        }

        self.validation_datasets = {task: data[:2] for task, data in self.datasets.items()}

# ### Task-Aware CEM Optimizer
class TaskAwareCEMOptimizer:
    def __init__(self, config: FixedEnhancedConfig):
        self.config = config
        self.task_params = config.cem_params
        self.momentum = config.cem_momentum

    def get_task_params(self, task_type: str) -> Dict[str, Any]:
        """Get task-specific CEM parameters"""
        return self.task_params.get(task_type, self.task_params["general"])

    def optimize_adaptation(self, model, input_batch, target_batch, adaptation_dim: int,
                          task_type: str = "general", max_steps: int = None):
        """Optimize adaptation parameters using task-specific CEM settings"""
        params = self.get_task_params(task_type)
        population_size = params["population_size"]
        elite_ratio = params["elite_ratio"]
        n_elite = max(1, int(population_size * elite_ratio))
        noise_std = params["noise_std"]
        max_steps = params["adaptation_steps"] if max_steps is None else max_steps
        convergence_threshold = params["convergence_threshold"]

        population_mean = torch.zeros(adaptation_dim, device=device)
        population_std = torch.ones(adaptation_dim, device=device) * noise_std
        best_params, best_score = None, float('-inf')
        convergence_history = []

        step_size = 1.0
        patience = 5 if task_type == "classification" else 3
        no_improve_count = 0

        with timer(f"CEM Optimization for {task_type}"):
            for step in range(max_steps):
                try:
                    # Generate population
                    population = torch.randn(population_size, adaptation_dim, device=device)
                    population = population * population_std * step_size + population_mean

                    if task_type == "classification":
                        population = torch.clamp(population, -1.5, 1.5)
                    else:
                        population = torch.clamp(population, -2.0, 2.0)

                    # Evaluate population
                    scores = self._batch_evaluate_adaptation_params(
                        model, input_batch, target_batch, population, task_type
                    )

                    valid_mask = torch.isfinite(scores) & (scores > -100)
                    if not valid_mask.any():
                        logger.warning(f"All CEM scores invalid at step {step} for {task_type}")
                        scores = torch.randn_like(scores) * 0.1 - 5.0
                        valid_mask = torch.ones_like(scores, dtype=torch.bool)

                    valid_scores = scores[valid_mask]
                    valid_population = population[valid_mask]

                    if len(valid_scores) > 0:
                        current_best_idx = torch.argmax(valid_scores)
                        current_best_score = valid_scores[current_best_idx].item()

                        if current_best_score > best_score:
                            best_score = current_best_score
                            best_params = valid_population[current_best_idx].clone()
                            no_improve_count = 0
                        else:
                            no_improve_count += 1

                    # Update distribution
                    n_elite_actual = min(n_elite, len(valid_scores))
                    if n_elite_actual > 0:
                        elite_indices = torch.topk(valid_scores, n_elite_actual)[1]
                        elite_samples = valid_population[elite_indices]

                        new_mean = elite_samples.mean(dim=0)
                        new_std = elite_samples.std(dim=0) + 1e-6

                        population_mean = self.momentum * population_mean + (1 - self.momentum) * new_mean
                        population_std = self.momentum * population_std + (1 - self.momentum) * new_std

                        if task_type == "classification":
                            population_std = torch.clamp(population_std, 0.02, 0.5)
                        else:
                            population_std = torch.clamp(population_std, 0.05, 1.0)

                        mean_change = torch.norm(new_mean - population_mean).item()
                        convergence_history.append(mean_change)

                        if no_improve_count >= patience:
                            step_size *= 0.8
                            no_improve_count = 0

                        if mean_change < convergence_threshold:
                            logger.info(f"CEM converged at step {step} for {task_type}")
                            break

                except Exception as e:
                    logger.error(f"CEM step {step} failed for {task_type}: {str(e)}")
                    continue

        return best_params, best_score, convergence_history

    def _batch_evaluate_adaptation_params(self, model, input_batch, target_batch, population, task_type):
        """Evaluate adaptation parameters"""
        scores = torch.full((len(population),), float('-inf'), device=device)

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

                    if target_batch is not None:
                        loss = F.cross_entropy(
                            outputs.logits.view(-1, outputs.logits.size(-1)),
                            target_batch.view(-1),
                            ignore_index=-100,
                            reduction='mean'
                        )
                    else:
                        shift_logits = outputs.logits[..., :-1, :].contiguous()
                        shift_labels = input_batch["input_ids"][..., 1:].contiguous()
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100,
                            reduction='mean'
                        )

                    score = -loss.item()

                    if task_type == "classification":
                        adaptation_magnitude = torch.norm(params).item()
                        if adaptation_magnitude > 10.0:
                            score -= 0.1 * (adaptation_magnitude - 10.0)

                    scores[i] = score if torch.isfinite(torch.tensor(score)) else -10.0

                except Exception as e:
                    logger.error(f"Evaluation of params {i} failed: {str(e)}")
                    scores[i] = -10.0

        return scores

# ### Episode Dataclass
@dataclass
class Episode:
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

# ### Fixed Model
class FixedEnhancedSelfAdaptiveGPT2(nn.Module):
    def __init__(self, config: FixedEnhancedConfig):
        super().__init__()
        self.config = config
        logger.info(f"Loading {config.model_name} on {device}")
        self.base_model = GPT2LMHeadModel.from_pretrained(config.model_name).to(device)

        # Enable gradient checkpointing
        self.base_model.gradient_checkpointing_enable()

        # Initialize fixed KV cache
        self.kv_cache = FixedEnhancedPagedKVCache(
            max_seq_len=config.max_length * 2,
            hidden_size=self.base_model.config.hidden_size,
            num_heads=self.base_model.config.num_attention_heads,
            block_size=config.paged_block_size,
            max_blocks=config.max_cache_blocks,
            enable_prefix_caching=config.enable_prefix_caching
        ) if config.enable_paged_attention else None

        logger.info(f"PagedAttention enabled with {config.max_cache_blocks} blocks"
                   if config.enable_paged_attention else "PagedAttention disabled")

        self.svd_components = {}
        self.adaptation_params = nn.ParameterDict()
        self.value_network = EnhancedValueNetwork(self.base_model.config.hidden_size)
        self.task_classifier = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, config.num_experts),
            nn.Softmax(dim=-1)
        ).to(device)

        self.cem_optimizer = TaskAwareCEMOptimizer(config)
        self._initialize_svd_decomposition()
        self.current_adaptation = None
        self.adaptation_history = deque(maxlen=50)
        self.sequence_counter = 0
        self.current_temperature = config.temperature
        self.temperature_decay = 0.95 if config.temperature_annealing else 1.0

        logger.info(f"Model initialized with {self._count_parameters():,} parameters")

    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _initialize_svd_decomposition(self):
        """Initialize SVD decomposition for key layers"""
        logger.info("Initializing SVD decomposition")

        target_patterns = [
            f'transformer.h.{i}.mlp.c_fc' for i in range(4)
        ] + [
            f'transformer.h.{i}.mlp.c_proj' for i in range(4)
        ] + [
            f'transformer.h.{i}.attn.c_attn' for i in range(2)
        ]

        decomposed_layers = 0
        for name, module in self.base_model.named_modules():
            if any(pattern in name for pattern in target_patterns) and hasattr(module, 'weight'):
                try:
                    weight = module.weight.data.to(device)
                    original_dtype = weight.dtype

                    U, S, V = StabilizedSVDDecomposer.decompose_weight(
                        weight,
                        self.config.svd_rank_ratio,
                        self.config.svd_min_singular_value
                    )

                    if U is None:
                        logger.warning(f"SVD failed for {name}")
                        continue

                    self.svd_components[name] = {
                        'U': U.to(device),
                        'S': S.to(device),
                        'V': V.to(device),
                        'original_shape': weight.shape,
                        'original_dtype': original_dtype
                    }

                    param_name = name.replace('.', '_')
                    self.adaptation_params[param_name] = nn.Parameter(
                        torch.zeros(len(S), device=device, dtype=torch.float32),
                        requires_grad=True
                    )
                    decomposed_layers += 1

                except Exception as e:
                    logger.error(f"SVD failed for {name}: {str(e)}")

        logger.info(f"SVD decomposition completed for {decomposed_layers} layers")
        logger.info(f"Total adaptation parameters: {sum(p.numel() for p in self.adaptation_params.values()):,}")

    def apply_adaptation_params(self, global_params: torch.Tensor):
        """Apply adaptation parameters to model"""
        try:
            global_params = global_params.to(device).float()
            param_idx = 0

            for name in sorted(self.svd_components.keys()):
                param_name = name.replace('.', '_')
                if param_name in self.adaptation_params:
                    param_size = self.adaptation_params[param_name].size(0)
                    if param_idx + param_size <= len(global_params):
                        self.adaptation_params[param_name].data = global_params[param_idx:param_idx + param_size].float()
                        param_idx += param_size
                    else:
                        logger.warning(f"Insufficient parameters for {param_name}")
                        break

        except Exception as e:
            logger.error(f"Error applying adaptation params: {str(e)}")

    def get_total_adaptation_dim(self) -> int:
        return sum(param.size(0) for param in self.adaptation_params.values())

    def forward_with_adaptation(self, input_ids, attention_mask=None, use_adaptation=True, sequence_id=None):
        """Forward pass with adaptation"""
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        if not use_adaptation or not self.svd_components:
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        original_weights = {}
        try:
            # Apply adapted weights
            for name, components in self.svd_components.items():
                module = dict(self.base_model.named_modules())[name]
                original_weights[name] = module.weight.data.clone()

                param_name = name.replace('.', '_')
                if param_name in self.adaptation_params:
                    adapted_weight = StabilizedSVDDecomposer.reconstruct_weight(
                        components['U'],
                        components['S'],
                        components['V'],
                        self.adaptation_params[param_name],
                        target_dtype=components['original_dtype']
                    )
                    if adapted_weight is not None:
                        module.weight.data = adapted_weight

            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            return outputs

        except Exception as e:
            logger.error(f"Forward with adaptation failed: {str(e)}")
            return None

        finally:
            # Restore original weights
            for name, original_weight in original_weights.items():
                dict(self.base_model.named_modules())[name].weight.data = original_weight

    def generate_episode(self, input_ids, attention_mask, max_new_tokens=None, task_type="general"):
        """Generate episode with variable length and better cache management"""
        self.eval()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Variable episode length
        if max_new_tokens is None:
            max_new_tokens = random.randint(
                self.config.min_episode_length,
                self.config.max_episode_length
            )

        sequence_id = f"seq_{self.sequence_counter}_{task_type}"
        self.sequence_counter += 1

        try:
            with torch.no_grad():
                # Allocate cache
                if self.config.enable_paged_attention and self.kv_cache:
                    initial_length = input_ids.size(1)
                    success = self.kv_cache.allocate_sequence(sequence_id, initial_length, input_ids)
                    if not success:
                        logger.debug(f"Cache allocation failed for {sequence_id}, proceeding without cache")

                # Initial forward pass
                initial_outputs = self.forward_with_adaptation(
                    input_ids, attention_mask, sequence_id=sequence_id
                )
                if initial_outputs is None:
                    raise ValueError("Initial forward pass failed")

                values = self.value_network(initial_outputs.hidden_states[-1])

                # Generation config
                current_temp = self.current_temperature * (self.temperature_decay ** (self.sequence_counter // 100))

                generation_config = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "temperature": max(current_temp, 0.3),
                    "top_p": self.config.top_p,
                    "top_k": 50,
                    "repetition_penalty": self.config.repetition_penalty,
                    "no_repeat_ngram_size": 3,
                    "pad_token_id": self.base_model.config.eos_token_id,
                    "eos_token_id": self.base_model.config.eos_token_id,
                    "return_dict_in_generate": True,
                    "output_scores": True
                }

                generated = self.base_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **generation_config
                )

                # Process generated tokens
                generated_tokens = generated.sequences[:, input_ids.size(1):]
                actual_length = generated_tokens.size(1)

                # Extract log probabilities
                log_probs = []
                for i, score in enumerate(generated.scores):
                    if i < actual_length:
                        log_prob = F.log_softmax(score, dim=-1).gather(
                            1, generated_tokens[:, i:i+1]
                        ).squeeze(-1)
                        log_probs.append(log_prob)

                log_probs = torch.stack(log_probs, dim=1) if log_probs else torch.zeros(
                    generated_tokens.size(), device=device, dtype=torch.float32
                )

                rewards = torch.zeros(
                    generated_tokens.size(), device=device, dtype=torch.float32
                )

                return Episode(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generated_tokens=generated_tokens,
                    log_probs=log_probs,
                    rewards=rewards,
                    values=values,
                    task_type=task_type,
                    sequence_id=sequence_id,
                    episode_length=actual_length
                )

        except Exception as e:
            logger.error(f"Episode generation failed: {str(e)}")
            # Return minimal episode on failure
            dummy_tokens = torch.zeros((input_ids.size(0), 1), device=device, dtype=torch.long)
            return Episode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generated_tokens=dummy_tokens,
                log_probs=torch.zeros_like(dummy_tokens, dtype=torch.float32),
                rewards=torch.zeros_like(dummy_tokens, dtype=torch.float32),
                values=torch.zeros(input_ids.size(0), device=device, dtype=torch.float32),
                task_type=task_type,
                sequence_id=sequence_id,
                episode_length=1
            )
        finally:
            # Deallocate sequence from cache
            if self.config.enable_paged_attention and self.kv_cache and sequence_id:
                self.kv_cache.deallocate_sequence(sequence_id)

    def compute_grpo_loss(self, episodes: List[Episode]):
        """Compute GRPO loss with improved stability"""
        if not episodes:
            return torch.tensor(0.0, requires_grad=True, device=device)

        # Group episodes by task type
        grouped_episodes = defaultdict(list)
        for episode in episodes:
            grouped_episodes[episode.task_type].append(episode)

        total_loss = torch.tensor(0.0, requires_grad=True, device=device)
        total_episodes = 0

        for task_type, task_episodes in grouped_episodes.items():
            if len(task_episodes) < 1:
                continue

            # Collect all rewards
            all_rewards = torch.cat([
                ep.rewards.flatten() for ep in task_episodes if ep.rewards.numel() > 0
            ])

            if not all_rewards.numel():
                continue

            # Normalization
            if len(all_rewards) > 1 and self.config.grpo_reward_normalization:
                reward_mean = all_rewards.mean()
                reward_std = torch.clamp(all_rewards.std() + 1e-6, min=0.1, max=10.0)
            else:
                reward_mean, reward_std = 0.0, 1.0

            task_loss = torch.tensor(0.0, requires_grad=True, device=device)
            valid_episodes = 0

            for episode in task_episodes:
                try:
                    if episode.rewards.numel() == 0 or episode.log_probs.numel() == 0:
                        continue

                    # Normalize rewards
                    normalized_rewards = episode.rewards
                    if self.config.grpo_reward_normalization:
                        normalized_rewards = (episode.rewards - reward_mean) / reward_std

                    # Apply clipping
                    normalized_rewards = torch.clamp(
                        normalized_rewards,
                        -self.config.clip_rewards,
                        self.config.clip_rewards
                    ) * self.config.reward_scaling

                    # Compute values if needed
                    if not episode.values.requires_grad:
                        outputs = self.forward_with_adaptation(
                            episode.input_ids,
                            episode.attention_mask,
                            sequence_id=episode.sequence_id
                        )
                        if outputs is not None:
                            episode.values = self.value_network(outputs.hidden_states[-1])

                    # Compute advantages
                    if normalized_rewards.dim() == 2 and episode.values.dim() == 1:
                        values_expanded = episode.values.unsqueeze(1).expand_as(normalized_rewards)
                    elif normalized_rewards.dim() == 1 and episode.values.dim() == 1:
                        if len(normalized_rewards) > len(episode.values):
                            normalized_rewards = normalized_rewards.mean()
                            values_expanded = episode.values.mean()
                        else:
                            values_expanded = episode.values[:len(normalized_rewards)]
                    else:
                        values_expanded = episode.values

                    advantages = normalized_rewards - values_expanded.detach()

                    # Normalize advantages
                    if advantages.numel() > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

                    # Policy loss
                    policy_loss = -(episode.log_probs * advantages).mean()

                    # Value loss
                    value_loss = F.mse_loss(values_expanded, normalized_rewards.detach())

                    # Entropy bonus
                    entropy_loss = -episode.log_probs.mean()

                    # Combine losses
                    episode_loss = (
                        policy_loss +
                        self.config.grpo_value_loss_coeff * value_loss +
                        self.config.grpo_entropy_coeff * entropy_loss
                    )

                    if torch.isfinite(episode_loss):
                        task_loss = task_loss + episode_loss
                        valid_episodes += 1

                except Exception as e:
                    logger.error(f"Episode loss computation failed: {str(e)}")

            if valid_episodes > 0:
                total_loss = total_loss + task_loss / valid_episodes
                total_episodes += valid_episodes

        return total_loss / max(len(grouped_episodes), 1) if total_episodes > 0 else torch.tensor(
            0.0, requires_grad=True, device=device
        )

    def adapt_for_inference(self, input_batch, target_batch=None, task_type="general"):
        """Adapt model for inference using task-aware CEM"""
        logger.info(f"Performing CEM adaptation for {task_type} task")

        try:
            adaptation_dim = self.get_total_adaptation_dim()
            if adaptation_dim == 0:
                logger.warning("No adaptation parameters available")
                return 0.0, []

            # Move batch to device
            for key in input_batch:
                if isinstance(input_batch[key], torch.Tensor):
                    input_batch[key] = input_batch[key].to(device)
            if target_batch is not None:
                target_batch = target_batch.to(device)

            # Perform task-aware CEM optimization
            best_params, best_score, history = self.cem_optimizer.optimize_adaptation(
                self, input_batch, target_batch, adaptation_dim, task_type
            )

            if best_params is not None:
                self.apply_adaptation_params(best_params)
                self.current_adaptation = best_params.clone()
                self.adaptation_history.append({
                    'params': best_params.detach().cpu().numpy(),
                    'score': best_score,
                    'task_type': task_type,
                    'convergence_history': history
                })
                logger.info(f"CEM adaptation completed for {task_type}. Score: {best_score:.4f}")
            else:
                logger.warning(f"CEM adaptation failed for {task_type}")
                best_score, history = -10.0, []

            return best_score, history

        except Exception as e:
            logger.error(f"CEM adaptation error: {str(e)}")
            return -10.0, []

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics"""
        stats = {
            "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            "gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
        }

        if self.config.enable_paged_attention and self.kv_cache:
            stats.update(self.kv_cache.get_memory_stats())

        return stats

    def cleanup_cache(self):
        """Manual cache cleanup"""
        if self.config.enable_paged_attention and self.kv_cache:
            # Force aggressive cleanup to free memory
            self.kv_cache._cleanup_lru_sequences(self.kv_cache.max_blocks // 4, aggressive=True)

# ### Fixed Trainer
class FixedEnhancedGRPOTrainer:
    def __init__(self, config: FixedEnhancedConfig):
        self.config = config
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Initializing fixed enhanced model")
        self.model = FixedEnhancedSelfAdaptiveGPT2(config)

        # Separate optimizers
        adaptation_params = list(self.model.adaptation_params.values())
        other_params = list(self.model.task_classifier.parameters())

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

        # Learning rate schedulers
        total_steps = config.num_epochs * 100  # Estimate
        self.policy_scheduler = get_cosine_schedule_with_warmup(
            self.policy_optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        self.value_scheduler = get_cosine_schedule_with_warmup(
            self.value_optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )

        # Initialize metrics
        self.training_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "adaptation_magnitude": [],
            "cem_scores": defaultdict(list),
            "task_rewards": defaultdict(list),
            "gpu_memory_usage": [],
            "training_speed": [],
            "learning_rates": [],
            "gradient_norms": [],
            "episode_lengths": [],
            "episode_length_distribution": defaultdict(list),
            "policy_ratios": [],
            "advantage_distributions": [],
            "cem_convergence": defaultdict(list),
            "dataset_stats": {},
            "paged_attention_stats": [],
            "cache_hit_rates": [],
            "temperature_schedule": []
        }

        # Load datasets
        self.dataset_loader = RobustDatasetLoader(config)
        self.datasets = self.dataset_loader.load_all_datasets()
        self.training_metrics["dataset_stats"] = {
            "successful_downloads": self.dataset_loader.successful_downloads,
            "failed_downloads": self.dataset_loader.failed_downloads,
            "total_samples": sum(len(data) for data in self.datasets.values()),
            "task_distribution": {task: len(data) for task, data in self.datasets.items()}
        }

        # Initialize reward function
        self.reward_function = TaskSpecificRewardFunction()

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        logger.info("Trainer initialized successfully")

    def create_optimized_dataloader(self, data, batch_size, is_validation=False):
        """Create an optimized DataLoader"""
        from torch.utils.data import Dataset, DataLoader

        class OptimizedDataset(Dataset):
            def __init__(self, data, tokenizer, max_length):
                self.data = [
                    (item[0], item[1], item[2]) for item in data
                    if isinstance(item, (list, tuple)) and len(item) >= 3 and
                    isinstance(item[0], str) and isinstance(item[1], str) and isinstance(item[2], str) and
                    len(item[0].strip()) > 0 and len(item[1].strip()) > 0
                ]
                self.tokenizer = tokenizer
                self.max_length = max_length
                logger.info(f"Dataset created with {len(self.data)} valid items")

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                try:
                    if idx >= len(self.data):
                        return None

                    input_text, target_text, task_type = self.data[idx]

                    inputs = self.tokenizer(
                        str(input_text),
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_length,
                        padding="max_length",
                        add_special_tokens=True
                    )

                    targets = self.tokenizer(
                        str(target_text),
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_length // 2,
                        padding="max_length",
                        add_special_tokens=True
                    )

                    return {
                        'input_ids': inputs['input_ids'].squeeze(0),
                        'attention_mask': inputs['attention_mask'].squeeze(0),
                        'target_ids': targets['input_ids'].squeeze(0),
                        'task_type': str(task_type),
                        'input_text': str(input_text)[:200],
                        'target_text': str(target_text)[:100]
                    }
                except Exception as e:
                    logger.error(f"Dataset item {idx} processing failed: {str(e)}")
                    return None

        dataset = OptimizedDataset(data, self.tokenizer, self.config.max_length)
        valid_items = [item for i in range(len(dataset)) if (item := dataset[i]) is not None]

        if not valid_items:
            logger.error("No valid dataset items found!")
            return None

        logger.info(f"Created DataLoader with {len(valid_items)} valid items")

        return DataLoader(
            valid_items,
            batch_size=batch_size,
            shuffle=not is_validation,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        """Custom collate function"""
        batch = [item for item in batch if item is not None]
        if not batch:
            return None

        collated = {}
        for key in batch[0].keys():
            if key in ['input_ids', 'attention_mask', 'target_ids']:
                collated[key] = torch.stack([item[key] for item in batch])
            else:
                collated[key] = [item[key] for item in batch]
        return collated

    def train_enhanced_grpo(self):
        """Train the model with fixed GRPO and monitoring"""
        logger.info("Starting fixed enhanced GRPO training")

        # Initialize wandb if configured
        if self.config.wandb_project:
            try:
                wandb.init(
                    project=self.config.wandb_project,
                    name=f"enhanced-grpo-fixed-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    config=self.config.__dict__
                )
                wandb.log(self.training_metrics["dataset_stats"])
            except Exception as e:
                logger.warning(f"Wandb initialization failed: {str(e)}")

        # Prepare data
        all_data = [item for task_data in self.datasets.values() for item in task_data]

        if not all_data:
            logger.error("No training data available")
            return

        logger.info(f"Total training samples: {len(all_data)}")

        # Create dataloader
        dataloader = self.create_optimized_dataloader(all_data, self.config.batch_size)
        if dataloader is None:
            logger.error("Failed to create training dataloader")
            return

        self.model.train()
        global_step = 0
        batch_count = 0

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            epoch_metrics = {
                'policy_loss': 0.0,
                'episodes': 0,
                'valid_episodes': 0,
                'total_reward': 0.0,
                'cache_hits': 0,
                'cache_misses': 0
            }

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

            for batch_idx, batch in enumerate(progress_bar):
                if batch is None:
                    continue

                batch_count += 1

                # Periodic cache cleanup
                if batch_count % self.config.cache_cleanup_interval == 0:
                    self.model.cleanup_cache()

                try:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    target_ids = batch['target_ids'].to(device)
                    task_types = batch['task_type']

                    # Generate episodes
                    episodes = []
                    for i in range(len(input_ids)):
                        try:
                            episode = self.model.generate_episode(
                                input_ids[i:i+1].contiguous(),
                                attention_mask[i:i+1].contiguous(),
                                max_new_tokens=None,
                                task_type=task_types[i]
                            )

                            if episode.generated_tokens.numel() > 0:
                                generated_text = self.tokenizer.decode(
                                    episode.generated_tokens[0],
                                    skip_special_tokens=True
                                )

                                reward = self.reward_function.compute_reward(
                                    generated_text,
                                    batch['target_text'][i],
                                    task_types[i]
                                )

                                episode.rewards = torch.full(
                                    episode.generated_tokens.size(),
                                    reward,
                                    device=device,
                                    dtype=torch.float32
                                )

                                episodes.append(episode)

                                # Track metrics
                                self.training_metrics["task_rewards"][task_types[i]].append(reward)
                                self.training_metrics["episode_lengths"].append(episode.episode_length)
                                self.training_metrics["episode_length_distribution"][task_types[i]].append(
                                    episode.episode_length
                                )
                                epoch_metrics['total_reward'] += reward

                        except Exception as e:
                            logger.error(f"Episode generation failed for sample {i}: {str(e)}")

                    if not episodes:
                        continue

                    # Compute GRPO loss
                    grpo_loss = self.model.compute_grpo_loss(episodes)

                    if not torch.isfinite(grpo_loss) or not grpo_loss.requires_grad:
                        logger.warning("Invalid GRPO loss, skipping batch")
                        continue

                    # Backward pass (no mixed precision)
                    (grpo_loss / self.config.gradient_accumulation_steps).backward()

                    # Gradient accumulation and optimizer step
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        # Gradient clipping
                        policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.policy_optimizer.param_groups[0]['params'],
                            self.config.max_grad_norm
                        )
                        value_grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.value_optimizer.param_groups[0]['params'],
                            self.config.max_grad_norm
                        )

                        # Optimizer step
                        self.policy_optimizer.step()
                        self.value_optimizer.step()

                        # Learning rate scheduling
                        if self.config.adaptive_learning_rate:
                            self.policy_scheduler.step()
                            self.value_scheduler.step()

                        # Zero gradients
                        self.policy_optimizer.zero_grad()
                        self.value_optimizer.zero_grad()

                        # Track gradient norms
                        self.training_metrics["gradient_norms"].append({
                            'policy': policy_grad_norm.item() if torch.isfinite(policy_grad_norm) else 0.0,
                            'value': value_grad_norm.item() if torch.isfinite(value_grad_norm) else 0.0
                        })

                        global_step += 1

                    # Update metrics
                    epoch_metrics['policy_loss'] += grpo_loss.item()
                    epoch_metrics['episodes'] += len(episodes)
                    epoch_metrics['valid_episodes'] += len([e for e in episodes if e.rewards.sum() != 0])

                    # Memory and cache statistics
                    memory_stats = self.model.get_memory_stats()
                    self.training_metrics["gpu_memory_usage"].append(memory_stats["gpu_memory_allocated"])

                    if self.config.enable_paged_attention:
                        paged_stats = {
                            'utilization': memory_stats.get("utilization", 0),
                            'active_sequences': memory_stats.get("active_sequences", 0),
                            'memory_mb': memory_stats.get("memory_mb", 0),
                            'allocation_count': memory_stats.get("allocation_count", 0),
                            'deallocation_count': memory_stats.get("deallocation_count", 0),
                            'cleanup_count': memory_stats.get("cleanup_count", 0),
                            'cache_hit_rate': memory_stats.get("cache_hit_rate", 0),
                            'cache_hits': memory_stats.get("cache_hits", 0),
                            'cache_misses': memory_stats.get("cache_misses", 0)
                        }
                        self.training_metrics["paged_attention_stats"].append(paged_stats)
                        self.training_metrics["cache_hit_rates"].append(paged_stats["cache_hit_rate"])
                        epoch_metrics['cache_hits'] += paged_stats['cache_hits']
                        epoch_metrics['cache_misses'] += paged_stats['cache_misses']

                    # Update progress bar
                    progress_data = {
                        'Loss': f'{grpo_loss.item():.4f}',
                        'Eps': len(episodes),
                        'Rwd': f'{epoch_metrics["total_reward"]/max(epoch_metrics["episodes"], 1):.3f}',
                        'GPU': f'{memory_stats["gpu_memory_allocated"]*1000:.0f}M'
                    }

                    if self.config.enable_paged_attention and self.training_metrics["paged_attention_stats"]:
                        latest_paged_stats = self.training_metrics["paged_attention_stats"][-1]
                        progress_data['Cache'] = f'{latest_paged_stats["utilization"]*100:.0f}%'
                        progress_data['Free'] = latest_paged_stats.get("free_blocks", 0)

                    progress_bar.set_postfix(progress_data)

                    # Logging
                    if batch_idx % self.config.log_interval == 0 and global_step > 0:
                        self._log_training_progress(epoch, batch_idx, grpo_loss.item(), global_step)

                    # Periodic memory cleanup
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {str(e)}")
                    traceback.print_exc()

            # End of epoch
            epoch_metrics['policy_loss'] /= max(len(dataloader), 1)
            self.training_metrics["policy_loss"].append(epoch_metrics['policy_loss'])
            self.training_metrics["learning_rates"].append({
                'policy': self.policy_optimizer.param_groups[0]['lr'],
                'value': self.value_optimizer.param_groups[0]['lr']
            })

            logger.info(
                f"Epoch {epoch + 1} completed:\n"
                f"  - Avg Policy Loss: {epoch_metrics['policy_loss']:.4f}\n"
                f"  - Episodes: {epoch_metrics['episodes']}\n"
                f"  - Avg Reward: {epoch_metrics['total_reward']/max(epoch_metrics['episodes'], 1):.3f}"
            )

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch + 1, global_step)

        logger.info("Training completed")

        # Final cleanup
        if self.config.enable_paged_attention and self.model.kv_cache:
            self.model.kv_cache.force_cleanup(keep_ratio=0.0)
        torch.cuda.empty_cache()
        gc.collect()

    def _log_training_progress(self, epoch, batch_idx, loss, global_step):
        """Log training progress to wandb"""
        if self.config.wandb_project and global_step > 0:
            try:
                log_dict = {
                    "train/policy_loss": loss,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                    "train/learning_rate": self.policy_optimizer.param_groups[0]['lr']
                }

                if self.training_metrics["gpu_memory_usage"]:
                    log_dict["train/gpu_memory_gb"] = self.training_metrics["gpu_memory_usage"][-1]

                if self.config.enable_paged_attention and self.training_metrics["paged_attention_stats"]:
                    latest_stats = self.training_metrics["paged_attention_stats"][-1]
                    log_dict.update({
                        "train/cache_utilization": latest_stats["utilization"],
                        "train/cache_hit_rate": latest_stats["cache_hit_rate"],
                        "train/free_blocks": latest_stats.get("free_blocks", 0),
                        "train/active_sequences": latest_stats["active_sequences"]
                    })

                # Log task-specific rewards
                for task, rewards in self.training_metrics["task_rewards"].items():
                    if rewards:
                        recent_rewards = rewards[-50:]
                        log_dict[f"train/reward_{task}"] = np.mean(recent_rewards)

                wandb.log(log_dict, step=global_step)

            except Exception as e:
                logger.warning(f"Wandb logging failed: {str(e)}")

    def save_checkpoint(self, epoch: int, global_step: int):
        """Save training checkpoint"""
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
                'value_optimizer_state_dict': self.value_optimizer.state_dict(),
                'training_metrics': self.training_metrics,
                'config': self.config
            }

            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")

# ### Main Execution
def main():
    """Main execution function"""
    logger.info("Starting Fixed Enhanced Pipeline")

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Initialize configuration
    config = FixedEnhancedConfig()

    logger.info(f"Configuration: {config.model_name}, Batch: {config.batch_size}, Epochs: {config.num_epochs}")

    try:
        # Initialize trainer
        trainer = FixedEnhancedGRPOTrainer(config)

        # Check data
        total_samples = sum(len(data) for data in trainer.datasets.values())
        if total_samples == 0:
            logger.error("No training data loaded!")
            return

        logger.info(f"Loaded {total_samples:,} training samples")

        # Start training
        trainer.train_enhanced_grpo()

        logger.info("Pipeline completed successfully!")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        traceback.print_exc()
    finally:
        # Cleanup
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
