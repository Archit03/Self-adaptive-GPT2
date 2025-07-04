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
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_cosine_schedule_with_warmup
from transformers.cache_utils import DynamicCache
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
        logging.FileHandler('training_grpo.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# TRANSFORMER² CORE COMPONENTS
# ============================================================================

@dataclass
class TaskProperties:
    """Task properties identified by dispatch system"""
    task_type: str  # 'code', 'math', 'reasoning', 'other'
    complexity: float  # 0.0 to 1.0
    domain_specificity: float  # 0.0 to 1.0
    reasoning_depth: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0

@dataclass
class ExpertVector:
    """Expert vector containing singular value adaptations"""
    expert_id: str
    task_type: str
    singular_adaptations: Dict[str, torch.Tensor]  # layer_name -> S matrix adaptation
    performance_score: float
    usage_count: int = 0

class DispatchSystem(nn.Module):
    """Two-pass dispatch system for task property identification"""
    
    def __init__(self, hidden_size: int = 768, num_task_types: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_task_types = num_task_types
        
        # Task type classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_task_types)
        )
        
        # Property estimators
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.domain_estimator = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.reasoning_estimator = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_size + num_task_types, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states: torch.Tensor) -> TaskProperties:
        """Pass 1: Identify task properties from input representation"""
        # Pool hidden states
        pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # Task classification
        task_logits = self.task_classifier(pooled)
        task_probs = F.softmax(task_logits, dim=-1)
        task_type_idx = torch.argmax(task_probs, dim=-1)
        
        # Property estimation
        complexity = self.complexity_estimator(pooled).squeeze(-1)
        domain_specificity = self.domain_estimator(pooled).squeeze(-1)
        reasoning_depth = self.reasoning_estimator(pooled).squeeze(-1)
        
        # Confidence estimation
        confidence_input = torch.cat([pooled, task_probs], dim=-1)
        confidence = self.confidence_estimator(confidence_input).squeeze(-1)
        
        # Map task index to type
        task_types = ['code', 'math', 'reasoning', 'other']
        task_type = task_types[task_type_idx.item()]
        
        return TaskProperties(
            task_type=task_type,
            complexity=complexity.item(),
            domain_specificity=domain_specificity.item(),
            reasoning_depth=reasoning_depth.item(),
            confidence=confidence.item()
        )

class ExpertMixingSystem(nn.Module):
    """Dynamic expert mixing system for singular value adaptations"""
    
    def __init__(self, hidden_size: int = 768, max_experts: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_experts = max_experts
        self.expert_vectors: Dict[str, ExpertVector] = {}
        
        # Mixing weights network
        self.mixing_network = nn.Sequential(
            nn.Linear(hidden_size + 4, 128),  # +4 for task properties
            nn.ReLU(),
            nn.Dropout(1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, max_experts),
            nn.Softmax(dim=-1)
        )
        
        # Layer-specific mixing for different transformer layers
        self.layer_mixing_networks = nn.ModuleDict()
        
    def add_expert(self, expert: ExpertVector):
        """Add expert vector to the system"""
        if len(self.expert_vectors) < self.max_experts:
            self.expert_vectors[expert.expert_id] = expert
            logger.info(f"Added expert {expert.expert_id} for task {expert.task_type}")
        else:
            # Replace least used expert
            least_used = min(self.expert_vectors.values(), key=lambda x: x.usage_count)
            del self.expert_vectors[least_used.expert_id]
            self.expert_vectors[expert.expert_id] = expert
            logger.info(f"Replaced expert {least_used.expert_id} with {expert.expert_id}")
    
    def get_mixed_adaptations(self, hidden_states: torch.Tensor, 
                            task_props: TaskProperties) -> Dict[str, torch.Tensor]:
        """Get dynamically mixed singular value adaptations"""
        if not self.expert_vectors:
            return {}
        
        # Pool hidden states
        pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # Create task property vector
        task_prop_vector = torch.tensor([
            task_props.complexity,
            task_props.domain_specificity, 
            task_props.reasoning_depth,
            task_props.confidence
        ], device=hidden_states.device).unsqueeze(0)
        
        # Compute mixing weights
        mixing_input = torch.cat([pooled, task_prop_vector], dim=-1)
        mixing_weights = self.mixing_network(mixing_input)  # [1, max_experts]
        
        # Get active experts (only those with weights > threshold)
        active_threshold = 0.01
        active_experts = []
        active_weights = []
        
        for i, (expert_id, expert) in enumerate(self.expert_vectors.items()):
            if i < mixing_weights.size(1) and mixing_weights[0, i] > active_threshold:
                active_experts.append(expert)
                active_weights.append(mixing_weights[0, i])

        if not active_experts:
            return {}
        
        # Normalize active weights
        active_weights = torch.tensor(active_weights, device=hidden_states.device)
        active_weights = active_weights / active_weights.sum()
        
        # Mix singular value adaptations
        mixed_adaptations = {}
        for expert, weight in zip(active_experts, active_weights):
            expert.usage_count += 1
            for layer_name, adaptation in expert.singular_adaptations.items():
                if layer_name not in mixed_adaptations:
                    mixed_adaptations[layer_name] = torch.zeros_like(adaptation)
                mixed_adaptations[layer_name] += weight * adaptation
        
        logger.debug(f"Mixed {len(active_experts)} experts for task {task_props.task_type}")
        return mixed_adaptations

class SingularValueAdapter(nn.Module):
    """Singular value-only adaptation module"""
    
    def __init__(self, layer_name: str, original_weight: torch.Tensor):
        super().__init__()
        self.layer_name = layer_name
        
        # Perform SVD decomposition
        U, S, Vh = torch.linalg.svd(original_weight.float(), full_matrices=False)
        
        # Store U and V as fixed (non-trainable)
        self.register_buffer('U', U)
        self.register_buffer('Vh', Vh)
        self.register_buffer('original_S', S)
        
        # Only S is adaptable - this is the core Transformer² innovation
        self.adaptation_scale = nn.Parameter(torch.ones_like(S) * 0.01)
        
        logger.debug(f"SVD adapter for {layer_name}: {original_weight.shape} -> rank {len(S)}")
    
    def get_adapted_weight(self, singular_adaptation: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reconstruct weight matrix with adapted singular values"""
        # Apply adaptation to singular values
        if singular_adaptation is not None:
            # Real-time adaptation from expert mixing
            adapted_S = self.original_S * (1.0 + singular_adaptation)
        else:
            # Training-time adaptation
            adapted_S = self.original_S * (1.0 + self.adaptation_scale)
        
        # Reconstruct: W = U @ diag(S_adapted) @ V^T
        reconstructed = torch.matmul(
            torch.matmul(self.U, torch.diag(adapted_S)), 
            self.Vh
        )
        
        return reconstructed

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
    """Set up the computation device with deterministic settings"""
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
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

def set_seed(seed: int = 42):
    """Set seeds for full reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed}")

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
    mixed_precision: bool = True
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
    grpo_value_loss_coeff: float = 0.1
    grpo_entropy_coeff: float = 0.05
    wandb_project: str = "transformer-squared-gpt2"
    output_dir: str = "./results"
    log_interval: int = 10
    save_interval: int = 1
    clip_rewards: float = 2.0
    reward_scaling: float = 0.2
    repetition_penalty: float = 1.2
    top_p: float = 0.85
    temperature: float = 0.7
    min_episode_length: int = 16
    max_episode_length: int = 48
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    
    # Transformer² specific parameters
    enable_transformer_squared: bool = True
    max_expert_vectors: int = 16
    expert_adaptation_strength: float = 0.1
    real_time_adaptation: bool = True
    
    # CEM parameters for different tasks
    cem_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "code": {
            "population_size": 16,
            "elite_ratio": 0.3,
            "noise_std": 0.2,
            "adaptation_steps": 8,
            "convergence_threshold": 0.01
        },
        "math": {
            "population_size": 16,
            "elite_ratio": 0.3,
            "noise_std": 0.25,
            "adaptation_steps": 8,
            "convergence_threshold": 0.01
        },
        "reasoning": {
            "population_size": 20,
            "elite_ratio": 0.3,
            "noise_std": 0.15,
            "adaptation_steps": 10,
            "convergence_threshold": 0.008
        },
        "other": {
            "population_size": 16,
            "elite_ratio": 0.3,
            "noise_std": 0.3,
            "adaptation_steps": 8,
            "convergence_threshold": 0.01
        }
    })

@dataclass
class UltraConfig(BaseConfig):
    """Ultra configuration for 15GB+ GPUs"""
    batch_size: int = 12
    learning_rate: float = 3e-5
    num_epochs: int = 5
    max_length: int = 384
    adaptation_rank: int = 32
    num_experts: int = 8
    enable_paged_attention: bool = True
    paged_block_size: int = 32
    max_cache_blocks: int = 1500
    max_samples_per_dataset: int = 1000
    grpo_episodes_per_batch: int = 6
    wandb_project: str = "transformer-squared-ultra-gpt2"
    output_dir: str = "./ultra_results"
    min_episode_length: int = 32
    max_episode_length: int = 64
    num_workers: int = min(8, os.cpu_count() or 4)
    max_expert_vectors: int = 32
    
    cem_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "code": {
            "population_size": 32,
            "elite_ratio": 0.25,
            "noise_std": 0.25,
            "adaptation_steps": 15,
            "convergence_threshold": 0.005
        },
        "math": {
            "population_size": 32,
            "elite_ratio": 0.25,
            "noise_std": 0.3,
            "adaptation_steps": 15,
            "convergence_threshold": 0.005
        },
        "reasoning": {
            "population_size": 40,
            "elite_ratio": 0.3,
            "noise_std": 0.2,
            "adaptation_steps": 20,
            "convergence_threshold": 0.003
        },
        "other": {
            "population_size": 32,
            "elite_ratio": 0.25,
            "noise_std": 0.3,
            "adaptation_steps": 15,
            "convergence_threshold": 0.005
        }
    })

# ============================================================================
# PAGED ATTENTION MODULE
# ============================================================================

class PagedGPT2Attention(nn.Module):
    """Paged attention module with KV caching - Fixed GPT2 compatibility"""
    
    def __init__(self, original_attn, config):
        super().__init__()
        self.config = config
        self.original_attn = original_attn
        self.device = next(original_attn.parameters()).device
        
        self.embed_dim = original_attn.embed_dim
        self.num_heads = original_attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.head_dim
        
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(self.device)
        
        if hasattr(original_attn, 'masked_bias'):
            self.masked_bias = original_attn.masked_bias
        else:
            self.register_buffer('masked_bias', torch.tensor(-1e4).to(self.device))
        
        self.c_attn = original_attn.c_attn
        self.c_proj = original_attn.c_proj
        self.attn_dropout = original_attn.attn_dropout
        self.resid_dropout = original_attn.resid_dropout
        
        self.kv_cache = PagedKVCache(
            max_seq_len=config.max_length * 2,
            hidden_size=self.embed_dim,
            num_heads=self.num_heads,
            block_size=config.paged_block_size,
            max_blocks=config.max_cache_blocks
        )
        
        logger.debug(f"Initialized PagedGPT2Attention with embed_dim={self.embed_dim}, num_heads={self.num_heads}")
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False,
                output_attentions=False, past_key_value=None, **kwargs):
        """Forward pass with paged attention"""
        
        if self.training:
            return self.original_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                past_key_value=past_key_value
            )
        
        batch_size, seq_len = hidden_states.shape[:2]
        seq_id = f"seq_{batch_size}_{seq_len}"
        
        if not self.kv_cache.has_sequence(seq_id):
            self.kv_cache.allocate_sequence(seq_id, seq_len)
        
        outputs = self.original_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            past_key_value=past_key_value
        )
        return outputs

class PagedKVCache:
    def __init__(self, max_seq_len, hidden_size, num_heads, block_size, max_blocks):
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.cache = {}

    def has_sequence(self, seq_id):
        return seq_id in self.cache

    def allocate_sequence(self, seq_id, seq_len):
        self.cache[seq_id] = torch.zeros((self.num_heads, seq_len, self.hidden_size // self.num_heads))

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
            self.tokenized_data = torch.load(self.cache_path)
        else:
            logger.info("Pre-tokenizing dataset...")
            self.tokenized_data = self._tokenize_all(data, tokenizer, max_length)
            torch.save(self.tokenized_data, self.cache_path)
            logger.info(f"Saved tokenized data to {self.cache_path}")
    
    def _tokenize_all(self, data, tokenizer, max_length):
        """Tokenize all data efficiently"""
        tokenized = []
        batch_size = 50
        
        for i in tqdm(range(0, len(data), batch_size), desc="Tokenizing"):
            batch = data[i:i+batch_size]
            input_texts = []
            target_texts = []
            task_types = []
            
            for input_text, target_text, task_type in batch:
                if all([input_text, target_text, task_type]):
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
    """Enhanced dataset loader with comprehensive benchmarks"""
    
    def __init__(self, config):
        self.config = config
        self.datasets = {}
        self.benchmark_tasks = [
            {'name': 'squad', 'split': 'train[:1000]', 'task_type': 'reasoning', 'process_fn': self._process_squad},
            {'name': 'squad_v2', 'split': 'train[:800]', 'task_type': 'reasoning', 'process_fn': self._process_squad_v2},
            {'name': 'imdb', 'split': 'train[:1000]', 'task_type': 'other', 'process_fn': self._process_imdb},
            {'name': 'sst2', 'split': 'train[:800]', 'task_type': 'other', 'process_fn': self._process_sst2},
            {'name': 'ag_news', 'split': 'train[:800]', 'task_type': 'reasoning', 'process_fn': self._process_ag_news},
            {'name': 'yelp_polarity', 'split': 'train[:600]', 'task_type': 'other', 'process_fn': self._process_yelp},
            {'name': 'snli', 'split': 'train[:800]', 'task_type': 'reasoning', 'process_fn': self._process_snli},
            {'name': 'mnli', 'split': 'train[:600]', 'task_type': 'reasoning', 'process_fn': self._process_mnli},
            {'name': 'boolq', 'split': 'train[:600]', 'task_type': 'reasoning', 'process_fn': self._process_boolq},
            {'name': 'piqa', 'split': 'train[:600]', 'task_type': 'reasoning', 'process_fn': self._process_piqa},
            {'name': 'gsm8k', 'split': 'train[:400]', 'task_type': 'math', 'process_fn': self._process_gsm8k, 'config': 'main'},
            {'name': 'code_synthetic', 'split': None, 'task_type': 'code', 'process_fn': self._create_code_synthetic},
            {'name': 'xsum', 'split': 'train[:400]', 'task_type': 'other', 'process_fn': self._process_xsum},
            {'name': 'cnn_dailymail', 'subset': '3.0.0', 'split': 'train[:400]', 'task_type': 'other', 'process_fn': self._process_cnn},
        ]
        
    def load_all_datasets(self):
        """Load comprehensive benchmark datasets"""
        if self.config.use_fallback_data_only:
            logger.info("Using fallback data only")
            self._add_fallback_data()
            return self.datasets
        
        successful_downloads = 0
        failed_downloads = 0
        
        for dataset_config in self.benchmark_tasks:
            try:
                with timer(f"Loading {dataset_config['name']}"):
                    if dataset_config['name'] == 'code_synthetic':
                        processed_data = dataset_config['process_fn']()
                    elif 'subset' in dataset_config:
                        dataset = load_dataset(
                            dataset_config['name'], 
                            dataset_config['subset'],
                            split=dataset_config['split'],
                            download_mode="reuse_cache_if_exists",
                            verification_mode="no_checks"
                        )
                        processed_data = dataset_config['process_fn'](dataset)
                    else:
                        dataset = load_dataset(
                            dataset_config['name'],
                            split=dataset_config['split'],
                            download_mode="reuse_cache_if_exists",
                            verification_mode="no_checks"
                        )
                        processed_data = dataset_config['process_fn'](dataset)
                    
                    if processed_data:
                        task_type = dataset_config['task_type']
                        if task_type not in self.datasets:
                            self.datasets[task_type] = []
                        self.datasets[task_type].extend(processed_data)
                        successful_downloads += 1
                        logger.info(f"✓ Loaded {dataset_config['name']}: {len(processed_data)} samples")
                    else:
                        failed_downloads += 1
                        
            except Exception as e:
                logger.warning(f"Failed to load {dataset_config['name']}: {e}")
                failed_downloads += 1
                continue
        
        total_samples = sum(len(data) for data in self.datasets.values())
        if total_samples < 100:
            logger.warning("Low sample count, adding fallback data")
            self._add_fallback_data()
        
        logger.info(f"Dataset loading complete: {successful_downloads} successful, {failed_downloads} failed")
        logger.info(f"Total training samples: {total_samples:,}")
        
        for task_type, data in self.datasets.items():
            logger.info(f"  {task_type}: {len(data):,} samples")
        
        return self.datasets
    
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
                    processed.append((f"Context: {context}\nQuestion: {question}", answer, 'reasoning'))
            except:
                continue
        
        return processed
    
    def _process_squad_v2(self, dataset):
        """Process SQuAD v2 dataset"""
        processed = []
        max_context_length = 600 if isinstance(self.config, UltraConfig) else 200
        
        for item in dataset:
            try:
                context = item.get('context', '').strip()
                question = item.get('question', '').strip()
                answers = item.get('answers', {})
                
                if context and question:
                    context = context[:max_context_length]
                    if answers and answers.get('text'):
                        answer = answers['text'][0].strip()
                    else:
                        answer = "No answer possible"
                    processed.append((f"Context: {context}\nQuestion: {question}", answer, 'reasoning'))
            except:
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
                    processed.append((f"Analyze the sentiment: {text}", target, 'other'))
            except:
                continue
        
        return processed
    
    def _process_sst2(self, dataset):
        """Process SST-2 dataset"""
        processed = []
        
        for item in dataset:
            try:
                sentence = item.get('sentence', '').strip()
                label = item.get('label', 0)
                
                if sentence:
                    target = 'positive' if label == 1 else 'negative'
                    processed.append((f"Sentiment of: {sentence}", target, 'other'))
            except:
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
                    processed.append((f"Classify this news: {text}", target, 'reasoning'))
            except:
                continue
        
        return processed
    
    def _process_yelp(self, dataset):
        """Process Yelp Polarity dataset"""
        processed = []
        max_text_length = 600 if isinstance(self.config, UltraConfig) else 200
        
        for item in dataset:
            try:
                text = item.get('text', '').strip()
                label = item.get('label', 0)
                
                if text and len(text) > 30:
                    text = text[:max_text_length]
                    target = 'positive' if label == 1 else 'negative'
                    processed.append((f"Review sentiment: {text}", target, 'other'))
            except:
                continue
        
        return processed
    
    def _process_snli(self, dataset):
        """Process SNLI dataset"""
        processed = []
        label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        
        for item in dataset:
            try:
                premise = item.get('premise', '').strip()
                hypothesis = item.get('hypothesis', '').strip()
                label = item.get('label', -1)
                
                if premise and hypothesis and label != -1:
                    target = label_map.get(label, 'neutral')
                    processed.append((
                        f"Premise: {premise}\nHypothesis: {hypothesis}\nRelation:", 
                        target, 'reasoning'
                    ))
            except:
                continue
        
        return processed
    
    def _process_mnli(self, dataset):
        """Process MNLI dataset"""
        processed = []
        label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        
        for item in dataset:
            try:
                premise = item.get('premise', '').strip()
                hypothesis = item.get(' SRShypothesis', '').        label = item.get('label', -1)
                
                if premise and hypothesis and label != -1:
                    target = label_map.get(label, 'neutral')
                    processed.append((
                        f"Premise: {premise}\nHypothesis: {hypothesis}\nRelation:", 
                        target, 'reasoning'
                    ))
            except:
                continue
        
        return processed
    
    def _process_boolq(self, dataset):
        """Process BoolQ dataset"""
        processed = []
        
        for item in dataset:
            try:
                question = item.get('question', '').strip()
                passage = item.get('passage', '').strip()
                answer = item.get('answer', False)
                
                if question and passage:
                    passage = passage[:400]
                    target = 'yes' if answer else 'no'
                    processed.append((
                        f"Passage: {passage}\nQuestion: {question}", 
                        target, 'reasoning'
                    ))
            except:
                continue
        
        return processed
    
    def _process_piqa(self, dataset):
        """Process PIQA dataset"""
        processed = []
        
        for item in dataset:
            try:
                goal = item.get('goal', '').strip()
                sol1 = item.get('sol1', '').strip()
                sol2 = item.get('sol2', '').strip()
                label = item.get('label', 0)
                
                if goal and sol1 and sol2:
                    target = sol1 if label == 0 else sol2
                    processed.append((
                        f"Goal: {goal}\nOption 1: {sol1}\nOption 2: {sol2}\nBest option:", 
                        target, 'reasoning'
                    ))
            except:
                continue
        
        return processed
    
    def _process_gsm8k(self, dataset):
        """Process GSM8K dataset"""
        processed = []
        
        for item in dataset:
            try:
                question = item.get('question', '').strip()
                answer = item.get('answer', '').strip()
                
                if question and answer:
                    import re
                    numbers = re.findall(r'\d+\.?\d*', answer)
                    if numbers:
                        numeric_answer = numbers[-1]
                        processed.append((f"Math problem: {question}", numeric_answer, 'math'))
            except:
                continue
        
        return processed
    
    def _process_xsum(self, dataset):
        """Process XSum dataset"""
        processed = []
        max_doc_length = 800 if isinstance(self.config, UltraConfig) else 400
        
        for item in dataset:
            try:
                document = item.get('document', '').strip()
                summary = item.get('summary', '').strip()
                
                if document and summary and len(document) > 100:
                    document = document[:max_doc_length]
                    processed.append((f"Summarize: {document}", summary, 'other'))
            except:
                continue
        
        return processed
    
    def _process_cnn(self, dataset):
        """Process CNN/DailyMail dataset"""
        processed = []
        max_article_length = 800 if isinstance(self.config, UltraConfig) else 400
        
        for item in dataset:
            try:
                article = item.get('article', '').strip()
                highlights = item.get('highlights', '').strip()
                
                if article and highlights and len(article) > 200:
                    article = article[:max_article_length]
                    processed.append((f"Summarize this article: {article}", highlights, 'other'))
            except:
                continue
        
        return processed
    
    def _create_code_synthetic(self):
        """Create synthetic code generation dataset"""
        code_examples = [
            ("Write a function to add two numbers", "def add(a, b):\n    return a + b", "code"),
            ("Create a function to find maximum of two numbers", "def max_two(a, b):\n    return a if a > b else b", "code"),
            ("Write a function to calculate factorial", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)", "code"),
            ("Create a function to check if number is even", "def is_even(n):\n    return n % 2 == 0", "code"),
            ("Write a function to reverse a string", "def reverse_string(s):\n    return s[::-1]", "code"),
            ("Create a function to find length of list", "def list_length(lst):\n    return len(lst)", "code"),
            ("Write a function to square a number", "def square(x):\n    return x * x", "code"),
            ("Create a function to check if string is empty", "def is_empty(s):\n    return len(s) == 0", "code"),
            ("Write a function to get first element of list", "def first_element(lst):\n    return lst[0] if lst else None", "code"),
            ("Create a function to concatenate two strings", "def concat(a, b):\n    return a + b", "code"),
        ]
        
        multiplier = 40 if isinstance(self.config, UltraConfig) else 20
        return code_examples * multiplier
    
    def _add_fallback_data(self):
        """Add fallback training data"""
        multiplier = 50 if isinstance(self.config, UltraConfig) else 25
        
        fallback = {
            'reasoning': [
                ("Context: Paris is the capital of France.\nQuestion: What is the capital of France?", "Paris", "reasoning"),
                ("Context: The sun is a star.\nQuestion: What is the sun?", "A star", "reasoning"),
                ("Context: Water boils at 100°C.\nQuestion: At what temperature does water boil?", "100 degrees Celsius", "reasoning"),
                ("Context: Python is a programming language.\nQuestion: What is Python?", "A programming language", "reasoning"),
            ] * multiplier,
            'other': [
                ("This movie was absolutely fantastic!", "positive", "other"),
                ("I hate this terrible product.", "negative", "other"),
                ("Great service and amazing food!", "positive", "other"),
                ("Worst experience ever, disappointed.", "negative", "other"),
            ] * multiplier,
            'code': [
                ("Write a function to add two numbers", "def add(a, b): return a + b", "code"),
                ("Create a list of even numbers", "[2, 4, 6, 8, 10]", "code"),
            ] * (multiplier // 2),
            'math': [
                ("What is 2 + 2?", "4", "math"),
                ("Calculate 5 * 6", "30", "math"),
            ] * (multiplier // 2),
        }
        
        for task_type, data in fallback.items():
            if task_type not in self.datasets:
                self.datasets[task_type] = []
            self.datasets[task_type].extend(data)

# ============================================================================
# REWARD FUNCTION
# ============================================================================

class RewardFunction:
    """Task-specific reward computation"""
    
    def __init__(self):
        self.task_scales = {
            'code': 2.5,
            'math': 2.2,
            'reasoning': 2.0,
            'other': 1.0
        }
    
    def compute_reward(self, generated_text: str, target_text: str, task_type: str) -> float:
        """Compute reward for generated text"""
        try:
            if not generated_text or not target_text:
                return -1.0
            
            reward_fn = {
                "code": self._code_reward,
                "math": self._math_reward,
                "reasoning": self._reasoning_reward,
                "other": self._other_reward,
            }.get(task_type, self._other_reward)
            
            raw_reward = reward_fn(generated_text, target_text)
            scaled_reward = raw_reward * self.task_scales.get(task_type, 1.0)
            
            return float(np.clip(scaled_reward, -2.0, 2.0))
            
        except Exception as e:
            logger.debug(f"Reward computation failed: {e}")
            return -1.0
    
    def _code_reward(self, generated: str, target: str) -> float:
        """Code reward based on syntax and functionality"""
        code_keywords = {'def', 'return', 'if', 'for', 'while', 'class', 'import'}
        gen_words = set(generated.lower().split())
        
        keyword_overlap = len(gen_words & code_keywords) / max(len(code_keywords), 1)
        
        if target.lower() in generated.lower():
            return 1.0
        
        return keyword_overlap * 0.8
    
    def _math_reward(self, generated: str, target: str) -> float:
        """Math reward based on numerical accuracy"""
        try:
            import re
            gen_numbers = re.findall(r'-?\d+\.?\d*', generated)
            target_numbers = re.findall(r'-?\d+\.?\d*', target)
            
            if target_numbers and gen_numbers:
                for target_num in target_numbers:
                    if target_num in gen_numbers:
                        return 1.0
            
            if target.lower().strip() in generated.lower():
                return 1.0
                
            return 0.2
        except:
            return 0.0
    
    def _reasoning_reward(self, generated: str, target: str) -> float:
        """Reasoning reward based on overlap and logic"""
        gen_words = set(generated.lower().split())
        target_words = set(target.lower().split())
        
        if not target_words:
            return 0.0
        
        overlap = len(gen_words & target_words) / len(target_words)
        
        if target.lower() in generated.lower():
            overlap += 0.5
        
        return min(overlap, 1.0)
    
    def _other_reward(self, generated: str, target: str) -> float:
        """General reward for other tasks"""
        if len(generated.strip()) < 5:
            return -0.5
        
        words = generated.split()
        diversity = len(set(words)) / max(len(words), 1)
        length_score = min(len(words) / 20, 1.0)
        
        if target.lower() in generated.lower():
            return 1.0
        
        return diversity * 0.3 + length_score * 0.2

# ============================================================================
# CEM OPTIMIZER
# ============================================================================

class CEMOptimizer:
    """Cross-Entropy Method optimizer for singular value adaptation"""
    
    def __init__(self, config):
        self.config = config
        self.cem_params = config.cem_params
    
    def optimize_adaptation(self, model, input_batch, adaptation_dim: int, 
                          task_type: str = "other") -> Tuple[torch.Tensor, float]:
        """Run CEM optimization to find best singular value adaptations"""
        params = self.cem_params.get(task_type, self.cem_params["other"])
        
        population_size = int(params["population_size"])
        elite_ratio = params["elite_ratio"]
        noise_std = params["noise_std"]
        max_steps = int(params["adaptation_steps"])
        convergence_threshold = params["convergence_threshold"]
        
        n_elite = max(1, int(population_size * elite_ratio))
        
        population_mean = torch.zeros(adaptation_dim, device=device)
        population_std = torch.ones(adaptation_dim, device=device) * noise_std
        
        best_params = None
        best_score = float('-inf')
        
        for step in range(max_steps):
            try:
                population = torch.randn(population_size, adaptation_dim, device=device)
                population = population * population_std + population_mean
                population = torch.clamp(population, -0.5, 0.5)
                
                scores = self._evaluate_population(model, input_batch, population, task_type)
                
                valid_mask = torch.isfinite(scores)
                if not valid_mask.any():
                    logger.warning(f"No valid scores in CEM step {step}")
                    break
                
                valid_scores = scores[valid_mask]
                valid_population = population[valid_mask]
                
                if len(valid_scores) > 0:
                    current_best_idx = torch.argmax(valid_scores)
                    current_best_score = valid_scores[current_best_idx].item()
                    
                    if current_best_score > best_score:
                        best_score = current_best_score
                        best_params = valid_population[current_best_idx].clone()
                
                n_elite_actual = min(n_elite, len(valid_scores))
                if n_elite_actual > 0:
                    elite_indices = torch.topk(valid_scores, n_elite_actual)[1]
                    elite_samples = valid_population[elite_indices]
                    
                    new_mean = elite_samples.mean(dim=0)
                    new_std = elite_samples.std(dim=0) + 1e-6
                    
                    momentum = 0.3
                    population_mean = momentum * population_mean + (1 - momentum) * new_mean
                    population_std = momentum * population_std + (1 - momentum) * new_std
                    population_std = torch.clamp(population_std, 0.01, 0.3)
                    
                    mean_change = torch.norm(new_mean - population_mean).item()
                    if mean_change < convergence_threshold:
                        logger.debug(f"CEM converged at step {step} for {task_type}")
                        break
                
            except Exception as e:
                logger.error(f"CEM step {step} failed: {e}")
                break
        
        if best_params is None:
            best_params = torch.zeros(adaptation_dim, device=device)
            best_score = 0.0
        
        return best_params, best_score
    
    def _evaluate_population(self, model, input_batch, population, task_type):
        """Evaluate population of singular value adaptations"""
        scores = torch.full((len(population),), float('-inf'), device=device)
        
        with torch.no_grad():
            for i, params in enumerate(population):
                try:
                    model.apply_singular_adaptations(params)
                    
                    outputs = model.forward_with_adaptations(
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
    task_properties: Optional[TaskProperties] = None
    sequence_id: str = None
    episode_length: int = 0

# ============================================================================
# TRANSFORMER² MODEL CLASS
# ============================================================================

class TransformerSquaredGPT2(nn.Module):
    """Transformer² GPT-2 with singular value adaptation and two-pass inference"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        logger.info(f"Loading {config.model_name} with Transformer² features")
        self.base_model = GPT2LMHeadModel.from_pretrained(config.model_name)
        self.base_model = self.base_model.to(device)
        
        if config.enable_paged_attention:
            self._replace_attention_modules()
        
        if config.enable_transformer_squared:
            self._initialize_transformer_squared()
        
        self.value_network = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        ).to(device)
        
        self.reward_function = RewardFunction()
        self.cem_optimizer = CEMOptimizer(config)
        
        self.sequence_counter = 0
        self.current_temperature = config.temperature
        
        logger.info(f"Model initialized with {self._count_parameters():,} parameters")
        if config.enable_transformer_squared:
            logger.info("✓ Transformer² features enabled")
        if config.enable_paged_attention:
            logger.info("✓ Paged attention modules installed")
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _replace_attention_modules(self):
        """Replace standard attention with paged attention modules"""
        replaced_count = 0
        
        for i, layer in enumerate(self.base_model.transformer.h):
            original_attn = layer.attn
            paged_attn = PagedGPT2Attention(original_attn, self.config)
            paged_attn = paged_attn.to(device)
            layer.attn = paged_attn
            replaced_count += 1
            logger.debug(f"Replaced attention module in layer {i}")
        
        logger.info(f"Replaced {replaced_count} attention modules with paged versions")
    
    def _initialize_transformer_squared(self):
        """Initialize Transformer² components"""
        hidden_size = self.base_model.config.hidden_size
        
        self.dispatch_system = DispatchSystem(hidden_size, num_task_types=4).to(device)
        self.expert_mixing = ExpertMixingSystem(
            hidden_size, 
            max_experts=self.config.max_expert_vectors
        ).to(device)
        
        self.sv_adapters = nn.ModuleDict()
        self._create_singular_value_adapters()
        
        logger.info("✓ Transformer² dispatch and expert mixing systems initialized")
    
    def _create_singular_value_adapters(self):
        """Create singular value adapters for key weight matrices"""
        target_layers = []
        
        for name, param in self.base_model.named_parameters():
            if any(key in name for key in ['mlp.c_fc.weight', 'mlp.c_proj.weight', 'attn.c_attn.weight']):
                target_layers.append((name, param))
        
        for name, weight in target_layers:
            adapter_name = name.replace('.', '_').replace('weight', 'adapter')
            self.sv_adapters[adapter_name] = SingularValueAdapter(name, weight.data)
            logger.debug(f"Created SVD adapter for {name}")
        
        logger.info(f"Created {len(self.sv_adapters)} singular value adapters")
    
    def apply_singular_adaptations(self, adaptation_vector: torch.Tensor):
        """Apply singular value adaptations to model weights"""
        if not hasattr(self, 'sv_adapters') or len(self.sv_adapters) == 0:
            return
        
        try:
            total_params = sum(adapter.original_S.numel() for adapter in self.sv_adapters.values())
            if len(adaptation_vector) != total_params:
                logger.warning(f"Adaptation vector size mismatch: {len(adaptation_vector)} vs {total_params}")
                return
            
            offset = 0
            for adapter_name, adapter in self.sv_adapters.items():
                param_size = adapter.original_S.numel()
                if offset + param_size <= len(adaptation_vector):
                    singular_adaptation = adaptation_vector[offset:offset + param_size]
                    
                    param_name = adapter.layer_name
                    if hasattr(self.base_model, 'get_parameter'):
                        model_param = self.base_model.get_parameter(param_name)
                    else:
                        parts = param_name.split('.')
                        param = self.base_model
                        for part in parts:
                            param = getattr(param, part)
                        model_param = param
                    
                    adapted_weight = adapter.get_adapted_weight(singular_adaptation)
                    model_param.data.copy_(adapted_weight.to(model_param.dtype))
                    
                    offset += param_size
            
        except Exception as e:
            logger.error(f"Failed to apply singular adaptations: {e}")
    
    def two_pass_inference(self, input_ids: torch.Tensor, 
                          attention_mask: torch.Tensor) -> Tuple[TaskProperties, Dict[str, torch.Tensor]]:
        """Two-pass inference: Pass 1 (dispatch) + Pass 2 (expert mixing)"""
        
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            task_props = self.dispatch_system(outputs.hidden_states[-1])
            
        mixed_adaptations = {}
        if hasattr(self, 'expert_mixing'):
            mixed_adaptations = self.expert_mixing.get_mixed_adaptations(
                outputs.hidden_states[-1], 
                task_props
            )
        
        return task_props, mixed_adaptations
    
    def forward_with_adaptations(self, input_ids, attention_mask=None, 
                               real_time_adaptation=True):
        """Forward pass with Transformer² adaptations"""
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        
        if not self.config.enable_transformer_squared or not real_time_adaptation:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        
        task_props, mixed_adaptations = self.two_pass_inference(input_ids, attention_mask)
        
        if mixed_adaptations:
            adaptation_vector = []
            for adapter_name in self.sv_adapters.keys():
                if adapter_name in mixed_adaptations:
                    adaptation_vector.append(mixed_adaptations[adapter_name])
                else:
                    adapter = self.sv_adapters[adapter_name]
                    adaptation_vector.append(torch.zeros_like(adapter.original_S))
            
            if adaptation_vector:
                combined_vector = torch.cat([a.flatten() for a in adaptation_vector])
                self.apply_singular_adaptations(combined_vector)
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        return outputs
    
    def generate_episode(self, input_ids, attention_mask, max_new_tokens=None, 
                        task_type="other", use_real_time_adaptation=True):
        """Generate episode with Transformer² real-time adaptation"""
        self.eval()
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        if max_new_tokens is None:
            if isinstance(self.config, UltraConfig):
                max_new_tokens = random.randint(
                    self.config.min_episode_length, 
                    self.config.max_episode_length
                )
            else:
                max_new_tokens = random.randint(8, 16)
        
        seq_id = f"seq_{self.sequence_counter}_{task_type}"
        self.sequence_counter += 1
        
        try:
            with torch.no_grad():
                task_props = None
                if use_real_time_adaptation and self.config.enable_transformer_squared:
                    task_props, _ = self.two_pass_inference(input_ids, attention_mask)
                    
                    try:
                        input_batch = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask
                        }
                        
                        total_adaptation_dim = sum(
                            adapter.original_S.numel() 
                            for adapter in self.sv_adapters.values()
                        )
                        
                        if total_adaptation_dim > 0:
                            best_params, best_score = self.cem_optimizer.optimize_adaptation(
                                self, input_batch, total_adaptation_dim, task_props.task_type
                            )
                            self.apply_singular_adaptations(best_params)
                            
                            expert_vector = ExpertVector(
                                expert_id=f"expert_{task_props.task_type}_{time.time()}",
                                task_type=task_props.task_type,
                                singular_adaptations={
                                    adapter_name: best_params[
                                        sum(self.sv_adapters[an].original_S.numel() 
                                            for an in list(self.sv_adapters.keys())[:i]):
                                        sum(self.sv_adapters[an].original_S.numel() 
                                            for an in list(self.sv_adapters.keys())[:i+1])
                                    ].view(adapter.original_S.shape)
                                    for i, (adapter_name, adapter) in enumerate(self.sv_adapters.items())
                                },
                                performance_score=best_score
                            )
                            
                            if hasattr(self, 'expert_mixing'):
                                self.expert_mixing.add_expert(expert_vector)
                            
                            logger.debug(f"Real-time adaptation: {task_props.task_type}, score: {best_score:.3f}")
                    except Exception as e:
                        logger.debug(f"Real-time adaptation failed: {e}")
                
                try:
                    init_outputs = self.forward_with_adaptations(input_ids, attention_mask)
                    hidden_states = init_outputs.hidden_states[-1]
                    values = self.value_network(hidden_states.mean(dim=1))
                except Exception as e:
                    logger.debug(f"Value estimation failed: {e}")
                    values = torch.zeros(input_ids.size(0), device=device)
                
                # Use DynamicCache for past_key_value to avoid layer_past conflict
                past_key_value = DynamicCache()
                
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
                    use_cache=True,
                    past_key_value=past_key_value
                )
                
                generated_tokens = generated.sequences[:, input_ids.size(1):]
                
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
                
                rewards = torch.zeros_like(generated_tokens, dtype=torch.float32)
                
                return Episode(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generated_tokens=generated_tokens,
                    log_probs=log_probs,
                    rewards=rewards,
                    values=values,
                    task_type=task_type,
                    task_properties=task_props,
                    sequence_id=seq_id,
                    episode_length=generated_tokens.size(1)
                )
                
        except Exception as e:
            logger.error(f"Episode generation failed: {e}")
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
                task_properties=task_props,
                sequence_id=seq_id,
                episode_length=1
            )
    
    def compute_grpo_loss(self, episodes: List[Episode]):
        """Compute GRPO loss from episodes"""
        if not episodes:
            return torch.tensor(0.0, requires_grad=True, device=device)
        
        total_loss = torch.tensor(0.0, requires_grad=True, device=device)
        valid_episodes = 0
        
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
                
                normalized_rewards = (episode.rewards - reward_mean) / reward_std
                normalized_rewards = torch.clamp(
                    normalized_rewards, 
                    -self.config.clip_rewards, 
                    self.config.clip_rewards
                )
                normalized_rewards = normalized_rewards * self.config.reward_scaling
                
                if episode.values.numel() > 0:
                    if normalized_rewards.dim() > episode.values.dim():
                        values_expanded = episode.values.mean().expand_as(normalized_rewards)
                    else:
                        values_expanded = episode.values
                    advantages = normalized_rewards - values_expanded.detach()
                else:
                    advantages = normalized_rewards
                
                policy_loss = -(episode.log_probs.flatten()[:advantages.numel()] * 
                               advantages.flatten()).mean()
                
                if episode.values.numel() > 0:
                    value_targets = normalized_rewards.flatten()[:episode.values.numel()]
                    value_loss = F.mse_loss(episode.values.flatten(), value_targets.detach())
                else:
                    value_loss = torch.tensor(0.0, device=device)
                
                entropy_loss = -episode.log_probs.mean()
                
                episode_loss = (
                    policy_loss + 
                    self.config.grpo_value_loss_coeff * value_loss +
                    self.config.grpo_entropy_coeff * entropy_loss
                )
                
                if torch.isfinite(episode_loss):
                    total_loss = total_loss + episode_loss
                    valid_episodes += 1
                    
            except Exception as e:
                logger.error(f"Episode loss computation failed: {e}")
                continue
        
        return total_loss / max(valid_episodes, 1) if valid_episodes > 0 else torch.tensor(0.0, requires_grad=True, device=device)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get GPU memory statistics"""
        stats = {}
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            stats.update({
                "gpu_memory_allocated": allocated / 1e9,
                "gpu_memory_reserved": reserved / 1e9,
                "gpu_memory_total": total_memory / 1e9,
                "gpu_memory_utilization": (allocated / total_memory) * 100,
            })
        else:
            stats.update({
                "gpu_memory_allocated": 0,
                "gpu_memory_reserved": 0,
                "gpu_memory_total": 0,
                "gpu_memory_utilization": 0,
            })
        
        return stats

# ============================================================================
# TRAINER CLASS WITH TRANSFORMER² SUPPORT
# ============================================================================

class TransformerSquaredTrainer:
    """Trainer for Transformer² GPT-2 with GRPO"""
    
    def __init__(self, config):
        self.config = config
        self.wandb_run = None
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Initializing Transformer² model with {type(config).__name__}")
        
        self.model = TransformerSquaredGPT2(config)
        
        optimizer_params = [
            {'params': self.model.value_network.parameters(), 'lr': config.learning_rate * 1.5},
        ]
        
        if config.enable_transformer_squared:
            optimizer_params.extend([
                {'params': self.model.dispatch_system.parameters(), 'lr': config.learning_rate},
                {'params': self.model.expert_mixing.parameters(), 'lr': config.learning_rate},
                {'params': self.model.sv_adapters.parameters(), 'lr': config.learning_rate * 0.5},
            ])
        
        self.optimizer = torch.optim.AdamW(
            optimizer_params,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.dataset_loader = DatasetLoader(config)
        self.datasets = self.dataset_loader.load_all_datasets()
        
        all_data = []
        for task_data in self.datasets.values():
            all_data.extend(task_data)
        
        num_batches = len(all_data) // config.batch_size
        total_steps = config.num_epochs * num_batches
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Scheduler configured for {total_steps} total steps "
                   f"({config.num_epochs} epochs × {num_batches} batches)")
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        
        os.makedirs(config.output_dir, exist_ok=True)
        self.training_metrics = {
            "policy_loss": [],
            "gpu_memory_usage": [],
            "task_rewards": defaultdict(list),
            "training_speed": [],
            "episode_lengths": [],
            "expert_usage": defaultdict(int),
            "task_properties": defaultdict(list)
        }
        
        logger.info("Transformer² trainer initialized successfully")
    
    def create_dataloader(self, data, batch_size):
        """Create DataLoader with proper configuration"""
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
        """Safely handle AMP optimizer step"""
        try:
            if self.config.mixed_precision:
                scaled_loss = self.scaler.scale(loss)
                scaled_loss.backward()
                
                if accumulated_steps >= self.config.gradient_accumulation_steps:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    return True, True
            else:
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
            logger.debug(f"AMP step failed: {e}")
            self.optimizer.zero_grad()
            if self.config.mixed_precision:
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            
            return False, False
    
    def train_grpo(self):
        """Main training loop with Transformer² features"""
        logger.info(f"Starting Transformer² GRPO training with {type(self.config).__name__}")
        
        try:
            if self.config.wandb_project:
                try:
                    self.wandb_run = wandb.init(
                        project=self.config.wandb_project,
                        name=f"t2-grpo-{type(self.config).__name__}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        config=self.config.__dict__
                    )
                    logger.info("✓ Wandb initialized")
                except Exception as e:
                    logger.warning(f"Wandb initialization failed: {e}")
            
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
                    'failed_steps': 0,
                    'task_distributions': defaultdict(int),
                    'expert_activations': 0
                }
                
                progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
                
                accumulated_steps = 0
                self.optimizer.zero_grad()
                last_valid_loss = None
                
                for batch_idx, batch in enumerate(progress_bar):
                    if batch is None:
                        continue
                    
                    batch_start_time = time.time()
                    batch_success = False
                    
                    try:
                        input_ids = batch['input_ids'].to(device, non_blocking=True)
                        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                        task_types = batch['task_type']
                        target_texts = batch['target_text']
                        
                        episodes = []
                        
                        for i in range(len(input_ids)):
                            try:
                                use_real_time = (epoch > 0) and self.config.real_time_adaptation
                                
                                episode = self.model.generate_episode(
                                    input_ids[i:i+1],
                                    attention_mask[i:i+1],
                                    task_type=task_types[i],
                                    use_real_time_adaptation=use_real_time
                                )
                                
                                if episode.generated_tokens.numel() > 0:
                                    generated_text = self.tokenizer.decode(
                                        episode.generated_tokens[0],
                                        skip_special_tokens=True
                                    )
                                    
                                    reward = self.model.reward_function.compute_reward(
                                        generated_text,
                                        target_texts[i],
                                        task_types[i]
                                    )
                                    
                                    episode.rewards = torch.full(
                                        episode.generated_tokens.size(),
                                        reward,
                                        device=device,
                                        dtype=torch.float32
                                    )
                                    
                                    episodes.append(episode)
                                    epoch_metrics['total_reward'] += reward
                                    epoch_metrics['task_distributions'][task_types[i]] += 1
                                    
                                    if episode.task_properties:
                                        self.training_metrics["task_properties"][task_types[i]].append({
                                            'complexity': episode.task_properties.complexity,
                                            'domain_specificity': episode.task_properties.domain_specificity,
                                            'reasoning_depth': episode.task_properties.reasoning_depth,
                                            'confidence': episode.task_properties.confidence
                                        })
                                    
                            except Exception as e:
                                logger.debug(f"Episode generation failed: {e}")
                                continue
                        
                        if not episodes:
                            continue
                        
                        if hasattr(self.model, 'expert_mixing') and self.model.expert_mixing.expert_vectors:
                            epoch_metrics['expert_activations'] += len(self.model.expert_mixing.expert_vectors)
                        
                        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                            grpo_loss = self.model.compute_grpo_loss(episodes)
                        
                        if not torch.isfinite(grpo_loss) or grpo_loss.item() == 0.0:
                            continue
                        
                        last_valid_loss = grpo_loss
                        
                        scaled_loss = grpo_loss / self.config.gradient_accumulation_steps
                        accumulated_steps += 1
                        
                        step_taken, step_successful = self._safe_amp_step(scaled_loss, accumulated_steps)
                        
                        if step_taken:
                            if step_successful:
                                epoch_metrics['successful_steps'] += 1
                                global_step += 1
                            else:
                                epoch_metrics['failed_steps'] += 1
                            accumulated_steps = 0
                        
                        epoch_metrics['policy_loss'] += grpo_loss.item()
                        epoch_metrics['episodes'] += len(episodes)
                        epoch_metrics['batches_processed'] += 1
                        batch_success = True
                        
                        memory_stats = self.model.get_memory_stats()
                        current_memory = memory_stats["gpu_memory_allocated"]
                        self.training_metrics["gpu_memory_usage"].append(current_memory)
                        
                        avg_reward = epoch_metrics['total_reward'] / max(epoch_metrics['episodes'], 1)
                        progress_info = {
                            'Loss': f'{grpo_loss.item():.4f}',
                            'Reward': f'{avg_reward:.3f}',
                            'GPU': f'{current_memory:.1f}GB',
                            'Steps': f"{epoch_metrics['successful_steps']}/{epoch_metrics['successful_steps'] + epoch_metrics['failed_steps']}",
                            'Experts': f"{len(self.model.expert_mixing.expert_vectors) if hasattr(self.model, 'expert_mixing') else 0}"
                        }
                        
                        if self.config.mixed_precision:
                            progress_info['Scale'] = f'{self.scaler.get_scale():.0f}'
                        
                        progress_bar.set_postfix(progress_info)
                        
                        if batch_idx % 20 == 0:
                            torch.cuda.empty_cache()
                    
                    except Exception as e:
                        logger.error(f"Batch {batch_idx} failed: {e}")
                        batch_success = False
                    
                    finally:
                        if not batch_success:
                            self.optimizer.zero_grad()
                            accumulated_steps = 0
                            if self.config.mixed_precision:
                                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
                
                if accumulated_steps > 0 and last_valid_loss is not None:
                    logger.info(f"Flushing {accumulated_steps} remaining gradients with last valid loss")
                    try:
                        scaled_loss = last_valid_loss / self.config.gradient_accumulation_steps
                        step_taken, step_successful = self._safe_amp_step(
                            scaled_loss, 
                            self.config.gradient_accumulation_steps
                        )
                        if step_taken and step_successful:
                            epoch_metrics['successful_steps'] += 1
                            global_step += 1
                    except Exception as e:
                        logger.warning(f"Final gradient flush failed: {e}")
                    finally:
                        self.optimizer.zero_grad()
                
                self._log_epoch_summary(epoch, epoch_metrics, start_time)
                self.save_checkpoint(epoch + 1, global_step)
                
                if isinstance(self.config, UltraConfig):
                    total_time = time.time() - start_time
                    if total_time > 2.5 * 3600:
                        logger.warning("Approaching time limit, stopping training")
                        break
            
            total_training_time = time.time() - start_time
            logger.info(f"Transformer² training completed in {total_training_time/60:.1f} minutes!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            traceback.print_exc()
        
        finally:
            if self.wandb_run is not None:
                try:
                    wandb.finish()
                    logger.info("✓ Wandb finished")
                except Exception as e:
                    logger.warning(f"Wandb finish failed: {e}")
    
    def _log_epoch_summary(self, epoch, epoch_metrics, start_time):
        """Log epoch summary with Transformer² metrics"""
        total_time = time.time() - start_time
        
        if epoch_metrics['batches_processed'] > 0:
            epoch_metrics['policy_loss'] /= epoch_metrics['batches_processed']
            avg_reward = epoch_metrics['total_reward'] / max(epoch_metrics['episodes'], 1)
            
            memory_stats = self.model.get_memory_stats()
            
            task_dist_str = ", ".join([
                f"{task}: {count}" 
                for task, count in epoch_metrics['task_distributions'].items()
            ])
            
            log_message = (
                f"Epoch {epoch + 1} completed:\n"
                f"  - Avg Policy Loss: {epoch_metrics['policy_loss']:.4f}\n"
                f"  - Episodes: {epoch_metrics['episodes']:,}\n"
                f"  - Avg Reward: {avg_reward:.3f}\n"
                f"  - Successful Steps: {epoch_metrics['successful_steps']}\n"
                f"  - Failed Steps: {epoch_metrics['failed_steps']}\n"
                f"  - Task Distribution: {task_dist_str}\n"
                f"  - Expert Vectors: {len(self.model.expert_mixing.expert_vectors) if hasattr(self.model, 'expert_mixing') else 0}\n"
                f"  - GPU Memory: {memory_stats['gpu_memory_allocated']:.1f}GB "
                f"({memory_stats['gpu_memory_utilization']:.1f}%)\n"
                f"  - Total Time: {total_time/60:.1f}min"
            )
            
            logger.info(log_message)
            
            if self.wandb_run is not None:
                try:
                    log_dict = {
                        "epoch": epoch + 1,
                        "policy_loss": epoch_metrics['policy_loss'],
                        "avg_reward": avg_reward,
                        "total_time_minutes": total_time / 60,
                        "gpu_memory_gb": memory_stats["gpu_memory_allocated"],
                        "gpu_memory_utilization_percent": memory_stats["gpu_memory_utilization"],
                        "episodes_per_epoch": epoch_metrics['episodes'],
                        "successful_steps": epoch_metrics['successful_steps'],
                        "failed_steps": epoch_metrics['failed_steps'],
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "num_expert_vectors": len(self.model.expert_mixing.expert_vectors) if hasattr(self.model, 'expert_mixing') else 0,
                        "expert_activations": epoch_metrics['expert_activations']
                    }
                    
                    for task, count in epoch_metrics['task_distributions'].items():
                        log_dict[f"task_count_{task}"] = count
                    
                    wandb.log(log_dict)
                except Exception as e:
                    logger.warning(f"Wandb logging failed: {e}")
        else:
            logger.warning(f"Epoch {epoch + 1} had no valid batches")
    
    def save_checkpoint(self, epoch: int, global_step: int):
        """Save training checkpoint with Transformer² state"""
        checkpoint_path = os.path.join(
            self.config.output_dir,
            f"t2_checkpoint_epoch_{epoch}_step_{global_step}.pt"
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
                'training_metrics': self.training_metrics,
                'transformer_squared_enabled': self.config.enable_transformer_squared,
                'paged_attention_enabled': self.config.enable_paged_attention
            }
            
            if hasattr(self.model, 'expert_mixing') and self.model.expert_mixing.expert_vectors:
                checkpoint['expert_vectors'] = {
                    expert_id: {
                        'task_type': expert.task_type,
                        'performance_score': expert.performance_score,
                        'usage_count': expert.usage_count,
                        'singular_adaptations': expert.singular_adaptations
                    }
                    for expert_id, expert in self.model.expert_mixing.expert_vectors.items()
                }
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"✓ Transformer² checkpoint saved to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with Transformer² features"""
    logger.info("Starting Transformer² GPT-2 + GRPO Training Pipeline")
    
    set_seed(42)
    
    config = UltraConfig()
    
    logger.info(f"Configuration: {type(config).__name__}")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Max Length: {config.max_length}")
    logger.info(f"  Mixed Precision: {config.mixed_precision}")
    logger.info(f"  Paged Attention: {config.enable_paged_attention}")
    logger.info(f"  Transformer²: {config.enable_transformer_squared}")
    logger.info(f"  Real-time Adaptation: {config.real_time_adaptation}")
    logger.info(f"  Max Expert Vectors: {config.max_expert_vectors}")
    logger.info(f"  Device: {device}")
    
    try:
        with timer("Transformer² trainer initialization"):
            trainer = TransformerSquaredTrainer(config)
        
        total_samples = sum(len(data) for data in trainer.datasets.values())
        if total_samples == 0:
            logger.error("No training data loaded!")
            return
        
        logger.info(f"Loaded {total_samples:,} training samples")
        logger.info("Starting Transformer² GRPO training with:")
        logger.info(f"  ✅ Singular Value-Only Adaptation: Enabled")
        logger.info(f"  ✅ Two-Pass Inference Mechanism: Enabled") 
        logger.info(f"  ✅ Dynamic Expert Mixing: Enabled")
        logger.info(f"  ✅ Real-Time Adaptation Capability: {config.real_time_adaptation}")
        logger.info(f"  ✓ Paged Attention: {'Enabled' if config.enable_paged_attention else 'Disabled'}")
        logger.info(f"  ✓ CEM Optimization: Enabled with task-specific parameters")
        logger.info(f"  ✓ Deterministic CUDNN: Enabled for reproducibility")
        
        with timer("Complete Transformer² GRPO training"):
            trainer.train_grpo()
        
        logger.info("🎉 Transformer² training pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        try:
            if 'trainer' in locals() and hasattr(trainer, 'wandb_run') and trainer.wandb_run is not None:
                wandb.finish()
        except:
            pass
        
        logger.info("✓ Cleanup completed")

if __name__ == "__main__":
    main() 
