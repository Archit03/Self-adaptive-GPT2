# Enhanced GPU-Optimized Self-Adaptive GPT2 with GRPO Training + CEM Inference

This implementation combines Generalized Reinforcement Learning from Policy Optimization (GRPO) with Cross-Entropy Method (CEM) inference for self-adaptive GPT-2 models using SVD decomposition and multi-dataset training.

## Requirements

```bash
pip install torch>=1.9.0 transformers>=4.20.0 datasets>=2.0.0 wandb
pip install numpy>=1.21.0 scipy>=1.7.0 matplotlib>=3.3.0 seaborn>=0.11.0
pip install accelerate>=0.12.0 evaluate>=0.2.0 rouge-score>=0.1.0
pip install sacrebleu>=2.0.0 bert-score>=0.3.0 scikit-learn>=1.0.0
pip install pandas>=1.3.0 tqdm>=4.60.0 sentence-transformers>=2.0.0
pip install requests>=2.25.0 tokenizers>=0.12.0 huggingface-hub>=0.10.0
```

## Usage

```python
python enhanced_grpo_cem_pipeline.py
```

## Configuration

The `EnhancedConfig` class contains all configuration parameters:

- `model_name`: "gpt2" - Base model to use
- `batch_size`: 16 - Training batch size
- `learning_rate`: 5e-5 - Learning rate
- `num_epochs`: 5 - Number of training epochs
- `max_length`: 256 - Maximum sequence length
- `adaptation_rank`: 32 - SVD adaptation rank
- `num_experts`: 8 - Number of expert networks
- `mixed_precision`: False - Enable mixed precision training
- `gradient_accumulation_steps`: 4 - Gradient accumulation steps
- `max_grad_norm`: 0.5 - Gradient clipping threshold
- `warmup_steps`: 100 - Learning rate warmup steps
- `weight_decay`: 0.01 - Weight decay
- `max_samples_per_dataset`: 500 - Maximum samples per dataset
- `use_fallback_data_only`: False - Use only fallback data
- `enable_internet_check`: True - Check internet connectivity
- `dataset_download_timeout`: 300 - Dataset download timeout
- `max_download_retries`: 3 - Maximum download retries
- `enable_qa_datasets`: True - Enable question answering datasets
- `enable_sentiment_datasets`: True - Enable sentiment analysis datasets
- `enable_summarization_datasets`: True - Enable summarization datasets
- `enable_classification_datasets`: True - Enable classification datasets
- `enable_generation_datasets`: True - Enable text generation datasets
- `grpo_episodes_per_batch`: 8 - GRPO episodes per batch
- `grpo_reward_normalization`: True - Enable reward normalization
- `grpo_kl_coeff`: 0.01 - KL divergence coefficient
- `grpo_value_loss_coeff`: 0.1 - Value loss coefficient
- `grpo_entropy_coeff`: 0.08 - Entropy coefficient
- `cem_population_size`: 100 - CEM population size
- `cem_elite_ratio`: 0.3 - CEM elite ratio
- `cem_noise_std`: 0.3 - CEM noise standard deviation
- `cem_adaptation_steps`: 50 - CEM adaptation steps
- `cem_convergence_threshold`: 5e-3 - CEM convergence threshold
- `cem_momentum`: 0.3 - CEM momentum
- `svd_rank_ratio`: 0.8 - SVD rank ratio
- `svd_min_singular_value`: 1e-5 - Minimum singular value for SVD
- `wandb_project`: "enhanced-grpo-cem-gpt2" - Weights & Biases project name
- `output_dir`: "./enhanced_results" - Output directory
- `log_interval`: 10 - Logging interval
- `save_interval`: 1 - Model saving interval
- `clip_rewards`: 3.0 - Reward clipping threshold
- `reward_scaling`: 0.1 - Reward scaling factor
- `temperature_annealing`: True - Enable temperature annealing
- `adaptive_learning_rate`: True - Enable adaptive learning rate
- `repetition_penalty`: 1.3 - Repetition penalty for generation
- `top_p`: 0.85 - Top-p sampling parameter
- `temperature`: 0.6 - Temperature for generation

## Supported Datasets

### Question Answering
- SQuAD v1.1
- MS MARCO v1.1

### Sentiment Analysis
- IMDB
- Amazon Polarity
- Yelp Review Full

### Summarization
- XSum
- CNN/DailyMail 3.0.0

### Classification
- AG News
- SetFit/20_newsgroups

### Text Generation
- WikiText-2 Raw v1
- TinyStories

The system automatically falls back to synthetic data if HuggingFace datasets fail to load.

## Components

### StabilizedSVDDecomposer
Performs SVD decomposition of model weights with regularization and reconstruction capabilities.

### EnhancedValueNetwork
Multi-layer value network with attention-based pooling for GRPO training.

### ImprovedTaskRewardFunction
Task-specific reward computation using ROUGE scores, semantic similarity, and task-specific heuristics.

### RobustDatasetLoader
Downloads and processes multiple HuggingFace datasets with retry logic and fallback data.

### EnhancedCEMOptimizer
Cross-Entropy Method optimization for adaptation parameter search with population-based optimization.

### EnhancedSelfAdaptiveGPT2
Main model class combining GPT-2 with SVD-based adaptation, value networks, and task classification.

### EnhancedGRPOTrainer
Training pipeline implementing GRPO with CEM adaptation, dataset loading, and comprehensive evaluation.

## Output Files

- `enhanced_checkpoint_epoch_X_step_Y.pt` - Model checkpoints
- `enhanced_comprehensive_report.png` - Training visualization
- `enhanced_training_summary.txt` - Detailed training summary
- `enhanced_evaluation_results.json` - Evaluation results
- `training.log` - Training logs

## Features

- GPU optimization with mixed precision training
- SVD-based parameter-efficient adaptation
- Multi-task learning across 5 NLP task categories
- Population-based CEM optimization for inference
- Comprehensive dataset loading with fallback system
- Real-time monitoring and visualization
- Automatic hyperparameter scheduling
- Gradient clipping and training stabilization
