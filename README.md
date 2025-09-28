# Student-Aware Knowledge Distillation

## Overview

A complete implementation of **Student-Aware Knowledge Distillation** using a novel routing mechanism to distill knowledge from the **Huihui-MoE-1B** teacher model to the **SmolLM-135M** student model. This implementation is optimized for Kaggle environments with P100 GPU support.

### Key Features

- 🚀 **Novel Student-Aware Router**: Adaptive routing based on student learning capacity
- 🔥 **Multi-Component Distillation**: KL divergence, attention transfer, layer-wise, and contrastive losses
- 💾 **Memory Efficient**: Mixed precision training with gradient accumulation
- 📊 **Comprehensive Evaluation**: Perplexity, knowledge retention, and compression metrics
- 🎯 **Progressive Training**: Dynamic routing complexity scheduling

## Architecture

### Teacher Model: Huihui-MoE-1B
- Mixture of Experts (MoE) architecture
- 1B parameters with 8 experts
- Sparse activation (top-2 experts)

### Student Model: SmolLM-135M
- Compact transformer architecture
- 135M parameters
- ~7.4x compression ratio

### Student-Aware Router
The router adapts knowledge transfer based on:
1. **Capacity Estimation**: Assesses student's current learning capacity
2. **Knowledge Gap Analysis**: Identifies areas where student needs more guidance
3. **Expert Importance Scoring**: Weights teacher experts by relevance
4. **Progressive Scheduling**: Gradually increases routing complexity

## Installation

```bash
# Clone the repository
cd student_aware_distillation

# Install requirements
pip install -r requirements.txt
```

## Quick Start

### Basic Training

```bash
python train.py
```

### Custom Configuration

```bash
python train.py --config configs/default_config.json --batch-size 4 --epochs 3
```

### Resume Training

```bash
python train.py --resume checkpoints/checkpoint_epoch_1
```

## Project Structure

```
student_aware_distillation/
├── models/
│   ├── student_aware_router.py     # Novel routing mechanism
│   └── distillation_framework.py   # Main distillation framework
├── data/
│   └── data_loader.py              # Data loading and preprocessing
├── utils/
│   ├── training.py                 # Training loop with mixed precision
│   └── evaluation.py               # Evaluation metrics and reporting
├── configs/
│   └── default_config.json         # Default configuration
├── train.py                        # Main training script
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 4 | Training batch size (optimized for P100) |
| `gradient_accumulation_steps` | 8 | Effective batch = 32 |
| `learning_rate` | 5e-5 | Student model learning rate |
| `router_lr` | 1e-4 | Router learning rate |
| `num_epochs` | 3 | Training epochs |
| `alpha_kd` | 0.7 | KL divergence loss weight |
| `temperature` | 4.0 | Distillation temperature |
| `initial_top_k` | 1 | Initial experts to route |
| `final_top_k` | 4 | Final experts to route |

### Loss Components

1. **KL Divergence Loss** (70%): Main distillation objective
2. **Feature Loss** (10%): Hidden state alignment
3. **Attention Loss** (10%): Attention pattern transfer
4. **Layer-wise Loss** (5%): Progressive layer matching
5. **Contrastive Loss** (5%): Representation learning

## Training on Kaggle

### Kaggle Notebook Setup

```python
# Cell 1: Install dependencies
!pip install -q transformers accelerate datasets einops

# Cell 2: Clone and setup
!git clone <your-repo-url> student_aware_distillation
%cd student_aware_distillation

# Cell 3: Run training
!python train.py --batch-size 4 --epochs 3
```

### Memory Optimization for P100

- Batch size: 4 with gradient accumulation (effective batch: 32)
- Mixed precision (FP16) training enabled
- Dynamic memory management with automatic cleanup
- Gradient checkpointing for large models

## Evaluation Metrics

### Model Compression
- **Parameter Reduction**: ~7.4x (1B → 135M)
- **Size Reduction**: ~86% smaller
- **Inference Speedup**: ~5-6x faster

### Performance Metrics
- **Perplexity**: Measures language modeling quality
- **Knowledge Retention**: KL divergence and top-k overlap
- **Attention Transfer**: Attention pattern similarity

## Novel Techniques

### 1. Student Capacity Estimation
```python
- Analyzes student hidden states
- Computes learning capacity scores
- Adapts routing based on capacity
```

### 2. Progressive Routing Schedule
```python
- Starts with single expert (top-1)
- Gradually increases to multiple experts (top-4)
- Temperature annealing for exploration
```

### 3. Knowledge Gap Analysis
```python
- Identifies discrepancies between models
- Focuses distillation on challenging areas
- Dynamic weight adjustment
```

## Results

### Expected Performance

| Metric | Teacher (MoE-1B) | Student (SmolLM-135M) |
|--------|------------------|----------------------|
| Parameters | 1B | 135M |
| Perplexity | ~15-20 | ~25-35 |
| Inference Time | 1.0x | ~0.17x |
| Memory Usage | ~4GB | ~0.5GB |

### Training Progress

- **Epoch 1**: Focus on basic knowledge transfer
- **Epoch 2**: Refine with attention and feature alignment
- **Epoch 3**: Fine-tune with all loss components

## Advanced Usage

### Custom Datasets

```python
python train.py --datasets wikitext openwebtext c4
```

### Hyperparameter Tuning

```python
# Create custom config
config = {
    "learning_rate": 1e-4,
    "alpha_kd": 0.8,
    "temperature": 5.0,
    "initial_top_k": 2,
    "final_top_k": 6
}

# Save and use
with open('custom_config.json', 'w') as f:
    json.dump(config, f)

python train.py --config custom_config.json
```

### Monitoring with WandB

```python
# Enable WandB logging
python train.py --config configs/default_config.json \
                --use-wandb \
                --project-name "my-distillation"
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` to 2
- Increase `gradient_accumulation_steps` to 16
- Reduce `max_length` to 256
- Disable `use_dynamic_batching`

### Slow Training
- Ensure CUDA is available
- Check mixed precision is enabled
- Reduce `dataset_subset_size`
- Use fewer workers (`num_workers: 1`)

### Poor Convergence
- Lower learning rate to 1e-5
- Increase warmup steps to 1000
- Adjust temperature (try 3.0 or 5.0)
- Increase `alpha_kd` to 0.8

## Citation

If you use this implementation, please cite:

```bibtex
@software{student_aware_distillation,
  title = {Student-Aware Knowledge Distillation},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/student_aware_distillation}
}
```

## Key Innovations

1. **Adaptive Routing**: Routes knowledge based on student capacity
2. **Progressive Complexity**: Gradually increases task difficulty
3. **Multi-Loss Framework**: Comprehensive distillation objectives
4. **Memory Efficiency**: Optimized for limited GPU resources
5. **Production Ready**: Complete implementation with evaluation

## Future Improvements

- [ ] Quantization support (INT8/INT4)
- [ ] ONNX export for deployment
- [ ] Distributed training support
- [ ] Additional student architectures
- [ ] Curriculum learning strategies

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Huihui-MoE model by huihui-ai
- SmolLM model by HuggingFaceTB
- PyTorch and HuggingFace teams

---

**Note**: This implementation is optimized for Kaggle P100 GPU environments. Adjust batch sizes and memory settings for different hardware configurations.