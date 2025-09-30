#!/usr/bin/env python3
"""
Main training script for Student-Aware Knowledge Distillation
Distills from Huihui-MoE-1B to SmolLM-135M using novel routing mechanisms
"""

import torch

import argparse
import json
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer
from models.distillation_framework import StudentAwareDistillationFramework
from data.data_loader import (
    create_distillation_dataloader,
    prepare_eval_dataloader,
    get_recommended_datasets
)
from utils.training import create_trainer
from utils.evaluation import (
    MetricsTracker,
    DistillationEvaluator,
    WandBLogger,
    create_evaluation_report
)


def setup_environment():
    """Setup environment for optimal performance"""
    # Set environment variables for better performance
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('TORCH_NCCL_DEBUG', 'WARN')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF_MAX_SPLIT_SIZE_MB', '64')

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # Set memory growth for better memory management
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("WARNING: CUDA not available, training will be slow on CPU")


def load_config(config_path: str = None) -> dict:
    """Load configuration from file or use defaults"""
    default_config = {
        # Model configuration
        'teacher_model': 'huihui-ai/Huihui-MoE-1B-A0.6B',
        'student_model': 'HuggingFaceTB/SmolLM-135M',
        'num_experts': 8,
        'top_k': 2,

        # Training configuration
        'batch_size': 4,  # Small batch for P100 GPU
        'gradient_accumulation_steps': 8,  # Effective batch size = 32
        'learning_rate': 5e-5,
        'router_lr': 1e-4,
        'num_epochs': 3,
        'warmup_steps': 500,
        'eval_steps': 500,
        'save_epochs': 1,
        'max_length': 512,

        # Loss weights
        'alpha_kd': 0.7,
        'alpha_feature': 0.1,
        'alpha_attention': 0.1,
        'alpha_layerwise': 0.05,
        'alpha_contrastive': 0.05,

        # Distillation settings
        'temperature': 4.0,
        'contrastive_temp': 0.07,

        # Router configuration
        'initial_top_k': 1,
        'final_top_k': 4,
        'load_balance_weight': 0.01,
        'noise_std': 0.1,

        # Optimization
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'scheduler_type': 'cosine',
        'use_amp': True,
        'amp_dtype': 'bfloat16',
        'loss_chunk_size': 128,
        'attention_layers': 4,

        # Data configuration
        'dataset_subset_size': 50000,  # Limit dataset size for Kaggle
        'num_workers': 2,
        'use_dynamic_batching': True,

        # Paths
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
        'cache_dir': './cache',

        # Memory management
        'memory_threshold': 0.85,

        # Logging
        'use_wandb': False,  # Set to True if WandB is configured
        'project_name': 'student-aware-distillation'
    }

    if config_path is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            custom_config = json.load(f)
            default_config.update(custom_config)

    return default_config


def main(args):
    """Main training function"""
    # Setup environment
    setup_environment()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate

    print("\n" + "="*60)
    print("STUDENT-AWARE KNOWLEDGE DISTILLATION")
    print("="*60)
    print(f"Teacher Model: {config['teacher_model']}")
    print(f"Student Model: {config['student_model']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Gradient Accumulation: {config['gradient_accumulation_steps']}")
    print(f"Effective Batch Size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Epochs: {config['num_epochs']}")
    print("="*60 + "\n")

    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['student_model'],
        cache_dir=config['cache_dir']
    )

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create data loaders
    print("\nPreparing datasets...")
    dataset_names = args.datasets if args.datasets else get_recommended_datasets()

    try:
        train_dataloader = create_distillation_dataloader(
            dataset_names=dataset_names,
            tokenizer=tokenizer,
            batch_size=config['batch_size'],
            max_length=config['max_length'],
            subset_size=config['dataset_subset_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            use_dynamic_batching=config['use_dynamic_batching'],
            cache_dir=config['cache_dir']
        )
    except Exception as e:
        print(f"Warning: Failed to load preferred datasets ({e})")
        print("Falling back to wikitext-2 only...")
        train_dataloader = create_distillation_dataloader(
            dataset_names=["wikitext"],
            tokenizer=tokenizer,
            batch_size=config['batch_size'],
            max_length=config['max_length'],
            subset_size=config['dataset_subset_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            use_dynamic_batching=config['use_dynamic_batching'],
            cache_dir=config['cache_dir']
        )

    try:
        eval_dataloader = prepare_eval_dataloader(
            tokenizer=tokenizer,
            batch_size=config['batch_size'] * 2,  # Larger batch for evaluation
            max_length=config['max_length'],
            cache_dir=config['cache_dir']
        )
    except Exception as e:
        print(f"Warning: Failed to load evaluation dataset ({e})")
        print("Using a smaller evaluation set...")
        eval_dataloader = prepare_eval_dataloader(
            tokenizer=tokenizer,
            batch_size=config['batch_size'],  # Smaller batch size
            max_length=config['max_length'],
            cache_dir=config['cache_dir']
        )

    try:
        print(f"Training samples: {len(train_dataloader.dataset)}")
        print(f"Evaluation samples: {len(eval_dataloader.dataset)}")
    except:
        print("Training and evaluation datasets prepared successfully")

    # Initialize distillation framework
    print("\nInitializing models...")
    print("Loading teacher model (this may take a while)...")

    framework_config = {
        **config,
        'total_steps': len(train_dataloader) * config['num_epochs']
    }

    try:
        model = StudentAwareDistillationFramework(framework_config)
    except Exception as e:
        print(f"Error initializing distillation framework: {e}")
        print("This might be due to model compatibility issues.")
        raise

    print("Models loaded successfully!")
    print(f"Teacher parameters: {sum(p.numel() for p in model.teacher_model.parameters()):,}")
    print(f"Student parameters: {sum(p.numel() for p in model.student_model.parameters()):,}")

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(log_dir=config['log_dir'])

    # Initialize WandB logger if enabled
    wandb_logger = None
    if config['use_wandb']:
        wandb_logger = WandBLogger(
            project_name=config['project_name'],
            config=config
        )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = create_trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Start training
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")

    try:
        # Train the model
        trainer.train()

        # Save final checkpoint
        print("\nSaving final model...")
        final_path = os.path.join(config['checkpoint_dir'], 'final_model')
        trainer.save_checkpoint(final_path)

        # Create evaluation report
        if not args.skip_eval:
            print("\n" + "="*60)
            print("FINAL EVALUATION")
            print("="*60 + "\n")

            evaluator = DistillationEvaluator(
                teacher_model=model.teacher_model,
                student_model=model.student_model,
                tokenizer=tokenizer,
                device=str(trainer.device)
            )

            report_path = os.path.join(config['log_dir'], 'evaluation_report.txt')
            evaluation_report = create_evaluation_report(
                evaluator=evaluator,
                eval_dataloader=eval_dataloader,
                save_path=report_path
            )

            # Log final metrics
            if wandb_logger:
                wandb_logger.log(evaluation_report)
                wandb_logger.log_model(final_path, "final_distilled_model")

        # Save metrics
        metrics_tracker.save_metrics()

        # Plot training curves
        plot_path = os.path.join(config['log_dir'], 'training_curves.png')
        # Note: Plotting might fail in headless environment
        try:
            metrics_tracker.plot_metrics(save_path=plot_path)
        except:
            print("Could not generate plots (headless environment)")

        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final model saved to: {final_path}")
        if not args.skip_eval:
            print(f"Evaluation report saved to: {report_path}")
        print(f"Training logs saved to: {config['log_dir']}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving checkpoint...")
        interrupt_path = os.path.join(config['checkpoint_dir'], 'interrupted_checkpoint')
        trainer.save_checkpoint(interrupt_path)
        print(f"Checkpoint saved to: {interrupt_path}")

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

        # Try to save emergency checkpoint
        try:
            emergency_path = os.path.join(config['checkpoint_dir'], 'emergency_checkpoint')
            trainer.save_checkpoint(emergency_path)
            print(f"Emergency checkpoint saved to: {emergency_path}")
        except:
            print("Could not save emergency checkpoint")

    finally:
        # Cleanup
        if wandb_logger:
            wandb_logger.finish()

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\nCleanup completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Student-Aware Knowledge Distillation Model"
    )

    # Configuration arguments
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration JSON file'
    )

    # Training arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Training batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate'
    )

    # Data arguments
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='List of dataset names to use'
    )

    # Checkpoint arguments
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    # Other arguments
    parser.add_argument(
        '--skip-eval',
        action='store_true',
        help='Skip final evaluation'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Run main training
    main(args)
