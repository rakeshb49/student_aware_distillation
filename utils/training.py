"""
Training Loop Implementation with Mixed Precision and Memory Optimization
Optimized for P100 GPU with gradient accumulation and efficient memory management
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Optional, Tuple, Callable
from tqdm import tqdm
import numpy as np
import time
import psutil
import gc
import os


class MemoryManager:
    """Manages GPU memory during training"""
    
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.cleanup_counter = 0
        self.cleanup_frequency = 50  # Clean every N steps
        
    def check_memory(self) -> Dict:
        """Check current memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_memory,
                'usage_percent': allocated / reserved if reserved > 0 else 0
            }
        else:
            # CPU memory
            memory = psutil.virtual_memory()
            return {
                'used_gb': memory.used / 1024**3,
                'available_gb': memory.available / 1024**3,
                'usage_percent': memory.percent / 100
            }
    
    def cleanup(self, force: bool = False):
        """Clean up memory if needed"""
        self.cleanup_counter += 1
        
        if force or self.cleanup_counter >= self.cleanup_frequency:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            self.cleanup_counter = 0
    
    def optimize_batch_size(self, current_batch_size: int, 
                           memory_info: Dict) -> int:
        """Dynamically adjust batch size based on memory usage"""
        usage = memory_info.get('usage_percent', 0)
        
        if usage > self.threshold:
            # Reduce batch size
            return max(1, current_batch_size // 2)
        elif usage < 0.6:
            # Can potentially increase batch size
            return min(current_batch_size * 2, 32)  # Cap at 32
        
        return current_batch_size


class GradientAccumulator:
    """Handles gradient accumulation for effective larger batch sizes"""
    
    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
        
    def should_step(self) -> bool:
        """Check if optimizer should step"""
        self.step_count += 1
        return self.step_count % self.accumulation_steps == 0
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient accumulation"""
        return loss / self.accumulation_steps
    
    def reset(self):
        """Reset step counter"""
        self.step_count = 0


class DistillationTrainer:
    """Main trainer class for knowledge distillation with mixed precision"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict,
                 train_dataloader,
                 eval_dataloader=None,
                 device: Optional[str] = None):
        """
        Initialize the trainer
        
        Args:
            model: The distillation framework model
            config: Training configuration
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            device: Device to use (auto-detected if None)
        """
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup mixed precision training
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        amp_dtype = config.get('amp_dtype', 'bfloat16')
        bf16_supported = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()

        if amp_dtype == 'bfloat16' and bf16_supported:
            self.amp_dtype = torch.bfloat16
            self.scaler = None  # GradScaler not needed for bf16
        else:
            if amp_dtype == 'bfloat16' and torch.cuda.is_available() and not bf16_supported:
                print("[Info] Requested bf16 autocast but this GPU does not support it; falling back to fp16.")
            self.amp_dtype = torch.float16
            self.scaler = GradScaler() if self.use_amp else None
        
        # Setup gradient accumulation
        self.gradient_accumulator = GradientAccumulator(
            accumulation_steps=config.get('gradient_accumulation_steps', 4)
        )
        
        # Setup memory manager
        self.memory_manager = MemoryManager(
            threshold=config.get('memory_threshold', 0.9)
        )
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Metrics tracking
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create and configure optimizer"""
        # Separate parameters for different learning rates
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        
        # Get student model parameters (only these need gradients)
        student_params = []
        router_params = []
        
        for name, param in self.model.named_parameters():
            if "student_model" in name:
                student_params.append((name, param))
            elif "router" in name:
                router_params.append((name, param))
        
        # Group parameters
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in student_params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.get('weight_decay', 0.01),
                "lr": self.config.get('learning_rate', 5e-5)
            },
            {
                "params": [p for n, p in student_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.config.get('learning_rate', 5e-5)
            },
            {
                "params": [p for n, p in router_params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.get('weight_decay', 0.01),
                "lr": self.config.get('router_lr', 1e-4)
            },
            {
                "params": [p for n, p in router_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.config.get('router_lr', 1e-4)
            }
        ]
        
        # Filter out empty parameter groups
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=self.config.get('betas', (0.9, 0.999)),
            eps=self.config.get('eps', 1e-8)
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_type = self.config.get('scheduler_type', 'cosine')
        num_training_steps = max(1, len(self.train_dataloader) * self.config.get('num_epochs', 3))
        num_warmup_steps = min(self.config.get('warmup_steps', 1000), num_training_steps // 10)
        
        if scheduler_type == 'cosine':
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == 'cosine_restarts':
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=num_training_steps // 4,
                T_mult=2,
                eta_min=1e-7
            )
        elif scheduler_type == 'onecycle':
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.get('learning_rate', 5e-5),
                total_steps=num_training_steps,
                pct_start=0.1
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'total': [],
            'kd': [],
            'feature': [],
            'attention': [],
            'routing': []
        }
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device with dual tokenization
            student_input_ids = batch['student_input_ids'].to(self.device)
            student_attention_mask = batch['student_attention_mask'].to(self.device)
            teacher_input_ids = batch['teacher_input_ids'].to(self.device)
            teacher_attention_mask = batch['teacher_attention_mask'].to(self.device)
            labels = batch.get('labels')
            if labels is not None:
                labels = labels.to(self.device)
            
            # Memory check
            if batch_idx % 10 == 0:
                memory_info = self.memory_manager.check_memory()
                if memory_info.get('usage_percent', 0) > self.memory_manager.threshold:
                    print(f"[Warning] High GPU memory usage detected ({memory_info['usage_percent']:.2f}).")
                    if self.config.get('use_dynamic_batching', True):
                        print("[Action] Clearing CUDA cache to free memory.")
                    self.memory_manager.cleanup(force=True)
                else:
                    self.memory_manager.cleanup()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
                    outputs = self.model(
                        student_input_ids=student_input_ids,
                        student_attention_mask=student_attention_mask,
                        teacher_input_ids=teacher_input_ids,
                        teacher_attention_mask=teacher_attention_mask,
                        labels=labels,
                        step=self.global_step
                    )
                    loss = outputs['loss']

                    # Scale loss for gradient accumulation
                    loss = self.gradient_accumulator.scale_loss(loss)

                # Backward pass with scaled gradients when scaler is available
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Optimizer step with gradient accumulation
                if self.gradient_accumulator.should_step():
                    # Gradient clipping
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('max_grad_norm', 1.0)
                    )

                    # Optimizer step
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Scheduler step
                    if self.scheduler is not None:
                        self.scheduler.step()
            else:
                # Standard training without mixed precision
                outputs = self.model(
                    student_input_ids=student_input_ids,
                    student_attention_mask=student_attention_mask,
                    teacher_input_ids=teacher_input_ids,
                    teacher_attention_mask=teacher_attention_mask,
                    labels=labels,
                    step=self.global_step
                )
                loss = outputs['loss']
                loss = self.gradient_accumulator.scale_loss(loss)
                
                loss.backward()
                
                if self.gradient_accumulator.should_step():
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.get('max_grad_norm', 1.0)
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler is not None:
                        self.scheduler.step()
            
            # Track losses
            epoch_losses['total'].append(loss.item() * self.gradient_accumulator.accumulation_steps)
            for loss_name, loss_value in outputs.get('losses', {}).items():
                key = loss_name.replace('_loss', '').replace('routing_', '')
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(loss_value.item())
            
            # Update progress bar
            avg_loss = np.mean(epoch_losses['total'][-100:])  # Running average
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'step': self.global_step
            })
            
            self.global_step += 1
            
            # Periodic evaluation
            if self.eval_dataloader and self.global_step % self.config.get('eval_steps', 500) == 0:
                eval_metrics = self.evaluate()
                self.model.train()  # Back to training mode
        
        # Compute epoch metrics
        epoch_metrics = {
            f'train_{key}': np.mean(values) 
            for key, values in epoch_losses.items() if values
        }
        
        return epoch_metrics
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Evaluate the model"""
        if not self.eval_dataloader:
            return {}
        
        self.model.eval()
        eval_losses = []
        
        eval_progress = tqdm(self.eval_dataloader, desc="Evaluating", leave=False)

        for batch in eval_progress:
            student_input_ids = batch['student_input_ids'].to(self.device)
            student_attention_mask = batch['student_attention_mask'].to(self.device)
            teacher_input_ids = batch['teacher_input_ids'].to(self.device)
            teacher_attention_mask = batch['teacher_attention_mask'].to(self.device)
            labels = batch.get('labels')
            if labels is not None:
                labels = labels.to(self.device)

            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
                    outputs = self.model(
                        student_input_ids=student_input_ids,
                        student_attention_mask=student_attention_mask,
                        teacher_input_ids=teacher_input_ids,
                        teacher_attention_mask=teacher_attention_mask,
                        labels=labels
                    )
            else:
                outputs = self.model(
                    student_input_ids=student_input_ids,
                    student_attention_mask=student_attention_mask,
                    teacher_input_ids=teacher_input_ids,
                    teacher_attention_mask=teacher_attention_mask,
                    labels=labels
                )
            
            eval_losses.append(outputs['loss'].item())
            eval_progress.set_postfix({'loss': f"{eval_losses[-1]:.4f}"})
        
        eval_loss = np.mean(eval_losses)
        eval_progress.close()
        
        # Calculate perplexity
        perplexity = np.exp(eval_loss)
        
        metrics = {
            'eval_loss': eval_loss,
            'eval_perplexity': perplexity
        }
        
        print(f"[Eval] loss: {metrics['eval_loss']:.4f}, ppl: {metrics['eval_perplexity']:.2f}")

        # Check if this is the best model
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.save_checkpoint(best=True)
        
        return metrics
    
    def train(self) -> Dict:
        """Main training loop"""
        num_epochs = self.config.get('num_epochs', 3)
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Gradient Accumulation Steps: {self.gradient_accumulator.accumulation_steps}")
        
        all_metrics = []
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train epoch
            epoch_metrics = self.train_epoch()
            
            # Evaluate
            if self.eval_dataloader:
                eval_metrics = self.evaluate()
                epoch_metrics.update(eval_metrics)
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_epochs', 1) == 0:
                self.save_checkpoint()
            
            # Print metrics
            print(f"\nEpoch {epoch + 1} Metrics:")
            for key, value in epoch_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            all_metrics.append(epoch_metrics)
            
            # Clean memory after epoch
            self.memory_manager.cleanup(force=True)
        
        print("\nTraining completed!")
        return all_metrics
    
    def save_checkpoint(self, path: Optional[str] = None, best: bool = False):
        """Save model checkpoint"""
        if path is None:
            checkpoint_dir = self.config.get('checkpoint_dir', './checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            if best:
                path = os.path.join(checkpoint_dir, 'best_model')
            else:
                path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{self.epoch}')
        
        # Save student model
        self.model.save_student(path)
        
        # Save training state
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_eval_loss': self.best_eval_loss,
            'config': self.config
        }, os.path.join(path, 'training_state.pt'))
        
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        # Load training state
        state_path = os.path.join(path, 'training_state.pt')
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.epoch = state['epoch']
            self.global_step = state['global_step']
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            if self.scheduler and state.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(state['scheduler_state_dict'])
            self.best_eval_loss = state.get('best_eval_loss', float('inf'))
            
            print(f"Checkpoint loaded from {path}")
            print(f"Resuming from epoch {self.epoch}, step {self.global_step}")


def create_trainer(model, config, train_dataloader, eval_dataloader=None):
    """Factory function to create a trainer"""
    return DistillationTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )