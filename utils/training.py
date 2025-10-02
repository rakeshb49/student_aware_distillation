"""
Training Loop Implementation with Mixed Precision and Memory Optimization
Optimized for P100 GPU with gradient accumulation and efficient memory management
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Optional, Tuple, Callable
from tqdm import tqdm
import numpy as np
import time
import psutil
import gc
import os


class EarlyStopping:
    """Early stopping to halt training when validation metric stops improving"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait for improvement (FIX ISSUE #5: increased from 3 to 10)
            min_delta: Minimum change to qualify as improvement (FIX ISSUE #5: reduced from 0.01 to 0.001)
            mode: 'min' for metrics that should decrease (loss), 'max' for metrics that should increase
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric: float) -> bool:
        """
        Check if training should stop

        Args:
            metric: Current metric value

        Returns:
            True if should stop, False otherwise
        """
        score = -metric if self.mode == 'min' else metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop

    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


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

            # FIX: Use actual GPU memory for accurate usage percentage
            free_b, total_b = torch.cuda.mem_get_info()
            used_b = total_b - free_b

            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_memory,
                'total_gb': total_b / 1024**3,
                'used_gb': used_b / 1024**3,
                'usage_percent': used_b / total_b
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


class ModelEMA:
    """FIX ISSUE #6: Exponential Moving Average for model weights

    Maintains a moving average of model parameters for more stable checkpoints.
    SOTA practice from papers like MoCo, BYOL, and modern vision transformers.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[str] = None):
        """
        Args:
            model: Model to track
            decay: EMA decay rate (higher = slower update)
            device: Device for EMA parameters
        """
        self.decay = decay
        self.device = device if device else next(model.parameters()).device

        # Create shadow parameters
        self.shadow_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone().to(self.device)

    def update(self, model: nn.Module):
        """Update EMA parameters"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    self.shadow_params[name].mul_(self.decay).add_(
                        param.data.to(self.device), alpha=1 - self.decay
                    )

    def apply_shadow(self, model: nn.Module):
        """Apply EMA parameters to model (for evaluation/saving)"""
        backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                backup[name] = param.data.clone()
                # FIX: Ensure device and shape compatibility
                shadow_param = self.shadow_params[name]
                if shadow_param.shape == param.data.shape:
                    param.data.copy_(shadow_param.to(param.device))
                else:
                    # Shape mismatch - skip this parameter
                    print(f"Warning: EMA shape mismatch for {name}: {shadow_param.shape} vs {param.data.shape}")
        return backup

    def restore(self, model: nn.Module, backup: Dict):
        """Restore original parameters"""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def state_dict(self):
        """Get EMA state for checkpointing"""
        return {
            'decay': self.decay,
            'shadow_params': self.shadow_params
        }

    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint"""
        self.decay = state_dict.get('decay', self.decay)
        self.shadow_params = state_dict['shadow_params']


class DistillationTrainer:
    """Main trainer class for knowledge distillation with mixed precision"""

    def __init__(self,
                 model: nn.Module,
                 config: Dict,
                 train_dataloader,
                 eval_dataloader=None,
                 device: Optional[str] = None,
                 metrics_tracker=None):
        """
        Initialize the trainer

        Args:
            model: The distillation framework model
            config: Training configuration
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            device: Device to use (auto-detected if None)
            metrics_tracker: CRITICAL FIX #3 - Optional MetricsTracker for logging
        """
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.metrics_tracker = metrics_tracker  # CRITICAL FIX #3: Store metrics tracker

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

        # Setup early stopping
        self.use_early_stopping = config.get('use_early_stopping', True)
        if self.use_early_stopping:
            # FIX ISSUE #5: More patient early stopping (10 instead of 3, 0.001 instead of 0.01)
            self.early_stopping = EarlyStopping(
                patience=config.get('early_stopping_patience', 10),
                min_delta=config.get('early_stopping_min_delta', 0.001),
                mode='min'  # For loss/perplexity
            )
        else:
            self.early_stopping = None

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # FIX ISSUE #6: Setup EMA for model weights
        self.use_ema = config.get('use_ema', True)
        if self.use_ema:
            self.ema = ModelEMA(
                self.model,
                decay=config.get('ema_decay', 0.9999),
                device=self.device
            )
            print(f"EMA enabled with decay={config.get('ema_decay', 0.9999)}")
        else:
            self.ema = None

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

        student_params = []
        router_params = []
        aux_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "student_model" in name:
                student_params.append((name, param))
            elif "router" in name:
                router_params.append((name, param))
            else:
                aux_params.append((name, param))

        def build_groups(param_list, lr):
            if not param_list:
                return []
            decay_params = [p for n, p in param_list if not any(nd in n for nd in no_decay)]
            nodecay_params = [p for n, p in param_list if any(nd in n for nd in no_decay)]
            groups = []
            if decay_params:
                groups.append({
                    "params": decay_params,
                    "weight_decay": self.config.get('weight_decay', 0.01),
                    "lr": lr
                })
            if nodecay_params:
                groups.append({
                    "params": nodecay_params,
                    "weight_decay": 0.0,
                    "lr": lr
                })
            return groups

        optimizer_grouped_parameters = []
        optimizer_grouped_parameters.extend(
            build_groups(student_params, self.config.get('learning_rate', 5e-5))
        )
        optimizer_grouped_parameters.extend(
            build_groups(router_params, self.config.get('router_lr', 1e-4))
        )
        aux_lr = self.config.get('aux_lr', self.config.get('learning_rate', 5e-5))
        optimizer_grouped_parameters.extend(
            build_groups(aux_params, aux_lr)
        )

        optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=self.config.get('betas', (0.9, 0.999)),
            eps=self.config.get('eps', 1e-8)
        )

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_type = self.config.get('scheduler_type', 'cosine')

        # CRITICAL FIX: Calculate actual optimizer steps, not batch steps
        # The scheduler.step() is called only when optimizer steps (after gradient accumulation)
        total_batches = len(self.train_dataloader) * self.config.get('num_epochs', 3)
        grad_accum_steps = self.config.get('gradient_accumulation_steps', 1)
        num_training_steps = max(1, total_batches // grad_accum_steps)

        num_warmup_steps = min(self.config.get('warmup_steps', 1000), num_training_steps // 10)

        # Debug logging to verify scheduler configuration
        print(f"[Scheduler] Total batches: {total_batches:,}")
        print(f"[Scheduler] Gradient accumulation: {grad_accum_steps}")
        print(f"[Scheduler] Optimizer steps: {num_training_steps:,}")
        print(f"[Scheduler] Warmup steps: {num_warmup_steps:,} ({num_warmup_steps/num_training_steps*100:.1f}%)")

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

        # FIX ISSUE #1: Initialize epoch_metrics early to avoid UnboundLocalError
        epoch_metrics = {}
        early_stop_triggered = False

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
                    # FIX ISSUE #2: Use set_to_none=True for better memory efficiency
                    self.optimizer.zero_grad(set_to_none=True)

                    # FIX ISSUE #6: Update EMA after optimizer step
                    if self.ema is not None:
                        self.ema.update(self.model)

                    # CRITICAL FIX: Scheduler should step only when optimizer steps
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
                    # FIX ISSUE #2: Use set_to_none=True for better memory efficiency
                    self.optimizer.zero_grad(set_to_none=True)

                    # FIX ISSUE #6: Update EMA after optimizer step
                    if self.ema is not None:
                        self.ema.update(self.model)

                    # CRITICAL FIX: Scheduler should step only when optimizer steps
                    if self.scheduler is not None:
                        self.scheduler.step()

            # Track losses
            epoch_losses['total'].append(loss.item() * self.gradient_accumulator.accumulation_steps)
            for loss_name, loss_value in outputs.get('losses', {}).items():
                key = loss_name.replace('_loss', '').replace('routing_', '')
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(loss_value.item())

            # DEBUG: Comprehensive loss component logging
            if batch_idx % 100 == 0:
                print(f"\n{'='*60}")
                print(f"[TRAIN DEBUG] Step {self.global_step}, Batch {batch_idx}")
                print(f"{'='*60}")
                print(f"  Progress bar loss: {loss.item() * self.gradient_accumulator.accumulation_steps:.4f}")
                print(f"\n  Raw loss components:")
                for key, value in outputs.get('losses', {}).items():
                    print(f"    {key}: {value.item():.4f}")
                total_from_components = sum(v.item() for v in outputs.get('losses', {}).values())
                print(f"\n  Sum of components: {total_from_components:.4f}")
                print(f"  Total loss (outputs['loss']): {outputs['loss'].item():.4f}")
                print(f"  Scaled for grad accum: {outputs['loss'].item() * self.gradient_accumulator.accumulation_steps:.4f}")
                print(f"{'='*60}\n")

            # FIX ISSUE #8: Monitor gradient norms
            if self.gradient_accumulator.should_step() and batch_idx % 100 == 0:
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                if total_norm > 10.0:
                    print(f"\n[Warning] High gradient norm: {total_norm:.4f}")

            # Update progress bar
            avg_loss = np.mean(epoch_losses['total'][-100:])  # Running average
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.3e}',  # Changed from .2e to .3e for better visibility during warmup
                'step': self.global_step
            })

            self.global_step += 1

            # Periodic evaluation
            if self.eval_dataloader and self.global_step % self.config.get('eval_steps', 500) == 0:
                eval_metrics = self.evaluate()
                self.model.train()  # Back to training mode

                # FIX ISSUE #1: Check early stopping mid-epoch using flag
                if eval_metrics.get('early_stop', False):
                    print(f"\n[Early Stop] Triggered at step {self.global_step} (mid-epoch)")
                    early_stop_triggered = True
                    break  # Exit epoch early

        # Compute epoch metrics
        epoch_metrics.update({
            f'train_{key}': np.mean(values)
            for key, values in epoch_losses.items() if values
        })

        # FIX ISSUE #1: Add early stop flag after metrics are computed
        if early_stop_triggered:
            epoch_metrics['early_stop'] = True

        # CRITICAL FIX #3: Log metrics to tracker if available
        if self.metrics_tracker is not None:
            # Log without 'train_' prefix for cleaner metrics
            clean_metrics = {k.replace('train_', ''): v for k, v in epoch_metrics.items()}
            self.metrics_tracker.update(
                clean_metrics,
                split='train',
                step=self.global_step
            )

        # FIX ISSUE #8: Print component loss breakdown
        component_losses = {k: v for k, v in epoch_metrics.items() if k.startswith('train_') and k != 'train_total'}
        if component_losses:
            print(f"\n[Train] Component losses:")
            for key, value in component_losses.items():
                print(f"  {key}: {value:.4f}")

        return epoch_metrics

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Evaluate the model"""
        if not self.eval_dataloader:
            return {}

        self.model.eval()
        eval_losses = []

        # FIX ISSUE #8: Track component losses during evaluation
        eval_component_losses = {
            'kd': [],
            'feature': [],
            'attention': [],
            'routing': []
        }

        # FIX ISSUE #6: Use EMA weights for evaluation if available
        ema_backup = None
        if self.ema is not None:
            ema_backup = self.ema.apply_shadow(self.model)

        total_batches = len(self.eval_dataloader)
        print(f"[Eval] Starting evaluation ({total_batches} batches)...")

        for batch_idx, batch in enumerate(self.eval_dataloader):
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

            # FIX ISSUE #8: Track component losses
            for loss_name, loss_value in outputs.get('losses', {}).items():
                key = loss_name.replace('_loss', '').replace('routing_', '')
                if key in eval_component_losses:
                    eval_component_losses[key].append(loss_value.item())

            # Print progress every 10% or every 20 batches, whichever is more frequent
            progress_interval = max(1, min(20, total_batches // 10))
            if (batch_idx + 1) % progress_interval == 0 or (batch_idx + 1) == total_batches:
                progress_pct = (batch_idx + 1) / total_batches * 100
                current_loss = eval_losses[-1]
                avg_loss = np.mean(eval_losses)
                print(f"[Eval] {batch_idx + 1}/{total_batches} ({progress_pct:.1f}%) - current: {current_loss:.4f}, avg: {avg_loss:.4f}")

        eval_loss = np.mean(eval_losses)

        # FIX ISSUE #2: Calculate perplexity with overflow protection
        perplexity = np.exp(min(eval_loss, 20.0))  # Cap at exp(20) ≈ 485M
        if eval_loss > 20.0:
            print(f"[Warning] Loss {eval_loss:.2f} too high for meaningful perplexity (capped at 20)")

        metrics = {
            'eval_loss': eval_loss,
            'eval_perplexity': perplexity
        }

        # FIX ISSUE #8: Add component losses to metrics
        for key, values in eval_component_losses.items():
            if values:
                metrics[f'eval_{key}_loss'] = np.mean(values)

        print(f"[Eval] loss: {metrics['eval_loss']:.4f}, ppl: {metrics['eval_perplexity']:.2f}")

        # FIX ISSUE #8: Print component losses
        component_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k.startswith('eval_') and k not in ['eval_loss', 'eval_perplexity']])
        if component_str:
            print(f"[Eval] components: {component_str}")

        # FIX ISSUE #6: Restore original weights after evaluation with EMA
        if ema_backup is not None:
            self.ema.restore(self.model, ema_backup)

        # CRITICAL FIX #3: Log evaluation metrics to tracker if available
        if self.metrics_tracker is not None:
            self.metrics_tracker.update(
                metrics,
                split='eval',
                step=self.global_step
            )

        # Check if this is the best model
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.save_checkpoint(best=True)

        # Check early stopping
        if self.early_stopping is not None:
            if self.early_stopping(eval_loss):
                print(f"\n[Early Stopping] No improvement for {self.early_stopping.patience} evaluations. Stopping training.")
                metrics['early_stop'] = True

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

                # Check early stopping
                if eval_metrics.get('early_stop', False):
                    print("\nEarly stopping triggered. Training halted.")
                    break

            # Save checkpoint
            if (epoch + 1) % self.config.get('save_epochs', 1) == 0:
                self.save_checkpoint()

            # Print metrics
            print(f"\nEpoch {epoch + 1} Metrics:")
            for key, value in epoch_metrics.items():
                if key != 'early_stop':
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

        # CRITICAL FIX: Save model state (excluding frozen teacher)
        model_state = self.model.state_dict()
        model_state = {k: v for k, v in model_state.items() if not k.startswith('teacher_model.')}

        # Save training state (FIX ISSUE #1: include gradient accumulator state)
        checkpoint_state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_eval_loss': self.best_eval_loss,
            'gradient_accumulator_step_count': self.gradient_accumulator.step_count,
            'model_state_dict': model_state,  # Save framework weights
            'config': self.config
        }

        # Add scaler state if using mixed precision
        if self.scaler is not None:
            checkpoint_state['scaler_state_dict'] = self.scaler.state_dict()

        # FIX ISSUE #6: Add EMA state to checkpoint
        if self.ema is not None:
            checkpoint_state['ema_state_dict'] = self.ema.state_dict()

        torch.save(checkpoint_state, os.path.join(path, 'training_state.pt'))

        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        # CRITICAL FIX: Reload student model weights from saved directory
        try:
            from transformers import AutoModelForCausalLM
            student_model = AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=True,
                torch_dtype=torch.float32  # Keep student in fp32 for AMP
            )
            student_model.to(self.device)
            self.model.student_model = student_model
            print("Restored student model weights from saved directory.")
        except Exception as e:
            print(f"Warning: could not restore student model from {path}: {e}")

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

            # CRITICAL FIX: Restore framework weights (router, projectors, etc.)
            if 'model_state_dict' in state:
                missing, unexpected = self.model.load_state_dict(state['model_state_dict'], strict=False)
                if missing or unexpected:
                    print(f"Model state restored with missing={len(missing)}, unexpected={len(unexpected)}")
                else:
                    print("Model state fully restored")

            # FIX ISSUE #1: Restore gradient accumulator state
            if 'gradient_accumulator_step_count' in state:
                self.gradient_accumulator.step_count = state['gradient_accumulator_step_count']
                print(f"Restored gradient accumulator step count: {self.gradient_accumulator.step_count}")

            # Restore scaler state if present
            if self.scaler is not None and 'scaler_state_dict' in state:
                self.scaler.load_state_dict(state['scaler_state_dict'])
                print(f"Restored gradient scaler state")

            # FIX ISSUE #6: Restore EMA state if present
            if self.ema is not None and 'ema_state_dict' in state:
                self.ema.load_state_dict(state['ema_state_dict'])
                print(f"Restored EMA state")

            print(f"Checkpoint loaded from {path}")
            print(f"Resuming from epoch {self.epoch}, step {self.global_step}")


def create_trainer(model, config, train_dataloader, eval_dataloader=None, metrics_tracker=None):
    """Factory function to create a trainer

    Args:
        model: Distillation framework model
        config: Training configuration
        train_dataloader: Training data loader
        eval_dataloader: Optional evaluation data loader
        metrics_tracker: Optional MetricsTracker for logging (CRITICAL FIX #3)
    """
    return DistillationTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        metrics_tracker=metrics_tracker  # CRITICAL FIX #3: Pass metrics tracker
    )
