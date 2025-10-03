"""
Evaluation and Monitoring Utilities for Knowledge Distillation
Includes metrics computation, visualization, and progress tracking
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Optional, Tuple
import json
import wandb
from datetime import datetime
import os
from pathlib import Path


class MetricsTracker:
    """Track and compute various metrics during training"""

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            'train': {},
            'eval': {},
            'distillation': {}
        }

        self.step_metrics = []
        self.epoch_metrics = []

    def update(self, metrics: Dict, split: str = 'train', step: Optional[int] = None):
        """Update metrics for a given split"""
        for key, value in metrics.items():
            if key not in self.metrics[split]:
                self.metrics[split][key] = []

            self.metrics[split][key].append({
                'value': value,
                'step': step
            })

    def compute_average(self, metric_name: str, split: str = 'train',
                       last_n: Optional[int] = None) -> float:
        """Compute average of a metric"""
        if metric_name not in self.metrics[split]:
            return 0.0

        values = [m['value'] for m in self.metrics[split][metric_name]]

        if last_n is not None:
            values = values[-last_n:]

        return np.mean(values) if values else 0.0

    def save_metrics(self, filename: Optional[str] = None):
        """Save metrics to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.log_dir / f"metrics_{timestamp}.json"
        else:
            filename = self.log_dir / filename

        # Convert to serializable format
        serializable_metrics = {}
        for split, split_metrics in self.metrics.items():
            serializable_metrics[split] = {}
            for key, values in split_metrics.items():
                serializable_metrics[split][key] = values

        with open(filename, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        print(f"Metrics saved to {filename}")

    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training and evaluation metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Plot different metrics
        metrics_to_plot = [
            ('train', 'total', 'Training Loss'),
            ('eval', 'eval_loss', 'Validation Loss'),
            ('train', 'kd', 'KD Loss'),
            ('train', 'attention', 'Attention Loss'),
            ('train', 'feature', 'Feature Loss'),
            ('eval', 'eval_perplexity', 'Validation Perplexity')
        ]

        for idx, (split, metric, title) in enumerate(metrics_to_plot):
            if idx >= len(axes):
                break

            ax = axes[idx]

            if split in self.metrics and metric in self.metrics[split]:
                data = self.metrics[split][metric]
                steps = [d['step'] for d in data if d['step'] is not None]
                values = [d['value'] for d in data]

                if steps and values:
                    ax.plot(steps, values, label=title)
                    ax.set_xlabel('Step')
                    ax.set_ylabel(title)
                    ax.set_title(title)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
            else:
                ax.text(0.5, 0.5, f'No data for {title}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Plots saved to {save_path}")
        else:
            plt.show()

        plt.close()


class DistillationEvaluator:
    """Comprehensive evaluation for distilled models"""

    def __init__(self, teacher_model, student_model, student_tokenizer, teacher_tokenizer,
                 device='cuda', logit_projector=None, temperature: float = 4.0):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.device = device
        self.logit_projector = logit_projector
        self.temperature = temperature

    @torch.no_grad()
    def compute_perplexity(self, model, dataloader) -> float:
        """Compute perplexity on a dataset"""
        model.eval()
        total_loss = 0
        total_tokens = 0

        for batch in dataloader:
            if model is self.teacher_model:
                input_ids = batch['teacher_input_ids'].to(self.device)
                attention_mask = batch['teacher_attention_mask'].to(self.device)
                labels = input_ids
            else:
                input_ids = batch['student_input_ids'].to(self.device)
                attention_mask = batch['student_attention_mask'].to(self.device)
                labels = batch.get('labels')
                labels = labels.to(self.device) if labels is not None else None

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Get loss
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                loss = outputs['loss']

            # Count tokens
            num_tokens = (attention_mask == 1).sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        return perplexity

    @torch.no_grad()
    def compute_knowledge_retention(self, dataloader, top_k: int = 5) -> Dict:
        """Measure how well student retains teacher's knowledge"""
        self.teacher_model.eval()
        self.student_model.eval()

        total_kl_div = 0
        total_top_k_overlap = 0
        total_samples = 0

        for batch in dataloader:
            student_input_ids = batch['student_input_ids'].to(self.device)
            student_attention_mask = batch['student_attention_mask'].to(self.device)
            teacher_input_ids = batch['teacher_input_ids'].to(self.device)
            teacher_attention_mask = batch['teacher_attention_mask'].to(self.device)

            # Get teacher outputs on teacher tokenization
            teacher_outputs = self.teacher_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask
            )
            teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs['logits']

            # Get student outputs on student tokenization
            student_outputs = self.student_model(
                input_ids=student_input_ids,
                attention_mask=student_attention_mask
            )
            student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs['logits']

            # Because vocabularies differ, align teacher logits via projection if available
            if self.logit_projector is not None:
                teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
                # PRIORITY 1 FIX: Wrap in autocast to handle dtype mismatch
                # Teacher is float16, projector should be too (converted in __init__), but use autocast for safety
                with autocast(dtype=torch.float16 if torch.cuda.is_available() else torch.float32):
                    aligned_teacher_logits = self.logit_projector(teacher_probs)
            else:
                teacher_vocab = teacher_logits.size(-1)
                student_vocab = student_logits.size(-1)
                if teacher_vocab >= student_vocab:
                    aligned_teacher_logits = teacher_logits[..., :student_vocab]
                else:
                    pad_size = student_vocab - teacher_vocab
                    pad = torch.zeros(*teacher_logits.shape[:-1], pad_size, device=teacher_logits.device,
                                      dtype=teacher_logits.dtype)
                    aligned_teacher_logits = torch.cat([teacher_logits, pad], dim=-1)

            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(aligned_teacher_logits, dim=-1)
            kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            total_kl_div += kl_div.item()

            # Compute top-k overlap
            teacher_topk = torch.topk(aligned_teacher_logits, k=top_k, dim=-1).indices
            student_topk = torch.topk(student_logits, k=top_k, dim=-1).indices

            # Calculate overlap
            batch_size, seq_len, _ = teacher_topk.shape
            for b in range(batch_size):
                for s in range(seq_len):
                    teacher_set = set(teacher_topk[b, s].cpu().numpy())
                    student_set = set(student_topk[b, s].cpu().numpy())
                    overlap = len(teacher_set.intersection(student_set)) / top_k
                    total_top_k_overlap += overlap

            total_samples += batch_size * seq_len

        return {
            'avg_kl_divergence': total_kl_div / len(dataloader),
            'top_k_overlap': total_top_k_overlap / total_samples,
            'knowledge_retention_score': 1 - (total_kl_div / len(dataloader))  # Higher is better
        }

    @torch.no_grad()
    def compute_compression_metrics(self) -> Dict:
        """Compute model compression metrics"""
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())

        teacher_params = count_parameters(self.teacher_model)
        student_params = count_parameters(self.student_model)

        # Compute model sizes in MB
        teacher_size = teacher_params * 4 / (1024 * 1024)  # Assuming float32
        student_size = student_params * 4 / (1024 * 1024)

        return {
            'teacher_parameters': teacher_params,
            'student_parameters': student_params,
            'compression_ratio': teacher_params / student_params,
            'teacher_size_mb': teacher_size,
            'student_size_mb': student_size,
            'size_reduction_percent': (1 - student_size / teacher_size) * 100
        }

    @torch.no_grad()
    def compute_inference_speed(self, dataloader, num_batches: int = 10) -> Dict:
        """Compare inference speeds"""
        import time

        self.teacher_model.eval()
        self.student_model.eval()

        teacher_times = []
        student_times = []

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            student_input_ids = batch['student_input_ids'].to(self.device)
            student_attention_mask = batch['student_attention_mask'].to(self.device)
            teacher_input_ids = batch['teacher_input_ids'].to(self.device)
            teacher_attention_mask = batch['teacher_attention_mask'].to(self.device)

            # Teacher inference time
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            _ = self.teacher_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask
            )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            teacher_times.append(time.time() - start)

            # Student inference time
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            _ = self.student_model(
                input_ids=student_input_ids,
                attention_mask=student_attention_mask
            )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            student_times.append(time.time() - start)

        return {
            'teacher_avg_time': np.mean(teacher_times),
            'student_avg_time': np.mean(student_times),
            'speedup': np.mean(teacher_times) / np.mean(student_times),
            'teacher_throughput': 1 / np.mean(teacher_times),
            'student_throughput': 1 / np.mean(student_times)
        }


class WandBLogger:
    """Weights & Biases logging integration"""

    def __init__(self, project_name: str = "student-aware-distillation",
                 config: Optional[Dict] = None):
        self.enabled = False
        try:
            wandb.init(project=project_name, config=config)
            self.enabled = True
        except:
            print("WandB logging disabled (not configured or error)")

    def log(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics to WandB"""
        if self.enabled:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)

    def log_model(self, model_path: str, name: str = "distilled_model"):
        """Log model artifact"""
        if self.enabled:
            artifact = wandb.Artifact(name, type='model')
            artifact.add_dir(model_path)
            wandb.log_artifact(artifact)

    def finish(self):
        """Finish WandB run"""
        if self.enabled:
            wandb.finish()


def create_evaluation_report(evaluator: DistillationEvaluator,
                            eval_dataloader,
                            save_path: str = "evaluation_report.txt") -> Dict:
    """Create comprehensive evaluation report"""
    print("Creating evaluation report...")

    # Compute all metrics
    compression_metrics = evaluator.compute_compression_metrics()

    print("Computing perplexity...")
    teacher_perplexity = evaluator.compute_perplexity(
        evaluator.teacher_model, eval_dataloader
    )
    student_perplexity = evaluator.compute_perplexity(
        evaluator.student_model, eval_dataloader
    )

    print("Computing knowledge retention...")
    retention_metrics = evaluator.compute_knowledge_retention(eval_dataloader)

    print("Computing inference speed...")
    speed_metrics = evaluator.compute_inference_speed(eval_dataloader)

    # Create report
    report = {
        'compression': compression_metrics,
        'perplexity': {
            'teacher': teacher_perplexity,
            'student': student_perplexity,
            'perplexity_increase': student_perplexity - teacher_perplexity
        },
        'knowledge_retention': retention_metrics,
        'inference_speed': speed_metrics
    }

    # Write report to file
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("KNOWLEDGE DISTILLATION EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("MODEL COMPRESSION METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Teacher Parameters: {compression_metrics['teacher_parameters']:,}\n")
        f.write(f"Student Parameters: {compression_metrics['student_parameters']:,}\n")
        f.write(f"Compression Ratio: {compression_metrics['compression_ratio']:.2f}x\n")
        f.write(f"Size Reduction: {compression_metrics['size_reduction_percent']:.1f}%\n\n")

        f.write("PERPLEXITY METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Teacher Perplexity: {teacher_perplexity:.2f}\n")
        f.write(f"Student Perplexity: {student_perplexity:.2f}\n")
        f.write(f"Perplexity Increase: {student_perplexity - teacher_perplexity:.2f}\n\n")

        f.write("KNOWLEDGE RETENTION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"KL Divergence: {retention_metrics['avg_kl_divergence']:.4f}\n")
        f.write(f"Top-5 Overlap: {retention_metrics['top_k_overlap']:.2%}\n")
        f.write(f"Retention Score: {retention_metrics['knowledge_retention_score']:.4f}\n\n")

        f.write("INFERENCE SPEED:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Teacher Avg Time: {speed_metrics['teacher_avg_time']:.4f}s\n")
        f.write(f"Student Avg Time: {speed_metrics['student_avg_time']:.4f}s\n")
        f.write(f"Speedup: {speed_metrics['speedup']:.2f}x\n")
        f.write(f"Student Throughput: {speed_metrics['student_throughput']:.2f} batches/s\n")

    print(f"Report saved to {save_path}")
    return report
