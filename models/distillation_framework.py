"""
Core Distillation Framework for Student-Aware Knowledge Transfer
Implements SOTA distillation techniques with novel enhancements
"""

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    MixtralForCausalLM
)
from typing import Dict, Optional, Tuple, List
import math
from einops import rearrange
from .student_aware_router import StudentAwareDistillationRouter


class VocabularyAligner(nn.Module):
    """Handles vocabulary size mismatches between teacher and student models"""

    def __init__(self, teacher_vocab_size: int, student_vocab_size: int):
        super().__init__()
        self.teacher_vocab_size = teacher_vocab_size
        self.student_vocab_size = student_vocab_size

        if teacher_vocab_size != student_vocab_size:
            # Create a mapping from teacher vocab to student vocab
            # For simplicity, we'll use truncation or padding
            self.needs_alignment = True

            if teacher_vocab_size > student_vocab_size:
                # Teacher has larger vocab - we'll truncate
                self.alignment_type = "truncate"
            else:
                # Student has larger vocab - we'll pad with zeros
                self.alignment_type = "pad"
                self.padding = nn.Parameter(
                    torch.zeros(student_vocab_size - teacher_vocab_size),
                    requires_grad=False
                )
        else:
            self.needs_alignment = False

    def align_teacher_logits(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Align teacher logits to match student vocabulary size"""
        if not self.needs_alignment:
            return teacher_logits

        batch_size, seq_len, teacher_vocab = teacher_logits.shape

        if self.alignment_type == "truncate":
            # Truncate teacher logits to student vocab size
            aligned_logits = teacher_logits[:, :, :self.student_vocab_size]
        else:  # pad
            # Pad teacher logits to match student vocab size
            padding_shape = (batch_size, seq_len, self.student_vocab_size - teacher_vocab)
            padding_tensor = self.padding.expand(padding_shape).to(teacher_logits.device)
            aligned_logits = torch.cat([teacher_logits, padding_tensor], dim=-1)

        return aligned_logits


class FeatureProjector(nn.Module):
    """Projects features between different dimensional spaces"""

    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()

        if num_layers == 1:
            self.projector = nn.Linear(input_dim, output_dim)
        else:
            layers = []
            hidden_dim = (input_dim + output_dim) // 2

            # First layer
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])

            # Middle layers
            for _ in range(num_layers - 2):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1)
                ])

            # Final layer
            layers.append(nn.Linear(hidden_dim, output_dim))

            self.projector = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class AttentionTransferModule(nn.Module):
    """Transfers attention patterns from teacher to student"""

    def __init__(self, student_heads: int, teacher_heads: int):
        super().__init__()
        self.student_heads = student_heads
        self.teacher_heads = teacher_heads

        # Attention head alignment
        if student_heads != teacher_heads:
            self.head_projector = nn.Linear(teacher_heads, student_heads, bias=False)
        else:
            self.head_projector = nn.Identity()

        # Temperature for attention transfer
        self.temperature = nn.Parameter(torch.ones(1) * 4.0)

    def forward(self,
                student_attn: torch.Tensor,
                teacher_attn: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_attn: [batch_size, student_heads, seq_len, seq_len]
            teacher_attn: [batch_size, teacher_heads, seq_len, seq_len]

        Returns:
            attention_loss: Scalar tensor
        """
        batch_size, _, seq_len, _ = student_attn.shape

        # Align teacher attention heads to student
        if self.student_heads != self.teacher_heads:
            # Reshape for projection
            teacher_attn = rearrange(teacher_attn, 'b h s1 s2 -> b (s1 s2) h')
            teacher_attn = self.head_projector(teacher_attn)
            teacher_attn = rearrange(
                teacher_attn, 'b (s1 s2) h -> b h s1 s2',
                s1=seq_len, s2=seq_len
            )

        # Apply temperature scaling
        student_attn = student_attn / self.temperature
        teacher_attn = teacher_attn / self.temperature

        # Compute attention transfer loss
        student_attn_log = F.log_softmax(student_attn, dim=-1)
        teacher_attn_soft = F.softmax(teacher_attn, dim=-1)

        # KL divergence loss
        loss = F.kl_div(
            student_attn_log,
            teacher_attn_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return loss


class LayerwiseDistillationLoss(nn.Module):
    """Computes layer-wise distillation loss with adaptive weighting"""

    def __init__(self,
                 student_layers: int,
                 teacher_layers: int,
                 student_dim: int,
                 teacher_dim: int):
        super().__init__()
        self.student_layers = student_layers
        self.teacher_layers = teacher_layers

        # Layer mapping strategy
        self.layer_mapping = self._compute_layer_mapping()

        # Feature projectors for each mapped layer pair
        self.projectors = nn.ModuleList([
            FeatureProjector(student_dim, teacher_dim)
            for _ in range(len(self.layer_mapping))
        ])

        # Adaptive layer weights
        self.layer_weights = nn.Parameter(
            torch.ones(len(self.layer_mapping)) / len(self.layer_mapping)
        )

    def _compute_layer_mapping(self) -> List[Tuple[int, int]]:
        """Compute optimal mapping between student and teacher layers"""
        mapping = []

        if self.student_layers <= self.teacher_layers:
            # Map each student layer to corresponding teacher layer
            ratio = self.teacher_layers / self.student_layers
            for s_idx in range(self.student_layers):
                t_idx = int(s_idx * ratio)
                mapping.append((s_idx, t_idx))
        else:
            # Map subset of student layers to teacher layers
            ratio = self.student_layers / self.teacher_layers
            for t_idx in range(self.teacher_layers):
                s_idx = int(t_idx * ratio)
                mapping.append((s_idx, t_idx))

        return mapping

    def forward(self,
                student_hidden_states: List[torch.Tensor],
                teacher_hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute layer-wise distillation loss

        Args:
            student_hidden_states: List of [batch_size, seq_len, student_dim]
            teacher_hidden_states: List of [batch_size, seq_len, teacher_dim]

        Returns:
            layer_loss: Scalar tensor
        """
        total_loss = 0.0
        weights = F.softmax(self.layer_weights, dim=0)

        for idx, (s_idx, t_idx) in enumerate(self.layer_mapping):
            # Project student hidden states
            student_proj = self.projectors[idx](student_hidden_states[s_idx])
            teacher_hidden = teacher_hidden_states[t_idx]

            # Compute MSE loss for this layer pair
            layer_loss = F.mse_loss(student_proj, teacher_hidden)

            # Apply adaptive weight
            total_loss += weights[idx] * layer_loss

        return total_loss


class ContrastiveDistillationLoss(nn.Module):
    """Contrastive loss for better representation learning"""

    def __init__(self, temperature: float = 0.07, student_dim: int = None, teacher_dim: int = None):
        super().__init__()
        self.temperature = temperature

        # Add projection layer if dimensions don't match
        if student_dim is not None and teacher_dim is not None and student_dim != teacher_dim:
            self.needs_projection = True
            self.teacher_projector = nn.Linear(teacher_dim, student_dim)
        else:
            self.needs_projection = False

    def forward(self,
                student_embeddings: torch.Tensor,
                teacher_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive distillation loss

        Args:
            student_embeddings: [batch_size, embedding_dim]
            teacher_embeddings: [batch_size, embedding_dim]

        Returns:
            contrastive_loss: Scalar tensor
        """
        # Project teacher embeddings to student space if needed
        if self.needs_projection:
            teacher_embeddings = self.teacher_projector(teacher_embeddings)

        # Normalize embeddings
        student_norm = F.normalize(student_embeddings, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_embeddings, p=2, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(student_norm, teacher_norm.t()) / self.temperature

        # Labels are diagonal (positive pairs)
        batch_size = student_embeddings.size(0)
        labels = torch.arange(batch_size, device=student_embeddings.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity, labels)

        return loss


class StudentAwareDistillationFramework(nn.Module):
    """Main framework for student-aware knowledge distillation"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.loss_chunk_size = config.get('loss_chunk_size', None)
        self.max_attention_layers = config.get('attention_layers', None)

        # Load teacher model (Huihui-MoE)
        self.teacher_model = self._load_teacher_model()
        self.teacher_model.eval()

        # Load student model (SmolLM)
        self.student_model = self._load_student_model()

        # Get model dimensions
        self._extract_model_dimensions()

        # Student-aware router
        router_config = {
            'student_dim': self.student_dim,
            'teacher_dim': self.teacher_dim,
            'num_experts': config.get('num_experts', 8),
            'top_k': config.get('top_k', 2),
            'num_heads': config.get('num_heads', 4),
            'initial_top_k': config.get('initial_top_k', 1),
            'final_top_k': config.get('final_top_k', 4),
            'warmup_steps': config.get('warmup_steps', 1000),
            'total_steps': config.get('total_steps', 10000)
        }
        self.router = StudentAwareDistillationRouter(router_config)

        # Loss components
        self.kd_loss = nn.KLDivLoss(reduction='batchmean')

        self.attention_transfer = AttentionTransferModule(
            student_heads=self.student_heads,
            teacher_heads=self.teacher_heads
        )

        self.layerwise_loss = LayerwiseDistillationLoss(
            student_layers=self.student_layers,
            teacher_layers=self.teacher_layers,
            student_dim=self.student_dim,
            teacher_dim=self.teacher_dim
        )

        self.contrastive_loss = ContrastiveDistillationLoss(
            temperature=config.get('contrastive_temp', 0.07),
            student_dim=self.student_dim,
            teacher_dim=self.teacher_dim
        )

        # Loss weights
        self.alpha_kd = config.get('alpha_kd', 0.7)
        self.alpha_feature = config.get('alpha_feature', 0.1)
        self.alpha_attention = config.get('alpha_attention', 0.1)
        self.alpha_layerwise = config.get('alpha_layerwise', 0.05)
        self.alpha_contrastive = config.get('alpha_contrastive', 0.05)

        # Temperature for KD
        self.temperature = config.get('temperature', 4.0)

        # Track which losses already emitted non-finite warnings
        self._loss_warnings_printed = set()

    def _sanitize_tensor(self, tensor: Optional[torch.Tensor], name: str,
                          clamp_range: Optional[Tuple[float, float]] = None) -> Optional[torch.Tensor]:
        if tensor is None:
            return None

        if clamp_range is not None:
            tensor = torch.clamp(tensor, min=clamp_range[0], max=clamp_range[1])

        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            if name not in self._loss_warnings_printed:
                print(f"[Warning] Detected NaN/Inf values in {name}; applying nan_to_num.")
                self._loss_warnings_printed.add(name)
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

        return tensor

    def _ensure_finite_loss(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if torch.isnan(value) or torch.isinf(value):
            if name not in self._loss_warnings_printed:
                print(f"[Warning] {name} produced non-finite value ({value.item()}); clamping to zero.")
                self._loss_warnings_printed.add(name)
            return torch.zeros_like(value)
        return value

    def _load_teacher_model(self):
        """Load and configure teacher model"""
        model_name = self.config.get('teacher_model', 'huihui-ai/Huihui-MoE-1B-A0.6B')

        # Load with automatic device mapping for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None,
            trust_remote_code=True,
            attn_implementation="eager"  # Fix attention implementation warning
        )

        # Freeze teacher model
        for param in model.parameters():
            param.requires_grad = False

        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False

        return model

    def _load_student_model(self):
        """Load and configure student model"""
        model_name = self.config.get('student_model', 'HuggingFaceTB/SmolLM-135M')

        # Important: keep student weights in fp32 even when using AMP.
        # GradScaler expects fp32 params; loading directly in fp16 causes
        # "Attempting to unscale FP16 gradients" during optimizer steps.
        student_dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=student_dtype,
            trust_remote_code=True,
            attn_implementation="eager"  # Fix attention implementation warning
        )

        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False

        return model

    def _extract_model_dimensions(self):
        """Extract dimensions from models"""
        # Teacher dimensions
        teacher_config = self.teacher_model.config
        self.teacher_dim = getattr(teacher_config, 'hidden_size', 1024)
        self.teacher_layers = getattr(teacher_config, 'num_hidden_layers', 24)
        self.teacher_heads = getattr(teacher_config, 'num_attention_heads', 16)
        self.teacher_vocab_size = getattr(teacher_config, 'vocab_size', 151936)

        # Student dimensions
        student_config = self.student_model.config
        self.student_dim = getattr(student_config, 'hidden_size', 576)
        self.student_layers = getattr(student_config, 'num_hidden_layers', 30)
        self.student_heads = getattr(student_config, 'num_attention_heads', 9)
        self.student_vocab_size = getattr(student_config, 'vocab_size', 49152)

        # Enable gradient checkpointing to trade compute for memory
        if hasattr(self.student_model, 'gradient_checkpointing_enable'):
            self.student_model.gradient_checkpointing_enable()

        # Initialize vocabulary aligner for handling vocab size mismatches
        self.vocab_aligner = VocabularyAligner(
            teacher_vocab_size=self.teacher_vocab_size,
            student_vocab_size=self.student_vocab_size
        )

    @torch.no_grad()
    def get_teacher_outputs(self,
                           input_ids: torch.Tensor,
                           attention_mask: torch.Tensor) -> Dict:
        """Get teacher model outputs"""
        outputs = self.teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )

        # Extract MoE expert outputs if available
        expert_outputs = []
        if hasattr(self.teacher_model, 'model') and hasattr(self.teacher_model.model, 'layers'):
            for layer in self.teacher_model.model.layers:
                if hasattr(layer, 'block_sparse_moe'):
                    # This is a MoE layer
                    with torch.no_grad():
                        hidden_states = outputs.hidden_states[0]  # Use input hidden states
                        # Note: Actual expert extraction would require model internals access
                        expert_outputs.append(hidden_states)

        return {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            'expert_outputs': expert_outputs if expert_outputs else [outputs.hidden_states[-1]]
        }

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                step: Optional[int] = None) -> Dict:
        """
        Forward pass with distillation

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for language modeling
            step: Current training step

        Returns:
            Dictionary containing losses and outputs
        """
        # Get teacher outputs (no gradient)
        with torch.no_grad():
            teacher_outputs = self.get_teacher_outputs(input_ids, attention_mask)
            teacher_logits = self._sanitize_tensor(
                teacher_outputs['logits'],
                'teacher_logits',
                clamp_range=(-30.0, 30.0)
            )
            teacher_hidden = [
                self._sanitize_tensor(h, f'teacher_hidden_{idx}')
                for idx, h in enumerate(teacher_outputs['hidden_states'])
            ]
            teacher_hidden = [torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0) for h in teacher_hidden]

            teacher_attention = []
            for idx, a in enumerate(teacher_outputs['attentions']):
                if a is None:
                    teacher_attention.append(None)
                else:
                    sanitized_attention = self._sanitize_tensor(a, f'teacher_attention_{idx}')
                    teacher_attention.append(torch.nan_to_num(sanitized_attention, nan=0.0, posinf=0.0, neginf=0.0))

            raw_teacher_experts = teacher_outputs.get('expert_outputs', [teacher_hidden[-1]])
            teacher_experts = [
                self._sanitize_tensor(exp, f'teacher_expert_{idx}')
                for idx, exp in enumerate(raw_teacher_experts)
            ]

            # Release reference to original outputs to free GPU memory early
            del teacher_outputs

        # Get student outputs (with gradient)
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )

        # Make sure student outputs are in float32 for stable loss computation
        student_logits = self._sanitize_tensor(
            student_outputs.logits.to(torch.float32),
            'student_logits',
            clamp_range=(-30.0, 30.0)
        )
        student_logits = torch.nan_to_num(student_logits, nan=0.0, posinf=0.0, neginf=0.0)
        student_hidden = [
            self._sanitize_tensor(h.to(torch.float32), f'student_hidden_{idx}')
            for idx, h in enumerate(student_outputs.hidden_states)
        ]
        student_hidden = [torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0) for h in student_hidden]
        student_attention = []
        for idx, a in enumerate(student_outputs.attentions):
            if a is None:
                student_attention.append(None)
            else:
                sanitized_attn = self._sanitize_tensor(a.to(torch.float32), f'student_attention_{idx}')
                student_attention.append(torch.nan_to_num(sanitized_attn, nan=0.0, posinf=0.0, neginf=0.0))

        # Initialize losses dictionary
        losses = {}

        # 1. KL Divergence Loss (main distillation loss)
        # Handle vocabulary size mismatch
        aligned_teacher_logits = self.vocab_aligner.align_teacher_logits(teacher_logits)

        # Perform KD computations in float32 to avoid half-precision underflow/overflow
        student_logits_fp32 = student_logits.float()
        teacher_logits_fp32 = aligned_teacher_logits.float()

        # Clamp logits for stability, especially with mixed precision
        clamp_min, clamp_max = -30.0, 30.0
        student_logits_clamped = torch.clamp(student_logits_fp32, min=clamp_min, max=clamp_max)
        teacher_logits_clamped = torch.clamp(teacher_logits_fp32, min=clamp_min, max=clamp_max)

        student_log_probs = F.log_softmax(student_logits_clamped / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits_clamped / self.temperature, dim=-1)

        # Ensure teacher_probs don't have NaNs from softmax and remain normalized
        teacher_probs = torch.nan_to_num(teacher_probs, nan=0.0)
        teacher_probs = teacher_probs / torch.clamp(teacher_probs.sum(dim=-1, keepdim=True), min=1e-8)

        kd_loss = self.kd_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)
        losses['kd_loss'] = self._ensure_finite_loss('kd_loss', kd_loss * self.alpha_kd)

        # 2. Student-aware routing and feature losses
        if len(student_hidden) > 0 and len(teacher_hidden) > 0:
            # Use last hidden state for routing
            # Format teacher outputs for router compatibility
            teacher_outputs_formatted = {
                'hidden_states': teacher_hidden[-1],  # Use last layer hidden states
                'expert_outputs': teacher_experts
            }

            routing_outputs = self.router(
                student_hidden[-1],
                teacher_outputs_formatted,
                step=step
            )

            # Add routing losses
            for loss_name, loss_value in routing_outputs['losses'].items():
                scaled = self._ensure_finite_loss(f'routing_{loss_name}', loss_value * self.alpha_feature)
                losses[f'routing_{loss_name}'] = scaled

        # 3. Attention transfer loss
        if student_attention and teacher_attention and self.alpha_attention > 0:
            # Average attention transfer loss across layers
            attn_losses = []
            min_layers = min(len(student_attention), len(teacher_attention))
            layers_to_use = min_layers
            if self.max_attention_layers:
                layers_to_use = min(min_layers, self.max_attention_layers)

            # Use most recent attention maps to maintain signal
            student_attn_subset = student_attention[-layers_to_use:]
            teacher_attn_subset = teacher_attention[-layers_to_use:]

            for s_attn, t_attn in zip(student_attn_subset, teacher_attn_subset):
                if s_attn is not None and t_attn is not None:
                    attn_loss = self.attention_transfer(
                        s_attn,
                        t_attn
                    )
                    attn_losses.append(attn_loss)

            if attn_losses:
                attn_mean = torch.stack(attn_losses).mean()
                losses['attention_loss'] = self._ensure_finite_loss('attention_loss', attn_mean * self.alpha_attention)

        # 4. Layer-wise distillation loss
        if len(student_hidden) > 1 and len(teacher_hidden) > 1 and self.alpha_layerwise > 0:
            layerwise_loss = self.layerwise_loss(
                list(student_hidden),
                list(teacher_hidden)
            )
            losses['layerwise_loss'] = self._ensure_finite_loss('layerwise_loss', layerwise_loss * self.alpha_layerwise)

        # 5. Contrastive loss (using CLS or mean pooling)
        if self.alpha_contrastive > 0:
            # Mean pooling over sequence dimension
            student_embed = student_hidden[-1].mean(dim=1)
            teacher_embed = teacher_hidden[-1].mean(dim=1)

            contrastive_loss = self.contrastive_loss(student_embed, teacher_embed)
            losses['contrastive_loss'] = self._ensure_finite_loss('contrastive_loss', contrastive_loss * self.alpha_contrastive)

        # 6. Language modeling loss (if labels provided)
        if labels is not None:
            lm_loss = self._chunked_cross_entropy(student_logits, labels)
            losses['lm_loss'] = self._ensure_finite_loss('lm_loss', lm_loss * (1 - self.alpha_kd))

        # Compute total loss
        total_loss = sum(losses.values())
        total_loss = self._ensure_finite_loss('total_loss', total_loss)

        return {
            'loss': total_loss,
            'losses': losses,
            'student_logits': student_logits,
            'teacher_logits': teacher_logits,
            'routing_info': routing_outputs.get('routing_info', {}) if 'routing_outputs' in locals() else {}
        }

    def _chunked_cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross entropy in chunks to reduce activation memory footprint."""
        vocab_size = logits.size(-1)
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)

        chunk_size = self.loss_chunk_size or flat_logits.size(0)

        if chunk_size >= flat_logits.size(0):
            return F.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=-100
            )

        total_loss = logits.new_zeros(())
        total_weight = logits.new_zeros(())

        for start in range(0, flat_logits.size(0), chunk_size):
            end = min(start + chunk_size, flat_logits.size(0))
            logits_chunk = flat_logits[start:end]
            labels_chunk = flat_labels[start:end]

            mask = labels_chunk != -100
            if mask.any():
                loss_chunk = F.cross_entropy(
                    logits_chunk[mask],
                    labels_chunk[mask],
                    reduction='sum'
                )
                total_loss = total_loss + loss_chunk
                total_weight = total_weight + mask.sum()

        return total_loss / torch.clamp(total_weight, min=1.0)

    def save_student(self, save_path: str):
        """Save the distilled student model"""
        self.student_model.save_pretrained(save_path)

    def get_student_model(self):
        """Get the student model for inference"""
        return self.student_model
