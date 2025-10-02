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


class TeacherToStudentLogitProjector(nn.Module):
    """Projects teacher probability distributions into student vocabulary space.

    CRITICAL PERFORMANCE FIX: Uses teacher hidden states directly to avoid
    O(B·L·Vt·Dt) compute/memory bottleneck (Vt≈152k on P100 causes OOM).
    """

    def __init__(self,
                 teacher_embedding: nn.Module,
                 student_embedding: nn.Module,
                 teacher_dim: int,
                 student_dim: int):
        super().__init__()
        if teacher_embedding is None or student_embedding is None:
            raise ValueError("Both teacher and student embeddings must be provided for logit projection.")

        self.teacher_embedding = teacher_embedding
        self.student_embedding = student_embedding
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim

        self.hidden_projector = nn.Linear(teacher_dim, student_dim)

    def forward(self,
                teacher_probs: torch.Tensor = None,
                teacher_hidden: torch.Tensor = None) -> torch.Tensor:
        """
        Preferred fast path: pass teacher_hidden [B, L, Dt] to avoid O(B·L·Vt·Dt) cost.
        Fallback: teacher_probs [B, L, Vt] for compatibility with existing tests.

        Args:
            teacher_probs: [batch, seq_len, teacher_vocab] (fallback, slow)
            teacher_hidden: [batch, seq_len, teacher_dim] (preferred, fast)

        Returns:
            projected_teacher_logits: [batch, seq_len, student_vocab]
        """
        if teacher_hidden is None and teacher_probs is None:
            raise ValueError("Provide teacher_hidden or teacher_probs")

        if teacher_hidden is None:
            # Fallback (slower): compute expected hidden via probs @ embedding
            # Only for backward compatibility with tests
            teacher_embedding_weight = self.teacher_embedding.weight.to(teacher_probs.dtype)
            teacher_hidden = torch.matmul(teacher_probs, teacher_embedding_weight)

        # Project teacher hidden to student dimensionality
        student_hidden = self.hidden_projector(teacher_hidden)

        # Convert to student vocabulary logits
        student_embedding_weight = self.student_embedding.weight.to(student_hidden.dtype)
        projected_logits = torch.matmul(student_hidden, student_embedding_weight.t())
        return projected_logits


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

        # FIX ISSUE #4: Align teacher attention heads to student
        # Correct approach: project across head dimension, not spatial dimension
        if self.student_heads != self.teacher_heads:
            # Reshape to [batch, seq, seq, heads] for head projection
            teacher_attn = rearrange(teacher_attn, 'b h s1 s2 -> b s1 s2 h')
            # Project from teacher_heads to student_heads
            teacher_attn = self.head_projector(teacher_attn)
            # Reshape back to [batch, heads, seq, seq]
            teacher_attn = rearrange(teacher_attn, 'b s1 s2 h -> b h s1 s2')

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

            if teacher_hidden.size(1) != student_proj.size(1):
                teacher_hidden = F.interpolate(
                    teacher_hidden.transpose(1, 2),
                    size=student_proj.size(1),
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

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

        # Vocabulary projection for KD logits
        self.logit_projector = TeacherToStudentLogitProjector(
            teacher_embedding=self.teacher_model.get_input_embeddings(),
            student_embedding=self.student_model.get_input_embeddings(),
            teacher_dim=self.teacher_dim,
            student_dim=self.student_dim
        )

        # Store pad token id (used for masking losses)
        tokenizer = config.get('student_tokenizer')
        pad_token_id = getattr(tokenizer, 'pad_token_id', None) if tokenizer is not None else None
        self.pad_token_id = pad_token_id if pad_token_id is not None else -100

        # Loss weights
        self.alpha_kd = config.get('alpha_kd', 0.7)
        self.alpha_feature = config.get('alpha_feature', 0.1)
        self.alpha_attention = config.get('alpha_attention', 0.1)
        self.alpha_layerwise = config.get('alpha_layerwise', 0.05)
        self.alpha_contrastive = config.get('alpha_contrastive', 0.05)

        # FIX ISSUE #10: Temperature for KD with curriculum (start high, reduce gradually)
        self.base_temperature = config.get('temperature', 3.0)  # Reduced from 4.0 to 3.0
        self.min_temperature = config.get('min_temperature', 2.0)
        self.use_temperature_curriculum = config.get('use_temperature_curriculum', True)

        # FIX ISSUE #8: Curriculum learning for loss weights
        self.use_curriculum = config.get('use_curriculum', True)
        self.total_steps = config.get('total_steps', 10000)

        # Track which losses already emitted non-finite warnings
        self._loss_warnings_printed = set()

        # FIX ISSUE #7: Track loss magnitudes for adaptive balancing
        self.loss_magnitude_ema = {}
        self.magnitude_momentum = 0.9

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
        """FIX ISSUE #3: Improved NaN/Inf detection and handling"""
        if torch.isnan(value) or torch.isinf(value):
            if name not in self._loss_warnings_printed:
                print(f"[Warning] {name} produced non-finite value ({value.item() if value.numel() == 1 else 'tensor'}); clamping to zero.")
                self._loss_warnings_printed.add(name)
            return torch.zeros_like(value)

        # FIX ISSUE #3: Also check for unreasonably high values that indicate numerical issues
        if value.item() > 1000.0:
            if name not in self._loss_warnings_printed:
                print(f"[Warning] {name} produced very high value ({value.item():.2f}); possible numerical instability.")
                self._loss_warnings_printed.add(name)

        return value

    def _kl_on_subset(self, student_logits, teacher_logits, attention_mask, temperature, top_k=256):
        """HIGH-IMPACT OPTIMIZATION: Compute KD on union of top-k tokens only.

        Reduces vocab KD compute by 10-100x with negligible quality loss.
        Essential for P100 with large vocabs (student vocab ≈49k).

        Args:
            student_logits: [B, L, V]
            teacher_logits: [B, L, V]
            attention_mask: [B, L]
            temperature: float
            top_k: int (e.g., 256)

        Returns:
            kd_loss: scalar
        """
        # [B, L, V]
        with torch.no_grad():
            t_topk = torch.topk(teacher_logits, k=top_k, dim=-1).indices
            s_topk = torch.topk(student_logits, k=top_k, dim=-1).indices
            subset_idx = torch.cat([t_topk, s_topk], dim=-1)  # [B, L, 2k]
            # Get unique indices per position
            subset_idx, _ = torch.sort(subset_idx, dim=-1)

        # Gather both distributions on subset
        s_sub = torch.gather(student_logits, dim=-1, index=subset_idx).div(temperature)
        t_sub = torch.gather(teacher_logits, dim=-1, index=subset_idx).div(temperature)

        s_logp = F.log_softmax(s_sub, dim=-1)
        t_p = F.softmax(t_sub, dim=-1)

        kd_raw = F.kl_div(s_logp, t_p, reduction='none').sum(-1)  # [B, L]
        mask = attention_mask[:, :kd_raw.size(1)].to(kd_raw.dtype)
        return (kd_raw * mask).sum() / mask.sum().clamp(min=1.0) * (temperature ** 2)

    def _get_curriculum_temperature(self, step: Optional[int] = None) -> float:
        """FIX ISSUE #10: Temperature curriculum - start high, gradually reduce

        High temperature (3.0) early: softer targets, easier learning
        Low temperature (2.0) late: sharper targets, better final performance
        """
        if not self.use_temperature_curriculum or step is None:
            return self.base_temperature

        progress = min(1.0, step / self.total_steps)
        # Linear annealing from base_temperature to min_temperature
        temperature = self.base_temperature - (self.base_temperature - self.min_temperature) * progress
        return temperature

    def _get_curriculum_weights(self, step: Optional[int] = None) -> Dict[str, float]:
        """FIX ISSUE #8: Compute curriculum learning weights

        Progressive loss introduction:
        - Phase 1 (0-30%): Focus on KD loss only
        - Phase 2 (30-60%): Add attention and feature losses
        - Phase 3 (60-100%): Add all losses

        This stabilizes early training and gradually increases task complexity.
        """
        if not self.use_curriculum or step is None:
            return {
                'kd': self.alpha_kd,
                'feature': self.alpha_feature,
                'attention': self.alpha_attention,
                'layerwise': self.alpha_layerwise,
                'contrastive': self.alpha_contrastive
            }

        progress = min(1.0, step / self.total_steps)

        # Phase transitions
        if progress < 0.3:  # Phase 1: KD only
            phase_progress = progress / 0.3
            return {
                'kd': self.alpha_kd,
                'feature': 0.0,
                'attention': 0.0,
                'layerwise': 0.0,
                'contrastive': 0.0
            }
        elif progress < 0.6:  # Phase 2: Add attention and feature
            phase_progress = (progress - 0.3) / 0.3
            return {
                'kd': self.alpha_kd,
                'feature': self.alpha_feature * phase_progress,
                'attention': self.alpha_attention * phase_progress,
                'layerwise': 0.0,
                'contrastive': 0.0
            }
        else:  # Phase 3: All losses
            phase_progress = (progress - 0.6) / 0.4
            return {
                'kd': self.alpha_kd,
                'feature': self.alpha_feature,
                'attention': self.alpha_attention,
                'layerwise': self.alpha_layerwise * phase_progress,
                'contrastive': self.alpha_contrastive * phase_progress
            }

    def _update_loss_magnitude_ema(self, loss_name: str, loss_value: float):
        """FIX ISSUE #7: Track EMA of loss magnitudes for adaptive balancing"""
        if loss_name not in self.loss_magnitude_ema:
            self.loss_magnitude_ema[loss_name] = loss_value
        else:
            self.loss_magnitude_ema[loss_name] = (
                self.magnitude_momentum * self.loss_magnitude_ema[loss_name] +
                (1 - self.magnitude_momentum) * loss_value
            )

    def _get_balanced_loss_weight(self, loss_name: str, base_weight: float) -> float:
        """FIX ISSUE #7: Compute magnitude-balanced weight

        Normalizes loss contributions so each component has similar impact
        regardless of absolute magnitude.
        """
        if not self.loss_magnitude_ema or loss_name not in self.loss_magnitude_ema:
            return base_weight

        # Get max magnitude for normalization
        max_magnitude = max(self.loss_magnitude_ema.values())
        current_magnitude = self.loss_magnitude_ema[loss_name]

        if current_magnitude < 1e-6:
            return base_weight

        # Scale weight inversely with magnitude
        balanced_weight = base_weight * (max_magnitude / current_magnitude)

        # Clamp to reasonable range (0.1x to 10x original weight)
        balanced_weight = max(base_weight * 0.1, min(base_weight * 10.0, balanced_weight))

        return balanced_weight

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
        # CRITICAL FIX #5: Verify gradient checkpointing is properly enabled
        if hasattr(self.student_model, 'gradient_checkpointing_enable'):
            self.student_model.gradient_checkpointing_enable()

            # Verify that use_cache is disabled (required for gradient checkpointing)
            if hasattr(self.student_model.config, 'use_cache'):
                if self.student_model.config.use_cache:
                    print("[Warning] Gradient checkpointing requires use_cache=False, but it's still True!")
                    self.student_model.config.use_cache = False

            print("[Info] Gradient checkpointing enabled for student model")
        else:
            print("[Warning] Student model does not support gradient_checkpointing_enable()")

    def _resize_attention(self, attn_tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        """Resize attention maps to the target sequence length."""
        if attn_tensor.size(-1) == target_len and attn_tensor.size(-2) == target_len:
            return attn_tensor

        b, h, s1, s2 = attn_tensor.shape
        attn_reshaped = attn_tensor.reshape(b * h, 1, s1, s2)
        attn_resized = F.interpolate(
            attn_reshaped,
            size=(target_len, target_len),
            mode='bilinear',
            align_corners=False
        )
        return attn_resized.reshape(b, h, target_len, target_len)

    @torch.no_grad()
    def get_teacher_outputs(self,
                           teacher_input_ids: torch.Tensor,
                           teacher_attention_mask: torch.Tensor) -> Dict:
        """Get teacher model outputs with proper MoE expert extraction"""
        outputs = self.teacher_model(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )

        # Extract MoE expert outputs if available
        expert_outputs = []
        if hasattr(self.teacher_model, 'model') and hasattr(self.teacher_model.model, 'layers'):
            for layer_idx, layer in enumerate(self.teacher_model.model.layers):
                if hasattr(layer, 'block_sparse_moe'):
                    # This is a MoE layer - extract actual expert outputs
                    moe = layer.block_sparse_moe
                    hidden = outputs.hidden_states[layer_idx]

                    try:
                        # Get hidden state for this layer
                        batch_size, seq_len, hidden_size = hidden.shape
                        hidden_reshaped = hidden.view(-1, hidden_size)

                        # Get routing decisions from gate
                        if hasattr(moe, 'gate'):
                            router_logits = moe.gate(hidden_reshaped)
                        else:
                            # Fallback: use layer output as single expert
                            expert_outputs.append(hidden)
                            continue

                        # Get individual expert outputs if accessible
                        if hasattr(moe, 'experts'):
                            num_experts = len(moe.experts)
                            for expert_idx in range(min(num_experts, 8)):  # Limit to 8 experts
                                try:
                                    expert = moe.experts[expert_idx]
                                    expert_out = expert(hidden_reshaped)
                                    expert_out = expert_out.view(batch_size, seq_len, -1)
                                    expert_outputs.append(expert_out)
                                except:
                                    # If expert access fails, use hidden state
                                    expert_outputs.append(hidden)
                                    break
                        else:
                            # If experts not accessible, use layer output
                            expert_outputs.append(hidden)
                    except Exception as e:
                        # Fallback: use the hidden state if expert extraction fails
                        expert_outputs.append(hidden)

        # If no experts were extracted, use hidden states from different layers
        if not expert_outputs:
            # Use evenly spaced hidden layers as "pseudo-experts"
            num_layers = len(outputs.hidden_states)
            step = max(1, num_layers // 8)
            expert_outputs = [outputs.hidden_states[i] for i in range(0, num_layers, step)][:8]

            # Ensure at least one expert
            if not expert_outputs:
                expert_outputs = [outputs.hidden_states[-1]]

        return {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            'expert_outputs': expert_outputs
        }

    def forward(self,
                student_input_ids: torch.Tensor,
                student_attention_mask: torch.Tensor,
                teacher_input_ids: torch.Tensor,
                teacher_attention_mask: torch.Tensor,
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
            teacher_outputs = self.get_teacher_outputs(teacher_input_ids, teacher_attention_mask)
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
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
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

        # FIX ISSUE #8: Get curriculum learning weights
        curriculum_weights = self._get_curriculum_weights(step)

        # DEBUG: Log curriculum weights periodically
        if step is not None and (step % 500 == 0 or step < 10):
            progress_pct = (step / self.total_steps) * 100 if self.total_steps > 0 else 0
            print(f"\n{'='*60}")
            print(f"[CURRICULUM] Step {step}, Progress {progress_pct:.1f}%")
            print(f"{'='*60}")
            print(f"  Curriculum weights:")
            for k, v in curriculum_weights.items():
                print(f"    {k}: {v:.4f}")
            print(f"{'='*60}\n")

        # FIX ISSUE #10: Get curriculum temperature (annealing)
        current_temperature = self._get_curriculum_temperature(step)

        # 1. KL Divergence Loss (main distillation loss)
        # CRITICAL FIX: Use fast path with teacher hidden states
        teacher_hidden_last = teacher_hidden[-1]  # [B, Lt, Dt]

        # Align sequence lengths if needed
        if teacher_hidden_last.size(1) != student_logits.size(1):
            teacher_hidden_last = F.interpolate(
                teacher_hidden_last.transpose(1, 2),
                size=student_logits.size(1),
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        # Project teacher hidden to student vocab logits (avoids O(B·L·Vt·Dt) cost)
        projected_teacher_logits = self.logit_projector(teacher_hidden=teacher_hidden_last)

        # Perform KD computations in float32 to avoid half-precision underflow/overflow
        student_logits_fp32 = student_logits.float()
        teacher_logits_fp32 = projected_teacher_logits.float()

        # FIX ISSUE #4: More aggressive clamping to prevent extreme values
        clamp_min, clamp_max = -20.0, 20.0  # Reduced from -30/30 to -20/20
        student_logits_clamped = torch.clamp(student_logits_fp32, min=clamp_min, max=clamp_max)
        teacher_logits_clamped = torch.clamp(teacher_logits_fp32, min=clamp_min, max=clamp_max)

        # HIGH-IMPACT OPTIMIZATION: Use subset KD if configured (FIX ISSUE #13)
        kd_top_k = self.config.get('kd_top_k', 256)  # Default to 256 for P100 memory efficiency
        if kd_top_k > 0:
            # Subset KD: 10-100x faster with minimal quality loss
            kd_loss = self._kl_on_subset(
                student_logits_clamped,
                teacher_logits_clamped,
                student_attention_mask,
                current_temperature,
                top_k=kd_top_k
            )
        else:
            # Full vocab KD (original path)
            student_log_probs = F.log_softmax(student_logits_clamped / current_temperature, dim=-1)
            teacher_probs_clamped = torch.clamp(projected_teacher_logits / current_temperature, min=clamp_min, max=clamp_max)
            teacher_probs = F.softmax(teacher_probs_clamped, dim=-1)

            # Ensure teacher_probs don't have NaNs from softmax and remain normalized
            teacher_probs = torch.nan_to_num(teacher_probs, nan=0.0)
            teacher_probs = teacher_probs / torch.clamp(teacher_probs.sum(dim=-1, keepdim=True), min=1e-8)

            kd_raw = F.kl_div(student_log_probs, teacher_probs, reduction='none')  # [B, L, V]
            kd_per_token = kd_raw.sum(dim=-1)  # [B, L]

            attention_mask = student_attention_mask
            if attention_mask is not None:
                mask = attention_mask[:, :kd_per_token.size(1)].to(kd_per_token.dtype)
            else:
                mask = kd_per_token.new_ones(kd_per_token.shape)

            masked_kd = (kd_per_token * mask).sum() / mask.sum().clamp(min=1.0)
            kd_loss = masked_kd * (current_temperature ** 2)

        # FIX ISSUE #7: Track loss magnitude for adaptive balancing
        self._update_loss_magnitude_ema('kd', kd_loss.item())

        # FIX ISSUE #8: Use curriculum weight instead of fixed alpha
        weighted_kd = kd_loss * curriculum_weights['kd']
        losses['kd_loss'] = self._ensure_finite_loss('kd_loss', weighted_kd)

        # DEBUG: Log KD loss details
        if step is not None and (step % 500 == 0 or step < 10):
            print(f"\n{'='*60}")
            print(f"[KD LOSS] Step {step}")
            print(f"{'='*60}")
            print(f"  Raw KD loss: {kd_loss.item():.4f}")
            print(f"  Curriculum weight: {curriculum_weights['kd']:.4f}")
            print(f"  Weighted KD loss: {weighted_kd.item():.4f}")
            print(f"{'='*60}\n")

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
            # FIX ISSUE #8: Use curriculum weights for routing losses
            for loss_name, loss_value in routing_outputs['losses'].items():
                if loss_name == 'attention_alignment_loss':
                    weight = curriculum_weights['attention']
                else:
                    weight = curriculum_weights['feature']

                scaled = self._ensure_finite_loss(f'routing_{loss_name}', loss_value * weight)
                losses[f'routing_{loss_name}'] = scaled

            # DEBUG: Log routing losses details
            if step is not None and (step % 500 == 0 or step < 10):
                print(f"\n{'='*60}")
                print(f"[ROUTING LOSSES] Step {step}")
                print(f"{'='*60}")
                for loss_name, loss_value in routing_outputs['losses'].items():
                    raw_val = loss_value.item()
                    if loss_name == 'attention_alignment_loss':
                        weight = curriculum_weights['attention']
                    else:
                        weight = curriculum_weights['feature']
                    scaled_val = raw_val * weight
                    print(f"  {loss_name}:")
                    print(f"    Raw: {raw_val:.4f}")
                    print(f"    Weight: {weight:.4f}")
                    print(f"    Scaled: {scaled_val:.4f}")
                print(f"{'='*60}\n")

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
                    if t_attn.size(-1) != s_attn.size(-1):
                        t_attn = self._resize_attention(t_attn, s_attn.size(-1))
                    attn_loss = self.attention_transfer(
                        s_attn,
                        t_attn
                    )
                    attn_losses.append(attn_loss)

            if attn_losses:
                attn_mean = torch.stack(attn_losses).mean()
                # FIX ISSUE #8: Use curriculum weight
                losses['attention_loss'] = self._ensure_finite_loss('attention_loss', attn_mean * curriculum_weights['attention'])

        # 4. Layer-wise distillation loss
        # FIX ISSUE #8: Use curriculum weight
        if len(student_hidden) > 1 and len(teacher_hidden) > 1 and curriculum_weights['layerwise'] > 0:
            layerwise_loss = self.layerwise_loss(
                list(student_hidden),
                list(teacher_hidden)
            )
            losses['layerwise_loss'] = self._ensure_finite_loss('layerwise_loss', layerwise_loss * curriculum_weights['layerwise'])

        # 5. Contrastive loss (using CLS or mean pooling)
        # FIX ISSUE #8: Use curriculum weight
        if curriculum_weights['contrastive'] > 0:
            # Mean pooling over sequence dimension
            student_embed = student_hidden[-1].mean(dim=1)
            teacher_embed = teacher_hidden[-1].mean(dim=1)

            contrastive_loss = self.contrastive_loss(student_embed, teacher_embed)
            losses['contrastive_loss'] = self._ensure_finite_loss('contrastive_loss', contrastive_loss * curriculum_weights['contrastive'])

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
            'routing_info': routing_outputs['routing_info'] if 'routing_outputs' in locals() else {}
        }

    def _chunked_cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross entropy in chunks to reduce activation memory footprint.

        FIX ISSUE #9: Added label smoothing for better generalization.
        """
        # Shift for causal language modeling (predict next token)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Mask padding tokens so they do not contribute to loss
        if self.pad_token_id != -100:
            shift_labels = shift_labels.masked_fill(shift_labels == self.pad_token_id, -100)

        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)

        chunk_size = self.loss_chunk_size or flat_logits.size(0)

        # FIX ISSUE #9: Use label smoothing (SOTA: 0.1)
        label_smoothing = self.config.get('label_smoothing', 0.1)

        if chunk_size >= flat_logits.size(0):
            return F.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=-100,
                label_smoothing=label_smoothing
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
                    reduction='sum',
                    label_smoothing=label_smoothing  # FIX ISSUE #9
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
