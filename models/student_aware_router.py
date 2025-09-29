"""
Student-Aware Router for Knowledge Distillation
Implements novel routing mechanism that adapts to student model capacity and learning dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np
from einops import rearrange, repeat


class StudentCapacityEstimator(nn.Module):
    """Estimates student model's current learning capacity and knowledge gaps"""

    def __init__(self, student_dim: int, teacher_dim: int, num_experts: int = 8):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.num_experts = num_experts

        # Capacity estimation network
        self.capacity_net = nn.Sequential(
            nn.Linear(student_dim, student_dim * 2),
            nn.LayerNorm(student_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(student_dim * 2, student_dim),
            nn.LayerNorm(student_dim),
            nn.GELU(),
            nn.Linear(student_dim, num_experts)
        )

        # Teacher to student dimension projection for gap analysis
        self.teacher_to_student_proj = nn.Linear(teacher_dim, student_dim)

        # Knowledge gap analyzer
        self.gap_analyzer = nn.Sequential(
            nn.Linear(student_dim + student_dim, teacher_dim),  # Both inputs now student_dim
            nn.LayerNorm(teacher_dim),
            nn.GELU(),
            nn.Linear(teacher_dim, num_experts)
        )

        # Adaptive temperature for capacity-aware routing
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, student_hidden: torch.Tensor,
                teacher_hidden: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            student_hidden: [batch_size, seq_len, student_dim]
            teacher_hidden: [batch_size, seq_len, teacher_dim] (optional)

        Returns:
            Dictionary containing capacity scores and routing weights
        """
        batch_size, seq_len, _ = student_hidden.shape

        # Estimate current capacity
        capacity_scores = self.capacity_net(student_hidden)  # [B, L, num_experts]
        capacity_scores = F.softmax(capacity_scores / self.temperature, dim=-1)

        # Analyze knowledge gaps if teacher hidden states are available
        gap_scores = None
        if teacher_hidden is not None:
            batch_size, student_seq_len, student_dim = student_hidden.shape
            teacher_batch_size, teacher_seq_len, teacher_dim_actual = teacher_hidden.shape

            # Step 1: Align sequence lengths using interpolation
            if teacher_seq_len != student_seq_len:
                # Interpolate teacher sequence to match student sequence length
                teacher_seq_aligned = F.interpolate(
                    teacher_hidden.transpose(1, 2),  # [B, D, L]
                    size=student_seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # [B, L, D]
            else:
                teacher_seq_aligned = teacher_hidden

            # Step 2: Align hidden dimensions for concatenation
            # We need to project teacher to student dimensions for gap analysis
            teacher_proj = self.teacher_to_student_proj(teacher_seq_aligned)

            # Now both tensors have shape [B, student_seq_len, student_dim]
            combined = torch.cat([student_hidden, teacher_proj], dim=-1)

            gap_scores = self.gap_analyzer(combined)
            gap_scores = F.softmax(gap_scores, dim=-1)

        return {
            'capacity_scores': capacity_scores,
            'gap_scores': gap_scores,
            'temperature': self.temperature
        }


class AdaptiveExpertRouter(nn.Module):
    """Routes knowledge from teacher experts to student based on adaptive capacity"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_experts = config.get('num_experts', 8)
        self.student_dim = config['student_dim']
        self.teacher_dim = config['teacher_dim']
        self.top_k = config.get('top_k', 2)

        # Student capacity estimator
        self.capacity_estimator = StudentCapacityEstimator(
            self.student_dim, self.teacher_dim, self.num_experts
        )

        # Expert importance scoring (adapted to work with teacher dimensions)
        self.expert_scorer = nn.Sequential(
            nn.Linear(self.teacher_dim, self.teacher_dim // 2),
            nn.GELU(),
            nn.Linear(self.teacher_dim // 2, self.num_experts)
        )

        # Routing gate with load balancing
        self.routing_gate = nn.Linear(self.student_dim, self.num_experts)

        # Load balancing loss weight
        self.load_balance_weight = config.get('load_balance_weight', 0.01)

        # Noise for exploration during training
        self.noise_std = config.get('noise_std', 0.1)

    def compute_routing_weights(self,
                               student_hidden: torch.Tensor,
                               teacher_hidden: torch.Tensor,
                               training: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Compute adaptive routing weights based on student capacity

        Args:
            student_hidden: [batch_size, seq_len, student_dim]
            teacher_hidden: [batch_size, seq_len, teacher_dim]
            training: Whether in training mode

        Returns:
            routing_weights: [batch_size, seq_len, num_experts]
            aux_info: Dictionary with auxiliary information
        """
        batch_size, seq_len, _ = student_hidden.shape

        # Get student capacity estimates
        capacity_info = self.capacity_estimator(student_hidden, teacher_hidden)
        capacity_scores = capacity_info['capacity_scores']
        gap_scores = capacity_info['gap_scores']

        # Compute base routing scores
        routing_logits = self.routing_gate(student_hidden)

        # Add noise for exploration during training
        if training and self.noise_std > 0:
            noise = torch.randn_like(routing_logits) * self.noise_std
            routing_logits = routing_logits + noise

        # Align teacher hidden with student for expert scoring
        batch_size, student_seq_len, student_dim = student_hidden.shape
        teacher_batch_size, teacher_seq_len, teacher_dim_actual = teacher_hidden.shape

        # Align teacher sequence length to student
        if teacher_seq_len != student_seq_len:
            teacher_for_scoring = F.interpolate(
                teacher_hidden.transpose(1, 2),
                size=student_seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            teacher_for_scoring = teacher_hidden

        # Compute expert importance from aligned teacher
        expert_importance = self.expert_scorer(teacher_for_scoring)
        expert_importance = F.softmax(expert_importance, dim=-1)

        # Combine routing signals
        if gap_scores is not None:
            # Weighted combination of capacity, gaps, and importance
            combined_scores = (
                0.4 * capacity_scores +
                0.3 * gap_scores +
                0.3 * expert_importance
            )
        else:
            combined_scores = 0.6 * capacity_scores + 0.4 * expert_importance

        # Apply top-k selection
        topk_scores, topk_indices = torch.topk(
            combined_scores, self.top_k, dim=-1
        )

        # Create sparse routing weights
        routing_weights = torch.zeros_like(combined_scores)
        routing_weights.scatter_(-1, topk_indices, topk_scores)

        # Normalize weights
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute load balancing loss
        load_balance_loss = self.compute_load_balance_loss(routing_weights)

        aux_info = {
            'load_balance_loss': load_balance_loss,
            'capacity_scores': capacity_scores,
            'gap_scores': gap_scores,
            'expert_importance': expert_importance,
            'selected_experts': topk_indices
        }

        return routing_weights, aux_info

    def compute_load_balance_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss to ensure even expert utilization (MSE version for stability)"""
        # Average routing weights across batch and sequence
        avg_routing = routing_weights.mean(dim=[0, 1])

        # Target uniform distribution
        uniform_target = torch.full_like(avg_routing, 1.0 / self.num_experts)

        # Use Mean Squared Error loss for numerical stability (avoids log)
        load_balance_loss = F.mse_loss(avg_routing, uniform_target)

        return load_balance_loss * self.load_balance_weight

    def forward(self, student_hidden: torch.Tensor,
                teacher_expert_outputs: List[torch.Tensor],
                training: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Route teacher expert outputs to student based on capacity

        Args:
            student_hidden: [batch_size, seq_len, student_dim]
            teacher_expert_outputs: List of [batch_size, seq_len, teacher_dim]
            training: Whether in training mode

        Returns:
            routed_output: [batch_size, seq_len, teacher_dim]
            routing_info: Dictionary with routing information
        """
        batch_size, seq_len, _ = student_hidden.shape

        # Align expert outputs to student sequence length
        aligned_expert_outputs = []
        for expert_output in teacher_expert_outputs:
            expert_batch, expert_seq, expert_dim = expert_output.shape
            if expert_seq != seq_len:
                # Interpolate to match student sequence length
                aligned_expert = F.interpolate(
                    expert_output.transpose(1, 2),  # [B, D, L]
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # [B, L, D]
            else:
                aligned_expert = expert_output
            aligned_expert_outputs.append(aligned_expert)

        # Stack aligned expert outputs
        expert_stack = torch.stack(aligned_expert_outputs, dim=2)  # [B, L, E, D]

        # Use first aligned expert output as representative for routing
        representative_teacher = aligned_expert_outputs[0]

        # Compute routing weights
        routing_weights, aux_info = self.compute_routing_weights(
            student_hidden, representative_teacher, training
        )

        # Apply routing weights
        routing_weights = routing_weights.unsqueeze(-1)  # [B, L, E, 1]
        routed_output = (expert_stack * routing_weights).sum(dim=2)  # [B, L, D]

        routing_info = {
            'routing_weights': routing_weights.squeeze(-1),
            'aux_info': aux_info
        }

        return routed_output, routing_info


class ProgressiveRoutingScheduler:
    """Schedules routing complexity based on student learning progress"""

    def __init__(self,
                 initial_top_k: int = 1,
                 final_top_k: int = 4,
                 warmup_steps: int = 1000,
                 total_steps: int = 10000):
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

    def get_current_top_k(self) -> int:
        """Get current top-k value based on training progress"""
        if self.current_step < self.warmup_steps:
            return self.initial_top_k

        progress = min(1.0, (self.current_step - self.warmup_steps) /
                      (self.total_steps - self.warmup_steps))

        # Smooth progression
        progress = np.sin(progress * np.pi / 2)  # Smooth acceleration

        current_k = self.initial_top_k + int(
            progress * (self.final_top_k - self.initial_top_k)
        )

        return current_k

    def step(self):
        """Update step counter"""
        self.current_step += 1

    def get_routing_temperature(self) -> float:
        """Get temperature for routing softmax based on progress"""
        if self.current_step < self.warmup_steps:
            return 1.0

        progress = min(1.0, self.current_step / self.total_steps)
        # Temperature annealing: start hot, cool down
        return max(0.1, 1.0 - 0.9 * progress)


class StudentAwareDistillationRouter(nn.Module):
    """Main router for student-aware knowledge distillation"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Core routing module
        self.adaptive_router = AdaptiveExpertRouter(config)

        # Progressive scheduler
        self.scheduler = ProgressiveRoutingScheduler(
            initial_top_k=config.get('initial_top_k', 1),
            final_top_k=config.get('final_top_k', 4),
            warmup_steps=config.get('warmup_steps', 1000),
            total_steps=config.get('total_steps', 10000)
        )

        # Feature alignment layers
        self.student_projector = nn.Linear(
            config['student_dim'], config['teacher_dim']
        )

        self.teacher_compressor = nn.Sequential(
            nn.Linear(config['teacher_dim'], config['teacher_dim'] // 2),
            nn.GELU(),
            nn.Linear(config['teacher_dim'] // 2, config['student_dim'])
        )

        # Attention transfer module
        self.attention_transfer = nn.MultiheadAttention(
            embed_dim=config['student_dim'],
            num_heads=config.get('num_heads', 4),
            dropout=0.1,
            batch_first=True
        )

    def forward(self,
                student_hidden: torch.Tensor,
                teacher_outputs: Dict,
                step: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Perform student-aware routing and knowledge transfer

        Args:
            student_hidden: Student model hidden states
            teacher_outputs: Dictionary containing teacher model outputs
            step: Current training step

        Returns:
            Dictionary containing routed outputs and auxiliary losses
        """
        if step is not None:
            self.scheduler.current_step = step
            self.adaptive_router.top_k = self.scheduler.get_current_top_k()
            temperature = self.scheduler.get_routing_temperature()
            self.adaptive_router.capacity_estimator.temperature.data = torch.tensor(temperature)

        # Extract teacher expert outputs if available (for MoE)
        if 'expert_outputs' in teacher_outputs:
            teacher_expert_outputs = teacher_outputs['expert_outputs']
        else:
            # Treat single model as single expert
            teacher_expert_outputs = [teacher_outputs['hidden_states']]

        # Route teacher knowledge
        routed_knowledge, routing_info = self.adaptive_router(
            student_hidden, teacher_expert_outputs, self.training
        )

        # Project student to teacher space
        student_projected = self.student_projector(student_hidden)

        # Compress teacher knowledge to student space
        teacher_compressed = self.teacher_compressor(routed_knowledge)

        # Attention-based knowledge transfer
        attended_knowledge, attention_weights = self.attention_transfer(
            student_hidden, teacher_compressed, teacher_compressed
        )

        # Compute alignment losses
        feature_loss = F.mse_loss(student_projected, routed_knowledge)
        attention_loss = F.mse_loss(student_hidden, attended_knowledge)

        outputs = {
            'routed_knowledge': routed_knowledge,
            'teacher_compressed': teacher_compressed,
            'attended_knowledge': attended_knowledge,
            'routing_info': routing_info,
            'losses': {
                'feature_loss': feature_loss,
                'attention_loss': attention_loss,
                'load_balance_loss': routing_info['aux_info']['load_balance_loss']
            },
            'attention_weights': attention_weights
        }

        return outputs
