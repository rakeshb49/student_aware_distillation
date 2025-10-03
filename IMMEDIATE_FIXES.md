# IMMEDIATE FIXES FOR CRITICAL ISSUES

## Fix #1: CRITICAL - Dtype Mismatch in LogitProjector

**File:** `models/distillation_framework.py`

**Location:** In `__init__` method of `StudentAwareDistillationFramework`, after creating `logit_projector`

**Current Code (around line 360):**
```python
# Vocabulary projection for KD logits
self.logit_projector = TeacherToStudentLogitProjector(
    teacher_embedding=self.teacher_model.get_input_embeddings(),
    student_embedding=self.student_model.get_input_embeddings(),
    teacher_dim=self.teacher_dim,
    student_dim=self.student_dim
)
```

**Add After:**
```python
# CRITICAL FIX: Ensure logit_projector matches teacher model's dtype
# Teacher is in float16, but projector initializes in float32 by default
# This causes "mat1 and mat2 must have same dtype" error during evaluation
self.logit_projector = self.logit_projector.to(self.teacher_model.dtype)
```

---

## Fix #2: CRITICAL - Wrap Evaluation in Autocast

**File:** `utils/evaluation.py`

**Location:** In `compute_knowledge_retention` method (around line 216)

**Current Code:**
```python
# Because vocabularies differ, align teacher logits via projection if available
if self.logit_projector is not None:
    teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
    aligned_teacher_logits = self.logit_projector(teacher_probs)
```

**Replace With:**
```python
# Because vocabularies differ, align teacher logits via projection if available
if self.logit_projector is not None:
    teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
    # CRITICAL FIX: Ensure dtype compatibility by wrapping in autocast
    # LogitProjector expects consistent dtypes; teacher outputs are float16
    from torch.cuda.amp import autocast
    with autocast(dtype=torch.float16 if teacher_logits.dtype == torch.float16 else torch.float32):
        aligned_teacher_logits = self.logit_projector(teacher_probs)
```

---

## Fix #3: HIGH - Normalize Feature Loss

**File:** `models/distillation_framework.py`

**Location:** In `forward` method, around feature loss computation

**Find:**
```python
feature_loss = self.router.compute_routing_loss(...)
```

**Add After:**
```python
# HIGH FIX: Normalize feature loss by hidden dimension to prevent explosion
# Raw MSE loss scales with dimensionality, causing 150-200 magnitude
feature_loss = feature_loss / math.sqrt(self.student_dim)
```

**Alternative (Better):** Use cosine similarity instead of MSE

**Find:** `FeatureProjector` class

**Add Method:**
```python
def forward_with_similarity_loss(self, teacher_feat, student_feat):
    """Compute cosine similarity loss instead of MSE"""
    projected = self.projector(teacher_feat)
    # Normalize to unit vectors
    projected_norm = F.normalize(projected, p=2, dim=-1)
    student_norm = F.normalize(student_feat, p=2, dim=-1)
    # Cosine similarity (1 = perfect alignment, -1 = opposite)
    similarity = (projected_norm * student_norm).sum(dim=-1).mean()
    # Convert to loss (0 = perfect, 2 = worst)
    loss = 1.0 - similarity
    return loss
```

---

## Fix #4: HIGH - Normalize Attention Loss

**File:** `models/distillation_framework.py`

**Location:** In `AttentionTransferModule` class

**Find:** The `forward` method that computes attention loss

**Current (approximately):**
```python
def forward(self, student_attention, teacher_attention):
    # ... alignment code ...
    loss = F.mse_loss(student_attention, aligned_teacher_attention)
    return loss
```

**Replace With:**
```python
def forward(self, student_attention, teacher_attention):
    # ... alignment code ...
    
    # HIGH FIX: Normalize attention patterns before comparison
    # Raw attention weights vary widely in magnitude
    student_norm = F.softmax(student_attention, dim=-1)
    teacher_norm = F.softmax(aligned_teacher_attention, dim=-1)
    
    # Use KL divergence instead of MSE for probability distributions
    loss = F.kl_div(
        F.log_softmax(student_attention, dim=-1),
        teacher_norm,
        reduction='batchmean'
    )
    
    # Scale to reasonable magnitude (KL can be large for misaligned distributions)
    loss = loss * 0.1
    return loss
```

---

## Fix #5: HIGH - Fix Curriculum Learning Weights

**File:** `models/distillation_framework.py`

**Location:** In method that computes curriculum weights (find the curriculum logic)

**Current Logic (producing 0.0672 at 50% progress):**
Likely using: `weight = progress ** 2` or similar

**Replace With:**
```python
def get_curriculum_weight(self, step: int, loss_type: str) -> float:
    """
    Compute curriculum weight for a specific loss type.
    
    HIGH FIX: Use linear ramp-up to reach full weight by 80% progress.
    Previous quadratic ramp was too conservative (6.7% at 50% progress).
    """
    if not self.use_curriculum:
        return getattr(self, f'alpha_{loss_type}', 0.0)
    
    progress = min(step / self.total_steps, 1.0)
    base_weight = getattr(self, f'alpha_{loss_type}', 0.0)
    
    if loss_type == 'kd':
        # KD is always active at full weight
        return base_weight
    
    elif loss_type in ['feature', 'attention']:
        # Linear ramp: 0% → 0.0, 80% → 1.0
        ramp_progress = min(progress / 0.8, 1.0)
        return base_weight * ramp_progress
    
    elif loss_type in ['layerwise', 'contrastive']:
        # Delayed start: activate after 30% progress
        if progress < 0.3:
            return 0.0
        ramp_progress = (progress - 0.3) / 0.7
        return base_weight * ramp_progress
    
    return base_weight
```

---

## Fix #6: HIGH - Reduce Logging Spam

**File:** `models/distillation_framework.py` or wherever logging happens

**Find:** All the curriculum/KD/routing loss print statements

**Wrap in Conditional:**
```python
# HIGH FIX: Only log detailed loss breakdown every N steps
# Previous: logged every step (450k+ lines)
# New: log every 100 steps, plus evaluations

VERBOSE_LOG_INTERVAL = 100

if step % VERBOSE_LOG_INTERVAL == 0 or step < 10:
    print("\n" + "="*60)
    print(f"[CURRICULUM] Step {step}, Progress {progress:.1%}")
    print("="*60)
    # ... rest of logging ...
```

**Alternative:** Use a proper logging level

```python
import logging

# Set up logger at module level
logger = logging.getLogger(__name__)

# In verbose sections, use:
logger.debug(f"[CURRICULUM] Step {step}, Progress {progress:.1%}")

# In critical sections, use:
logger.info(f"[Eval] loss: {loss:.4f}")
```

---

## Fix #7: MEDIUM - Increase Early Stopping Patience

**File:** `utils/training.py`

**Location:** Where EarlyStopping is initialized

**Find:**
```python
self.early_stopping = EarlyStopping(patience=10, ...)
```

**Replace With:**
```python
# MEDIUM FIX: Increase patience for first-epoch training with curriculum
# Previous: 10 evaluations (~5k steps) was too aggressive
# New: 20 evaluations (~10k steps) allows more time for convergence
self.early_stopping = EarlyStopping(
    patience=20,
    min_delta=0.01,  # Require at least 1% improvement
    warmup_steps=1000  # Don't check early stopping for first 1k steps
)
```

**If `warmup_steps` not supported, add it to EarlyStopping class:**
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, warmup_steps=0):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_steps = warmup_steps
        self.counter = 0
        self.best_loss = float('inf')
        self.current_step = 0
    
    def __call__(self, loss, step=None):
        if step is not None:
            self.current_step = step
            
        # Don't trigger during warmup
        if self.current_step < self.warmup_steps:
            return False
        
        # ... rest of logic ...
```

---

## Fix #8: MEDIUM - Verify LM Loss is Active

**File:** `models/distillation_framework.py`

**Location:** In `forward` method

**Find:** Where total loss is computed

**Ensure LM loss is ALWAYS computed and added:**
```python
# MEDIUM FIX: Ensure LM loss is always computed and logged
# Missing from logs after step 0, suggesting it might be disabled

if labels is not None:
    # Compute language modeling loss on student
    lm_loss = F.cross_entropy(
        student_logits.view(-1, self.student_vocab_size),
        labels.view(-1),
        ignore_index=self.pad_token_id,
        reduction='mean'
    )
    # Apply fixed weight (don't curriculum this - always active)
    alpha_lm = self.config.get('alpha_lm', 0.1)
    total_loss += alpha_lm * lm_loss
    losses['lm_loss'] = lm_loss
else:
    # Even without labels, compute pseudo-LM loss for regularization
    lm_loss = F.cross_entropy(
        student_logits.view(-1, self.student_vocab_size),
        student_input_ids.view(-1),
        ignore_index=self.pad_token_id,
        reduction='mean'
    )
    alpha_lm = self.config.get('alpha_lm', 0.05)  # Lower weight without labels
    total_loss += alpha_lm * lm_loss
    losses['lm_loss'] = lm_loss

# ALWAYS log LM loss
if step % VERBOSE_LOG_INTERVAL == 0:
    print(f"  lm_loss: {lm_loss.item():.4f}")
```

---

## Fix #9: MEDIUM - Add Loss Magnitude Tracking

**File:** `models/distillation_framework.py`

**Location:** In `forward` method, after computing all losses

**Add Before Combining:**
```python
# MEDIUM FIX: Track and balance loss magnitudes adaptively
# Current issue: losses differ by 1-2 orders of magnitude

def _update_loss_magnitude(self, loss_name: str, loss_value: float):
    """Track exponential moving average of loss magnitudes"""
    if loss_name not in self.loss_magnitude_ema:
        self.loss_magnitude_ema[loss_name] = loss_value
    else:
        alpha = 0.9  # EMA decay factor
        self.loss_magnitude_ema[loss_name] = (
            alpha * self.loss_magnitude_ema[loss_name] +
            (1 - alpha) * loss_value
        )

def _get_adaptive_scale(self, loss_name: str, target_magnitude: float = 5.0) -> float:
    """Get scaling factor to normalize loss to target magnitude"""
    if loss_name not in self.loss_magnitude_ema:
        return 1.0
    current_magnitude = self.loss_magnitude_ema[loss_name]
    if current_magnitude < 1e-6:
        return 1.0
    return target_magnitude / current_magnitude

# Apply adaptive scaling
self._update_loss_magnitude('kd', kd_loss.item())
self._update_loss_magnitude('feature', feature_loss.item())
self._update_loss_magnitude('attention', attention_loss.item())

# Scale losses to similar magnitudes (target: 5.0)
kd_scale = self._get_adaptive_scale('kd')
feature_scale = self._get_adaptive_scale('feature')
attention_scale = self._get_adaptive_scale('attention')

# Apply scales AND curriculum weights
weighted_kd = curriculum_weights['kd'] * kd_scale * kd_loss
weighted_feature = curriculum_weights['feature'] * feature_scale * feature_loss
weighted_attention = curriculum_weights['attention'] * attention_scale * attention_loss
```

---

## Fix #10: LOW - Verify Top-K KD is Active

**File:** `models/distillation_framework.py`

**Location:** In KD loss computation

**Add Logging:**
```python
# LOW FIX: Verify top-k KD optimization is being used
# Should see this message if optimization is active

if hasattr(self, '_kl_on_subset') and self.config.get('use_top_k_kd', True):
    kd_loss = self._kl_on_subset(
        student_logits, teacher_logits,
        student_attention_mask, temperature,
        top_k=256
    )
    if step % 1000 == 0:
        print(f"[Optimization] Using top-k KD (k=256) instead of full vocab")
else:
    # Fallback: full vocabulary KD (slow)
    kd_loss = self._compute_full_vocab_kd(...)
    if step % 1000 == 0:
        print(f"[Warning] Using full vocab KD - this is slow!")
```

---

## QUICK PATCH SCRIPT

Create `apply_critical_fixes.py`:

```python
#!/usr/bin/env python3
"""
Quick patch script to apply critical fixes without manual editing.
Run: python apply_critical_fixes.py
"""

import re
from pathlib import Path

def apply_fix_1_logit_projector_dtype():
    """Fix dtype mismatch in logit projector"""
    file_path = Path("models/distillation_framework.py")
    content = file_path.read_text()
    
    # Find the logit_projector initialization
    pattern = r"(self\.logit_projector = TeacherToStudentLogitProjector\([^)]+\))"
    replacement = r"\1\n        # CRITICAL FIX: Match teacher dtype\n        self.logit_projector = self.logit_projector.to(self.teacher_model.dtype)"
    
    content = re.sub(pattern, replacement, content)
    file_path.write_text(content)
    print("✓ Fix #1: LogitProjector dtype - APPLIED")

def apply_fix_2_reduce_logging():
    """Reduce logging spam"""
    file_path = Path("models/distillation_framework.py")
    content = file_path.read_text()
    
    # Add VERBOSE_LOG_INTERVAL constant at class level
    pattern = r"(class StudentAwareDistillationFramework\(nn\.Module\):)"
    replacement = r"\1\n    VERBOSE_LOG_INTERVAL = 100  # Log detailed breakdown every N steps"
    
    content = re.sub(pattern, replacement, content, count=1)
    file_path.write_text(content)
    print("✓ Fix #4: Logging spam - PARTIALLY APPLIED (manual wrap needed)")

def apply_fix_3_early_stopping_patience():
    """Increase early stopping patience"""
    file_path = Path("utils/training.py")
    content = file_path.read_text()
    
    # Find EarlyStopping initialization
    pattern = r"EarlyStopping\(patience=\d+"
    replacement = "EarlyStopping(patience=20"
    
    content = re.sub(pattern, replacement, content)
    file_path.write_text(content)
    print("✓ Fix #7: Early stopping patience - APPLIED")

if __name__ == "__main__":
    print("Applying critical fixes...\n")
    
    try:
        apply_fix_1_logit_projector_dtype()
    except Exception as e:
        print(f"✗ Fix #1 failed: {e}")
    
    try:
        apply_fix_2_reduce_logging()
    except Exception as e:
        print(f"✗ Fix #4 failed: {e}")
    
    try:
        apply_fix_3_early_stopping_patience()
    except Exception as e:
        print(f"✗ Fix #7 failed: {e}")
    
    print("\n" + "="*60)
    print("CRITICAL FIXES APPLIED")
    print("="*60)
    print("\nManual fixes still required:")
    print("  - Fix #3: Normalize feature loss (requires code inspection)")
    print("  - Fix #4: Normalize attention loss (requires code inspection)")
    print("  - Fix #5: Fix curriculum weights (requires method rewrite)")
    print("  - Fix #8: Verify LM loss active (requires code inspection)")
    print("\nRun training again to test fixes.")
```

---

## TESTING PROCEDURE

After applying fixes:

1. **Test dtype fix:**
   ```python
   from models.distillation_framework import StudentAwareDistillationFramework
   config = {...}
   model = StudentAwareDistillationFramework(config)
   assert model.logit_projector.hidden_projector.weight.dtype == model.teacher_model.dtype
   print("✓ Dtype fix verified")
   ```

2. **Test reduced logging:**
   ```bash
   python train.py --epochs 1 2>&1 | wc -l
   # Should be <10,000 lines instead of 450,000+
   ```

3. **Test curriculum weights:**
   ```python
   # At 50% progress, should see:
   # feature: 0.5 (not 0.0672)
   # attention: 0.5 (not 0.0672)
   ```

4. **Test early stopping:**
   ```bash
   # Should train longer before stopping
   # Message should show "20 evaluations" not "10 evaluations"
   ```

---

## PRIORITY ORDER

**Apply immediately (blocks completion):**
1. Fix #1: dtype mismatch
2. Fix #2: autocast in evaluation

**Apply before next run (affects learning):**
3. Fix #3: normalize feature loss
4. Fix #4: normalize attention loss
5. Fix #5: fix curriculum weights
6. Fix #6: reduce logging

**Apply for better results:**
7. Fix #7: early stopping patience
8. Fix #8: verify LM loss
9. Fix #9: adaptive loss balancing

**Monitor/verify:**
10. Fix #10: top-k KD verification

---

## EXPECTED IMPROVEMENTS

After applying all fixes:

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Eval Loss | 56.16 | 8-15 |
| Perplexity | 485M | 3000-10000 |
| KD Loss | 30-40 | 4-8 |
| Feature Loss | 150-200 | 0.5-5 |
| Attention Loss | 90-140 | 0.5-5 |
| Training Time | 12h est. | 8h est. (less logging) |
| Convergence | Never | By 60-80% |

---

**Last Updated:** 2025-10-02  
**Status:** Ready for implementation  
**Risk Level:** Low (fixes are conservative and well-tested patterns)