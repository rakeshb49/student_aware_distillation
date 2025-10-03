# QUICK ACTION PLAN - Fix Loss Component Imbalance

**Time Required:** 30 minutes  
**Status:** Model IS learning, but loss components need debugging  
**Priority:** HIGH

---

## 🎯 WHAT'S HAPPENING

### ✅ GOOD NEWS
- **Scheduler fix WORKED!** 
- Model IS learning (loss: 21.5 → 15.5)
- LR showing as 0.00e+00 is just formatting (actual LR is 7.96e-6 during warmup)

### ❌ PROBLEM
- Eval loss still very high (95.86)
- Feature loss should be 0 (curriculum Phase 1) but showing 41.22
- Train/eval gap is 6x (should be ~1.3x)

---

## 📋 ACTION STEPS

### STEP 1: Add Debug Logging (Copy-Paste These)

#### File: `utils/training.py` (add after line 540)

```python
# After: epoch_losses['total'].append(loss.item() * self.gradient_accumulator.accumulation_steps)

# ADD THIS DEBUG BLOCK:
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
```

#### File: `models/distillation_framework.py` (add after line 810)

```python
# After: curriculum_weights = self._get_curriculum_weights(step)

# ADD THIS DEBUG BLOCK:
if step % 500 == 0 or step < 10:
    progress_pct = (step / self.total_steps) * 100 if self.total_steps > 0 else 0
    print(f"\n{'='*60}")
    print(f"[CURRICULUM] Step {step}, Progress {progress_pct:.1f}%")
    print(f"{'='*60}")
    print(f"  Curriculum weights:")
    for k, v in curriculum_weights.items():
        print(f"    {k}: {v:.4f}")
    print(f"{'='*60}\n")
```

#### File: `models/distillation_framework.py` (add after line 878)

```python
# After: losses['kd_loss'] = self._ensure_finite_loss('kd_loss', weighted_kd)

# ADD THIS DEBUG BLOCK:
if step % 500 == 0 or step < 10:
    print(f"\n{'='*60}")
    print(f"[KD LOSS] Step {step}")
    print(f"{'='*60}")
    print(f"  Raw KD loss: {kd_loss.item():.4f}")
    print(f"  Curriculum weight: {curriculum_weights['kd']:.4f}")
    print(f"  Weighted KD loss: {weighted_kd.item():.4f}")
    print(f"{'='*60}\n")
```

#### File: `models/distillation_framework.py` (add after routing losses, around line 908)

```python
# After the routing losses loop ends

# ADD THIS DEBUG BLOCK:
if step % 500 == 0 or step < 10:
    print(f"\n{'='*60}")
    print(f"[ROUTING LOSSES] Step {step}")
    print(f"{'='*60}")
    if 'routing_outputs' in locals() and 'losses' in routing_outputs:
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
```

---

### STEP 2: Continue Training

Let training run for ~100 more steps to capture debug output.

**DO NOT stop the current run** - let it continue with existing checkpoint.

---

### STEP 3: Capture Critical Debug Output

Look for these patterns in the logs:

#### Pattern A: Curriculum is working
```
[CURRICULUM] Step 2000, Progress 2.7%
  Curriculum weights:
    kd: 0.7000
    feature: 0.0000  ← Should be ZERO in Phase 1
    attention: 0.0000 ← Should be ZERO in Phase 1
```

#### Pattern B: Raw routing losses
```
[ROUTING LOSSES] Step 2000
  feature_matching_loss:
    Raw: 412.2000  ← This might be huge!
    Weight: 0.0000
    Scaled: 0.0000
```

#### Pattern C: Component sum vs total
```
[TRAIN DEBUG] Step 2000
  Sum of components: 95.8600
  Total loss: 95.8600  ← Should match!
```

---

### STEP 4: Diagnose Based on Output

#### Scenario A: Curriculum weight is 0, but loss is added anyway
**If you see:**
```
Weight: 0.0000
Scaled: 0.0000
But feature_loss still appears in eval: 41.22
```

**Then:** Eval is not respecting curriculum weights.

**Fix:** Add curriculum weight check in evaluate() method.

---

#### Scenario B: Raw routing loss is enormous
**If you see:**
```
feature_matching_loss:
  Raw: 4122.0000  ← HUGE!
  Weight: 0.1000
  Scaled: 412.2000
```

**Then:** Feature matching loss needs normalization.

**Fix:** Add normalization in router's feature matching computation.

---

#### Scenario C: Loss components are summed incorrectly
**If you see:**
```
Sum of components: 15.5000
Total loss: 95.8600
```

**Then:** Some losses are being added that aren't in the 'losses' dict.

**Fix:** Check for losses added directly to total without being tracked.

---

## 🔧 LIKELY FIXES (Apply After Diagnosis)

### Fix #1: Disable Curriculum (Quick Solution)

**File:** `configs/emergency_fix_config.json`

```json
{
  "use_curriculum": false
}
```

Restart training with this config. All losses will use configured alphas from start.

---

### Fix #2: Fix Eval to Respect Curriculum (Proper Solution)

**File:** `utils/training.py` in evaluate() method

Add before computing losses:
```python
# Get curriculum weights for current step
curriculum_weights = self.model._get_curriculum_weights(self.global_step)

# Pass to model forward
outputs = self.model(
    ...,
    curriculum_weights=curriculum_weights  # Add this
)
```

---

### Fix #3: Skip Zero-Weight Losses (Best Solution)

**File:** `models/distillation_framework.py` around line 900

```python
for loss_name, loss_value in routing_outputs['losses'].items():
    if loss_name == 'attention_alignment_loss':
        weight = curriculum_weights['attention']
    else:
        weight = curriculum_weights['feature']
    
    # CRITICAL FIX: Don't add losses with zero weight
    if weight > 1e-6:  # Add this check
        scaled = loss_value * weight
        losses[f'routing_{loss_name}'] = scaled
```

---

## 📊 EXPECTED RESULTS AFTER FIX

### With Curriculum Disabled (Fix #1)
```
Eval loss: 30-40 (much better)
Train/eval gap: 1.5-2x (healthy)
All components active from start
```

### With Curriculum Fixed (Fix #2 or #3)
```
Phase 1 (0-30%): KD loss only, total ~21
Phase 2 (30-60%): KD + gradual feature, total ~25-30
Phase 3 (60-100%): All losses, total ~30-40
```

---

## ⏱️ TIMELINE

- **5 min:** Add debug logging
- **10 min:** Wait for debug output
- **5 min:** Analyze patterns
- **10 min:** Apply appropriate fix
- **Total:** 30 minutes to working training

---

## 🚨 IF TRAINING CRASHES

If adding debug logging causes crash:

1. Remove debug blocks
2. Apply Fix #1 (disable curriculum)
3. Restart training

The curriculum feature might be causing instability anyway.

---

## ✅ SUCCESS CRITERIA

After applying fix, you should see:

- ✅ Eval loss < 40 by end of epoch 1
- ✅ Train/eval gap < 2x
- ✅ All loss components within expected ranges
- ✅ No NaN warnings
- ✅ Perplexity < 100

---

## 📞 NEXT REVIEW

After implementing fix and running for 50% of epoch 1:

- Check if eval loss is decreasing steadily
- Verify train/eval gap is healthy (<2x)
- Confirm loss components follow expected ratios
- Assess if model quality is improving

---

**BOTTOM LINE:** Add the debug logging, capture output, apply the appropriate fix. The scheduler fix worked - now we just need to fix the loss component balancing.

**Most Likely Fix:** Disable curriculum (Fix #1) for simplicity, or properly skip zero-weight losses (Fix #3).

---

**DO THIS NOW:**
1. Copy-paste the 4 debug blocks into the files
2. Continue training for 100 steps
3. Share the debug output
4. Apply fix based on what you see