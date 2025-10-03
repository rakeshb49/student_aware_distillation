# Resume Training Guide

**Date:** 2025-01-02  
**Status:** ✅ READY TO RESUME  
**Last Step:** 1999 (crashed during evaluation)  
**Checkpoint:** `./checkpoints/emergency_checkpoint`

---

## Quick Start

```bash
# Resume training from emergency checkpoint
cd /kaggle/working/student_aware_distillation
python train.py --resume ./checkpoints/emergency_checkpoint
```

Training will resume from step 1999 and complete the evaluation that crashed.

---

## What Was Fixed

### Critical Bug: TypeError at Evaluation
- **Problem:** `step` parameter was `None` during evaluation, causing `step % 500` to crash
- **Fix:** Added `None` checks before all modulo operations
- **Impact:** Training can now complete evaluations without crashing

### Enhancement: Curriculum During Evaluation
- **Change:** Evaluation now passes `step=self.current_step` to model
- **Benefit:** Curriculum weights are now applied during evaluation (more accurate metrics)

**See `BUGFIX_ANALYSIS.md` for full details.**

---

## Expected Behavior

### Immediate (Steps 2000-2500)
1. ✅ Evaluation completes successfully at step 2000
2. ✅ Training continues normally
3. ✅ Loss continues decreasing
4. ✅ Debug logs print every 100 steps

### Short Term (Steps 2500-24,941)
- Training progresses through Phase 1 (KD loss only)
- Loss should decrease from ~15.7 to ~10-12 range
- Epoch 1 completes at step 24,941 (33% progress)
- Evaluation every 2000 steps (configured as `eval_steps`)

### Phase 2 Transition (Step 22,447)
At 30% progress, curriculum transitions to Phase 2:
- ✅ Attention loss activates (gradual ramp-up)
- ✅ Feature loss activates (gradual ramp-up)
- You'll see curriculum weights change in debug logs

### Phase 3 Transition (Step 44,894)
At 60% progress, curriculum transitions to Phase 3:
- ✅ Layerwise loss activates
- ✅ Contrastive loss activates
- Full multi-objective training begins

---

## Monitoring Checklist

### ✅ Healthy Training Signs
- [ ] Loss decreases over time (expect ~15.7 → 10-12 by end of epoch 1)
- [ ] Eval loss within 2-3x of training loss
- [ ] No OOM errors (P100 has 16GB, should be fine)
- [ ] Curriculum phases activate at correct steps (30%, 60%)
- [ ] NaN warnings < 1% of batches (saw 1 NaN at step 1184, acceptable)

### 🚨 Warning Signs
- [ ] Loss plateaus for > 1000 steps → check learning rate schedule
- [ ] Eval loss > 5x training loss → possible overfitting or eval bug
- [ ] Frequent NaN warnings (> 1%) → reduce learning rate
- [ ] OOM errors → reduce batch size from 2 to 1

---

## Training Progress

| Metric | Value | Status |
|--------|-------|--------|
| **Current Step** | 1999 | 🔄 Ready to resume |
| **Total Steps** | 74,823 | 3 epochs |
| **Progress** | 2.7% | Early training |
| **Current Loss** | 15.70 | ✅ Decreasing |
| **Epoch** | 0/3 | First epoch |
| **Phase** | 1 (KD only) | Correct |

---

## Debug Output to Watch For

### Every 100 Steps
```
[TRAIN DEBUG] Step X, Batch X
  Progress bar loss: X.XXXX
  Raw loss components:
    kd_loss: X.XXXX
    routing_feature_loss: 0.0000  ← Should be 0.0 in Phase 1
    lm_loss: X.XXXX
  Sum of components: X.XXXX
  Total loss: X.XXXX
```

### Every 500 Steps
```
[CURRICULUM] Step X, Progress X.X%
  Curriculum weights:
    kd: 0.7000         ← Active in Phase 1
    feature: 0.0000    ← Disabled in Phase 1
    attention: 0.0000  ← Disabled in Phase 1
```

### At Evaluation (Every 2000 Steps)
```
[Eval] Starting evaluation (161 batches)...
[Eval] 161/161 (100.0%) - current: X.XXXX, avg: X.XXXX

==================================================
Evaluation Results (Step X)
==================================================
Eval Loss: X.XXXX
Perplexity: X.XXX
Component Losses:
  kd: X.XXXX
  lm: X.XXXX
==================================================
```

---

## Key Files

- **Training script:** `train.py`
- **Checkpoint:** `./checkpoints/emergency_checkpoint`
- **Logs:** Console output (redirectable to file)
- **Config:** Embedded in checkpoint
- **Bug analysis:** `BUGFIX_ANALYSIS.md`

---

## Troubleshooting

### If Training Crashes Again
```bash
# Check the error message carefully
# Look for the traceback location
# Create a GitHub issue with:
# 1. Full error message
# 2. Last 50 lines of log output
# 3. Step number where it crashed
```

### If Loss Stops Decreasing
```bash
# Check learning rate is not zero
# Look for "lr=X.XXe-XX" in progress bar
# If lr=0.0000e+00, scheduler bug detected
```

### If OOM Error
```python
# Edit train.py, line ~280
# Change batch_size from 2 to 1
config = {
    'batch_size': 1,  # ← Reduce from 2
    'gradient_accumulation_steps': 32,  # ← Double from 16
    # ... rest of config
}
```

### If NaN Warnings Are Frequent
```python
# Edit train.py, add gradient clipping
# In config around line ~290
config = {
    # ... existing config
    'max_grad_norm': 1.0,  # ← Add this line
    # ... rest of config
}
```

---

## Performance Expectations

### P100 GPU (16GB)
- **Speed:** ~2.4 it/s (iterations per second)
- **Time per epoch:** ~2.9 hours (24,941 steps ÷ 2.4 it/s)
- **Time for 3 epochs:** ~8.7 hours total
- **Memory usage:** ~14-15 GB (safe margin)

### Training Time Estimate
- **Completed:** 1999 steps = ~14 minutes
- **Remaining in Epoch 1:** 22,942 steps = ~2.7 hours
- **Epochs 2-3:** ~5.8 hours
- **Total remaining:** ~8.5 hours

---

## Success Criteria

Training is successful if:
1. ✅ All 3 epochs complete without crashes
2. ✅ Final loss < 10.0 (starting from ~17.3)
3. ✅ Final perplexity < 50 (down from initial ~485M)
4. ✅ Eval loss tracks training loss (within 2-3x)
5. ✅ All curriculum phases activate correctly
6. ✅ Model checkpoint saves successfully

---

## Next Steps After Training

1. **Validate checkpoint:**
   ```bash
   python validate_checkpoint.py --checkpoint ./checkpoints/final_checkpoint
   ```

2. **Test generation:**
   ```bash
   python test_generation.py --model ./checkpoints/final_checkpoint
   ```

3. **Compare to baseline:**
   - Baseline SmolLM-135M perplexity: ~20-30 (typical)
   - Target distilled model perplexity: <15 (better than baseline)

---

**Ready to resume training!** 🚀

Run the command and monitor the output. Training should complete in ~8.5 hours.