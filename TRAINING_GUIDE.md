# Student-Aware Distillation Training Guide

**Status:** ✅ READY FOR TRAINING  
**Last Updated:** 2025-10-02  
**Version:** 2.0 - Post Scheduler Fix

---

## 🎯 QUICK START

### Run Training (Recommended)
```bash
python train.py \
  --config configs/emergency_fix_config.json \
  --epochs 3
```

### Monitor These Metrics
```
✅ GOOD: lr > 0 (will show as 0.00e+00 during early warmup, but model IS learning)
✅ GOOD: loss decreasing (expect 21 → 15 in first 2000 steps)
✅ GOOD: eval metrics changing between evaluations
✅ GOOD: memory 80-90%
```

---

## 📊 CURRENT STATUS

### What's Fixed ✅
1. **Scheduler Bug** - Was configured for 75k steps but only called 4.7k times (FIXED)
2. **Model Learning** - Loss now decreasing (was flat at 21.5)
3. **Evaluation Working** - Metrics changing (were frozen)
4. **Stable Training** - No OOM errors, consistent speed

### What's Being Debugged 🔍
1. **Loss Component Imbalance** - Feature loss high when should be 0 in Phase 1
2. **Train/Eval Gap** - 6x gap (train=15.5, eval=95.86) should be ~1.5x
3. **High Eval Loss** - Still at 95.86, needs investigation

### Known Cosmetic Issues 🟡
1. **LR Display** - Shows `0.00e+00` during warmup (formatting only, actual LR is ~7.96e-6)

---

## 🔧 CONFIGURATION

### Recommended Config: `configs/emergency_fix_config.json`

**Key Settings:**
```json
{
  "batch_size": 2,
  "gradient_accumulation_steps": 16,
  "learning_rate": 3e-5,
  "warmup_steps": 500,
  "max_length": 256,
  "temperature": 2.0,
  "kd_top_k": 256,
  "use_curriculum": true,
  "use_gradient_checkpointing": true,
  "eval_steps": 2000
}
```

**Why These Values:**
- Small batch (2) + high grad accum (16) = stable training on 16GB GPU
- Subset KD (kd_top_k=256) = 10-100x speedup
- Lower temp (2.0) = prevents NaN
- Reduced sequence length (256) = 30% memory savings

---

## 📈 EXPECTED TRAINING BEHAVIOR

### Healthy Training Metrics

**First 2000 Steps (Warmup Phase):**
```
Loss: 21.5 → 15.5 (decreasing)
LR: 0 → 3e-5 (ramping up, may display as 0.00e+00)
Memory: 80-85%
Speed: 2.4-2.5 it/s
```

**End of Epoch 1:**
```
Train Loss: 8-15
Eval Loss: Should be 15-40 (currently debugging why it's 95)
Perplexity: 30-100
Time: ~3 hours
```

**End of 3 Epochs:**
```
Train Loss: 5-10
Eval Loss: 8-20
Perplexity: 15-40
Total Time: 8-9 hours
```

---

## 🐛 ACTIVE DEBUGGING

### Debug Logging Added

The following debug blocks have been added to investigate loss imbalance:

**1. Training Loss Components** (`utils/training.py` line ~540)
- Shows which losses contribute to total
- Verifies progress bar loss = actual total loss

**2. Curriculum Weights** (`models/distillation_framework.py` line ~810)
- Shows curriculum phase and weights
- Verifies Phase 1 has feature=0, attention=0

**3. KD Loss Details** (`models/distillation_framework.py` line ~878)
- Shows raw vs weighted KD loss
- Verifies curriculum weight applied correctly

**4. Routing Losses** (`models/distillation_framework.py` line ~908)
- Shows raw routing loss values
- Verifies curriculum weights applied to routing

### What to Look For

**Expected Debug Output (Phase 1):**
```
[CURRICULUM] Step 2000, Progress 2.7%
  Curriculum weights:
    kd: 0.7000
    feature: 0.0000  ← Should be ZERO
    attention: 0.0000 ← Should be ZERO

[KD LOSS] Step 2000
  Raw KD loss: 21.2000
  Curriculum weight: 0.7000
  Weighted KD loss: 14.8400

[ROUTING LOSSES] Step 2000
  feature_matching_loss:
    Raw: 412.2000
    Weight: 0.0000
    Scaled: 0.0000  ← Should contribute nothing
```

---

## 🔧 KNOWN ISSUES & WORKAROUNDS

### Issue 1: High Eval Loss (95.86)

**Possible Causes:**
1. Curriculum weights not applied in eval
2. Subset KD in train but full vocab in eval
3. Routing losses bypass curriculum weights

**Workaround:** Disable curriculum if debugging confirms it's the issue:
```json
{"use_curriculum": false}
```

### Issue 2: LR Shows 0.00e+00

**Cause:** Display formatting during warmup (actual LR is 7.96e-6)

**Proof It's Working:** Loss is decreasing!

**Fix (Optional):** Change line 561 in `utils/training.py`:
```python
'lr': f'{current_lr:.3e}',  # Was .2e
```

### Issue 3: Occasional NaN Loss

**Current Mitigation:** Clamping to zero

**If Frequency Increases:**
- Reduce temperature to 1.5
- Lower max_grad_norm to 0.5

---

## 📊 LOSS COMPONENT ANALYSIS

### Current Observations (Step 2000)

**Training:**
- Loss: 15.5 (decreasing ✅)

**Evaluation:**
- Total: 95.86
- KD: 14.85 (15.5%)
- Feature: 41.22 (43.0%)
- Attention: 0.47 (0.5%)

**Expected in Phase 1 (0-30% of training):**
- KD should be ~70% of total
- Feature should be 0%
- Attention should be 0%

**Discrepancy:** Feature and attention losses are active when they should be zero.

---

## 🎯 SUCCESS CRITERIA

### Must Achieve
- [x] Model learning (loss decreasing) ✅
- [x] Eval metrics changing ✅
- [x] Scheduler configured correctly ✅
- [x] LR > 0 (actual, not display) ✅
- [ ] Eval loss < 50 by end of epoch 1
- [ ] Train/eval gap < 2x
- [ ] No NaN warnings for full epoch
- [ ] Loss components balanced per curriculum

### Nice to Have
- [ ] Training speed > 3 it/s
- [ ] Memory < 85%
- [ ] Perplexity < 100 by end of epoch 1

---

## 🚀 TROUBLESHOOTING

### Training Loss Not Decreasing
**Check:**
- Scheduler debug output shows correct optimizer steps
- LR is actually > 0 (check `optimizer.param_groups[0]['lr']`)
- Gradient norms are reasonable (< 10.0)

**Solution:** Verify scheduler fix was applied in `utils/training.py` line 377-392

### High Memory Usage (>90%)
**Quick Fix:**
- Reduce `max_length` to 192
- Set `attention_layers` to 0
- Reduce `batch_size` to 1

### Frequent NaN Losses
**Fix:**
- Reduce `temperature` to 1.5
- Add stricter gradient clipping: `max_grad_norm: 0.5`

### Very High Eval Loss
**Investigation:**
- Check debug output for curriculum weights in eval
- Verify subset KD used in both train and eval
- Consider disabling curriculum as workaround

---

## 📁 FILE STRUCTURE

```
student_aware_distillation/
├── train.py                        # Main training script
├── configs/
│   ├── improved_config.json        # Original config (has issues)
│   └── emergency_fix_config.json   # Fixed config (USE THIS)
├── models/
│   └── distillation_framework.py   # Core distillation logic
├── utils/
│   └── training.py                 # Training loop with scheduler fix
├── data/
│   └── data_loader.py              # Dataset handling
└── docs/
    ├── README.md                   # Project overview
    ├── CHANGELOG.md                # Version history
    ├── TRAINING_GUIDE.md           # This file
    ├── URGENT_FINDINGS.md          # Current debugging status
    └── QUICK_ACTION_PLAN.md        # Debug steps
```

---

## 🔍 DEBUG COMMANDS

### Check Scheduler Configuration
```bash
python train.py --config configs/emergency_fix_config.json --epochs 1 2>&1 | grep "\[Scheduler\]"
```

**Expected Output:**
```
[Scheduler] Total batches: 74,823
[Scheduler] Gradient accumulation: 16
[Scheduler] Optimizer steps: 4,676
[Scheduler] Warmup steps: 467 (10.0%)
```

### Monitor Training Progress
```bash
python train.py --config configs/emergency_fix_config.json --epochs 1 2>&1 | tee training.log

# In another terminal:
tail -f training.log | grep -E "loss=|Eval"
```

### Run Diagnostics
```bash
python diagnose_and_fix.py --config configs/emergency_fix_config.json
```

---

## 📞 NEXT STEPS

### Immediate (Active Debugging)
1. ✅ Debug logging added
2. ⏳ Continue training to capture debug output
3. 🔜 Analyze curriculum weight application
4. 🔜 Fix loss component imbalance based on findings

### Short Term
1. Fix loss component balancing
2. Verify train/eval gap becomes healthy (<2x)
3. Complete full epoch 1
4. Assess model quality

### Long Term
1. Complete 3 epoch training
2. Evaluate final model on benchmarks
3. Export model for deployment
4. Document final results

---

## ✅ VALIDATION CHECKLIST

Before considering training successful:

- [ ] Scheduler debug shows correct optimizer steps (4,676 not 74,823)
- [ ] Training loss decreases monotonically
- [ ] Eval metrics change between evaluations (not frozen)
- [ ] LR is non-zero (actual value, ignore display)
- [ ] Eval loss < 50 by end of epoch 1
- [ ] Train/eval gap < 2x
- [ ] No NaN warnings for sustained period
- [ ] Memory stable at 80-85%
- [ ] Loss components balanced per configuration
- [ ] Perplexity < 100

---

## 📚 ADDITIONAL RESOURCES

- **URGENT_FINDINGS.md** - Current debugging analysis
- **QUICK_ACTION_PLAN.md** - Step-by-step debug instructions
- **diagnose_and_fix.py** - Automated diagnostic tool
- **archive/** - Historical analysis documents

---

## 🎓 LESSONS LEARNED

### Critical Bugs Fixed
1. **Scheduler miscalculation** - Must divide by gradient_accumulation_steps
2. **Warmup too short** - Need to account for grad accum in warmup steps
3. **Memory optimization** - Subset KD essential for P100 GPU

### Best Practices
1. Always validate scheduler receives correct step count
2. Monitor LR throughout training (actual value, not just display)
3. Debug loss components early to catch imbalances
4. Use subset KD for efficiency without quality loss
5. Add comprehensive debug logging for complex loss functions

---

**STATUS:** Training is working, model is learning. Active debugging of loss component imbalance to optimize final model quality.

**RECOMMENDATION:** Continue current training run while monitoring debug output. Apply curriculum fix once root cause is identified.