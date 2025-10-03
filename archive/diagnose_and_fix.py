#!/usr/bin/env python3
"""
Emergency Diagnostic and Fix Script for Student-Aware Distillation
Identifies and fixes critical training issues
"""

import json
import os
import sys
from pathlib import Path

def diagnose_config(config_path):
    """Diagnose configuration issues"""
    print("="*80)
    print("DIAGNOSTIC REPORT: Student-Aware Distillation Training Issues")
    print("="*80)

    with open(config_path, 'r') as f:
        config = json.load(f)

    issues = []
    warnings = []

    # Issue 1: Warmup steps vs gradient accumulation
    warmup_steps = config.get('warmup_steps', 1000)
    grad_accum = config.get('gradient_accumulation_steps', 8)
    batch_size = config.get('batch_size', 4)
    num_epochs = config.get('num_epochs', 3)
    dataset_size = config.get('dataset_subset_size', 50000)

    # Calculate effective steps
    batches_per_epoch = dataset_size // batch_size
    total_batches = batches_per_epoch * num_epochs
    optimizer_steps = total_batches // grad_accum

    print(f"\n📊 TRAINING CONFIGURATION ANALYSIS")
    print(f"   Dataset size: {dataset_size:,}")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {grad_accum}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batches per epoch: {batches_per_epoch:,}")
    print(f"   Total batches: {total_batches:,}")
    print(f"   Optimizer steps: {optimizer_steps:,}")

    # Check scheduler configuration
    print(f"\n🔧 SCHEDULER ANALYSIS")
    print(f"   Warmup steps (config): {warmup_steps}")

    # CRITICAL: The scheduler receives optimizer steps, not batch steps
    # But num_training_steps is calculated as total_batches!
    scheduler_total_steps = total_batches  # This is what's passed to scheduler
    print(f"   Scheduler total_steps: {scheduler_total_steps:,}")
    print(f"   Scheduler step() calls: {optimizer_steps:,} (actual)")

    if scheduler_total_steps != optimizer_steps:
        issues.append({
            'severity': 'CRITICAL',
            'issue': 'Scheduler step count mismatch',
            'details': f'Scheduler configured for {scheduler_total_steps:,} steps but only called {optimizer_steps:,} times',
            'impact': 'Learning rate schedule completes at wrong time',
            'fix': f'Divide num_training_steps by gradient_accumulation_steps'
        })

    # Check warmup percentage
    warmup_pct = (warmup_steps / optimizer_steps) * 100
    print(f"   Warmup percentage: {warmup_pct:.1f}%")

    if warmup_pct < 5:
        warnings.append({
            'severity': 'WARNING',
            'issue': 'Warmup too short',
            'details': f'Warmup is only {warmup_pct:.1f}% of training',
            'recommendation': f'Increase warmup_steps to {int(optimizer_steps * 0.1)} for 10% warmup'
        })
    elif warmup_pct > 30:
        warnings.append({
            'severity': 'WARNING',
            'issue': 'Warmup too long',
            'details': f'Warmup is {warmup_pct:.1f}% of training',
            'recommendation': f'Decrease warmup_steps to {int(optimizer_steps * 0.1)} for 10% warmup'
        })

    # Check learning rate
    lr = config.get('learning_rate', 5e-5)
    print(f"\n📈 LEARNING RATE ANALYSIS")
    print(f"   Initial LR: {lr:.2e}")

    if lr == 0:
        issues.append({
            'severity': 'CRITICAL',
            'issue': 'Zero learning rate',
            'details': 'Learning rate is set to 0',
            'impact': 'Model cannot learn',
            'fix': 'Set learning_rate to 3e-5 or 5e-5'
        })

    # Check temperature
    temperature = config.get('temperature', 4.0)
    print(f"\n🌡️  DISTILLATION TEMPERATURE")
    print(f"   Temperature: {temperature}")

    if temperature > 3.0:
        warnings.append({
            'severity': 'WARNING',
            'issue': 'High temperature',
            'details': f'Temperature of {temperature} may cause numerical instability',
            'recommendation': 'Reduce to 2.0-2.5 for better stability'
        })

    # Check memory settings
    print(f"\n💾 MEMORY CONFIGURATION")
    max_length = config.get('max_length', 512)
    kd_top_k = config.get('kd_top_k', None)
    use_gradient_checkpointing = config.get('use_gradient_checkpointing', False)

    print(f"   Max sequence length: {max_length}")
    print(f"   KD top-k (subset): {kd_top_k if kd_top_k else 'DISABLED (using full vocab)'}")
    print(f"   Gradient checkpointing: {use_gradient_checkpointing}")

    if kd_top_k is None:
        warnings.append({
            'severity': 'HIGH',
            'issue': 'Subset KD not enabled',
            'details': 'Computing KD loss over full vocabulary (~50k tokens)',
            'impact': 'High memory usage, slow training',
            'recommendation': 'Set kd_top_k to 256 or 512 for 10-100x speedup'
        })

    if max_length > 384:
        warnings.append({
            'severity': 'MEDIUM',
            'issue': 'Long sequence length',
            'details': f'Sequence length of {max_length} uses significant memory',
            'recommendation': 'Reduce to 256-384 for better memory efficiency'
        })

    # Check loss weights
    print(f"\n⚖️  LOSS COMPONENT WEIGHTS")
    alpha_kd = config.get('alpha_kd', 0.7)
    alpha_feature = config.get('alpha_feature', 0.1)
    alpha_attention = config.get('alpha_attention', 0.1)

    print(f"   KD loss weight: {alpha_kd}")
    print(f"   Feature loss weight: {alpha_feature}")
    print(f"   Attention loss weight: {alpha_attention}")

    total_alpha = alpha_kd + alpha_feature + alpha_attention
    total_alpha += config.get('alpha_layerwise', 0.0)
    total_alpha += config.get('alpha_contrastive', 0.0)

    print(f"   Total weight: {total_alpha}")

    if total_alpha > 1.1 or total_alpha < 0.9:
        warnings.append({
            'severity': 'MEDIUM',
            'issue': 'Loss weights sum != 1.0',
            'details': f'Total loss weight is {total_alpha}',
            'recommendation': 'Normalize weights to sum to 1.0'
        })

    # Print issues
    print(f"\n{'='*80}")
    print(f"🚨 CRITICAL ISSUES FOUND: {len(issues)}")
    print(f"{'='*80}")

    for idx, issue in enumerate(issues, 1):
        print(f"\n[{idx}] {issue['severity']}: {issue['issue']}")
        print(f"    Details: {issue['details']}")
        print(f"    Impact: {issue['impact']}")
        print(f"    Fix: {issue['fix']}")

    print(f"\n{'='*80}")
    print(f"⚠️  WARNINGS: {len(warnings)}")
    print(f"{'='*80}")

    for idx, warning in enumerate(warnings, 1):
        print(f"\n[{idx}] {warning['severity']}: {warning['issue']}")
        print(f"    Details: {warning['details']}")
        print(f"    Recommendation: {warning['recommendation']}")

    return issues, warnings, {
        'optimizer_steps': optimizer_steps,
        'warmup_steps': warmup_steps,
        'batch_size': batch_size,
        'grad_accum': grad_accum
    }


def generate_fixed_config(config_path, output_path, stats):
    """Generate a fixed configuration file"""
    print(f"\n{'='*80}")
    print(f"🔧 GENERATING FIXED CONFIGURATION")
    print(f"{'='*80}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Remove comments
    config = {k: v for k, v in config.items() if not k.startswith('_comment')}

    changes = []

    # Fix 1: Correct warmup steps
    recommended_warmup = int(stats['optimizer_steps'] * 0.1)
    if config.get('warmup_steps', 0) != recommended_warmup:
        config['warmup_steps'] = recommended_warmup
        changes.append(f"warmup_steps: {config.get('warmup_steps')} → {recommended_warmup}")

    # Fix 2: Enable subset KD if not already
    if 'kd_top_k' not in config or config['kd_top_k'] is None:
        config['kd_top_k'] = 256
        changes.append(f"kd_top_k: None → 256 (ENABLED)")

    # Fix 3: Reduce temperature if too high
    if config.get('temperature', 4.0) > 2.5:
        old_temp = config.get('temperature', 4.0)
        config['temperature'] = 2.0
        changes.append(f"temperature: {old_temp} → 2.0")

    # Fix 4: Reduce max_length if too long
    if config.get('max_length', 512) > 384:
        old_len = config.get('max_length', 512)
        config['max_length'] = 256
        changes.append(f"max_length: {old_len} → 256")

    # Fix 5: Enable gradient checkpointing
    if not config.get('use_gradient_checkpointing', False):
        config['use_gradient_checkpointing'] = True
        changes.append(f"use_gradient_checkpointing: False → True")

    # Fix 6: Reduce evaluation frequency
    if config.get('eval_steps', 500) < 2000:
        old_eval = config.get('eval_steps', 500)
        config['eval_steps'] = 2000
        changes.append(f"eval_steps: {old_eval} → 2000")

    # Fix 7: Increase early stopping patience
    if config.get('early_stopping_patience', 3) < 10:
        old_patience = config.get('early_stopping_patience', 3)
        config['early_stopping_patience'] = 10
        changes.append(f"early_stopping_patience: {old_patience} → 10")

    # Fix 8: Ensure proper learning rate
    if config.get('learning_rate', 0) == 0:
        config['learning_rate'] = 3e-5
        changes.append(f"learning_rate: 0 → 3e-5")

    # Fix 9: Reduce attention layers for memory
    if config.get('attention_layers', 4) > 1:
        old_layers = config.get('attention_layers', 4)
        config['attention_layers'] = 1
        changes.append(f"attention_layers: {old_layers} → 1")

    # Add metadata
    config['_fixed_by'] = 'diagnose_and_fix.py'
    config['_original_config'] = config_path

    # Save fixed config
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Fixed configuration saved to: {output_path}")
    print(f"\n📝 Changes made:")
    for idx, change in enumerate(changes, 1):
        print(f"   {idx}. {change}")

    return config


def generate_scheduler_fix():
    """Generate code fix for scheduler issue"""
    print(f"\n{'='*80}")
    print(f"🔧 SCHEDULER FIX REQUIRED")
    print(f"{'='*80}")

    print("""
The scheduler bug is in utils/training.py, line ~380:

CURRENT CODE (BUGGY):
```python
def _create_scheduler(self):
    scheduler_type = self.config.get('scheduler_type', 'cosine')
    num_training_steps = max(1, len(self.train_dataloader) * self.config.get('num_epochs', 3))
    num_warmup_steps = min(self.config.get('warmup_steps', 1000), num_training_steps // 10)
```

ISSUE:
- num_training_steps counts total BATCHES
- But scheduler.step() is called every gradient_accumulation_steps batches
- This causes scheduler to think it has more steps than it actually receives

FIXED CODE:
```python
def _create_scheduler(self):
    scheduler_type = self.config.get('scheduler_type', 'cosine')

    # CRITICAL FIX: Calculate actual optimizer steps, not batch steps
    total_batches = len(self.train_dataloader) * self.config.get('num_epochs', 3)
    grad_accum_steps = self.config.get('gradient_accumulation_steps', 1)
    num_training_steps = max(1, total_batches // grad_accum_steps)

    num_warmup_steps = min(self.config.get('warmup_steps', 1000), num_training_steps // 10)

    print(f"[Scheduler] Total batches: {total_batches:,}")
    print(f"[Scheduler] Gradient accumulation: {grad_accum_steps}")
    print(f"[Scheduler] Optimizer steps: {num_training_steps:,}")
    print(f"[Scheduler] Warmup steps: {num_warmup_steps:,} ({num_warmup_steps/num_training_steps*100:.1f}%)")
```

This fix ensures the scheduler receives the correct number of total steps.
""")


def main():
    """Main diagnostic function"""
    import argparse

    parser = argparse.ArgumentParser(description='Diagnose and fix training issues')
    parser.add_argument('--config', type=str, default='configs/improved_config.json',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='configs/emergency_fix_config.json',
                       help='Path to save fixed config')
    parser.add_argument('--fix', action='store_true',
                       help='Generate fixed configuration')

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Run diagnostics
    issues, warnings, stats = diagnose_config(args.config)

    # Generate fixes if requested
    if args.fix:
        fixed_config = generate_fixed_config(args.config, args.output, stats)
        generate_scheduler_fix()

        print(f"\n{'='*80}")
        print(f"✅ NEXT STEPS")
        print(f"{'='*80}")
        print(f"""
1. Apply the scheduler fix to utils/training.py (see above)

2. Run training with fixed config:
   python train.py --config {args.output} --epochs 1

3. Monitor these metrics:
   - Learning rate should be > 0 and increasing during warmup
   - Training loss should decrease monotonically
   - Eval metrics should change between evaluations
   - No NaN warnings should appear
   - Memory usage should be < 90%

4. Expected first epoch results:
   - Initial loss: ~20-30
   - End of epoch loss: ~8-15
   - Perplexity: < 100
   - Time per epoch: ~1.5-2 hours (with subset KD)

5. If issues persist, run:
   python diagnose_and_fix.py --config {args.output}
        """)
    else:
        print(f"\n{'='*80}")
        print(f"Run with --fix to generate corrected configuration")
        print(f"{'='*80}")

    # Return exit code based on issues
    sys.exit(len(issues))


if __name__ == '__main__':
    main()
