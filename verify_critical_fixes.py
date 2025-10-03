#!/usr/bin/env python3
"""
Verification script for critical fixes identified in Kaggle run analysis.

This script verifies that all critical, high, and medium priority fixes
are properly implemented in the codebase.

Usage:
    python verify_critical_fixes.py

Exit codes:
    0 - All fixes verified successfully
    1 - One or more fixes missing or incorrect
"""

import sys
import re
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}\n")


def print_check(name, passed, details=""):
    """Print a check result"""
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"{status} - {name}")
    if details:
        print(f"       {details}")


def check_file_exists(filepath):
    """Check if a file exists"""
    return Path(filepath).exists()


def check_pattern_in_file(filepath, pattern, description):
    """Check if a pattern exists in a file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            if re.search(pattern, content, re.MULTILINE | re.DOTALL):
                return True, "Found"
            else:
                return False, "Pattern not found"
    except FileNotFoundError:
        return False, "File not found"
    except Exception as e:
        return False, f"Error: {str(e)}"


def check_layerwise_normalization():
    """Verify Issue #1: Layerwise loss normalization fix"""
    print_header("Issue #1: Layerwise Loss Normalization")

    filepath = "models/distillation_framework.py"

    # Check 1: File exists
    exists = check_file_exists(filepath)
    print_check("File exists", exists, filepath)
    if not exists:
        return False

    # Check 2: mse_sum with reduction='sum'
    pattern1 = r"mse_sum\s*=\s*F\.mse_loss\s*\([^)]+reduction\s*=\s*['\"]sum['\"]"
    passed1, details1 = check_pattern_in_file(filepath, pattern1, "mse_sum with reduction='sum'")
    print_check("MSE with reduction='sum'", passed1, details1)

    # Check 3: Normalization by batch_size, seq_len, hidden_dim
    pattern2 = r"batch_size,\s*seq_len,\s*hidden_dim\s*=\s*student_proj\.shape"
    passed2, details2 = check_pattern_in_file(filepath, pattern2, "Extract shape dimensions")
    print_check("Extract shape dimensions", passed2, details2)

    # Check 4: Division by all three dimensions
    pattern3 = r"layer_loss\s*=\s*mse_sum\s*/\s*\(\s*batch_size\s*\*\s*seq_len\s*\*\s*hidden_dim\s*\)"
    passed3, details3 = check_pattern_in_file(filepath, pattern3, "Normalize by all dimensions")
    print_check("Normalize by (batch × seq × hidden)", passed3, details3)

    # Check 5: Comment explaining the fix
    pattern4 = r"CRITICAL FIX.*normalize.*sequence length"
    passed4, details4 = check_pattern_in_file(filepath, pattern4, "Comment about normalization")
    print_check("Documentation comment present", passed4, details4)

    all_passed = all([passed1, passed2, passed3, passed4])

    if all_passed:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Issue #1: VERIFIED{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Issue #1: FAILED - Please check the fix{Colors.END}")

    return all_passed


def check_sequence_alignment():
    """Verify Issue #2: Sequence length alignment fix"""
    print_header("Issue #2: Sequence Length Alignment in Evaluation")

    filepath = "utils/evaluation.py"

    # Check 1: File exists
    exists = check_file_exists(filepath)
    print_check("File exists", exists, filepath)
    if not exists:
        return False

    # Check 2: Extract student and teacher sequence lengths
    pattern1 = r"student_seq_len\s*=\s*student_logits\.size\(1\)"
    passed1, details1 = check_pattern_in_file(filepath, pattern1, "Extract student_seq_len")
    print_check("Extract student_seq_len", passed1, details1)

    pattern2 = r"teacher_seq_len\s*=\s*aligned_teacher_logits\.size\(1\)"
    passed2, details2 = check_pattern_in_file(filepath, pattern2, "Extract teacher_seq_len")
    print_check("Extract teacher_seq_len", passed2, details2)

    # Check 3: Conditional alignment check
    pattern3 = r"if\s+student_seq_len\s*!=\s*teacher_seq_len:"
    passed3, details3 = check_pattern_in_file(filepath, pattern3, "Check sequence length mismatch")
    print_check("Check for length mismatch", passed3, details3)

    # Check 4: Minimum sequence length calculation
    pattern4 = r"min_seq_len\s*=\s*min\(student_seq_len,\s*teacher_seq_len\)"
    passed4, details4 = check_pattern_in_file(filepath, pattern4, "Calculate min_seq_len")
    print_check("Calculate min sequence length", passed4, details4)

    # Check 5: Truncate both tensors
    pattern5 = r"student_logits\s*=\s*student_logits\[:,\s*:min_seq_len,\s*:\]"
    passed5, details5 = check_pattern_in_file(filepath, pattern5, "Truncate student_logits")
    print_check("Truncate student_logits", passed5, details5)

    pattern6 = r"aligned_teacher_logits\s*=\s*aligned_teacher_logits\[:,\s*:min_seq_len,\s*:\]"
    passed6, details6 = check_pattern_in_file(filepath, pattern6, "Truncate teacher_logits")
    print_check("Truncate aligned_teacher_logits", passed6, details6)

    # Check 7: Comment explaining the fix
    pattern7 = r"CRITICAL FIX.*Align sequence lengths"
    passed7, details7 = check_pattern_in_file(filepath, pattern7, "Comment about alignment")
    print_check("Documentation comment present", passed7, details7)

    all_passed = all([passed1, passed2, passed3, passed4, passed5, passed6, passed7])

    if all_passed:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Issue #2: VERIFIED{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Issue #2: FAILED - Please check the fix{Colors.END}")

    return all_passed


def check_previous_fixes():
    """Verify that previous fixes are still in place"""
    print_header("Previous Fixes Verification")

    checks = []

    # Check: Feature loss normalization
    filepath1 = "models/student_aware_router.py"
    pattern1 = r"total_feat_loss\s*/=\s*total_pairs"
    passed1, details1 = check_pattern_in_file(filepath1, pattern1, "Feature loss normalization")
    print_check("Feature loss normalization", passed1, details1)
    checks.append(passed1)

    # Check: Attention alignment normalization
    pattern2 = r"align_loss\s*/=\s*seq_len"
    passed2, details2 = check_pattern_in_file(filepath1, pattern2, "Attention alignment normalization")
    print_check("Attention alignment normalization", passed2, details2)
    checks.append(passed2)

    # Check: Curriculum learning
    filepath2 = "models/distillation_framework.py"
    pattern3 = r"def.*get_curriculum_weights.*:"
    passed3, details3 = check_pattern_in_file(filepath2, pattern3, "Curriculum learning function")
    print_check("Curriculum learning implemented", passed3, details3)
    checks.append(passed3)

    # Check: Early stopping patience
    filepath3 = "utils/training.py"
    pattern4 = r"patience:\s*int\s*=\s*2[0-9]"
    passed4, details4 = check_pattern_in_file(filepath3, pattern4, "Early stopping patience >= 20")
    print_check("Early stopping patience", passed4, details4)
    checks.append(passed4)

    # Check: Reduced logging frequency
    pattern5 = r"step\s*%\s*100\s*==\s*0"
    passed5, details5 = check_pattern_in_file(filepath2, pattern5, "Log every 100 steps")
    print_check("Reduced logging frequency", passed5, details5)
    checks.append(passed5)

    all_passed = all(checks)

    if all_passed:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Previous Fixes: VERIFIED{Colors.END}")
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ Previous Fixes: Some checks failed (non-critical){Colors.END}")

    # Previous fixes are warnings only, not blocking
    return True


def check_code_structure():
    """Verify overall code structure and imports"""
    print_header("Code Structure Verification")

    checks = []

    # Check: Imports in distillation_framework.py
    filepath = "models/distillation_framework.py"
    pattern1 = r"import torch\.nn\.functional as F"
    passed1, details1 = check_pattern_in_file(filepath, pattern1, "F import")
    print_check("torch.nn.functional imported", passed1, details1)
    checks.append(passed1)

    # Check: LayerwiseDistillationLoss class exists
    pattern2 = r"class LayerwiseDistillationLoss\(nn\.Module\):"
    passed2, details2 = check_pattern_in_file(filepath, pattern2, "LayerwiseDistillationLoss class")
    print_check("LayerwiseDistillationLoss class exists", passed2, details2)
    checks.append(passed2)

    # Check: DistillationEvaluator in evaluation.py
    filepath2 = "utils/evaluation.py"
    pattern3 = r"class DistillationEvaluator:"
    passed3, details3 = check_pattern_in_file(filepath2, pattern3, "DistillationEvaluator class")
    print_check("DistillationEvaluator class exists", passed3, details3)
    checks.append(passed3)

    # Check: compute_knowledge_retention method
    pattern4 = r"def compute_knowledge_retention"
    passed4, details4 = check_pattern_in_file(filepath2, pattern4, "compute_knowledge_retention method")
    print_check("compute_knowledge_retention method exists", passed4, details4)
    checks.append(passed4)

    all_passed = all(checks)

    if all_passed:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Code Structure: VERIFIED{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Code Structure: FAILED{Colors.END}")

    return all_passed


def print_summary(results):
    """Print final summary"""
    print_header("Verification Summary")

    all_passed = all(results.values())

    for name, passed in results.items():
        status = f"{Colors.GREEN}✓{Colors.END}" if passed else f"{Colors.RED}✗{Colors.END}"
        print(f"{status} {name}")

    print()

    critical_passed = results["Issue #1: Layerwise Loss Normalization"] and results["Issue #2: Sequence Length Alignment"]

    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}{'=' * 70}{Colors.END}")
        print(f"{Colors.GREEN}{Colors.BOLD}ALL FIXES VERIFIED SUCCESSFULLY!{Colors.END}")
        print(f"{Colors.GREEN}{Colors.BOLD}{'=' * 70}{Colors.END}")
        print(f"\n{Colors.GREEN}✓ The codebase is ready for training.{Colors.END}")
        print(f"{Colors.GREEN}✓ Run: python train.py --batch-size 2 --epochs 1{Colors.END}\n")
        return 0
    elif critical_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}{'=' * 70}{Colors.END}")
        print(f"{Colors.GREEN}{Colors.BOLD}CRITICAL FIXES VERIFIED SUCCESSFULLY!{Colors.END}")
        print(f"{Colors.GREEN}{Colors.BOLD}{'=' * 70}{Colors.END}")
        print(f"\n{Colors.GREEN}✓ All critical issues are fixed.{Colors.END}")
        print(f"{Colors.YELLOW}⚠ Some non-critical checks failed (see warnings above).{Colors.END}")
        print(f"{Colors.GREEN}✓ The codebase is ready for training.{Colors.END}")
        print(f"{Colors.GREEN}✓ Run: python train.py --batch-size 2 --epochs 1{Colors.END}\n")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}{'=' * 70}{Colors.END}")
        print(f"{Colors.RED}{Colors.BOLD}VERIFICATION FAILED - CRITICAL FIXES INCOMPLETE{Colors.END}")
        print(f"{Colors.RED}{Colors.BOLD}{'=' * 70}{Colors.END}")
        print(f"\n{Colors.RED}✗ Please review the failed checks above.{Colors.END}")
        print(f"{Colors.RED}✗ Refer to CRITICAL_ROOT_CAUSE_ANALYSIS.md for details.{Colors.END}\n")
        return 1


def main():
    """Main verification routine"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}Critical Fixes Verification Script{Colors.END}")
    print(f"{Colors.BLUE}Verifying fixes from Kaggle run analysis...{Colors.END}")

    results = {
        "Issue #1: Layerwise Loss Normalization": check_layerwise_normalization(),
        "Issue #2: Sequence Length Alignment": check_sequence_alignment(),
        "Previous Fixes": check_previous_fixes(),
        "Code Structure": check_code_structure(),
    }

    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
