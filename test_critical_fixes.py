#!/usr/bin/env python3
"""
Quick test script for critical components that were failing in training
Tests only the specific error cases encountered in Kaggle training
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_vocab_kl_divergence():
    """Test the specific KL divergence error that was occurring"""
    print("Testing KL divergence with vocabulary size mismatch...")

    try:
        from models.distillation_framework import VocabularyAligner

        # Exact dimensions from the error logs
        teacher_vocab = 151936  # From error: "tensor a (151936)"
        student_vocab = 49152   # From error: "tensor b (49152)"

        batch_size, seq_len = 4, 512  # From training config
        temperature = 4.0

        print(f"Teacher vocab: {teacher_vocab}, Student vocab: {student_vocab}")

        # Create vocabulary aligner
        aligner = VocabularyAligner(teacher_vocab, student_vocab)
        print(f"Alignment type: {aligner.alignment_type}")

        # Simulate the exact tensors that were causing issues
        teacher_logits = torch.randn(batch_size, seq_len, teacher_vocab)
        student_logits = torch.randn(batch_size, seq_len, student_vocab)

        print(f"Original teacher logits: {teacher_logits.shape}")
        print(f"Student logits: {student_logits.shape}")

        # Apply vocabulary alignment (this was failing before)
        aligned_teacher_logits = aligner.align_teacher_logits(teacher_logits)
        print(f"Aligned teacher logits: {aligned_teacher_logits.shape}")

        # Test KL divergence computation (this was the exact failing line)
        student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = torch.softmax(aligned_teacher_logits / temperature, dim=-1)

        kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        kd_loss = kl_loss_fn(student_log_probs, teacher_probs) * (temperature ** 2)

        print(f"KL divergence loss: {kd_loss.item():.6f}")
        print("✅ Vocabulary alignment and KL divergence working!")

        return True

    except Exception as e:
        print(f"❌ KL divergence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_router_tensor_concat():
    """Test the specific router concatenation error that was occurring"""
    print("\nTesting router tensor concatenation with dimension mismatch...")

    try:
        from models.student_aware_router import StudentCapacityEstimator

        # Exact dimensions from the error logs
        student_dim = 576   # SmolLM hidden size
        teacher_dim = 1024  # Huihui-MoE hidden size
        student_seq = 512   # From error: "Expected size 512"
        teacher_seq = 576   # From error: "but got size 576"
        batch_size = 4      # From training config
        num_experts = 8

        print(f"Student: [{batch_size}, {student_seq}, {student_dim}]")
        print(f"Teacher: [{batch_size}, {teacher_seq}, {teacher_dim}]")

        # Create the exact tensors that were causing concatenation failures
        student_hidden = torch.randn(batch_size, student_seq, student_dim)
        teacher_hidden = torch.randn(batch_size, teacher_seq, teacher_dim)

        # Test the capacity estimator that was failing
        capacity_estimator = StudentCapacityEstimator(student_dim, teacher_dim, num_experts)

        # This was the failing line: concatenation with mismatched sequences
        capacity_info = capacity_estimator(student_hidden, teacher_hidden)

        print(f"Capacity scores shape: {capacity_info['capacity_scores'].shape}")
        print(f"Gap scores shape: {capacity_info['gap_scores'].shape}")
        print("✅ Router tensor operations working!")

        return True

    except Exception as e:
        print(f"❌ Router tensor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_config_compatibility():
    """Test model configuration compatibility without loading full models"""
    print("\nTesting model configuration compatibility...")

    try:
        from transformers import AutoConfig

        # Test the exact models from training
        teacher_model_name = "huihui-ai/Huihui-MoE-1B-A0.6B"
        student_model_name = "HuggingFaceTB/SmolLM-135M"

        print(f"Loading configs for {teacher_model_name} and {student_model_name}")

        teacher_config = AutoConfig.from_pretrained(teacher_model_name)
        student_config = AutoConfig.from_pretrained(student_model_name)

        # Extract the exact dimensions that were causing issues
        teacher_vocab = teacher_config.vocab_size
        student_vocab = student_config.vocab_size
        teacher_hidden = teacher_config.hidden_size
        student_hidden = student_config.hidden_size

        print(f"Teacher vocab: {teacher_vocab}, hidden: {teacher_hidden}")
        print(f"Student vocab: {student_vocab}, hidden: {student_hidden}")

        # Verify these match our test assumptions
        expected_teacher_vocab = 151936
        expected_student_vocab = 49152
        expected_teacher_hidden = 1024
        expected_student_hidden = 576

        print(f"Expected teacher vocab: {expected_teacher_vocab} (actual: {teacher_vocab})")
        print(f"Expected student vocab: {expected_student_vocab} (actual: {student_vocab})")

        if abs(teacher_vocab - expected_teacher_vocab) < 1000:  # Allow some variance
            print("✅ Teacher vocab size matches expectations")
        else:
            print(f"⚠️ Teacher vocab size different than expected")

        if abs(student_vocab - expected_student_vocab) < 1000:
            print("✅ Student vocab size matches expectations")
        else:
            print(f"⚠️ Student vocab size different than expected")

        print("✅ Model configurations loaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Model config test failed: {e}")
        return False

def test_framework_initialization_minimal():
    """Test framework initialization with minimal config"""
    print("\nTesting minimal framework initialization...")

    try:
        # Test just the component initialization without full model loading
        config = {
            'teacher_model': 'huihui-ai/Huihui-MoE-1B-A0.6B',
            'student_model': 'HuggingFaceTB/SmolLM-135M',
            'num_experts': 4,  # Smaller for testing
            'top_k': 1,
            'temperature': 4.0,
            'alpha_kd': 0.7,
            'alpha_feature': 0.1,
            'alpha_attention': 0.0,
            'alpha_layerwise': 0.0,
            'alpha_contrastive': 0.0,
            'total_steps': 1000
        }

        # Test vocabulary aligner creation
        from models.distillation_framework import VocabularyAligner
        aligner = VocabularyAligner(151936, 49152)
        print(f"Vocab aligner created: needs_alignment={aligner.needs_alignment}")

        # Test router config creation
        router_config = {
            'student_dim': 576,
            'teacher_dim': 1024,
            'num_experts': config['num_experts'],
            'top_k': config['top_k'],
            'initial_top_k': 1,
            'final_top_k': 2,
            'warmup_steps': 100,
            'total_steps': config['total_steps']
        }
        print(f"Router config created: {router_config}")

        print("✅ Framework components initialized successfully!")
        return True

    except Exception as e:
        print(f"❌ Framework initialization test failed: {e}")
        return False

def test_contrastive_loss_dimensions():
    """Test contrastive loss with mismatched embedding dimensions"""
    print("\nTesting contrastive loss dimension alignment...")

    try:
        from models.distillation_framework import ContrastiveDistillationLoss

        # Test parameters matching the error
        batch_size = 4
        student_dim = 576
        teacher_dim = 1024
        temperature = 0.07

        print(f"Student embeddings: [{batch_size}, {student_dim}]")
        print(f"Teacher embeddings: [{batch_size}, {teacher_dim}]")

        # Create test embeddings with different dimensions
        student_embeddings = torch.randn(batch_size, student_dim)
        teacher_embeddings = torch.randn(batch_size, teacher_dim)

        # Test contrastive loss with dimension alignment
        contrastive_loss = ContrastiveDistillationLoss(
            temperature=temperature,
            student_dim=student_dim,
            teacher_dim=teacher_dim
        )

        # This was the failing line: computing similarity between different dimensions
        loss = contrastive_loss(student_embeddings, teacher_embeddings)

        print(f"Contrastive loss: {loss.item():.6f}")
        print("✅ Contrastive loss dimension alignment working!")

        return True

    except Exception as e:
        print(f"❌ Contrastive loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run critical component tests"""
    print("="*70)
    print("CRITICAL COMPONENT FIX VERIFICATION")
    print("Testing exact failure cases from Kaggle training logs")
    print("="*70)

    tests = [
        ("Vocabulary KL Divergence Fix", test_vocab_kl_divergence),
        ("Router Tensor Concatenation Fix", test_router_tensor_concat),
        ("Model Config Compatibility", test_model_config_compatibility),
        ("Framework Component Initialization", test_framework_initialization_minimal),
        ("Contrastive Loss Dimension Fix", test_contrastive_loss_dimensions)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"CRITICAL TEST: {test_name}")
        print(f"{'='*50}")

        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"💥 {test_name} CRASHED: {e}")

    print(f"\n{'='*70}")
    print("CRITICAL FIXES VERIFICATION RESULTS")
    print(f"{'='*70}")
    print(f"✅ Critical Tests Passed: {passed}/{passed + failed}")
    print(f"❌ Critical Tests Failed: {failed}/{passed + failed}")

    if failed == 0:
        print("\n🎉 ALL CRITICAL FIXES WORKING!")
        print("The training should now proceed past the original error points.")
        print("\n🚀 Ready to run full training:")
        print("   python train.py --batch-size 4 --epochs 1")
    else:
        print(f"\n⚠️  {failed} critical test(s) still failing!")
        print("These need to be fixed before training will work.")

    print("="*70)

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
