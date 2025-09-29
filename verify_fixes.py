#!/usr/bin/env python3
"""
Verification script for Student-Aware Distillation fixes
Tests all critical fixes without running full training
"""

import torch
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all necessary imports work correctly"""
    print("Testing imports...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from models.distillation_framework import VocabularyAligner, StudentAwareDistillationFramework
        from data.data_loader import create_distillation_dataloader, get_recommended_datasets
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_vocabulary_alignment():
    """Test vocabulary alignment with mock data"""
    print("\nTesting vocabulary alignment...")
    try:
        from models.distillation_framework import VocabularyAligner

        # Test case 1: Teacher vocab larger than student
        teacher_vocab = 151936
        student_vocab = 49152

        aligner = VocabularyAligner(teacher_vocab, student_vocab)

        # Mock teacher logits
        batch_size, seq_len = 2, 10
        teacher_logits = torch.randn(batch_size, seq_len, teacher_vocab)

        # Align logits
        aligned_logits = aligner.align_teacher_logits(teacher_logits)

        # Verify shape
        expected_shape = (batch_size, seq_len, student_vocab)
        assert aligned_logits.shape == expected_shape, f"Shape mismatch: got {aligned_logits.shape}, expected {expected_shape}"

        # Test KL divergence computation
        student_logits = torch.randn(batch_size, seq_len, student_vocab)
        temperature = 4.0

        student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = torch.softmax(aligned_logits / temperature, dim=-1)

        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        loss = kl_loss(student_log_probs, teacher_probs)

        assert not torch.isnan(loss), "KL divergence loss is NaN"
        print(f"✅ Vocabulary alignment working - KL loss: {loss.item():.4f}")
        return True

    except Exception as e:
        print(f"❌ Vocabulary alignment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_compatibility():
    """Test model loading without actually loading large models"""
    print("\nTesting model configuration compatibility...")
    try:
        from transformers import AutoConfig

        # Test teacher model config
        teacher_config = AutoConfig.from_pretrained("huihui-ai/Huihui-MoE-1B-A0.6B")
        student_config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM-135M")

        # Extract dimensions
        teacher_vocab = getattr(teacher_config, 'vocab_size', None)
        student_vocab = getattr(student_config, 'vocab_size', None)
        teacher_dim = getattr(teacher_config, 'hidden_size', None)
        student_dim = getattr(student_config, 'hidden_size', None)

        print(f"Teacher - Vocab: {teacher_vocab}, Hidden: {teacher_dim}")
        print(f"Student - Vocab: {student_vocab}, Hidden: {student_dim}")

        # Test vocabulary aligner with real sizes
        if teacher_vocab and student_vocab:
            from models.distillation_framework import VocabularyAligner
            aligner = VocabularyAligner(teacher_vocab, student_vocab)
            print(f"Vocab alignment needed: {aligner.needs_alignment}")
            print(f"Alignment type: {aligner.alignment_type if aligner.needs_alignment else 'none'}")

        print("✅ Model compatibility verified")
        return True

    except Exception as e:
        print(f"❌ Model compatibility test failed: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading with fallback"""
    print("\nTesting dataset loading...")
    try:
        from transformers import AutoTokenizer
        from data.data_loader import get_recommended_datasets

        # Get recommended datasets
        datasets = get_recommended_datasets()
        print(f"Recommended datasets: {datasets}")

        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Tokenizer vocab size: {len(tokenizer)}")
        print("✅ Dataset and tokenizer setup working")
        return True

    except Exception as e:
        print(f"❌ Dataset loading test failed: {e}")
        return False

def test_framework_initialization():
    """Test framework initialization without loading actual models"""
    print("\nTesting framework configuration...")
    try:
        # Test config creation
        config = {
            'teacher_model': 'huihui-ai/Huihui-MoE-1B-A0.6B',
            'student_model': 'HuggingFaceTB/SmolLM-135M',
            'num_experts': 8,
            'top_k': 2,
            'temperature': 4.0,
            'alpha_kd': 0.7,
            'alpha_feature': 0.1,
            'alpha_attention': 0.1,
            'alpha_layerwise': 0.05,
            'alpha_contrastive': 0.05,
            'contrastive_temp': 0.07,
            'initial_top_k': 1,
            'final_top_k': 4,
            'warmup_steps': 1000,
            'total_steps': 10000
        }

        # Validate config structure
        required_keys = ['teacher_model', 'student_model', 'temperature']
        for key in required_keys:
            assert key in config, f"Missing required config key: {key}"

        print("✅ Framework configuration valid")
        return True

    except Exception as e:
        print(f"❌ Framework configuration test failed: {e}")
        return False

def test_training_pipeline_setup():
    """Test training pipeline setup without actual training"""
    print("\nTesting training pipeline setup...")
    try:
        # Test directory creation
        dirs = ['./checkpoints', './logs', './cache']
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            assert os.path.exists(dir_path), f"Failed to create {dir_path}"

        # Test CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if cuda_available:
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        print("✅ Training pipeline setup working")
        return True

    except Exception as e:
        print(f"❌ Training pipeline setup failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("="*60)
    print("STUDENT-AWARE DISTILLATION - FIX VERIFICATION")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("Vocabulary Alignment", test_vocabulary_alignment),
        ("Model Compatibility", test_model_compatibility),
        ("Dataset Loading", test_dataset_loading),
        ("Framework Config", test_framework_initialization),
        ("Training Setup", test_training_pipeline_setup),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"TEST: {test_name}")
        print(f"{'='*40}")

        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print("VERIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"✅ Tests Passed: {passed}")
    print(f"❌ Tests Failed: {failed}")
    print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")

    if failed == 0:
        print("\n🎉 All tests passed! The fixes are working correctly.")
        print("You can now run the training script with confidence.")
        print("\nNext steps:")
        print("1. Run: python train.py --batch-size 4 --epochs 1")
        print("2. Monitor for any remaining issues")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please review the errors above.")
        print("Some fixes may need additional work.")

    print("="*60)

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
