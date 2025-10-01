#!/usr/bin/env python3
"""
Test script to verify vocabulary alignment fix
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from models.distillation_framework import TeacherToStudentLogitProjector

def test_vocab_alignment():
    """Test vocabulary alignment between teacher and student models"""
    print("Testing vocabulary alignment fix...")

    # Load tokenizers to get actual vocab sizes
    print("Loading tokenizers...")
    teacher_tokenizer = AutoTokenizer.from_pretrained("huihui-ai/Huihui-MoE-1B-A0.6B")
    student_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

    teacher_vocab_size = len(teacher_tokenizer)
    student_vocab_size = len(student_tokenizer)

    print(f"Teacher vocab size: {teacher_vocab_size}")
    print(f"Student vocab size: {student_vocab_size}")

    teacher_config = AutoConfig.from_pretrained("huihui-ai/Huihui-MoE-1B-A0.6B")
    student_config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM-135M")

    teacher_dim = getattr(teacher_config, 'hidden_size', 1024)
    student_dim = getattr(student_config, 'hidden_size', 576)

    teacher_embedding = torch.nn.Embedding(teacher_vocab_size, teacher_dim)
    student_embedding = torch.nn.Embedding(student_vocab_size, student_dim)

    logit_projector = TeacherToStudentLogitProjector(
        teacher_embedding=teacher_embedding,
        student_embedding=student_embedding,
        teacher_dim=teacher_dim,
        student_dim=student_dim
    )

    # Test with mock logits
    batch_size = 2
    seq_len = 10

    # Create mock teacher logits
    teacher_logits = torch.randn(batch_size, seq_len, teacher_vocab_size)
    print(f"Teacher logits shape: {teacher_logits.shape}")

    # Align teacher logits
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    aligned_logits = logit_projector(teacher_probs)
    print(f"Aligned logits shape: {aligned_logits.shape}")

    # Verify alignment
    expected_shape = (batch_size, seq_len, student_vocab_size)
    assert aligned_logits.shape == expected_shape, f"Expected {expected_shape}, got {aligned_logits.shape}"

    print("✓ Vocabulary alignment test passed!")

    # Test KL divergence computation
    print("\nTesting KL divergence with aligned logits...")

    # Create mock student logits
    student_logits = torch.randn(batch_size, seq_len, student_vocab_size)

    # Compute probabilities
    temperature = 4.0
    student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = torch.softmax(aligned_logits / temperature, dim=-1)

    # Compute KL divergence
    kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
    kl_loss = kl_loss_fn(student_log_probs, teacher_probs) * (temperature ** 2)

    print(f"KL divergence loss: {kl_loss.item():.4f}")
    print("✓ KL divergence computation test passed!")

    return True

def test_model_loading():
    """Test model loading with attention fix"""
    print("\nTesting model loading with attention implementation fix...")

    try:
        # Test teacher model loading
        print("Loading teacher model...")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            "huihui-ai/Huihui-MoE-1B-A0.6B",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        print("✓ Teacher model loaded successfully")

        # Test student model loading
        print("Loading student model...")
        student_model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM-135M",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        print("✓ Student model loaded successfully")

        # Get model configs
        teacher_config = teacher_model.config
        student_config = student_model.config

        print(f"Teacher vocab size: {getattr(teacher_config, 'vocab_size', 'unknown')}")
        print(f"Student vocab size: {getattr(student_config, 'vocab_size', 'unknown')}")
        print(f"Teacher hidden size: {getattr(teacher_config, 'hidden_size', 'unknown')}")
        print(f"Student hidden size: {getattr(student_config, 'hidden_size', 'unknown')}")

        return True

    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_framework_initialization():
    """Test distillation framework initialization"""
    print("\nTesting distillation framework initialization...")

    try:
        from models.distillation_framework import StudentAwareDistillationFramework

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

        print("Initializing distillation framework...")
        framework = StudentAwareDistillationFramework(config)
        print("✓ Framework initialized successfully")

        print(f"Teacher vocab size: {framework.teacher_vocab_size}")
        print(f"Student vocab size: {framework.student_vocab_size}")
        has_projector = hasattr(framework, "logit_projector")
        print(f"Logit projector present: {has_projector}")
        assert has_projector, "Distillation framework missing logit projector"

        return True

    except Exception as e:
        print(f"✗ Framework initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running vocabulary fix tests...\n")

    tests = [
        test_vocab_alignment,
        test_model_loading,
        test_framework_initialization
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Vocabulary fix is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    print("="*50)
