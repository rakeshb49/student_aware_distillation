#!/usr/bin/env python3
"""
Integration test to verify all fixed components work together
Tests the complete forward pass without full model loading
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_complete_integration():
    """Test complete integration of all fixed components"""
    print("Testing complete integration of all fixes...")

    try:
        from transformers import AutoConfig
        from models.distillation_framework import (
            TeacherToStudentLogitProjector,
            ContrastiveDistillationLoss,
            LayerwiseDistillationLoss,
            AttentionTransferModule
        )
        from models.student_aware_router import StudentAwareDistillationRouter

        # Test parameters matching actual model dimensions
        batch_size = 4
        seq_len = 512
        student_dim = 576
        teacher_dim = 1024
        student_vocab = 49152
        teacher_vocab = 151936
        num_experts = 4
        num_layers = 8

        print(f"Testing with batch_size={batch_size}, seq_len={seq_len}")
        print(f"Student: vocab={student_vocab}, dim={student_dim}")
        print(f"Teacher: vocab={teacher_vocab}, dim={teacher_dim}")

        # 1. Test Vocabulary Alignment
        print("\n1. Testing vocabulary projection...")
        teacher_config = AutoConfig.from_pretrained("huihui-ai/Huihui-MoE-1B-A0.6B")
        student_config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM-135M")

        teacher_dim = getattr(teacher_config, 'hidden_size', 1024)
        student_dim = getattr(student_config, 'hidden_size', 576)

        teacher_embedding = torch.nn.Embedding(teacher_vocab, teacher_dim)
        student_embedding = torch.nn.Embedding(student_vocab, student_dim)

        logit_projector = TeacherToStudentLogitProjector(
            teacher_embedding=teacher_embedding,
            student_embedding=student_embedding,
            teacher_dim=teacher_dim,
            student_dim=student_dim
        )

        teacher_logits = torch.randn(batch_size, seq_len, teacher_vocab)
        student_logits = torch.randn(batch_size, seq_len, student_vocab)

        temperature = 4.0
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        aligned_teacher_logits = logit_projector(teacher_probs)

        # KL divergence computation
        student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = torch.softmax(aligned_teacher_logits / temperature, dim=-1)

        kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        kd_loss = kl_loss_fn(student_log_probs, teacher_probs) * (temperature ** 2)

        print(f"   ✅ KL divergence loss: {kd_loss.item():.4f}")

        # 2. Test Router Integration
        print("\n2. Testing router integration...")
        router_config = {
            'student_dim': student_dim,
            'teacher_dim': teacher_dim,
            'num_experts': num_experts,
            'top_k': 2,
            'initial_top_k': 1,
            'final_top_k': 2,
            'warmup_steps': 100,
            'total_steps': 1000
        }

        router = StudentAwareDistillationRouter(router_config)

        # Create test hidden states
        student_hidden = torch.randn(batch_size, seq_len, student_dim)
        teacher_outputs = {
            'hidden_states': torch.randn(batch_size, seq_len + 64, teacher_dim),  # Different seq len
            'expert_outputs': [torch.randn(batch_size, seq_len + 64, teacher_dim)]
        }

        # Router forward pass
        routing_outputs = router(student_hidden, teacher_outputs, step=50)

        print(f"   ✅ Router output keys: {list(routing_outputs.keys())}")
        print(f"   ✅ Routed knowledge shape: {routing_outputs['routed_knowledge'].shape}")

        # 3. Test Contrastive Loss
        print("\n3. Testing contrastive loss...")
        contrastive_loss = ContrastiveDistillationLoss(
            temperature=0.07,
            student_dim=student_dim,
            teacher_dim=teacher_dim
        )

        student_embed = torch.randn(batch_size, student_dim)
        teacher_embed = torch.randn(batch_size, teacher_dim)

        c_loss = contrastive_loss(student_embed, teacher_embed)
        print(f"   ✅ Contrastive loss: {c_loss.item():.4f}")

        # 4. Test Layer-wise Distillation
        print("\n4. Testing layer-wise distillation...")
        layerwise_loss = LayerwiseDistillationLoss(
            student_layers=num_layers,
            teacher_layers=num_layers,
            student_dim=student_dim,
            teacher_dim=teacher_dim
        )

        student_hidden_list = [torch.randn(batch_size, seq_len, student_dim) for _ in range(num_layers)]
        teacher_hidden_list = [torch.randn(batch_size, seq_len, teacher_dim) for _ in range(num_layers)]

        lw_loss = layerwise_loss(student_hidden_list, teacher_hidden_list)
        print(f"   ✅ Layerwise loss: {lw_loss.item():.4f}")

        # 5. Test Attention Transfer
        print("\n5. Testing attention transfer...")
        attention_transfer = AttentionTransferModule(student_heads=9, teacher_heads=16)

        student_attn = torch.randn(batch_size, 9, seq_len, seq_len)
        teacher_attn = torch.randn(batch_size, 16, seq_len, seq_len)

        attn_loss = attention_transfer(student_attn, teacher_attn)
        print(f"   ✅ Attention loss: {attn_loss.item():.4f}")

        # 6. Test Complete Loss Computation
        print("\n6. Testing complete loss computation...")
        losses = {
            'kd_loss': kd_loss * 0.7,
            'routing_feature_loss': sum(routing_outputs['losses'].values()) * 0.1,
            'layerwise_loss': lw_loss * 0.05,
            'contrastive_loss': c_loss * 0.05,
            'attention_loss': attn_loss * 0.1
        }

        total_loss = sum(losses.values())

        print(f"   ✅ Individual losses:")
        for name, loss in losses.items():
            print(f"      {name}: {loss.item():.4f}")
        print(f"   ✅ Total loss: {total_loss.item():.4f}")

        # 7. Test Gradient Flow
        print("\n7. Testing gradient flow...")
        total_loss.backward()

        # Check if gradients exist
        grad_count = 0
        param_count = 0
        for component in [logit_projector, router, contrastive_loss, layerwise_loss, attention_transfer]:
            for param in component.parameters():
                param_count += 1
                if param.grad is not None:
                    grad_count += 1

        print(f"   ✅ Parameters with gradients: {grad_count}/{param_count}")

        print("\n🎉 COMPLETE INTEGRATION TEST PASSED!")
        print("All components work together seamlessly.")

        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """Test memory efficiency of the integrated system"""
    print("\nTesting memory efficiency...")

    try:
        import torch.cuda

        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()

            # Get initial memory
            initial_memory = torch.cuda.memory_allocated()
            print(f"Initial GPU memory: {initial_memory / 1024**2:.1f} MB")

            # Run integration test
            success = test_complete_integration()

            # Get final memory
            final_memory = torch.cuda.memory_allocated()
            print(f"Final GPU memory: {final_memory / 1024**2:.1f} MB")
            print(f"Memory increase: {(final_memory - initial_memory) / 1024**2:.1f} MB")

            # Clean up
            torch.cuda.empty_cache()

            return success
        else:
            print("CUDA not available, running CPU test only")
            return test_complete_integration()

    except Exception as e:
        print(f"Memory efficiency test failed: {e}")
        return False

def main():
    """Run integration tests"""
    print("="*70)
    print("COMPLETE INTEGRATION TEST")
    print("Testing all fixes working together")
    print("="*70)

    tests = [
        ("Complete Integration", test_complete_integration),
        ("Memory Efficiency", test_memory_efficiency)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"INTEGRATION TEST: {test_name}")
        print(f"{'='*50}")

        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")

    print(f"\n{'='*70}")
    print("INTEGRATION TEST RESULTS")
    print(f"{'='*70}")
    print(f"✅ Tests Passed: {passed}/{total}")

    if passed == total:
        print("\n🚀 READY FOR TRAINING!")
        print("All components integrated successfully.")
        print("The training pipeline should work end-to-end.")
    else:
        print(f"\n⚠️ {total - passed} integration test(s) failed.")
        print("Some components may not work together properly.")

    print("="*70)

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
