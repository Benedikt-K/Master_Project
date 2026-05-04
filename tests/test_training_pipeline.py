#!/usr/bin/env python
"""
test_training_pipeline.py
=========================
Comprehensive validation of the direction_learning training pipeline.

Tests:
1. Dataset loading and vocabulary building
2. Example encoding and batch collation
3. Model instantiation
4. Forward pass through the model
5. Backward pass and one training step
6. Evaluation on test batch

Exit code: 0 if all tests pass, 1 if any test fails.
"""

import sys
import json
from pathlib import Path

print("=" * 70)
print("TESTING DIRECTION_LEARNING TRAINING PIPELINE")
print("=" * 70)

# Test 1: Import modules
print("\n[1/7] Testing module imports...")
try:
    from direction_learning.dataset import (
        DirectionJsonlDataset,
        DirectionExample,
        encode_example,
        collate_encoded_examples,
        build_vocab_from_jsonl,
    )
    from direction_learning.model import SpacerDirectionTransformer, build_model
    from direction_learning.train import (
        split_groups,
    )
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Module import failed: {e}")
    sys.exit(1)

# Test 2: Load and validate test dataset
print("\n[2/7] Testing dataset loading...")
test_jsonl = Path("/tmp/direction_training_dataset_test.jsonl")
try:
    dataset = DirectionJsonlDataset(str(test_jsonl))
    print(f"✓ Loaded test JSONL with {len(dataset)} records")
    
    # Peek at a sample record
    sample = dataset[0]
    print(f"  - Sample type: {type(sample).__name__}")
    print(f"  - Array name: {sample.array_name}")
    print(f"  - Num spacers: {len(sample.spacers)}")
    print(f"  - Label: {sample.label}")
except Exception as e:
    print(f"✗ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Build vocabulary
print("\n[3/7] Testing vocabulary building...")
try:
    vocab = build_vocab_from_jsonl(str(test_jsonl))
    print(f"✓ Built vocabulary with {len(vocab)} unique characters")
    print(f"  - Characters: {sorted(vocab.keys())}")
except Exception as e:
    print(f"✗ Vocabulary building failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test example encoding and batch collation
print("\n[4/7] Testing example encoding and batch collation...")
try:
    # Get a few examples
    batch_size = 4
    raw_examples = [dataset[i] for i in range(min(batch_size, len(dataset)))]
    print(f"  - Loaded {len(raw_examples)} examples for batch")
    
    # Encode examples
    encoded = [encode_example(ex, vocab) for ex in raw_examples]
    print(f"  - Encoded {len(encoded)} examples")
    
    # Collate into batch (still in list format, not tensors)
    batch = collate_encoded_examples(encoded)
    print(f"✓ Batch collation successful")
    print(f"  - Spacer tokens (outer): {len(batch['spacer_tokens'])} items")
    if batch['spacer_tokens']:
        print(f"    - First item: {len(batch['spacer_tokens'][0])} spacers")
        if batch['spacer_tokens'][0]:
            print(f"    - First spacer length: {len(batch['spacer_tokens'][0][0])}")
    print(f"  - Spacer mask (outer): {len(batch['spacer_mask'])} items")
    print(f"  - Labels: {len(batch['label'])} items")
    print(f"  - Batch size: {len(batch['label'])}")
except Exception as e:
    print(f"✗ Example encoding/collation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Instantiate model
print("\n[5/7] Testing model instantiation...")
try:
    import torch
    
    model = build_model(vocab_size=len(vocab), include_flanks=False)
    print(f"✓ Model instantiated successfully")
    print(f"  - Model type: {model.__class__.__name__}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
except Exception as e:
    print(f"✗ Model instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Forward pass and backward pass
print("\n[6/7] Testing forward and backward passes...")
try:
    import torch
    import torch.nn as nn
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    
    # Convert batch lists to tensors
    spacer_tokens_tensor = torch.tensor(batch['spacer_tokens'], dtype=torch.long, device=device)
    spacer_mask_tensor = torch.tensor(batch['spacer_mask'], dtype=torch.long, device=device)
    labels_tensor = torch.tensor(batch['label'], dtype=torch.long, device=device)
    
    # Create batch dict for model
    batch_dict = {
        'spacer_tokens': spacer_tokens_tensor,
        'spacer_mask': spacer_mask_tensor,
    }
    
    print(f"  - Using device: {device}")
    print(f"  - Spacer tokens tensor shape: {spacer_tokens_tensor.shape}")
    print(f"  - Spacer mask tensor shape: {spacer_mask_tensor.shape}")
    print(f"  - Labels tensor shape: {labels_tensor.shape}")
    
    # Forward pass
    logits = model(batch_dict)
    print(f"✓ Forward pass successful")
    print(f"  - Logits shape: {logits.shape}")
    print(f"  - Logits dtype: {logits.dtype}")
    
    # Compute loss and backward pass
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, labels_tensor.float())
    print(f"✓ Loss computation successful")
    print(f"  - Loss value: {loss.item():.6f}")
    
    # Backward pass
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"✓ Backward pass and optimizer step successful")
    
except Exception as e:
    print(f"✗ Forward/backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Evaluation
print("\n[7/7] Testing evaluation metrics...")
try:
    import torch
    
    model.eval()
    with torch.no_grad():
        logits_eval = model(batch_dict)
        loss_eval = loss_fn(logits_eval, labels_tensor.float())
        
        # Compute accuracy
        preds = (torch.sigmoid(logits_eval) > 0.5).long().squeeze()
        accuracy = (preds == labels_tensor).float().mean().item()
    
    print(f"✓ Evaluation successful")
    print(f"  - Eval loss: {loss_eval.item():.6f}")
    print(f"  - Batch accuracy: {accuracy:.2%}")
    
except Exception as e:
    print(f"✗ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success
print("\n" + "=" * 70)
print("✓✓✓ ALL TESTS PASSED ✓✓✓")
print("=" * 70)
print("\nThe training pipeline is working correctly!")
print(f"\nNext steps:")
print(f"1. Use the full dataset: /tmp/direction_training_dataset_full.jsonl")
print(f"2. Example training loop:")
print(f"")
print(f"   from direction_learning.train import build_dataloader, split_groups")
print(f"   from direction_learning.dataset import DirectionJsonlDataset, encode_example, build_vocab_from_jsonl")
print(f"   from direction_learning.model import build_model")
print(f"")
print(f"   # Load dataset")
print(f"   dataset = DirectionJsonlDataset('/tmp/direction_training_dataset_full.jsonl')")
print(f"   vocab = build_vocab_from_jsonl('/tmp/direction_training_dataset_full.jsonl')")
print(f"")
print(f"   # Split into train/val/test")
print(f"   splits = split_groups(dataset.raw_examples, seed=42)")
print(f"   train_loader = build_dataloader(dataset, splits['train'], vocab, batch_size=32, shuffle=True)")
print(f"   val_loader = build_dataloader(dataset, splits['val'], vocab, batch_size=32, shuffle=False)")
print(f"")
print(f"   # Build and train model")
print(f"   model = build_model(vocab_size=len(vocab))")
print(f"   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)")
print(f"   loss_fn = torch.nn.BCEWithLogitsLoss()")
print("=" * 70)

sys.exit(0)
