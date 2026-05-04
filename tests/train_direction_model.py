#!/usr/bin/env python
"""
train_direction_model.py
========================
Complete training script for CRISPR direction prediction transformer.

Usage:
    conda activate Direction
    python train_direction_model.py \
        --dataset /tmp/direction_training_dataset.jsonl \
        --output_model /tmp/direction_model.pth \
        --batch_size 32 \
        --epochs 20 \
        --learning_rate 0.001
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

from direction_learning.dataset import DirectionJsonlDataset, build_vocab_from_jsonl
from direction_learning.train import split_groups, build_dataloader
from direction_learning.model import build_model


def main():
    parser = argparse.ArgumentParser(description="Train direction prediction model")
    parser.add_argument("--dataset", required=True, help="Path to training JSONL dataset")
    parser.add_argument("--output_model", default="/tmp/direction_model.pth", help="Path to save model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--include_flanks", action="store_true", help="Include flanking sequences")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Using device: {device}")
    
    # Check dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[!] Dataset not found: {args.dataset}")
        return 1
    
    # 1. Load dataset and vocabulary
    print(f"\n[1] Loading dataset from {args.dataset}...")
    try:
        dataset = DirectionJsonlDataset(str(dataset_path), include_flanks=args.include_flanks)
        vocab = build_vocab_from_jsonl(str(dataset_path))
    except Exception as e:
        print(f"[!] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"    Dataset size: {len(dataset)}")
    print(f"    Vocabulary size: {len(vocab)}")
    
    # Count labels
    label_counts = [0, 0]
    for record in dataset.records:
        label_counts[record.label] += 1
    print(f"    Label distribution: Reverse={label_counts[0]} ({label_counts[0]/len(dataset):.1%}), "
          f"Forward={label_counts[1]} ({label_counts[1]/len(dataset):.1%})")
    
    # 2. Create group-safe splits
    print(f"\n[2] Creating train/val/test splits (group-safe)...")
    try:
        splits = split_groups(dataset.records, seed=args.seed)
    except Exception as e:
        print(f"[!] Failed to create splits: {e}")
        return 1
    
    print(f"    Train: {len(splits['train'])} ({len(splits['train'])/len(dataset):.1%})")
    print(f"    Val:   {len(splits['val'])} ({len(splits['val'])/len(dataset):.1%})")
    print(f"    Test:  {len(splits['test'])} ({len(splits['test'])/len(dataset):.1%})")
    
    # 3. Create data loaders
    print(f"\n[3] Creating data loaders (batch_size={args.batch_size})...")
    try:
        train_loader = build_dataloader(
            dataset, splits['train'], vocab, 
            batch_size=args.batch_size, shuffle=True
        )
        val_loader = build_dataloader(
            dataset, splits['val'], vocab, 
            batch_size=args.batch_size, shuffle=False
        )
        test_loader = build_dataloader(
            dataset, splits['test'], vocab, 
            batch_size=args.batch_size, shuffle=False
        )
        print(f"    Train batches: {len(train_loader)}")
        print(f"    Val batches:   {len(val_loader)}")
        print(f"    Test batches:  {len(test_loader)}")
    except Exception as e:
        print(f"[!] Failed to create data loaders: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 4. Build model
    print(f"\n[4] Building model...")
    try:
        model = build_model(vocab_size=len(vocab), include_flanks=args.include_flanks)
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Model: SpacerDirectionTransformer")
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")
    except Exception as e:
        print(f"[!] Failed to build model: {e}")
        return 1
    
    # 5. Setup optimizer and loss
    print(f"\n[5] Setting up optimizer and loss function...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    print(f"    Optimizer: Adam (lr={args.learning_rate})")
    print(f"    Loss: BCEWithLogitsLoss")
    
    # 6. Training loop
    print(f"\n[6] Training for {args.epochs} epochs...\n")
    print("=" * 80)
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for batch in train_loader:
            try:
                batch_gpu = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                logits = model({
                    'spacer_tokens': batch_gpu['spacer_tokens'],
                    'spacer_mask': batch_gpu['spacer_mask'],
                })
                loss = loss_fn(logits, batch_gpu['label'].float())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
            except Exception as e:
                print(f"[!] Error during training batch: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        avg_train_loss = train_loss / train_steps if train_steps > 0 else 0.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    batch_gpu = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    
                    logits = model({
                        'spacer_tokens': batch_gpu['spacer_tokens'],
                        'spacer_mask': batch_gpu['spacer_mask'],
                    })
                    loss = loss_fn(logits, batch_gpu['label'].float())
                    val_loss += loss.item()
                    
                    preds = (torch.sigmoid(logits) > 0.5).long().squeeze()
                    if preds.dim() == 0:  # Handle single element
                        preds = preds.unsqueeze(0)
                    val_correct += (preds == batch_gpu['label']).sum().item()
                    val_total += batch_gpu['label'].shape[0]
                    val_steps += 1
                except Exception as e:
                    print(f"[!] Error during validation: {e}")
                    import traceback
                    traceback.print_exc()
                    return 1
        
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2%}")
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    
    print("=" * 80)
    
    # 7. Evaluate on test set
    print(f"\n[7] Evaluating on test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0.0
    test_steps = 0
    
    with torch.no_grad():
        for batch in test_loader:
            try:
                batch_gpu = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                logits = model({
                    'spacer_tokens': batch_gpu['spacer_tokens'],
                    'spacer_mask': batch_gpu['spacer_mask'],
                })
                loss = loss_fn(logits, batch_gpu['label'].float())
                test_loss += loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5).long().squeeze()
                if preds.dim() == 0:
                    preds = preds.unsqueeze(0)
                test_correct += (preds == batch_gpu['label']).sum().item()
                test_total += batch_gpu['label'].shape[0]
                test_steps += 1
            except Exception as e:
                print(f"[!] Error during test evaluation: {e}")
                return 1
    
    test_acc = test_correct / test_total if test_total > 0 else 0.0
    avg_test_loss = test_loss / test_steps if test_steps > 0 else 0.0
    
    print(f"    Test Loss: {avg_test_loss:.4f}")
    print(f"    Test Accuracy: {test_acc:.2%}")
    print(f"    Best validation accuracy: {best_val_acc:.2%} (epoch {best_epoch})")
    
    # 8. Save best model
    print(f"\n[8] Saving model to {args.output_model}...")
    try:
        output_path = Path(args.output_model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_state, str(output_path))
        print(f"    Saved successfully!")
    except Exception as e:
        print(f"[!] Failed to save model: {e}")
        return 1
    
    print(f"\n[✓] Training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
