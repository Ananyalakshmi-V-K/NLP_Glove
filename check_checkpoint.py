import torch

checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')

print("Checkpoint keys:", checkpoint.keys())
print(f"\nEpoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Train Loss: {checkpoint.get('train_loss', 'N/A'):.4f}" if 'train_loss' in checkpoint else "Train Loss: N/A")
print(f"Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}" if 'val_loss' in checkpoint else "Val Loss: N/A")

if 'training_history' in checkpoint:
    history = checkpoint['training_history']
    print(f"\nTraining epochs completed: {len(history.get('train_loss', []))}")
    if history.get('train_loss'):
        print(f"Loss progression:")
        for i, (tl, vl) in enumerate(zip(history['train_loss'], history['val_loss']), 1):
            print(f"  Epoch {i}: train {tl:.4f}, val {vl:.4f}")
