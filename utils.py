import torch
import matplotlib.pyplot as plt
from PIL import Image 
import numpy as np
import json
import os
from datetime import datetime

def save_checkpoint(model, optimizer, epoch, loss, path, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch' : epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss' : loss,
        'timestamp' : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **kwargs
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(model, optimizer, path, device='cpu'):
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0)

    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch, loss

def visualize_prediction(image_path, generated_caption, ground_truth=None, save_path=None):

    img = Image.open(image_path).convert('RGB')

    plt.figure(figsize=(10,8))
    plt.imshow(img)
    plt.axis('off')

    title=f"Generated: {generated_caption}"
    if ground_truth:
        title += f"\n\nGroun Truth: {ground_truth}"
    plt.title(title, fontsize=12, wrap=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight' , dpi=150)
        print(f"Visualization saved : {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_batch_predictions(images, generated_captions, ground_truths=None, save_path=None, max_display=6):

    n = min(len(images) , max_display)
    rows = (n+2) // 3
    cols = min(n,3)

    fig, axes = plt.subplots(rows, cols, figsize=(15,5*rows))
    if n==1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(n):
        if isinstance(images[i], str):
            img = Image.open(images[i]).convert('RGB')
        else:
            img = images[i]
        
        axes[i].imshow(img)
        axes[i].axis('off')

        title = f"Generated: \n{generated_captions[i]}"
        if ground_truths and i < len(ground_truths):
            title += f"\n\nGT: {ground_truths[i]}"
        axes[i].set_title(title, fontsize=9)
    
    for i in range(n, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Batch visualization saved: {save_path}")
    else:
        plt.show()
    plt.close()

def clean_caption(caption):

    caption = ' '.join(caption.split())

    if caption:
        caption = caption[0].upper() + caption[1:]
    
    if caption and caption[-1] not in ['.', '!', '?']:
        caption += '.'
    return caption

def extract_glove_features(caption):

    features = {
        'size':None,
        'pattern' : None, 
        'color' : None,
        'material' : None,
        'special_feature' : None
    }

    caption_lower = caption.lower()

    size_keywords = ['small', 'medium', 'large', 'xl', 'xxl', 's', 'm', 'l']
    for size in size_keywords:
        if size in caption_lower:
            features['size'] = size
            break

    pattern_keywords = ['striped', 'dotted', 'plain', 'patterned', 'checked']
    for pattern in pattern_keywords:
        if pattern in caption_lower:
            features['pattern'] = pattern
            break
    
    color_keywords = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'brown', 'gray']
    for color in color_keywords:
        if color in caption_lower:
            features['color'] = color
            break
    
    special_keywords = ['hole', 'torn', 'damaged', 'new', 'used', 'vintage']
    for special in special_keywords:
        if special in caption_lower:
            features['special_feature'] = special
            break
    return features

class TrainingLogger:

    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'training_{timestamp}.log')

        self.metrics = {
            'train_loss' : [],
            'val_loss' : [],
            'epoch' : []
        }

    def log(self, message, print_msg=True):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"

        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
        
        if print_msg:
            print(log_entry)
    
    def log_metrics(self, epoch, train_loss, val_loss=None):
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)

        msg = f"Epoch {epoch}: Train Loss = {train_loss:.4f}"
        if val_loss is not None:
            msg += f", Val Loss = {val_loss:.4f}"
        self.log(msg)
    
    def save_metrics(self, path=None):
        if path is None:
            path = os.path.join(self.log_dir, 'metrics.json')

        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {path}")

def plot_training_curves(metrics, save_path=None):

        plt.figure(figsize=(10, 6))
        epochs = metrics['epoch']
        plt.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss', linewidth=2)

        if 'val_loss' in metrics and metrics['val_loss']:
            plt.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss', linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Progress', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Training curves saved: {save_path}")
        else:
            plt.show()
        
        plt.close()

def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total trainable parameters: {total_params:,}")

        print("\nParameter breakdown: ")
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f" {name} : {params:,}")
        return total_params

if __name__ == "__main__":
    print("Testing utils.py functions")

    logger = TrainingLogger(log_dir='test_logs')
    logger.log("Test log message")
    logger.log_metrics(epoch=1, train_loss=2.5, val_loss=2.3)
    logger.log_metrics(epoch=2, train_loss=2.0, val_loss=2.1)

    caption = " size medium blue glove with hole "
    cleaned = clean_caption(caption)
    print(f"\nCleaned caption: '{cleaned}' ")

    features = extract_glove_features(cleaned)
    print(f"Extracted features: {features}")

    metrics = {
        'epoch' : [1,2,3,4,5],
        'train_loss':[2.5,2.0,1.7,1.5,1.3],
        'val_loss': [2.3,2.1,1.9,1.7,1.6]
    }

    plot_training_curves(metrics, save_path='test_logs/training_curve.png')
    print("\nAll tests passed")