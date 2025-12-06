"""Evaluation script for Jester gesture recognition model."""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dataset import JesterDataset
from models.gesture_model import GestureModel
from utils import get_device, calculate_accuracy, load_checkpoint
from torch.utils.data import DataLoader


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Normalized Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def evaluate(model, data_loader, device, class_names=None):
    """Evaluate model on dataset.
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_logits = []
    
    total_time = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for frames, labels in data_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Measure inference time
            start_time = time.time()
            logits = model(frames)
            end_time = time.time()
            
            total_time += (end_time - start_time)
            num_samples += frames.size(0)
            
            # Get predictions
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu())
    
    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = torch.cat(all_logits, dim=0).numpy()
    
    # Calculate metrics
    # Top-1 and Top-5 accuracy
    top1_correct = (all_preds == all_labels).sum()
    top1_acc = top1_correct / len(all_labels) * 100.0
    
    # Top-5 accuracy
    top5_correct = 0
    for i in range(len(all_labels)):
        top5_preds = np.argsort(all_logits[i])[-5:][::-1]
        if all_labels[i] in top5_preds:
            top5_correct += 1
    top5_acc = top5_correct / len(all_labels) * 100.0
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Average metrics
    avg_precision = precision.mean()
    avg_recall = recall.mean()
    avg_f1 = f1.mean()
    
    # Inference latency
    avg_latency = total_time / num_samples * 1000  # ms per sample
    
    results = {
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support,
        'avg_latency_ms': avg_latency,
        'predictions': all_preds,
        'labels': all_labels,
        'class_names': class_names
    }
    
    return results


def print_results(results, split='Validation'):
    """Print evaluation results."""
    print(f"\n{'='*60}")
    print(f"{split} Results")
    print(f"{'='*60}")
    print(f"Top-1 Accuracy: {results['top1_acc']:.2f}%")
    print(f"Top-5 Accuracy: {results['top5_acc']:.2f}%")
    print(f"\nAverage Metrics:")
    print(f"  Precision: {results['avg_precision']:.4f}")
    print(f"  Recall:    {results['avg_recall']:.4f}")
    print(f"  F1-Score:  {results['avg_f1']:.4f}")
    print(f"\nInference Latency: {results['avg_latency_ms']:.2f} ms per sample")
    
    if results['class_names']:
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        for i, class_name in enumerate(results['class_names']):
            print(f"{class_name:<20} "
                  f"{results['per_class_precision'][i]:<12.4f} "
                  f"{results['per_class_recall'][i]:<12.4f} "
                  f"{results['per_class_f1'][i]:<12.4f} "
                  f"{int(results['per_class_support'][i]):<10}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Jester Gesture Recognition Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--save-cm', action='store_true', help='Save confusion matrix plot')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataset
    data_config = config['data']
    dataset = JesterDataset(
        root_dir=data_config['root_dir'],
        annotation_file=os.path.join(data_config['root_dir'], 'annotations', f'{args.split}.csv'),
        num_frames=data_config['num_frames'],
        frame_size=data_config['frame_size'],
        split=args.split
    )
    
    # Get class names
    _, id_to_label = dataset.get_label_mapping()
    class_names = [id_to_label[i] for i in range(len(id_to_label))]
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    model_config = config['model']
    model = GestureModel(
        num_classes=model_config['num_classes'],
        vit_model=model_config['vit_model'],
        vit_pretrained=model_config['vit_pretrained'],
        ssm_layers=model_config['ssm_layers'],
        ssm_d_state=model_config['ssm_d_state'],
        ssm_d_conv=model_config['ssm_d_conv'],
        ssm_expand=model_config['ssm_expand']
    )
    
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    load_checkpoint(args.checkpoint, model, device=device)
    
    # Evaluate
    print(f"Evaluating on {args.split} split...")
    results = evaluate(model, data_loader, device, class_names=class_names)
    
    # Print results
    print_results(results, split=args.split.capitalize())
    
    # Save confusion matrix if requested
    if args.save_cm:
        os.makedirs(args.output_dir, exist_ok=True)
        cm_path = os.path.join(args.output_dir, f'confusion_matrix_{args.split}.png')
        plot_confusion_matrix(
            results['labels'],
            results['predictions'],
            class_names,
            cm_path
        )
    
    # Save results to file
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, f'eval_results_{args.split}.txt')
    with open(results_file, 'w') as f:
        f.write(f"{args.split.capitalize()} Evaluation Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Top-1 Accuracy: {results['top1_acc']:.2f}%\n")
        f.write(f"Top-5 Accuracy: {results['top5_acc']:.2f}%\n\n")
        f.write("Average Metrics:\n")
        f.write(f"  Precision: {results['avg_precision']:.4f}\n")
        f.write(f"  Recall:    {results['avg_recall']:.4f}\n")
        f.write(f"  F1-Score:  {results['avg_f1']:.4f}\n\n")
        f.write(f"Inference Latency: {results['avg_latency_ms']:.2f} ms per sample\n\n")
        
        if class_names:
            f.write("Per-Class Metrics:\n")
            f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
            f.write("-" * 70 + "\n")
            for i, class_name in enumerate(class_names):
                f.write(f"{class_name:<20} "
                       f"{results['per_class_precision'][i]:<12.4f} "
                       f"{results['per_class_recall'][i]:<12.4f} "
                       f"{results['per_class_f1'][i]:<12.4f} "
                       f"{int(results['per_class_support'][i]):<10}\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()

