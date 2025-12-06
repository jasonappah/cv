"""Distributed training script for Jester gesture recognition using PyTorch DDP."""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dataset import JesterDataset
from models.gesture_model import GestureModel
from utils import get_device, calculate_accuracy, save_checkpoint, setup_logging


def setup_distributed():
    """Initialize distributed training."""
    # Get environment variables set by torchrun
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Initialize process group
    # Use 'gloo' backend for MPS/CPU, 'nccl' for CUDA
    if torch.cuda.is_available():
        backend = 'nccl'
    else:
        backend = 'gloo'
    
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, rank):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total_samples = 0
    
    for batch_idx, (frames, labels) in enumerate(train_loader):
        # Move to device
        frames = frames.to(device)  # (B, T, C, H, W)
        labels = labels.to(device)  # (B,)
        
        # Forward pass
        logits = model(frames)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Calculate accuracy
        acc1, acc5 = calculate_accuracy(logits, labels, topk=(1, 5))
        top1_correct += acc1 * batch_size / 100.0
        top5_correct += acc5 * batch_size / 100.0
        
        if rank == 0 and batch_idx % 100 == 0:
            logging.info(
                f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                f"Loss: {loss.item():.4f}, Top-1: {acc1:.2f}%, Top-5: {acc5:.2f}%"
            )
    
    # Average metrics
    avg_loss = running_loss / total_samples
    avg_top1 = top1_correct / total_samples * 100.0
    avg_top5 = top5_correct / total_samples * 100.0
    
    return avg_loss, avg_top1, avg_top5


def validate(model, val_loader, criterion, device, rank):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for frames, labels in val_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            
            logits = model(frames)
            loss = criterion(logits, labels)
            
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            
            acc1, acc5 = calculate_accuracy(logits, labels, topk=(1, 5))
            top1_correct += acc1 * batch_size / 100.0
            top5_correct += acc5 * batch_size / 100.0
    
    # Average metrics
    avg_loss = running_loss / total_samples
    avg_top1 = top1_correct / total_samples * 100.0
    avg_top5 = top5_correct / total_samples * 100.0
    
    return avg_loss, avg_top1, avg_top5


def main():
    parser = argparse.ArgumentParser(description='Train Jester Gesture Recognition Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Setup logging (only rank 0)
    log_dir = config['paths'].get('log_dir', 'logs')
    setup_logging(log_dir=log_dir, rank=rank)
    
    # Get device
    device = get_device()
    if rank == 0:
        logging.info(f"Using device: {device}")
        logging.info(f"World size: {world_size}, Rank: {rank}")
    
    # Create datasets
    data_config = config['data']
    train_dataset = JesterDataset(
        root_dir=data_config['root_dir'],
        annotation_file=os.path.join(data_config['root_dir'], 'annotations', 'train.csv'),
        num_frames=data_config['num_frames'],
        frame_size=data_config['frame_size'],
        split='train'
    )
    
    val_dataset = JesterDataset(
        root_dir=data_config['root_dir'],
        annotation_file=os.path.join(data_config['root_dir'], 'annotations', 'val.csv'),
        num_frames=data_config['num_frames'],
        frame_size=data_config['frame_size'],
        split='val'
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        sampler=train_sampler,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        sampler=val_sampler,
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
    
    # Wrap in DDP
    model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    training_config = config['training']
    
    if training_config['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.01)
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=training_config['learning_rate'],
            momentum=0.9,
            weight_decay=training_config.get('weight_decay', 0.01)
        )
    
    # Learning rate scheduler (optional warmup)
    warmup_epochs = training_config.get('warmup_epochs', 0)
    total_epochs = training_config['epochs']
    
    if warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        checkpoint_path = args.resume
        if rank == 0:
            logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        if rank == 0:
            logging.info(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.2f}%")
    
    # Training loop
    checkpoint_dir = config['paths']['checkpoint_dir']
    
    for epoch in range(start_epoch, total_epochs):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            logging.info(f"\nEpoch {epoch+1}/{total_epochs}")
        
        # Train
        train_loss, train_top1, train_top5 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1, rank
        )
        
        # Validate
        val_loss, val_top1, val_top5 = validate(
            model, val_loader, criterion, device, rank
        )
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Log results (rank 0 only)
        if rank == 0:
            logging.info(
                f"Train - Loss: {train_loss:.4f}, Top-1: {train_top1:.2f}%, Top-5: {train_top5:.2f}%"
            )
            logging.info(
                f"Val   - Loss: {val_loss:.4f}, Top-1: {val_top1:.2f}%, Top-5: {val_top5:.2f}%"
            )
        
        # Save checkpoint
        if rank == 0:
            is_best = val_top1 > best_acc
            best_acc = max(best_acc, val_top1)
            
            checkpoint_state = {
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config
            }
            
            save_checkpoint(
                checkpoint_state,
                checkpoint_dir,
                filename=f'checkpoint_epoch_{epoch+1}.pth.tar',
                is_best=is_best
            )
            
            if is_best:
                logging.info(f"New best model! Top-1 accuracy: {best_acc:.2f}%")
    
    if rank == 0:
        logging.info("Training completed!")
    
    cleanup_distributed()


if __name__ == '__main__':
    main()

