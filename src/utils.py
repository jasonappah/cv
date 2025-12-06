"""Utility functions for gesture recognition training and evaluation."""

import torch
import logging
import os
from pathlib import Path


def get_device():
    """Get the best available device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def calculate_accuracy(outputs, targets, topk=(1, 5)):
    """Calculate top-k accuracy.
    
    Args:
        outputs: Model outputs (logits) of shape (batch_size, num_classes)
        targets: Ground truth labels of shape (batch_size,)
        topk: Tuple of k values to compute accuracy for
        
    Returns:
        List of accuracies for each k value
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth.tar', is_best=False):
    """Save model checkpoint.
    
    Args:
        state: Dictionary containing model state, optimizer state, epoch, etc.
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
        is_best: Whether this is the best model so far
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = checkpoint_dir / 'model_best.pth.tar'
        torch.save(state, best_filepath)


def load_checkpoint(checkpoint_path, model, optimizer=None, device=None):
    """Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        device: Device to map checkpoint to
        
    Returns:
        Dictionary with loaded state (epoch, best_acc, etc.)
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DDP models (remove 'module.' prefix if present)
    state_dict = checkpoint['state_dict']
    if any(key.startswith('module.') for key in state_dict.keys()):
        # Remove 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'best_acc': checkpoint.get('best_acc', 0.0),
        'state_dict': state_dict,
    }


def setup_logging(log_dir=None, rank=0):
    """Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files (optional)
        rank: Process rank (only rank 0 logs to console)
    """
    if rank != 0:
        # Disable logging for non-rank-0 processes
        logging.getLogger().setLevel(logging.CRITICAL)
        return
    
    # Configure root logger
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / 'training.log')
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

