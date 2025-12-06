"""Utility functions."""

import torch
import logging
from pathlib import Path

def get_device():
    # Force CPU for this test
    return torch.device("cpu")

def calculate_accuracy(outputs, targets, topk=(1, 5)):
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
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_dir / filename)

def setup_logging(log_dir=None, rank=0):
    if rank != 0:
        logging.getLogger().setLevel(logging.CRITICAL)
        return
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler()])
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        logging.getLogger().addHandler(logging.FileHandler(Path(log_dir) / 'training.log'))