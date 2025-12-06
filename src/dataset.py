"""Dataset implementation for Jester gesture recognition."""

import os
import csv
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np


class JesterDataset(Dataset):
    """Dataset for Jester gesture recognition.
    
    Each sample is a video clip represented as a fixed number of frames.
    """
    
    def __init__(self, root_dir, annotation_file, num_frames=16, frame_size=160, 
                 split='train', transform=None):
        """Initialize Jester dataset.
        
        Args:
            root_dir: Root directory containing 'rawframes' and 'annotations'
            annotation_file: Path to CSV annotation file (train.csv, val.csv, or test.csv)
            num_frames: Number of frames to sample per clip
            frame_size: Size to resize frames to (square)
            split: Dataset split ('train', 'val', or 'test')
            transform: Optional custom transform pipeline
        """
        self.root_dir = Path(root_dir)
        self.rawframes_dir = self.root_dir / 'rawframes'
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.split = split
        
        # Read annotation file
        self.samples = []
        self.label_to_id = {}
        self.id_to_label = {}
        
        with open(annotation_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                folder_id = row[0].strip()
                label = row[1].strip()
                
                # Build label mapping
                if label not in self.label_to_id:
                    label_id = len(self.label_to_id)
                    self.label_to_id[label] = label_id
                    self.id_to_label[label_id] = label
                
                folder_path = self.rawframes_dir / folder_id
                if folder_path.exists():
                    self.samples.append((folder_id, self.label_to_id[label]))
        
        # Setup transforms
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
        
        print(f"Loaded {len(self.samples)} samples from {split} split")
        print(f"Number of classes: {len(self.label_to_id)}")
    
    def _get_default_transform(self):
        """Get default transform pipeline based on split."""
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((self.frame_size, self.frame_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])  # ImageNet stats
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.frame_size, self.frame_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _load_frames(self, folder_id):
        """Load and sample frames from a folder.
        
        Args:
            folder_id: Folder ID (e.g., "1", "2", etc.)
            
        Returns:
            List of PIL Images
        """
        folder_path = self.rawframes_dir / folder_id
        
        # Get all frame files
        frame_files = sorted([f for f in folder_path.iterdir() 
                             if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if len(frame_files) == 0:
            raise ValueError(f"No frames found in {folder_path}")
        
        # Sample frames uniformly
        if len(frame_files) >= self.num_frames:
            indices = np.linspace(0, len(frame_files) - 1, self.num_frames, dtype=int)
        else:
            # If fewer frames than needed, repeat the last frame
            indices = list(range(len(frame_files)))
            while len(indices) < self.num_frames:
                indices.append(len(frame_files) - 1)
        
        frames = []
        for idx in indices:
            img = Image.open(frame_files[idx]).convert('RGB')
            frames.append(img)
        
        return frames
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Returns:
            frames: Tensor of shape (T, C, H, W) where T=num_frames
            label: Integer label (0 to num_classes-1)
        """
        folder_id, label = self.samples[idx]
        
        # Load frames
        frames = self._load_frames(folder_id)
        
        # Apply transforms
        transformed_frames = []
        for frame in frames:
            transformed_frame = self.transform(frame)
            transformed_frames.append(transformed_frame)
        
        # Stack frames: (T, C, H, W)
        frames_tensor = torch.stack(transformed_frames, dim=0)
        
        return frames_tensor, label
    
    def get_num_classes(self):
        """Get number of gesture classes."""
        return len(self.label_to_id)
    
    def get_label_mapping(self):
        """Get label to ID and ID to label mappings."""
        return self.label_to_id, self.id_to_label

