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
    def __init__(self, root_dir, annotation_file, num_frames=16, frame_size=160, 
                 split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.rawframes_dir = self.root_dir / 'rawframes'
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.split = split
        
        self.samples = []
        self.label_to_id = {}
        self.id_to_label = {}
        
        # NOTE: For testing without real data, ensure annotation files exist but rows can be dummy
        with open(annotation_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2: continue
                folder_id = row[0].strip()
                label = row[1].strip()
                
                if label not in self.label_to_id:
                    label_id = len(self.label_to_id)
                    self.label_to_id[label] = label_id
                    self.id_to_label[label_id] = label
                
                # Check if folder exists
                folder_path = self.rawframes_dir / folder_id
                if folder_path.exists():
                    self.samples.append((folder_id, self.label_to_id[label]))
        
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
            
    def _get_default_transform(self):
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((self.frame_size, self.frame_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.frame_size, self.frame_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _load_frames(self, folder_id):
        folder_path = self.rawframes_dir / folder_id
        frame_files = sorted([f for f in folder_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if len(frame_files) == 0:
            # Create a black dummy frame if no frames found (Testing Hack)
            return [Image.new('RGB', (self.frame_size, self.frame_size)) for _ in range(self.num_frames)]
        
        if len(frame_files) >= self.num_frames:
            indices = np.linspace(0, len(frame_files) - 1, self.num_frames, dtype=int)
        else:
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
        folder_id, label = self.samples[idx]
        frames = self._load_frames(folder_id)
        transformed_frames = [self.transform(frame) for frame in frames]
        frames_tensor = torch.stack(transformed_frames, dim=0)
        return frames_tensor, label