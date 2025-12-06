Plan: Jester Gesture Recognition with ViT + SSM on Apple Silicon (3-Mac PyTorch DDP Setup)
This document outlines a step-by-step plan to implement a Jester gesture recognition model using a Vision Transformer (ViT) encoder plus a State Space Model (SSM) backbone in PyTorch, running on Apple Silicon (M‑series) Macs. Training will be distributed using PyTorch Distributed Data Parallel (DDP) across 3 Macs.
Phase 0 – Project Skeleton

Suggested repository structure:
jester-ssm-gesture/
├─ data/
│  └─ jester/
│     ├─ rawframes/        # 1..148092 folders with JPG frames
│     └─ annotations/      # train/val/test CSV or txt
├─ src/
│  ├─ dataset.py
│  ├─ models/
│  │  ├─ vit_encoder.py
│  │  ├─ ssm_backbone.py
│  │  └─ gesture_model.py
│  ├─ train_ddp.py
│  ├─ eval.py
│  └─ utils.py
├─ experiments/
│  └─ config_*.yaml
└─ requirements.txt
Initialize a git repository, create a conda environment, and commit early.

Phase 1 – Apple Silicon + PyTorch Setup (Single Mac)

1. Install Miniforge or Miniconda if it is not already installed.
2. Create and activate a new environment (run on each Mac):
   conda create -n jester_env python=3.11 -y
   conda activate jester_env
3. Install PyTorch with Apple Silicon (MPS) support (run on each Mac):
   # Using conda:
   conda install pytorch torchvision torchaudio -c pytorch -y
   # Or using pip:
   # pip install torch torchvision torchaudio
4. Install additional dependencies (also on each Mac):
   pip install timm opencv-python pyyaml
5. Verify that MPS is available (you can run this snippet in a Python REPL):
   import torch
   if torch.backends.mps.is_available():
       device = torch.device("mps")
   elif torch.cuda.is_available():
       device = torch.device("cuda")
   else:
       device = torch.device("cpu")
   print("Using device:", device)
This single-Mac setup will later be used as the base for DDP across 3 Macs.

Phase 2 – Jester Dataset Download and Organization

1. Download the Jester dataset parts and annotations from the Qualcomm Jester page.
   You should receive multiple tar-split files (e.g., 20bn-jester-v1-01, 20bn-jester-v1-02, …)
   plus annotation files (train/val/test).
2. On one Mac (the data host), extract the frames into data/jester/rawframes:
   cd data/jester
   cat 20bn-jester-v1-?? | tar zx
   After extraction, you should have directories 1/, 2/, …, 148092/, each containing
   frame images (00001.jpg, 00002.jpg, …).
3. Place the annotation files under data/jester/annotations/, for example:
   data/jester/annotations/train.csv
   data/jester/annotations/val.csv
   data/jester/annotations/test.csv
4. The annotation files should map each sample (folder id) to a gesture label.
   You will convert labels to integer class indices in the dataset code.

Phase 3 – PyTorch Dataset & DataLoader

Goal: each training sample is a short video clip represented as a fixed number of frames
(e.g., T = 16 or 32), plus a gesture class label.
Design decisions:
• T (number of frames per clip): start with 16 or 32.
• Frame size: resize frames to something like 160×160 for faster training on MPS.
• Sequence layout: use either (B, T, C, H, W) or (B, C, T, H, W); pick one convention
  and keep it consistent.
In src/dataset.py, implement a JesterDataset class:
• __init__:
  – Read the train/val/test annotation files.
  – Build a mapping from gesture label string to integer id (0..28).
  – Store the root path to rawframes.
• __getitem__(index):
  – Look up the folder id for this index.
  – List all frame images and select T frames (uniformly spaced or center segment).
  – Load images with PIL or OpenCV, convert to tensors.
  – Apply transforms: resize, crop, random horizontal flip, normalization
    (e.g., ImageNet mean/std if using a pre-trained ViT).
  – Return a tensor of shape (T, C, H, W) and the integer label.
• Wrap the dataset in a DataLoader (in train_ddp.py and eval.py):
  – Choose a batch size (start small, e.g., 4–8 per process for MPS).
  – Use shuffle=True for training, shuffle=False for validation/test.

Phase 4 – Model Architecture: ViT Encoder + SSM Backbone

The model consists of three main parts:
1. Vision encoder (ViT):
   • Use timm to create a pre-trained Vision Transformer (e.g., vit_base_patch16_224).
   • Remove the classification head so that the model outputs a frame embedding vector.
   • For a batch of frames (B, T, C, H, W):
     – Reshape to (B*T, C, H, W), run through the ViT.
     – Reshape back to (B, T, D), where D is the embedding dimension.
2. State Space Model (SSM) backbone:
   • Implement a sequence model that takes (B, T, D) and models temporal dependencies.
   • You can start by using a Transformer encoder or GRU/LSTM block as a placeholder
     and later swap to a true SSM (e.g., Mamba-like block).
   • The SSM outputs either a sequence (B, T, D) or a pooled representation (B, D).
3. Classification head:
   • If the SSM returns a sequence, apply temporal pooling (e.g., mean over T).
   • Apply LayerNorm and a final Linear layer to get logits of shape (B, num_classes).
Files:
• src/models/vit_encoder.py: FrameViTEncoder (wraps timm ViT).
• src/models/ssm_backbone.py: SimpleSSMBackbone (stack of SSM or placeholder blocks).
• src/models/gesture_model.py: combines encoder, SSM, and classifier head.

Phase 5 – Training with PyTorch DDP Across 3 Macs

You will use PyTorch Distributed Data Parallel (DDP) in multi-node mode to train on
3 separate Macs, each using the MPS backend.
1. Shared codebase and environment:
   • Clone the same git repository onto all 3 Macs.
   • Ensure the same Python and library versions on all machines (ideally using the
     same conda environment name and package versions).
2. Networking setup:
   • Make sure the 3 Macs can reach each other over the network (same LAN / pingable).
   • Pick one Mac to act as rank 0 / master (set MASTER_ADDR to that machine's IP).
   • Make sure firewalls allow traffic on the master port (e.g., 29500).
3. DDP environment variables (for each Mac):
   • WORLD_SIZE: total number of processes across all nodes
     – If you use 1 process per Mac, WORLD_SIZE = 3.
   • RANK: unique rank id for each process (0, 1, 2).
   • MASTER_ADDR: IP address or hostname of the rank 0 machine.
   • MASTER_PORT: port used by PyTorch distributed (e.g., 29500).
   Example for node 0 (master):
   • RANK=0
   • WORLD_SIZE=3
   • MASTER_ADDR=<ip_of_node0>
   • MASTER_PORT=29500
   For node 1:
   • RANK=1
   • WORLD_SIZE=3
   • MASTER_ADDR=<ip_of_node0>
   • MASTER_PORT=29500
   For node 2:
   • RANK=2
   • WORLD_SIZE=3
   • MASTER_ADDR=<ip_of_node0>
   • MASTER_PORT=29500
4. DDP launch (torchrun):
   • On each Mac, run the same train_ddp.py script with torchrun.
     For 1 process per node, on each machine do something like:
     torchrun                --nnodes=3                --nproc_per_node=1                --node_rank=<this_node_rank>                --master_addr=<ip_of_node0>                --master_port=29500                src/train_ddp.py --config experiments/config_jester.yaml
   • node_rank is 0 on the master, 1 on the second Mac, and 2 on the third Mac.
5. Inside train_ddp.py:
   • Initialize PyTorch distributed:
     import torch
     import torch.distributed as dist
     from torch.nn.parallel import DistributedDataParallel as DDP
     def setup_distributed():
         dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
   • Pick the device on each Mac:
     – For Apple Silicon, use MPS:
       if torch.backends.mps.is_available():
           device = torch.device("mps")
       else:
           device = torch.device("cpu")
   • Move the model to the device, then wrap it in DDP:
     model.to(device)
     model = DDP(model)
   • Use DistributedSampler for your dataset so that each process sees a different
     subset of data:
     from torch.utils.data.distributed import DistributedSampler
     train_sampler = DistributedSampler(train_dataset)
     train_loader = DataLoader(
         train_dataset,
         batch_size=batch_size,
         sampler=train_sampler,
         num_workers=num_workers,
     )
   • At the start of each epoch, call train_sampler.set_epoch(epoch) so that shuffling
     is consistent across processes.
6. Checkpointing:
   • Only have rank 0 save checkpoints (to avoid conflicts).
   • Save model.module.state_dict() when using DDP.
   • You can either save checkpoints to a shared network folder or copy them from
     rank 0's local disk after training completes.

Phase 6 – Training Loop and Hyperparameters

1. Loss and optimizer:
   • Use CrossEntropyLoss for multi-class gesture classification.
   • Use AdamW or SGD with momentum as the optimizer.
   • Start with a learning rate around 3e-4 (AdamW) or 0.1 (SGD with warmup and decay).
2. Epoch structure:
   • For each epoch:
     – Set the model to train mode.
     – Set train_sampler.set_epoch(epoch) for DDP.
     – Loop over batches from the DataLoader:
       · Move frames and labels to device.
       · Compute logits = model(frames).
       · Compute loss = criterion(logits, labels).
       · Backpropagate loss.backward().
       · Optionally clip gradients.
       · optimizer.step() and optimizer.zero_grad().
3. Validation:
   • After each epoch, switch to eval mode and disable gradient computation.
   • Run over the validation dataset (non-distributed or with another DistributedSampler).
   • Compute top-1 and top-5 accuracy, and log results.
   • Only rank 0 should print or log metrics to avoid clutter.
4. Apple Silicon considerations:
   • Use small batch sizes per process (e.g., 4–8) and adjust based on memory usage.
   • Reduce image resolution (e.g., 160×160) if memory is tight.
   • If MPS has issues with certain operations, you can fall back to CPU for debugging.

Phase 7 – Evaluation, Baselines, and Comparisons

To show the benefit of the ViT + SSM model, implement and compare baselines:
1. Baseline models:
   • 3D CNN (e.g., a ResNet3D-style architecture).
   • ViT with temporal average pooling (no SSM).
   • ViT with GRU/LSTM sequence modeling.
2. Metrics:
   • Top-1 and top-5 accuracy on the Jester validation/test set.
   • Per-class precision, recall, and F1 score.
   • Inference latency (time per video) measured on a single Mac.
3. Evaluation script (eval.py):
   • Load a saved checkpoint (from rank 0).
   • Run inference on validation/test DataLoader.
   • Print and optionally save metrics and confusion matrix plots.

Phase 8 – Documentation and Report

Finally, document the full system and experiments:
1. System description:
   • Overall architecture: Jester dataset → ViT encoder → SSM backbone → classifier.
   • Data pipeline: frame extraction, frame selection, and preprocessing.
   • DDP setup: multi-node Apple Silicon training with PyTorch.
2. Experimental setup:
   • Hyperparameters (learning rate, batch size, number of epochs, sequence length T).
   • Model variants (ViT-only, ViT+GRU, ViT+SSM).
   • Hardware: 3 Apple Silicon Macs, each using MPS.
3. Results:
   • Tables of accuracy and other metrics for each model.
   • Plots of training/validation curves.
   • Discussion of which gestures are hardest/easiest to classify.
   • Strengths and limitations of SSMs vs. RNNs and Transformers on this dataset.
4. Future work:
   • Try larger ViT or deeper SSM backbones.
   • Explore data augmentation and regularization (mixup, CutMix, etc.).
   • Optimize inference for real-time gesture recognition on device.

