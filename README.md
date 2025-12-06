# Jester Gesture Recognition with ViT + SSM

A PyTorch implementation of gesture recognition on the Jester dataset using a Vision Transformer (ViT) encoder with a State Space Model (SSM) backbone. The system supports distributed training across multiple Apple Silicon Macs using PyTorch Distributed Data Parallel (DDP).

## Project Overview

This project implements a complete gesture recognition pipeline that:
- Uses a pre-trained Vision Transformer (ViT) to extract frame-level features
- Models temporal dependencies using a Mamba-like State Space Model (SSM)
- Trains on the Jester dataset with 27 gesture classes
- Supports distributed training across 3 Apple Silicon Macs using PyTorch DDP

## Architecture

The model consists of three main components:

1. **ViT Encoder** (`src/models/vit_encoder.py`): 
   - Uses `vit_base_patch16_224` from timm
   - Processes each frame independently
   - Outputs frame embeddings of dimension 768

2. **SSM Backbone** (`src/models/ssm_backbone.py`):
   - Mamba-like State Space Model blocks
   - Models temporal dependencies across frames
   - Configurable depth and state dimensions

3. **Classifier** (`src/models/gesture_model.py`):
   - Temporal pooling (mean over time)
   - LayerNorm + Linear layer
   - Outputs logits for 27 gesture classes

## Installation

### Prerequisites

- Python 3.11
- Miniforge or Miniconda
- Apple Silicon Mac (M-series) for MPS support

### Setup

1. **Create conda environment** (run on each Mac):
   ```bash
   conda create -n jester_env python=3.11 -y
   conda activate jester_env
   ```

2. **Install PyTorch with MPS support**:
   ```bash
   conda install pytorch torchvision torchaudio -c pytorch -y
   ```

3. **Install additional dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify MPS availability**:
   ```python
   import torch
   if torch.backends.mps.is_available():
       print("MPS is available!")
   ```

## Dataset Preparation

1. **Download the Jester dataset** from the [Qualcomm Jester page](https://20bn.com/datasets/jester)

2. **Extract frames** (on one Mac, the data host):
   ```bash
   cd data/jester
   cat 20bn-jester-v1-?? | tar zx
   ```
   This will create directories `1/`, `2/`, ..., `148092/`, each containing frame images.

3. **Place annotation files** in `data/jester/annotations/`:
   - `train.csv`
   - `val.csv`
   - `test.csv`
   
   Each CSV file should have two columns: folder_id and gesture_label.

## Project Structure

```
jester-ssm-gesture/
├── data/
│   └── jester/
│       ├── rawframes/        # Extracted frame folders (1..148092)
│       └── annotations/      # train.csv, val.csv, test.csv
├── src/
│   ├── dataset.py            # JesterDataset class
│   ├── models/
│   │   ├── vit_encoder.py    # FrameViTEncoder
│   │   ├── ssm_backbone.py   # SSM implementation
│   │   └── gesture_model.py  # Complete model
│   ├── train_ddp.py          # DDP training script
│   ├── eval.py               # Evaluation script
│   └── utils.py              # Utility functions
├── experiments/
│   └── config_jester.yaml    # Training configuration
├── requirements.txt
└── README.md
```

## Usage

### Single Mac Training

For testing on a single Mac:

```bash
python src/train_ddp.py --config experiments/config_jester.yaml
```

Note: This will still use DDP with a single process. For true single-process training, you can modify the script to skip DDP initialization when `WORLD_SIZE=1`.

### Multi-Mac Distributed Training

To train across 3 Macs using PyTorch DDP:

1. **Ensure all Macs can communicate**:
   - All Macs should be on the same network
   - Firewalls should allow traffic on port 29500
   - Test connectivity: `ping <master_ip>`

2. **Clone the repository** on all 3 Macs and ensure the same environment setup.

3. **Set environment variables** on each Mac:

   **On Mac 0 (master, rank 0)**:
   ```bash
   export RANK=0
   export WORLD_SIZE=3
   export MASTER_ADDR=<ip_of_mac0>
   export MASTER_PORT=29500
   ```

   **On Mac 1 (rank 1)**:
   ```bash
   export RANK=1
   export WORLD_SIZE=3
   export MASTER_ADDR=<ip_of_mac0>
   export MASTER_PORT=29500
   ```

   **On Mac 2 (rank 2)**:
   ```bash
   export RANK=2
   export WORLD_SIZE=3
   export MASTER_ADDR=<ip_of_mac0>
   export MASTER_PORT=29500
   ```

4. **Launch training** on each Mac simultaneously:
   ```bash
   torchrun \
       --nnodes=3 \
       --nproc_per_node=1 \
       --node_rank=<0, 1, or 2> \
       --master_addr=<ip_of_mac0> \
       --master_port=29500 \
       src/train_ddp.py --config experiments/config_jester.yaml
   ```

   Or use the environment variables:
   ```bash
   torchrun \
       --nnodes=3 \
       --nproc_per_node=1 \
       src/train_ddp.py --config experiments/config_jester.yaml
   ```

### Evaluation

Evaluate a trained model:

```bash
python src/eval.py \
    --config experiments/config_jester.yaml \
    --checkpoint checkpoints/model_best.pth.tar \
    --split val \
    --save-cm \
    --output-dir results
```

Options:
- `--split`: Dataset split to evaluate on (`train`, `val`, or `test`)
- `--save-cm`: Save confusion matrix plot
- `--output-dir`: Directory to save results

## Configuration

Edit `experiments/config_jester.yaml` to customize:

- **Model**: ViT variant, SSM layers, state dimensions
- **Data**: Number of frames, frame size, batch size
- **Training**: Learning rate, epochs, optimizer settings
- **Paths**: Checkpoint and log directories

## Model Details

### Input Format
- Frames: (B, T, C, H, W) where:
  - B = batch size
  - T = 16 frames per clip
  - C = 3 (RGB)
  - H, W = 160×160 pixels

### Output Format
- Logits: (B, 27) for 27 gesture classes

### Key Hyperparameters
- Frames per clip: 16
- Frame resolution: 160×160
- ViT model: `vit_base_patch16_224` (768-dim embeddings)
- SSM layers: 4
- SSM state dimension: 16
- Batch size: 4 per process (adjust for memory)

## Notes

- **MPS Limitations**: PyTorch DDP with MPS backend may have limitations. The code uses 'gloo' backend which is compatible with MPS/CPU.
- **Memory**: Start with small batch sizes (4-8) and adjust based on available memory.
- **Data**: The dataset download/extraction is manual. The code expects the data structure as described.
- **Checkpoints**: Only rank 0 saves checkpoints to avoid conflicts. Checkpoints are saved in `checkpoints/` directory.

## Results

After training, you can:
- View training logs in `logs/training.log` (rank 0 only)
- Find best model in `checkpoints/model_best.pth.tar`
- Evaluate and generate confusion matrices using `src/eval.py`

## Future Work

- Implement baseline models (3D CNN, ViT-only, ViT+GRU) for comparison
- Optimize SSM implementation with parallel scan
- Add data augmentation (mixup, CutMix)
- Optimize inference for real-time gesture recognition

## License

[Add your license here]

## Citation

If you use this code, please cite:
- Jester Dataset: [20BN-Jester Dataset](https://20bn.com/datasets/jester)
- Vision Transformer: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- Mamba/SSM: [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
