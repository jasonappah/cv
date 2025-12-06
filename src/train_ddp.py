"""Distributed training script for Jester gesture recognition using PyTorch DDP (Windows CPU Version)."""

import os
# CRITICAL: Set USE_LIBUV=0 BEFORE importing torch to prevent libuv issues
if os.name == 'nt':  # Windows
    os.environ['USE_LIBUV'] = '0'

import argparse
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging
import sys
import socket
from pathlib import Path

# Add src to path to handle imports correctly
sys.path.insert(0, str(Path(__file__).parent))

from dataset import JesterDataset
from models.gesture_model import GestureModel
from utils import calculate_accuracy, save_checkpoint, setup_logging

def get_local_ip():
    """Get the local IP address that can reach the master."""
    try:
        # Try to connect to a remote address to determine local IP
        # This doesn't actually connect, just determines which interface would be used
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to a public DNS (doesn't actually send data)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        # Fallback: try to get hostname IP
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            return local_ip
        except Exception:
            return "127.0.0.1"

def get_interface_name_from_ip(ip_address):
    """Get Windows network interface name from IP address."""
    if os.name != 'nt':  # Only for Windows
        return None
    
    try:
        import subprocess
        # Use PowerShell to get interface name
        ps_cmd = f'(Get-NetIPAddress -IPAddress {ip_address}).InterfaceAlias'
        result = subprocess.run(
            ['powershell', '-Command', ps_cmd],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            interface_name = result.stdout.strip()
            return interface_name
    except Exception as e:
        print(f"Could not get interface name from IP: {e}")
    
    return None

def setup_distributed():
    """Initialize distributed training for Windows CPU."""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # FORCE Gloo for Windows CPU Distributed Training
    backend = 'gloo'
    
    # Get master address first (needed for interface detection)
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    # Windows-specific Gloo configuration - MUST be set before any torch.distributed calls
    if os.name == 'nt':  # Windows
        # CRITICAL: Set USE_LIBUV=0 FIRST, before any distributed operations
        os.environ['USE_LIBUV'] = '0'
        
        # Set default values if not already set
        if 'GLOO_SOCKET_FAMILY' not in os.environ:
            os.environ['GLOO_SOCKET_FAMILY'] = 'INET'
        
        # Get local IP address for diagnostics and interface detection
        local_ip = get_local_ip()
        print(f"Detected local IP: {local_ip}")
        
        # CRITICAL: Set GLOO_SOCKET_IFNAME to interface name (not IP, not hostname)
        # This prevents Gloo from trying to resolve hostnames
        ifname = os.environ.get('GLOO_SOCKET_IFNAME', '')
        
        if ifname:
            if '.' in ifname:  # Looks like an IP address - remove it
                print(f"WARNING: Removing GLOO_SOCKET_IFNAME (was set to IP: {ifname})")
                del os.environ['GLOO_SOCKET_IFNAME']
                ifname = None
        else:
            # Try to auto-detect interface name from local IP
            ifname = get_interface_name_from_ip(local_ip)
            if ifname:
                os.environ['GLOO_SOCKET_IFNAME'] = ifname
                print(f"Auto-detected interface name: {ifname}")
            else:
                print("WARNING: Could not auto-detect interface name. Gloo may try hostname resolution.")
        
        if ifname:
            print(f"Using GLOO_SOCKET_IFNAME: {ifname}")
    
    # Validate master address
    if master_addr == '0.0.0.0':
        raise ValueError(
            "MASTER_ADDR cannot be '0.0.0.0'. "
            "Set it to the actual IP address of the master device (e.g., 172.20.10.2). "
            "On the master device, use its own IP address. "
            "On worker devices, use the master's IP address."
        )
    
    # Ensure MASTER_ADDR is an IP address, not a hostname
    # This prevents "makeDeviceForHostname" errors
    try:
        # Validate it's a valid IP format
        socket.inet_aton(master_addr)
        # It's a valid IP, good
    except socket.error:
        # It might be a hostname, try to resolve it and use the IP
        try:
            resolved_ip = socket.gethostbyname(master_addr)
            print(f"WARNING: MASTER_ADDR '{master_addr}' is a hostname, resolved to '{resolved_ip}'. "
                  f"Using resolved IP to avoid hostname resolution issues.")
            master_addr = resolved_ip  # Use the resolved IP
        except socket.gaierror:
            raise ValueError(
                f"MASTER_ADDR '{master_addr}' is neither a valid IP nor resolvable hostname. "
                f"Please set it to an IP address (e.g., 172.20.10.2)."
            )
    
    # Construct init_method - use IP directly, not hostname
    init_method = f'tcp://{master_addr}:{master_port}'
    
    # Print diagnostic information
    print(f"Rank {rank}: Initializing distributed training...")
    print(f"  Backend: {backend}")
    print(f"  MASTER_ADDR: {master_addr} (using IP directly)")
    print(f"  MASTER_PORT: {master_port}")
    print(f"  WORLD_SIZE: {world_size}")
    print(f"  USE_LIBUV: {os.environ.get('USE_LIBUV', 'NOT SET')}")
    print(f"  GLOO_SOCKET_FAMILY: {os.environ.get('GLOO_SOCKET_FAMILY', 'NOT SET')}")
    print(f"  GLOO_SOCKET_IFNAME: {os.environ.get('GLOO_SOCKET_IFNAME', 'NOT SET (auto-detect)')}")
    
    # Use init_method directly - more reliable for Gloo on Windows
    # Skip TCPStore to avoid libuv issues
    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.default_pg_timeout
        )
        print(f"Rank {rank}: Successfully initialized process group!")
    except Exception as e:
        error_msg = f"Failed to initialize process group: {e}\n"
        error_msg += f"  MASTER_ADDR: {master_addr}, MASTER_PORT: {master_port}\n"
        error_msg += f"  USE_LIBUV: {os.environ.get('USE_LIBUV', 'NOT SET')}\n"
        error_msg += f"  GLOO_SOCKET_FAMILY: {os.environ.get('GLOO_SOCKET_FAMILY', 'NOT SET')}\n"
        error_msg += f"  GLOO_SOCKET_IFNAME: {os.environ.get('GLOO_SOCKET_IFNAME', 'NOT SET')}\n"
        error_msg += "\nTroubleshooting tips:\n"
        error_msg += "  1. Ensure MASTER_ADDR is the actual IP of the master device (not 0.0.0.0, not hostname)\n"
        error_msg += "  2. Remove GLOO_SOCKET_IFNAME completely: Remove-Item Env:\\GLOO_SOCKET_IFNAME\n"
        error_msg += "  3. Ensure USE_LIBUV=0 is set before running the script\n"
        error_msg += "  4. Ensure both devices can ping each other using IP addresses\n"
        error_msg += "  5. Check Windows Firewall allows traffic on port 29500\n"
        error_msg += "  6. Try: ping <master_ip> from worker device\n"
        print(error_msg)
        raise
    
    return rank, world_size, local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, rank):
    model.train()
    running_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total_samples = 0
    
    for batch_idx, (frames, labels) in enumerate(train_loader):
        frames = frames.to(device)
        labels = labels.to(device)
        
        logits = model(frames)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        
        acc1, acc5 = calculate_accuracy(logits, labels, topk=(1, 5))
        top1_correct += acc1 * batch_size / 100.0
        top5_correct += acc5 * batch_size / 100.0
        
        if rank == 0 and batch_idx % 10 == 0:
            logging.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {acc1:.2f}%")
    
    avg_loss = running_loss / total_samples
    avg_top1 = top1_correct / total_samples * 100.0
    avg_top5 = top5_correct / total_samples * 100.0
    return avg_loss, avg_top1, avg_top5

def validate(model, val_loader, criterion, device, rank):
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
    
    return running_loss / total_samples, top1_correct / total_samples * 100.0, top5_correct / total_samples * 100.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    rank, world_size, local_rank = setup_distributed()
    setup_logging(log_dir=config['paths']['log_dir'], rank=rank)
    
    # FORCE CPU
    device = torch.device("cpu")
    if rank == 0:
        logging.info(f"Using device: {device} (Forced for CPU Testing)")
        logging.info(f"World size: {world_size}, Rank: {rank}, Backend: gloo")
    
    # Data
    data_cfg = config['data']
    # NOTE: Ensure you have at least a dummy folder structure for this to work!
    train_dataset = JesterDataset(
        root_dir=data_cfg['root_dir'],
        annotation_file=os.path.join(data_cfg['root_dir'], 'annotations', 'train.csv'),
        num_frames=data_cfg['num_frames'],
        frame_size=data_cfg['frame_size'],
        split='train'
    )
    val_dataset = JesterDataset(
        root_dir=data_cfg['root_dir'],
        annotation_file=os.path.join(data_cfg['root_dir'], 'annotations', 'val.csv'),
        num_frames=data_cfg['num_frames'],
        frame_size=data_cfg['frame_size'],
        split='val'
    )
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=data_cfg['batch_size'], sampler=train_sampler, 
                              num_workers=data_cfg['num_workers'], pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=data_cfg['batch_size'], sampler=val_sampler, 
                            num_workers=data_cfg['num_workers'], pin_memory=False)
    
    # Model
    m_cfg = config['model']
    model = GestureModel(
        num_classes=m_cfg['num_classes'],
        vit_model=m_cfg['vit_model'],
        vit_pretrained=m_cfg['vit_pretrained'],
        ssm_layers=m_cfg['ssm_layers'],
        ssm_d_state=m_cfg['ssm_d_state'],
        ssm_d_conv=m_cfg['ssm_d_conv'],
        ssm_expand=m_cfg['ssm_expand']
    )
    model = model.to(device)
    
    # DDP Wrapper (CPU mode requires no device_ids)
    model = DDP(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    for epoch in range(config['training']['epochs']):
        train_sampler.set_epoch(epoch)
        if rank == 0: logging.info(f"\nEpoch {epoch+1}")
        
        t_loss, t_acc1, t_acc5 = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1, rank)
        v_loss, v_acc1, v_acc5 = validate(model, val_loader, criterion, device, rank)
        
        if rank == 0:
            logging.info(f"Train - Loss: {t_loss:.4f}, Acc: {t_acc1:.2f}%")
            logging.info(f"Val   - Loss: {v_loss:.4f}, Acc: {v_acc1:.2f}%")
            
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, config['paths']['checkpoint_dir'], f'checkpoint_{epoch+1}.pth.tar')
            
    cleanup_distributed()

if __name__ == '__main__':
    main()