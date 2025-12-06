"""State Space Model (SSM) backbone for temporal sequence modeling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SSMBlock(nn.Module):
    """Mamba-like State Space Model block for sequence modeling.
    
    Implements a simplified SSM with learnable state space parameters
    and selective scan mechanism for efficient temporal modeling.
    """
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto"):
        """Initialize SSM block.
        
        Args:
            d_model: Model dimension (input/output dimension)
            d_state: State dimension for SSM
            d_conv: Convolution kernel size for selective scan
            expand: Expansion factor for hidden dimension
            dt_rank: Rank for delta parameter (default: "auto" = d_model // 16)
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        d_inner = self.d_inner
        
        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = dt_rank
        
        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        
        # Convolution for selective scan
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=d_inner,
            padding=d_conv - 1,
        )
        
        # Activation
        self.activation = "silu"
        self.act = nn.SiLU()
        
        # State space parameters
        self.x_proj = nn.Linear(d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        
        # Initialize A (state transition matrix) as negative values
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, D)
            
        Returns:
            output: Output tensor of shape (B, T, D)
        """
        B, T, D = x.shape
        residual = x
        x = self.norm(x)
        
        # Input projection
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (B, T, d_inner)
        
        # Convolution (selective scan preparation)
        x = x.transpose(1, 2)  # (B, d_inner, T)
        x = self.conv1d(x)[:, :, :T]  # (B, d_inner, T)
        x = x.transpose(1, 2)  # (B, T, d_inner)
        
        # Activation
        x = self.act(x)
        
        # State space parameters
        x_dbl = self.x_proj(x)  # (B, T, dt_rank + 2*d_state)
        dt, B_param, C_param = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)  # (B, T, d_inner)
        
        # Get A matrix
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Simplified selective scan (discretization)
        # For efficiency, we use a simplified version
        dt = F.softplus(dt)
        dA = torch.einsum("btd,nd->btdn", dt, A)  # (B, T, d_inner, d_state)
        dB = torch.einsum("btd,btn->btdn", dt, B_param)  # (B, T, d_inner, d_state)
        
        # Sequential scan (simplified)
        # In practice, this can be optimized with parallel scan
        y = torch.zeros(B, T, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        for t in range(T):
            h = h * torch.exp(dA[:, t]) + x[:, t:t+1, :, None] * dB[:, t]
            y[:, t] = torch.sum(h * C_param[:, t:t+1, None, :], dim=-1)
        
        y = y.sum(dim=-1)  # (B, T, d_inner)
        
        # Gating
        y = y * self.act(z)
        
        # Output projection
        output = self.out_proj(y)  # (B, T, D)
        
        # Residual connection
        output = output + residual
        
        return output


class SimpleSSMBackbone(nn.Module):
    """Stack of SSM blocks for temporal sequence modeling.
    
    Takes frame embeddings and models temporal dependencies.
    """
    
    def __init__(self, d_model, num_layers=4, d_state=16, d_conv=4, expand=2, pool_output=True):
        """Initialize SSM backbone.
        
        Args:
            d_model: Model dimension (should match ViT embedding dimension)
            num_layers: Number of SSM blocks to stack
            d_state: State dimension for each SSM block
            d_conv: Convolution kernel size
            expand: Expansion factor
            pool_output: If True, return pooled (B, D) output; else return (B, T, D)
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.pool_output = pool_output
        
        # Stack of SSM blocks
        self.blocks = nn.ModuleList([
            SSMBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, D)
            
        Returns:
            output: If pool_output=True: (B, D), else: (B, T, D)
        """
        # Pass through SSM blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Temporal pooling if requested
        if self.pool_output:
            # Mean pooling over time dimension
            x = x.mean(dim=1)  # (B, D)
        
        return x

