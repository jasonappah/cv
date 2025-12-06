import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SSMBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, 
                                groups=self.d_inner, padding=d_conv - 1)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        residual = x
        x = self.norm(x)
        x, z = self.in_proj(x).chunk(2, dim=-1)
        
        # Conv
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :T]
        x = x.transpose(1, 2)
        x = self.act(x)
        
        # SSM Parameters
        x_dbl = self.x_proj(x)
        dt, B_param, C_param = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        dt = F.softplus(dt)
        
        A = -torch.exp(self.A_log.float())
        dA = torch.einsum("btd,nd->btdn", dt, A)
        dB = torch.einsum("btd,btn->btdn", dt, B_param)
        
        # Simplified Scan (CPU Friendly Loop)
        y = torch.zeros(B, T, self.d_inner, self.d_state, device=x.device)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        
        for t in range(T):
            h = h * torch.exp(dA[:, t]) + x[:, t:t+1, :, None] * dB[:, t]
            y[:, t] = torch.sum(h * C_param[:, t:t+1, None, :], dim=-1)
            
        y = y.sum(dim=-1)
        y = y * self.act(z)
        return self.out_proj(y) + residual

class SimpleSSMBackbone(nn.Module):
    def __init__(self, d_model, num_layers=4, d_state=16, d_conv=4, expand=2, pool_output=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            SSMBlock(d_model, d_state, d_conv, expand) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pool_output = pool_output

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x.mean(dim=1) if self.pool_output else x