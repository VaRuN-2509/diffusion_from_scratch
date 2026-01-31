import torch
import torch.nn as nn
import numpy as np
import math

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32)

        t = t.float()

        if t.dim() == 0:
            t = t.unsqueeze(0)

        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        embeddings = t[:, None] * embeddings[None, :]
        
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
        

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        # The core residual connection: x + f(x)
        return self.activation(x + self.net(x))

class noise_predictor(nn.Module):
    def __init__(self, inp_dim=2, embed_dim=64, hidden_dim=256, num_res_blocks=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # 1. Time Embedding MLP (Projects sinusoidal vector to hidden space)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # 2. Initial Projection (Combined x and time)
        self.input_proj = nn.Linear(inp_dim + hidden_dim, hidden_dim)
        
        # 3. Series of Residual Blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        
        # 4. Final Output Layer
        self.final_proj = nn.Linear(hidden_dim, inp_dim)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, t_emb):
        # x: (batch, 2)
        # t_emb: (batch, embed_dim)
        
        # Project time to hidden dimension
        t_feat = self.time_mlp(t_emb)
        
        # Concatenate and project to hidden space
        h = torch.cat([x, t_feat], dim=-1)
        h = self.input_proj(h)
        
        # Pass through residual blocks
        for block in self.res_blocks:
            h = block(h)
            
        # Predict noise (epsilon)
        return self.final_proj(h)