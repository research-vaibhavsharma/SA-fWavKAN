import torch
import torch.nn as nn
from .ssm_gating import SelectiveKANGating
from .selective_scan import HardwareAwareRollout

class SAfWavKANLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, num_atoms: int = 16):
        """
        A single SA-fWavKAN architectural block representing the full pipeline.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Normalization layer
        self.norm = nn.LayerNorm(d_model)
        
        # Sequence Pipeline
        self.ssm_gating = SelectiveKANGating(d_model, d_state, num_atoms)
        self.temporal_rollout = HardwareAwareRollout()
        
        # Output projection
        self.out_proj = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, d_model)
        """
        residual = x
        x_norm = self.norm(x)
        
        # 1 & 2. Global Dictionary & Selective Gating / ZOH Discretization
        A_bar, B_bar, C_t = self.ssm_gating(x_norm)
        
        # 3. Hardware-Aware Temporal Rollout
        y_seq = self.temporal_rollout(A_bar, B_bar, C_t, x_norm)
        
        # Re-project to model dimension and add residual connection
        y_seq = y_seq.unsqueeze(-1) 
        out = self.out_proj(y_seq) + residual
        
        return out


class SAfWavKANSequenceModel(nn.Module):
    def __init__(self, d_model: int, num_layers: int, output_dim: int, d_state: int = 16, num_atoms: int = 16):
        """
        End-to-End Deep Sequence Model for Complex Chaotic Forecasting.
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            SAfWavKANLayer(d_model, d_state, num_atoms) 
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input sequence through multiple stacked SA-fWavKAN layers.
        """
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_norm(x)
        predictions = self.head(x)
        
        return predictions
      
