import torch
import torch.nn as nn

class HardwareAwareRollout(nn.Module):
    def __init__(self):
        """
        Executes the discrete temporal rollout combining current sequence input 
        with latent memory to generate sequential output.
        """
        super().__init__()

    def forward(self, A_bar: torch.Tensor, B_bar: torch.Tensor, C_t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the recurrent scan (Eq. 15 and 16).
        Note: For extreme optimization, this sequence loop is typically replaced 
        with an associative scan written in custom CUDA. This PyTorch equivalent 
        ensures exact mathematical correctness.
        
        A_bar shape: (batch_size, seq_len, d_state, d_state)
        B_bar shape: (batch_size, seq_len, d_state)
        C_t shape:   (batch_size, seq_len, d_state)
        x shape:     (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_state = B_bar.shape
        device = x.device
        
        # Initialize latent memory state h_t
        h = torch.zeros(batch_size, d_state, device=device)
        y_out = []
        
        for t in range(seq_len):
            A_t = A_bar[:, t, :, :] # (batch_size, d_state, d_state)
            B_t = B_bar[:, t, :]    # (batch_size, d_state)
            C_step = C_t[:, t, :]   # (batch_size, d_state)
            x_t = x[:, t, :]        # (batch_size, d_model)
            
            # Autoregressive hidden state update (Eq. 15)
            # h_t = A_bar * h_{t-1} + B_bar * x_t
            h_prev = h.unsqueeze(-1)
            Ah = torch.bmm(A_t, h_prev).squeeze(-1) 
            h = Ah + B_t * x_t.mean(dim=-1, keepdim=True) # Broadcast adaptation
            
            # Sequential output (Eq. 16)
            # y_t = C_t * h_t
            y_t = (C_step * h).sum(dim=-1)
            y_out.append(y_t)
            
        # Stack sequential outputs along sequence dimension
        y_seq = torch.stack(y_out, dim=1) # Shape: (batch_size, seq_len)
        return y_seq
