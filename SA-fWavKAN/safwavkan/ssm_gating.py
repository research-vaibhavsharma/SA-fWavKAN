import torch
import torch.nn as nn
from .dictionary import GlobalFractionalWaveletDictionary

class SelectiveKANGating(nn.Module):
    def __init__(self, d_model: int, d_state: int, num_atoms: int = 16):
        """
        Continuous Selective State Space with KAN Gating.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # HiPPO Initialization for continuous state matrix A
        A = self._init_hippo(d_state)
        self.A_log = nn.Parameter(torch.log(A)) # Parameterized in log space for stability
        
        # SA-fWavKAN projections for B, C, and Delta
        self.kan_B = GlobalFractionalWaveletDictionary(num_atoms, d_model, d_state)
        self.kan_C = GlobalFractionalWaveletDictionary(num_atoms, d_model, d_state)
        self.kan_Delta = GlobalFractionalWaveletDictionary(num_atoms, d_model, 1)

    def _init_hippo(self, n: int) -> torch.Tensor:
        """Constructs the HiPPO matrix for stable continuous memory."""
        A = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1):
                A[i, j] = (2 * j + 1) ** 0.5 * (2 * i + 1) ** 0.5
                if i == j:
                    A[i, j] -= 1
        return A

    def forward(self, x: torch.Tensor):
        """
        Executes gating and ZOH discretization.
        """
        # 1. Selective KAN Gating
        B_t = self.kan_B(x)                    # Shape: (B, L, d_state)
        C_t = self.kan_C(x)                    # Shape: (B, L, d_state)
        Delta_t = F.softplus(self.kan_Delta(x)) # Shape: (B, L, 1)
        
        A = torch.exp(self.A_log)
        
        # 2. ZOH Discretization
        # A_bar = exp(Delta * A)
        A_bar = torch.exp(Delta_t * A) 
        
        # B_bar = (Delta * A)^-1 * (exp(Delta * A) - I) * Delta * B_t
        # Using Taylor approximation for numerical stability when Delta * A is small
        B_bar = Delta_t * B_t 
        
        return A_bar, B_bar, C_t
