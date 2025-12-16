import torch
from torch.nn import Linear, Sequential, Sigmoid
import numpy as np


def create_kuhn_player(device='cpu'):
    """
    - Input: 12-dimensional one-hot state vector
    - Output: 1-dimensional probability (bet/call action)
    """
    return Sequential(
        Linear(12, 1, bias = False),
        Sigmoid()
    ).to(device)


class OptimalBotP2:
    """
    Hardcoded Nash equilibrium strategy for Player 2
    """
    def __init__(self, device='cpu', alpha=0):
        self.device = device
        self.alpha = alpha  # P1's bluffing frequency (0 to 1/3)
    
    def __call__(self, state_tensor):
        state = state_tensor.cpu().numpy() if isinstance(state_tensor, torch.Tensor) else state_tensor
        
        # Find which state is active (one-hot encoded)
        state_idx = np.argmax(state)
        
        # Determine action probability based on Nash equilibrium
        if 3 <= state_idx <= 5:
            # After P1 pass: P2 decides to bet or check
            card = state_idx - 3  # 0=Jack, 1=Queen, 2=King
            if card == 0:  # Jack: Bet with prob 1/3 (bluff)
                bet_prob = 1/3
            elif card == 1:  # Queen: Never bet
                bet_prob = 0.0
            elif card == 2:  # King: Always bet
                bet_prob = 1.0
            else:
                raise ValueError
        elif 9 <= state_idx <= 11:
            # After P1 bet: P2 decides to call or fold
            card = state_idx - 9  # 0=Jack, 1=Queen, 2=King
            if card == 0:  # Jack: Never call (always fold)
                bet_prob = 0.0
            elif card == 1:  # Queen: Call with prob (alpha + 1/3)
                bet_prob = self.alpha + 1/3
            elif card == 2:  # King: Always call
                bet_prob = 1.0
            else:
                raise ValueError
        else:
            # Other states .. shouldnt happen for P2's decisions
            raise ValueError
        
        return torch.tensor([bet_prob], dtype=torch.float32, device=self.device)
