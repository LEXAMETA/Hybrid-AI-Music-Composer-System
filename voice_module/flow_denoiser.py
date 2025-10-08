
import torch
import torch.nn as nn

class FlowDenoiser(nn.Module):
    """
    Toy continuous latent generator and denoiser for flow-based models.
    Uses simple MLP with SiLU activations and conditioning on timestep t ∈ [0,1].
    """
    def __init__(self, dim, hidden=512):
        """
        Args:
            dim (int): Dimensionality of input latent vector.
            hidden (int): Number of units in hidden layers.
        """
        super(FlowDenoiser, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x_t, t):
        """
        Args:
            x_t (torch.Tensor): Input latent tensor, shape (batch_size, dim)
            t (torch.Tensor): Continuous time conditioning scalar, shape (batch_size,)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, dim)
        """
        t = t.view(-1, 1)  # reshape for concat
        inp = torch.cat([x_t, t], dim=1)
        out = self.net(inp)
        return out
