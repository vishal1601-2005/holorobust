import torch
import torch.nn as nn
from typing import Tuple, Optional


class HoloRobustModel(nn.Module):

    def __init__(self, encoder=None, decoder=None, input_dim=64, latent_dim=16, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder = encoder if encoder is not None else self._build_encoder()
        self.decoder = decoder if decoder is not None else self._build_decoder()

    def _build_encoder(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.latent_dim),
        )

    def _build_decoder(self):
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def latent(self, x):
        with torch.no_grad():
            return self.encode(x)

    def anomaly_score(self, x):
        device = next(self.parameters()).device
        x = x.to(device)
        self.eval()
        with torch.no_grad():
            x_hat, _ = self.forward(x)
            score = torch.mean((x - x_hat) ** 2, dim=-1)
        return score

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (f"{self.__class__.__name__}(input_dim={self.input_dim}, "
                f"latent_dim={self.latent_dim}, params={self.count_parameters():,})")
