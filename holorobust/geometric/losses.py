import torch
import torch.nn as nn
import torch.nn.functional as F


class ArakelovLoss(nn.Module):
    """
    Geometric regularizer inspired by Lorentzian Arakelov theory.

    Core idea:
      Arakelov geometry studies arithmetic varieties by combining
      algebraic geometry at finite primes with analysis at the
      archimedean (real/complex) place. In ML terms we enforce:

      1. Height loss       -- penalises latent vectors with large
                             arithmetic 'height' (analogous to the
                             Arakelov height function on a variety).
                             Keeps embeddings in a bounded, well-defined
                             region of the manifold.

      2. Curvature loss    -- penalises large Jacobian norms of the
                             encoder map. High Jacobian norm = the encoder
                             is stretching or folding the manifold sharply,
                             which destabilises training and hurts
                             generalisation. We want a flat, smooth map.

      3. Lorentzian loss   -- inspired by the Lorentzian (indefinite) metric
                             in Arakelov theory. Enforces a light-cone
                             structure in the latent space: pairs of
                             representations should respect a causal
                             ordering rather than being uniformly spread.
                             Useful for encoding temporal or causal
                             structure in physics data.

    Args:
        height_weight     (float): weight on the height penalty.
        curvature_weight  (float): weight on the Jacobian / curvature penalty.
        lorentzian_weight (float): weight on the Lorentzian metric penalty.
        height_scale      (float): scale of the height function denominator.
        time_dim          (int):   which latent dimension is treated as the
                                   'time' coordinate for the Lorentzian metric.
    """

    def __init__(
        self,
        height_weight: float = 0.1,
        curvature_weight: float = 0.05,
        lorentzian_weight: float = 0.05,
        height_scale: float = 1.0,
        time_dim: int = 0,
    ):
        super().__init__()
        self.height_weight = height_weight
        self.curvature_weight = curvature_weight
        self.lorentzian_weight = lorentzian_weight
        self.height_scale = height_scale
        self.time_dim = time_dim

    def height_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Arakelov height function penalty.

        The Arakelov height of a point measures its arithmetic complexity.
        We model this as log(1 + ||z||^2 / scale^2) — a soft, logarithmic
        penalty that grows slowly for moderate norms but strongly for
        very large ones. This is the ML analogue of the Weil height.

        Minimising this encourages the encoder to map inputs to
        low-complexity, well-structured regions of the latent manifold.

        z : (batch, latent_dim)
        """
        norms_sq = torch.sum(z ** 2, dim=-1)               # (batch,)
        height = torch.log1p(norms_sq / (self.height_scale ** 2))
        return torch.mean(height)

    def curvature_loss(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        encoder: nn.Module,
    ) -> torch.Tensor:
        """
        Jacobian-based curvature penalty.

        Computes the Frobenius norm of the encoder Jacobian dz/dx.
        A large Jacobian norm means the encoder is locally stretching
        the input manifold — high curvature, unstable, hard to attack-proof.

        We use a finite-difference approximation to avoid full
        autograd Jacobian computation (which is expensive for large dims).
        A small perturbation eps is added to x; we measure how much
        z changes. This approximates the operator norm of J.

        z       : (batch, latent_dim)   latent from encoder(x)
        x       : (batch, input_dim)    original input
        encoder : nn.Module             the encoder being regularised
        """
        eps = 1e-3
        # Random unit-norm perturbation direction
        delta = torch.randn_like(x)
        delta = delta / (torch.norm(delta, dim=-1, keepdim=True) + 1e-8)

        with torch.no_grad():
            z_perturbed = encoder(x + eps * delta)

        # Finite-difference Jacobian-vector product
        jvp = (z_perturbed - z.detach()) / eps          # (batch, latent_dim)
        jacobian_norm = torch.norm(jvp, dim=-1)          # (batch,)
        return torch.mean(jacobian_norm ** 2)

    def lorentzian_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Lorentzian metric structure in latent space.

        Inspired by the indefinite metric signature (-,+,+,...,+) from
        Lorentzian Arakelov geometry. We designate one latent dimension
        as the 'time' coordinate and enforce a light-cone structure:

        For pairs of latent vectors (z_i, z_j), the Lorentzian inner
        product is:  <z_i, z_j>_L = -t_i*t_j + sum(s_i * s_j)
        where t is the time coordinate and s are spatial coordinates.

        We penalise violations of the causal constraint: pairs that
        should be spacelike (similar time) but have negative Lorentzian
        product are penalised. This encodes physical causality into the
        latent geometry.

        z : (batch, latent_dim)
        """
        # Split into time and space components
        t = z[:, self.time_dim:self.time_dim + 1]       # (batch, 1)
        # All other dims are spatial
        s_dims = list(range(z.shape[-1]))
        s_dims.remove(self.time_dim)
        s = z[:, s_dims]                                 # (batch, latent_dim-1)

        # Compute pairwise Lorentzian inner products (sample pairs in batch)
        # Use first half vs second half of batch for efficiency
        half = z.shape[0] // 2
        if half == 0:
            return torch.tensor(0.0, device=z.device)

        t1, t2 = t[:half], t[half:2*half]
        s1, s2 = s[:half], s[half:2*half]

        lorentzian_ip = -(t1 * t2) + torch.sum(s1 * s2, dim=-1, keepdim=True)

        # Penalise pairs with positive Lorentzian product that have
        # similar time coordinates (should be spacelike = negative product)
        delta_t = torch.abs(t1 - t2)
        spacelike_mask = (delta_t < 0.5).float()
        violation = F.relu(lorentzian_ip) * spacelike_mask
        return torch.mean(violation ** 2)

    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        encoder: nn.Module,
    ) -> dict:
        """
        Compute all Arakelov geometric penalties.

        Returns a dict so the trainer can log each term separately.

        z       : (batch, latent_dim)
        x       : (batch, input_dim)
        encoder : nn.Module  — the encoder being regularised
        """
        h_loss = self.height_loss(z)
        c_loss = self.curvature_loss(z, x, encoder)
        l_loss = self.lorentzian_loss(z)

        total = (
            self.height_weight * h_loss
            + self.curvature_weight * c_loss
            + self.lorentzian_weight * l_loss
        )

        return {
            "arakelov_total": total,
            "height":         h_loss,
            "curvature":      c_loss,
            "lorentzian":     l_loss,
        }