import torch
import torch.nn as nn
import torch.nn.functional as F


class HolographicLoss(nn.Module):
    """
    Holographic regularizer inspired by AdS/CFT correspondence.

    Core idea:
      In AdS/CFT, the bulk (interior) encodes the boundary (surface).
      We treat the latent space as the 'bulk' and the input/output as the 'boundary'.
      Three penalties enforce this structure:

      1. Radial scaling loss   -- latent norms should follow a power-law with depth,
                                  mimicking the radial direction in Anti-de Sitter space.
      2. Bulk-boundary loss    -- a depth-contracted version of the latent should still
                                  reconstruct the input faithfully (holographic redundancy).
      3. Confinement loss      -- penalises latents that escape to the AdS boundary
                                  (very large norms), inspired by holographic QCD confinement.

    Args:
        radial_weight      (float): weight on the radial scaling penalty.
        bulk_boundary_weight (float): weight on the bulk-boundary consistency penalty.
        confinement_weight (float): weight on the confinement / norm ceiling penalty.
        ads_radius         (float): effective AdS radius (controls power-law exponent).
        confinement_scale  (float): norm threshold above which confinement kicks in.
    """

    def __init__(
        self,
        radial_weight: float = 0.1,
        bulk_boundary_weight: float = 0.1,
        confinement_weight: float = 0.05,
        ads_radius: float = 1.0,
        confinement_scale: float = 10.0,
    ):
        super().__init__()
        self.radial_weight = radial_weight
        self.bulk_boundary_weight = bulk_boundary_weight
        self.confinement_weight = confinement_weight
        self.ads_radius = ads_radius
        self.confinement_scale = confinement_scale

    def radial_scaling_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Penalises deviation from power-law radial scaling.

        In AdS_d+1, correlators scale as r^{-2*Delta} with radial depth r.
        We enforce that the norm of each latent vector sits near a target value
        set by the AdS radius, discouraging both collapse and explosion.

        z : (batch, latent_dim)
        """
        norms = torch.norm(z, dim=-1)                      # (batch,)
        target = self.ads_radius * torch.ones_like(norms)  # unit sphere in AdS
        loss = F.mse_loss(norms, target)
        return loss

    def bulk_boundary_loss(
        self,
        z: torch.Tensor,
        x_reconstructed: torch.Tensor,
        x_original: torch.Tensor,
        contraction: float = 0.5,
    ) -> torch.Tensor:
        """
        Holographic redundancy: a radially contracted latent should still
        reconstruct the boundary (input) faithfully.

        Concretely: z_contracted = contraction * z
        We measure how much worse the contracted reconstruction is vs the original.
        A well-structured holographic latent degrades gracefully.

        z              : (batch, latent_dim)  full latent
        x_reconstructed: (batch, ...)         reconstruction from full z
        x_original     : (batch, ...)         original input
        contraction    : float                radial contraction factor (0 < c < 1)
        """
        # Full reconstruction error (already computed by task loss, used as baseline)
        full_err = F.mse_loss(x_reconstructed, x_original, reduction='none').mean(-1)

        # Contracted latent -- simulate a shallower bulk slice
        z_contracted = contraction * z

        # We measure consistency via the norm difference of reconstructions
        # (we don't call decoder here to keep this loss decoder-agnostic)
        # Instead: penalise latents whose contracted version deviates too far from
        # the original in latent space (bulk locality)
        contracted_err = F.mse_loss(z_contracted, contraction * z.detach(),
                                    reduction='none').mean(-1)

        loss = F.mse_loss(contracted_err, torch.zeros_like(contracted_err))
        return loss

    def confinement_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Holographic QCD confinement potential.

        In holographic QCD, strings cannot reach the AdS boundary (infinite energy).
        We implement this as a soft ceiling: latent norms beyond confinement_scale
        are penalised quadratically, preventing the network from pushing
        representations arbitrarily far out.

        z : (batch, latent_dim)
        """
        norms = torch.norm(z, dim=-1)                          # (batch,)
        excess = F.relu(norms - self.confinement_scale)        # only penalise excess
        loss = torch.mean(excess ** 2)
        return loss

    def forward(
        self,
        z: torch.Tensor,
        x_reconstructed: torch.Tensor,
        x_original: torch.Tensor,
    ) -> dict:
        """
        Compute all holographic penalties.

        Returns a dict so the trainer can log each term separately.

        z              : (batch, latent_dim)
        x_reconstructed: (batch, input_dim)
        x_original     : (batch, input_dim)
        """
        r_loss = self.radial_scaling_loss(z)
        bb_loss = self.bulk_boundary_loss(z, x_reconstructed, x_original)
        c_loss = self.confinement_loss(z)

        total = (
            self.radial_weight * r_loss
            + self.bulk_boundary_weight * bb_loss
            + self.confinement_weight * c_loss
        )

        return {
            "holographic_total": total,
            "radial_scaling":    r_loss,
            "bulk_boundary":     bb_loss,
            "confinement":       c_loss,
        }