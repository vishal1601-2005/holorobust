import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
import time

from holorobust.holographic.losses import HolographicLoss
from holorobust.geometric.losses import ArakelovLoss
from holorobust.core.model import HoloRobustModel


class HoloRobustTrainer:
    """
    Unified trainer that combines four loss components:

        1. Task loss        -- reconstruction MSE (standard autoencoder)
        2. Holographic loss -- AdS/CFT boundary-bulk penalties
        3. Arakelov loss    -- geometric height + curvature + Lorentzian
        4. Adversarial loss -- PGD attack resistance (optional)

    The total loss is:
        L = task + w_holo * L_holo + w_arakelov * L_arakelov
            + w_adv * L_adversarial

    All four weights are tunable. Setting a weight to 0.0 disables
    that component — so this trainer works as a plain autoencoder too,
    making ablation studies (proving your method works) trivial.

    Args:
        model             : HoloRobustModel instance
        lr                : learning rate (default 1e-3)
        holo_weight       : weight on total holographic loss
        arakelov_weight   : weight on total Arakelov loss
        adversarial_weight: weight on adversarial loss
        adv_eps           : PGD attack epsilon (perturbation budget)
        adv_steps         : PGD attack steps per training batch
        device            : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        model: HoloRobustModel,
        lr: float = 1e-3,
        holo_weight: float = 0.1,
        arakelov_weight: float = 0.1,
        adversarial_weight: float = 0.1,
        adv_eps: float = 0.1,
        adv_steps: int = 5,
        device: Optional[str] = None,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.holo_weight = holo_weight
        self.arakelov_weight = arakelov_weight
        self.adversarial_weight = adversarial_weight
        self.adv_eps = adv_eps
        self.adv_steps = adv_steps

        # Loss functions
        self.task_loss_fn = nn.MSELoss()
        self.holo_loss_fn = HolographicLoss()
        self.arakelov_loss_fn = ArakelovLoss()

        # Optimiser — AdamW is more stable than Adam under physics losses
        self.optimiser = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )

        # Scheduler — cosine annealing for smooth convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimiser, T_max=100, eta_min=1e-5
        )

        # Training history — logged per epoch for plotting
        self.history: Dict[str, List[float]] = {
            "total":            [],
            "task":             [],
            "holographic":      [],
            "arakelov":         [],
            "adversarial":      [],
            "epoch_time_sec":   [],
        }

        print(f"HoloRobustTrainer ready on {self.device.upper()}")
        print(f"Model parameters: {model.count_parameters():,}")
        print(f"Loss weights  ->  holo={holo_weight}  "
              f"arakelov={arakelov_weight}  adv={adversarial_weight}")

    # ------------------------------------------------------------------
    # PGD adversarial attack (built-in, no torchattacks dependency)
    # ------------------------------------------------------------------

    def _pgd_attack(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projected Gradient Descent attack for adversarial training.

        Generates adversarial examples by iteratively perturbing x
        in the direction that maximises the reconstruction loss.
        The perturbation is clipped to an L-inf ball of radius adv_eps.

        This is the inner maximisation step of adversarial training:
            max_{||delta||_inf <= eps} L(x + delta)

        A model trained against these examples is provably more robust
        than one trained on clean data alone.
        """
        x_adv = x.detach().clone().requires_grad_(True)
        step_size = self.adv_eps / self.adv_steps

        for _ in range(self.adv_steps):
            x_hat, _ = self.model(x_adv)
            loss = self.task_loss_fn(x_hat, x_adv)
            loss.backward()

            with torch.no_grad():
                # Step in gradient direction (maximise loss)
                x_adv = x_adv + step_size * x_adv.grad.sign()
                # Project back into epsilon ball around original x
                x_adv = torch.max(
                    torch.min(x_adv, x + self.adv_eps), x - self.adv_eps
                )
                x_adv = x_adv.detach().requires_grad_(True)

        return x_adv.detach()

    # ------------------------------------------------------------------
    # Single batch step
    # ------------------------------------------------------------------

    def _train_step(self, x: torch.Tensor) -> Dict[str, float]:
        """
        One training step on a single batch.
        Returns a dict of individual loss values for logging.
        """
        x = x.to(self.device)
        self.model.train()
        self.optimiser.zero_grad()

        # --- Forward pass (clean) ---
        x_hat, z = self.model(x)

        # --- Task loss ---
        task_loss = self.task_loss_fn(x_hat, x)

        # --- Holographic loss ---
        holo_dict = self.holo_loss_fn(z, x_hat, x)
        holo_loss = holo_dict["holographic_total"]

        # --- Arakelov loss ---
        arakelov_dict = self.arakelov_loss_fn(z, x, self.model.encoder)
        arakelov_loss = arakelov_dict["arakelov_total"]

        # --- Adversarial loss (optional) ---
        if self.adversarial_weight > 0.0:
            x_adv = self._pgd_attack(x)
            x_hat_adv, z_adv = self.model(x_adv)
            adv_loss = self.task_loss_fn(x_hat_adv, x)
        else:
            adv_loss = torch.tensor(0.0, device=self.device)

        # --- Total loss ---
        total_loss = (
            task_loss
            + self.holo_weight       * holo_loss
            + self.arakelov_weight   * arakelov_loss
            + self.adversarial_weight * adv_loss
        )

        # --- Backward + update ---
        total_loss.backward()
        # Gradient clipping — prevents explosions with physics losses
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimiser.step()

        return {
            "total":       total_loss.item(),
            "task":        task_loss.item(),
            "holographic": holo_loss.item(),
            "arakelov":    arakelov_loss.item(),
            "adversarial": adv_loss.item(),
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        dataloader: DataLoader,
        epochs: int = 50,
        print_every: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Train the model for a given number of epochs.

        Args:
            dataloader  : PyTorch DataLoader yielding batches of x
            epochs      : number of full passes over the data
            print_every : print loss summary every N epochs

        Returns:
            history dict with loss curves for all components
        """
        print(f"\nStarting training: {epochs} epochs")
        print("-" * 55)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            epoch_losses: Dict[str, List[float]] = {
                k: [] for k in ["total", "task", "holographic",
                                 "arakelov", "adversarial"]
            }

            for batch in dataloader:
                # Handle both (x,) tuples and raw tensors
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch

                step_losses = self._train_step(x)
                for k, v in step_losses.items():
                    epoch_losses[k].append(v)

            # Average across batches
            epoch_avg = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
            elapsed = time.time() - t0

            # Store in history
            for k in epoch_avg:
                self.history[k].append(epoch_avg[k])
            self.history["epoch_time_sec"].append(elapsed)

            # Step scheduler
            self.scheduler.step()

            # Print
            if epoch % print_every == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:>4}/{epochs} | "
                    f"total={epoch_avg['total']:.4f} | "
                    f"task={epoch_avg['task']:.4f} | "
                    f"holo={epoch_avg['holographic']:.4f} | "
                    f"arak={epoch_avg['arakelov']:.4f} | "
                    f"adv={epoch_avg['adversarial']:.4f} | "
                    f"{elapsed:.1f}s"
                )

        print("-" * 55)
        print("Training complete.")
        return self.history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate reconstruction loss on a dataloader (no grad).
        Returns mean task loss and mean anomaly score across the dataset.
        """
        self.model.eval()
        total_task = 0.0
        total_score = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)
                x_hat, z = self.model(x)
                total_task  += self.task_loss_fn(x_hat, x).item()
                total_score += self.model.anomaly_score(x).mean().item()
                n_batches   += 1

        return {
            "eval_task_loss":    total_task  / n_batches,
            "eval_anomaly_score": total_score / n_batches,
        }

    def save(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")