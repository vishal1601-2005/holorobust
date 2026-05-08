"""
HoloRobust v0.1
===============
Holographic & Geometric Physics-Informed Robust ML Framework.

Brings ideas from AdS/CFT holography, holographic QCD, and
Lorentzian Arakelov geometry into practical PyTorch training
for high-energy physics anomaly detection and adversarially
robust cybersecurity models.

Quick start
-----------
    from holorobust import HoloRobustModel, HoloRobustTrainer

    model   = HoloRobustModel(input_dim=64, latent_dim=16)
    trainer = HoloRobustTrainer(model, holo_weight=0.1, arakelov_weight=0.1)
    trainer.train(dataloader, epochs=50)

    scores = model.anomaly_score(x_test)   # higher = more anomalous
"""

__version__ = "0.1.0"
__author__  = "HoloRobust"
__license__ = "MIT"

# Core — always available
from holorobust.core.model   import HoloRobustModel
from holorobust.core.trainer import HoloRobustTrainer

# Physics losses — available individually
from holorobust.holographic.losses import HolographicLoss
from holorobust.geometric.losses   import ArakelovLoss

# Export utilities
from holorobust.utils.export import ModelExporter
from holorobust.utils.selection import tv_distance_test

__all__ = [
    "HoloRobustModel",
    "HoloRobustTrainer",
    "HolographicLoss",
    "ArakelovLoss",
    "ModelExporter",
    "tv_distance_test",
    "__version__",
]