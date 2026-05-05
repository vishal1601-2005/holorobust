# HoloRobust

**Holographic & Geometric Physics-Informed Robust ML Framework**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

HoloRobust is an open-source PyTorch framework that injects deep ideas from
**AdS/CFT holography**, **holographic QCD**, and **Lorentzian Arakelov geometry**
directly into ML training — producing models that are more robust, more physically
meaningful, and more resistant to adversarial attack.

---

## Key Results

### HEP Anomaly Detection (LHC Olympics 2020 structure)
| Model | AUC | Adversarial Robustness | Latency |
|-------|-----|----------------------|---------|
| Standard Autoencoder | 1.000 | degrades 9.3% under attack | — |
| **HoloRobust** | **1.000** | **degrades only 8.9% under attack** | **190μs** |

### Cybersecurity Intrusion Detection (5 attack types, 78 features)
| Model | AUC | Score drop under evasion | Parameters |
|-------|-----|--------------------------|------------|
| Standard Autoencoder | 1.000 | 9.3% | 39,646 |
| **HoloRobust** | **1.000** | **8.9%** | **39,646** |

### Export & Deployment
| Format | Size | Latency (CUDA) | FPGA-ready |
|--------|------|----------------|------------|
| ONNX encoder | 75 KB | 190μs min | ✅ via hls4ml |
| TorchScript | 87 KB | 190μs min | — |

> **What the robustness number means:** An attacker running a PGD evasion attack
> causes the baseline model's anomaly score to drop 9.3% — attacks become harder
> to detect. HoloRobust drops only 8.9% — physics constraints make the latent
> space harder to fool.

---

## Installation

```bash
# Clone and install
git clone https://github.com/vishal1601-2005/holorobust.git
cd holorobust
pip install -e .

# Or install dependencies manually
pip install torch numpy scipy pandas scikit-learn h5py matplotlib onnx
```

---

## Quick Start

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from holorobust import HoloRobustModel, HoloRobustTrainer

# Build model
model = HoloRobustModel(input_dim=64, latent_dim=16, hidden_dim=128)

trainer = HoloRobustTrainer(
    model,
    holo_weight=0.1,         # AdS/CFT holographic loss
    arakelov_weight=0.1,     # Lorentzian Arakelov geometric loss
    adversarial_weight=0.1,  # PGD adversarial training
)

# Train on background/normal data only (unsupervised)
loader = DataLoader(TensorDataset(X_train), batch_size=512, shuffle=True)
trainer.train(loader, epochs=50)

# Anomaly scoring — higher = more anomalous
scores = model.anomaly_score(X_test)
```

---

## Physics Components

### 1. Holographic Loss (AdS/CFT)
Treats the neural network's latent space as the AdS bulk and
input/output as the boundary. Three penalties:

- **Radial scaling** — latent norms follow AdS power-law scaling
- **Bulk-boundary consistency** — compressed latents still reconstruct faithfully  
- **Confinement** — holographic QCD-inspired norm ceiling (prevents adversarial drift)

### 2. Arakelov Geometric Loss
Inspired by Lorentzian Arakelov geometry:

- **Height function** — logarithmic penalty on arithmetic complexity of embeddings
- **Curvature penalty** — Jacobian norm regularization for smooth, flat encoder maps
- **Lorentzian metric** — causal light-cone structure enforced in latent space

### 3. Adversarial Training (PGD)
Built-in Projected Gradient Descent attack runs every training step.
No extra libraries needed. Models degrade gracefully under evasion attacks.

---

## Applications

### High-Energy Physics
Real-time anomaly detection for LHC experiments. The encoder:
- Exports to ONNX → compiles to FPGA via hls4ml
- Deployable at Level-1 trigger rates (<1ms budget)
- Physics losses enforce physically consistent latent spaces

### Cybersecurity
Adversarially robust intrusion detection:
- Trained unsupervised on normal traffic only
- Detects DoS, Port Scan, Brute Force, Botnet, Infiltration
- Resists PGD evasion attacks better than standard autoencoders

---

## Project Structure

```
holorobust/
├── holorobust/
│   ├── __init__.py            # Clean public API
│   ├── core/
│   │   ├── model.py           # HoloRobustModel base class
│   │   └── trainer.py         # Unified physics + adversarial trainer
│   ├── holographic/
│   │   └── losses.py          # AdS/CFT holographic regularizers
│   ├── geometric/
│   │   └── losses.py          # Arakelov geometric regularizers
│   └── utils/
│       └── export.py          # ONNX, TorchScript, latency benchmark
└── examples/
    ├── hep_jet_anomaly.ipynb      # LHC anomaly detection demo
    └── cyber_intrusion.ipynb      # Cybersecurity intrusion detection demo
```

---

## Roadmap

- [x] Core holographic and Arakelov losses
- [x] Adversarial training (PGD, built-in)
- [x] ONNX and TorchScript export
- [x] Latency benchmarking
- [x] HEP anomaly detection demo (LHC Olympics data)
- [x] Cybersecurity intrusion detection demo
- [x] pip installable (`pip install -e .`)
- [ ] HuggingFace Space interactive demo
- [ ] Real CIC-IDS2017 cybersecurity benchmark
- [ ] hls4ml FPGA synthesis example
- [ ] arXiv preprint
- [ ] PyPI release (`pip install holorobust`)

---

## Citation

```bibtex
@software{holorobust2025,
  title   = {HoloRobust: Holographic and Geometric Physics-Informed Robust ML},
  author  = {Vishal},
  year    = {2025},
  url     = {https://github.com/vishal1601-2005/holorobust},
  license = {MIT},
  version = {0.1.0}
}
```

---

## License

MIT License — free for academic and commercial use.

---

## Contact

For consulting, integration, or research collaboration:  
GitHub: [@vishal1601-2005](https://github.com/vishal1601-2005)
