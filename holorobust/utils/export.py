import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
import json


class ModelExporter:
    """
    Export trained HoloRobust models to deployment formats.

    Supported formats:
        1. ONNX         -- universal inference format, runs on CPU/GPU/edge
        2. TorchScript  -- PyTorch native, fast inference, no Python needed
        3. Model card   -- JSON metadata for HuggingFace / documentation

    Why this matters commercially:
        - ONNX lets cybersecurity vendors integrate your model into
          any language (C++, Go, Java) without a Python runtime.
        - TorchScript lets you ship a single file with no dependencies.
        - hls4ml can convert ONNX to FPGA firmware for HEP triggers.
          A model that runs at LHC Level-1 trigger rates (40 MHz)
          must be in ONNX or similar first.

    Usage:
        exporter = ModelExporter(model, input_dim=64)
        exporter.to_onnx('encoder.onnx', export_encoder_only=True)
        exporter.to_torchscript('model.pt')
        exporter.save_model_card('model_card.json')
    """

    def __init__(
        self,
        model: nn.Module,
        input_dim: int = 64,
        latent_dim: int = 16,
        device: Optional[str] = None,
    ):
        self.model = model
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def to_onnx(
        self,
        path: str,
        export_encoder_only: bool = False,
        batch_size: int = 1,
        opset_version: int = 14,
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            path               : output file path (e.g. 'encoder.onnx')
            export_encoder_only: if True, exports only the encoder
                                 (latent extractor). Smaller, faster,
                                 better for FPGA/edge deployment.
                                 if False, exports full autoencoder.
            batch_size         : dummy batch size for tracing (1 for edge)
            opset_version      : ONNX opset (14 is widely supported)

        Returns:
            path to the exported file
        """
        import onnx

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        dummy_input = torch.randn(batch_size, self.input_dim).to(self.device)

        if export_encoder_only:
            # Wrap encoder to return only z (not tuple)
            class EncoderWrapper(nn.Module):
                def __init__(self, encoder):
                    super().__init__()
                    self.encoder = encoder

                def forward(self, x):
                    return self.encoder(x)

            export_model = EncoderWrapper(self.model.encoder)
            input_names  = ['jet_features']   # HEP naming
            output_names = ['latent_z']
            dynamic_axes = {
                'jet_features': {0: 'batch_size'},
                'latent_z':     {0: 'batch_size'},
            }
        else:
            # Wrap full model to return only x_hat (not tuple)
            class AutoencoderWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    x_hat, _ = self.model(x)
                    return x_hat

            export_model = AutoencoderWrapper(self.model)
            input_names  = ['input_features']
            output_names = ['reconstruction']
            dynamic_axes = {
                'input_features': {0: 'batch_size'},
                'reconstruction': {0: 'batch_size'},
            }

        export_model.eval()

        torch.onnx.export(
            export_model,
            dummy_input,
            path,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,   # optimise constants at export time
        )

        # Verify the exported model is valid
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)

        size_kb = Path(path).stat().st_size / 1024
        print(f"ONNX export successful: {path}")
        print(f"  Format  : {'encoder-only' if export_encoder_only else 'full autoencoder'}")
        print(f"  Opset   : {opset_version}")
        print(f"  Size    : {size_kb:.1f} KB")
        print(f"  Input   : ({batch_size}, {self.input_dim})")
        print(f"  Output  : ({batch_size}, "
              f"{self.latent_dim if export_encoder_only else self.input_dim})")

        return path

    # ------------------------------------------------------------------
    # TorchScript export
    # ------------------------------------------------------------------

    def to_torchscript(self, path: str) -> str:
        """
        Export encoder to TorchScript via tracing.

        TorchScript produces a single .pt file that runs without
        Python — useful for C++ inference servers in cybersecurity
        pipelines and for shipping to clients without exposing source.

        Args:
            path : output file path (e.g. 'encoder.pt')

        Returns:
            path to the exported file
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        dummy_input = torch.randn(1, self.input_dim).to(self.device)

        # Trace the encoder only — lightweight, fast
        traced = torch.jit.trace(self.model.encoder, dummy_input)
        traced.save(path)

        size_kb = Path(path).stat().st_size / 1024
        print(f"TorchScript export successful: {path}")
        print(f"  Size: {size_kb:.1f} KB")

        return path

    # ------------------------------------------------------------------
    # Latency benchmark
    # ------------------------------------------------------------------

    def benchmark_latency(
        self,
        n_runs: int = 1000,
        batch_size: int = 1,
    ) -> dict:
        """
        Measure inference latency of the encoder.

        Runs n_runs forward passes and reports mean, min, and max
        latency in microseconds. Critical for:
          - HEP: Level-1 trigger has ~1 microsecond budget
          - Cyber: packet inspection needs sub-millisecond scoring

        Args:
            n_runs     : number of timing runs
            batch_size : batch size to benchmark (1 = single event)

        Returns:
            dict with latency statistics in microseconds
        """
        import time

        x = torch.randn(batch_size, self.input_dim).to(self.device)
        times = []

        # Warmup
        for _ in range(50):
            with torch.no_grad():
                _ = self.model.encoder(x)

        # Timed runs
        for _ in range(n_runs):
            if self.device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = self.model.encoder(x)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e6)  # microseconds

        results = {
            "mean_us":   sum(times) / len(times),
            "min_us":    min(times),
            "max_us":    max(times),
            "device":    self.device,
            "batch_size": batch_size,
        }

        print(f"\nLatency benchmark ({n_runs} runs, batch={batch_size}):")
        print(f"  Mean : {results['mean_us']:.2f} μs")
        print(f"  Min  : {results['min_us']:.2f} μs")
        print(f"  Max  : {results['max_us']:.2f} μs")
        print(f"  Device: {self.device.upper()}")

        return results

    # ------------------------------------------------------------------
    # Model card
    # ------------------------------------------------------------------

    def save_model_card(
        self,
        path: str,
        description: str = "",
        metrics: Optional[dict] = None,
    ) -> str:
        """
        Save a JSON model card with metadata.

        This is what you upload to HuggingFace alongside the model.
        It documents what the model does, its architecture, and
        its benchmark results — critical for adoption and for
        convincing commercial clients the model works.

        Args:
            path        : output path (e.g. 'model_card.json')
            description : one-paragraph description of this model
            metrics     : dict of benchmark results to include

        Returns:
            path to the saved model card
        """
        card = {
            "name":        "HoloRobust Autoencoder",
            "framework":   "HoloRobust v0.1",
            "description": description or (
                "Physics-informed autoencoder with holographic (AdS/CFT) "
                "and Arakelov geometric regularization. Trained with "
                "simultaneous PGD adversarial training for robustness."
            ),
            "architecture": {
                "type":       "Autoencoder",
                "input_dim":  self.input_dim,
                "latent_dim": self.latent_dim,
                "parameters": sum(
                    p.numel() for p in self.model.parameters()
                    if p.requires_grad
                ),
                "normalization": "LayerNorm",
                "activation":    "GELU",
            },
            "physics_components": {
                "holographic_loss":  "AdS radial scaling, bulk-boundary, confinement",
                "arakelov_loss":     "Height function, Jacobian curvature, Lorentzian metric",
                "adversarial":       "PGD (L-inf)",
            },
            "use_cases": [
                "HEP anomaly detection (LHC jet data)",
                "Cybersecurity intrusion detection",
                "Real-time FPGA deployment via hls4ml",
            ],
            "metrics": metrics or {},
            "license": "MIT",
            "citation": (
                "HoloRobust: Holographic & Geometric Physics-Informed "
                "Robust ML Framework. github.com/yourusername/holorobust"
            ),
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(card, f, indent=2)

        print(f"Model card saved: {path}")
        return path