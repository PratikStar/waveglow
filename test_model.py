from glow import WaveGlow, WaveGlowLoss
import torch

waveglow_config = {
        "n_mel_channels": 256,
        "n_flows": 12,
        "n_group": 8,
        "n_early_every": 4,
        "n_early_size": 2,
        "WN_config": {
            "n_layers": 8,
            "n_channels": 256,
            "kernel_size": 3
        }
    }

model = WaveGlow(**waveglow_config).cpu()

model((torch.randn(16, 256, 16), torch.randn(16, 4095)))