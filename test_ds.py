
from mel2samp import Mel2Samp
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
import numpy as np

c = {"training_files": "train_files.txt","segment_length": 16000,"sampling_rate": 22050,"filter_length": 1024,"hop_length": 256,"win_length": 1024,"mel_fmin": 0.0,"mel_fmax": 8000.0 }

ds = Mel2Samp(**c)
batch_size=12
train_loader = DataLoader(ds, num_workers=0, shuffle=False,
                          sampler=None,
                          batch_size=batch_size,
                          pin_memory=False,
                          drop_last=True)
i, batch = next(enumerate(train_loader))
spec = batch[0]
sample = batch[1]


from glow import WaveGlow, WaveGlowLoss

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

exit()

fig, ax = plt.subplots(dpi=40)

img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=22050,
                         fmax=8000, ax=ax)
img = librosa.display.specshow(spec[0],n_fft=1024,hop_length=256,
                                      y_axis='log', x_axis='s', ax=ax)



fig.colorbar(img, ax=ax, format="%+2.2f dB")

fig.savefig('here.png')
plt.close(fig)






