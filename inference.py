# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
# python3 inference.py -f <(ls mel_spectrograms/*.pt) -w waveglow_256channels_universal_v5.pt -o . -s 0.6
import os
import sys
from scipy.io.wavfile import write
import torch
from mel2samp import files_to_list, MAX_WAV_VALUE
from denoiser import Denoiser

sys.path.insert(0, '/Users/pratik/repos/TimbreSpace')
from datasets import AudioSTFTDataModule
from utils import dotdict

def main(waveglow_path, sigma, output_dir, sampling_rate, is_fp16,
         denoiser_strength):
    # mel_files = files_to_list(mel_files)
    waveglow = torch.load(waveglow_path, map_location=torch.device('cpu'))['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cpu().eval()
    if is_fp16:
        from apex import amp
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

    if denoiser_strength > 0:
        denoiser = Denoiser(waveglow).cuda()

    data = AudioSTFTDataModule(config = dotdict(
        {'dataset_path': '../../data/timbre',
         'stft': {'frame_size': 512, 'hop_length': 256, 'segment_duration': 0.18575},
         'saver': {'enabled': False, 'save_dir': '../out'},
         'visualizer': {'enabled': False, 'save_dir': '../out'},
         'csv': {'enabled': True, 'path': '../log'},
         'batch_size': 16,
         'num_workers': 0,
         # 'music_vae': {'checkpoint_path': '/work/gk77/k77021/repos/TimbreSpace/logs/MusicVAEFlat/version_11/checkpoints/last.ckpt'}
         'music_vae': {'checkpoint_path': '/Users/pratik/repos/TimbreSpace/logw/logs/MusicVAEFlat/version_11/checkpoints/last.ckpt'}
         }))
    data.setup()
    train_loader = data.train_dataloader()

    for i, batch in enumerate(train_loader):
        # file_name = os.path.splitext(os.path.basename(file_path))[0]
        # mel = torch.load(file_path)
        # mel = torch.autograd.Variable(mel.cpu())
        mel, audio, _, _, clipname, _, offset = batch
        mel = torch.squeeze(mel, 0)
        mel = torch.squeeze(mel, 1)
        # audio = torch.squeeze(audio, 0)
        # audio = torch.squeeze(audio, 1)
        mel = torch.autograd.Variable(mel.cpu())
        # audio = torch.autograd.Variable(audio.cpu())
        # mel=mel[0]
        mel_batch = []
        for k in range(mel.shape[0]):
            mel_batch.append(mel[k])
        mel_batch = tuple(mel_batch)
        mel_batch = torch.cat(mel_batch, 1)

        audio = torch.squeeze(audio)
        offset = float(offset.cpu().numpy())
        clipname = clipname[0]
        # mel = torch.unsqueeze(mel, 0)
        mel = mel.half() if is_fp16 else mel
        with torch.no_grad():
            audio_recons = waveglow.infer(mel, sigma=sigma)
            print(f"inference: {audio_recons.shape}")
            if denoiser_strength > 0:
                audio_recons = denoiser(audio_recons, denoiser_strength)
            audio_recons = audio_recons * MAX_WAV_VALUE
        audio_recons = audio_recons.squeeze()

        audio_recons_batch = []
        audio_batch = []
        for k in range(audio_recons.shape[0]):
            audio_recons_batch.append(audio_recons[k])
            audio_batch.append(audio[k])
        audio_recons_batch = torch.cat(tuple(audio_recons_batch), 0)
        audio_batch = torch.cat(tuple(audio_batch), 0)

        audio_batch = audio_batch.cpu().numpy()
        audio_recons_batch = audio_recons_batch.cpu().numpy()
        audio_recons_batch = audio_recons_batch.astype('int16')


        audio_path = os.path.join(
            output_dir, "{}-{:.2f}.wav".format(clipname, offset))
        audio_path_recons = os.path.join(
            output_dir, "{}-{:.2f}_synthesis.wav".format(clipname, offset))
        write(audio_path, sampling_rate, audio_batch)
        write(audio_path_recons, sampling_rate, audio_recons_batch)
        print(audio_path)
        # break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow decoder checkpoint with model')
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    parser.add_argument("--sampling_rate", default=22050, type=int)
    parser.add_argument("--is_fp16", action="store_true")
    parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')

    args = parser.parse_args()

    main(
        # args.filelist_path,
         args.waveglow_path, args.sigma, args.output_dir,
         args.sampling_rate, args.is_fp16, args.denoiser_strength)
