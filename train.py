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
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import argparse
import json
import os
import sys
import torch

#=====START: ADDED FOR DISTRIBUTED======
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
#=====END:   ADDED FOR DISTRIBUTED======

from torch.utils.data import DataLoader
from glow import WaveGlow, WaveGlowLoss
sys.path.insert(0, '/Users/pratik/repos/TimbreSpace')
sys.path.insert(0, '/work/gk77/k77021/repos/TimbreSpace')
from datasets import AudioSTFTDataModule
from utils import dotdict

def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")

def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath, device):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    model_for_saving = WaveGlow(**waveglow_config).to(device)
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)

def train(num_gpus, rank, group_name, output_directory, epochs, learning_rate,
          sigma, iters_per_checkpoint, batch_size, seed, fp16_run,
          checkpoint_path, with_tensorboard):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = 'cpu' if num_gpus == 0 else 'cuda'
    print(f"device is {device}")
    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    #=====END:   ADDED FOR DISTRIBUTED======

    criterion = WaveGlowLoss(sigma)
    model = WaveGlow(**waveglow_config, **{"device": device}).to(device)

    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)
    #=====END:   ADDED FOR DISTRIBUTED======

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        print(f"Loading model from: {checkpoint_path}")
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model,
                                                      optimizer)
        print(f"Resuming from iteration: {iteration}")
        iteration += 1  # next iteration is iteration + 1

    # trainset = Mel2Samp(**data_config)
    data = AudioSTFTDataModule(config = dotdict(
        {'dataset_path': '../../data/timbre',
         'stft': {'frame_size': 512, 'hop_length': 256, 'segment_duration': 0.18575},
         'saver': {'enabled': False, 'save_dir': '../out'},
         'visualizer': {'enabled': False, 'save_dir': '../out'},
         'csv': {'enabled': True, 'path': '../log'},
         'batch_size': 16,
         'num_workers': 0,
         'music_vae': {'checkpoint_path': '/work/gk77/k77021/repos/TimbreSpace/logs/MusicVAEFlat/version_11/checkpoints/last.ckpt'}
         # 'music_vae': {'checkpoint_path': '/Users/pratik/repos/TimbreSpace/logw/logs/MusicVAEFlat/version_11/checkpoints/last.ckpt'}
         }))
    data.setup()
    train_loader = data.train_dataloader()
    # =====START: ADDED FOR DISTRIBUTED======
    # train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None
    # =====END:   ADDED FOR DISTRIBUTED======
    # train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
    #                           sampler=train_sampler,
    #                           batch_size=batch_size,
    #                           pin_memory=False,
    #                           drop_last=True)

    # Get shared output_directory ready
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    if with_tensorboard and rank == 0:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(os.path.join(output_directory, 'logs'))

    model.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            model.zero_grad()

            mel, audio, _, _, _, key, _ = batch
            # print(f"Key: {key.numpy()}")
            mel = torch.squeeze(mel, 0)
            mel = torch.squeeze(mel, 1)
            audio = torch.squeeze(audio, 0)
            audio = torch.squeeze(audio, 1)
            mel = torch.autograd.Variable(mel.to(device))
            audio = torch.autograd.Variable(audio.to(device))
            outputs = model((mel, audio))


            loss = criterion(outputs)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()

            if fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            print("Step: {}\tLoss: {:.9f}".format(iteration, reduced_loss))
            if with_tensorboard and rank == 0:
                logger.add_scalar('training_loss', reduced_loss, i + len(train_loader) * epoch)

            if (iteration % iters_per_checkpoint == 0):
                if rank == 0:
                    checkpoint_path = "{}/waveglow_{}".format(
                        output_directory, iteration)
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path, device)

            iteration += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global waveglow_config
    waveglow_config = config["waveglow_config"]
    # global audiostft_config
    # audiostft_config = config["audiostft_config"]

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(num_gpus, args.rank, args.group_name, **train_config)
