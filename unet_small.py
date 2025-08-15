#%%

import os 
from utils import file_location

os.chdir(file_location)

import math

import torch
import torch.nn as nn
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F

from utils import default_args
from utils_for_torch import init_weights, ConstrainedConv2d, var, sample, Multi_Kernel_Conv, add_position_layers, \
    SpaceToDepth, DepthToSpace, ResBlock, AttnBlock



# Let's make a Latent-Space UNet ε-predictor (UNET)!
class UNET(nn.Module):
    def __init__(self, args = default_args):
        super(UNET, self).__init__()
        
        self.args = args
        
        # This is my kludgey way to see qualities that layers should have.
        example = torch.zeros(self.args.batch_size, self.args.latent_channels, 8, 8)
        B = example.shape[0]
        print(f"\nStart of UNET:\t{example.shape}")
                
        self.a = nn.Sequential(
            Multi_Kernel_Conv(
                in_channels = example.shape[1], 
                out_channels = [32], 
                kernel_sizes = [3], 
                args = self.args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            Multi_Kernel_Conv(
                in_channels = 32, 
                out_channels = [32], 
                kernel_sizes = [3], 
                args = self.args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            Multi_Kernel_Conv(
                in_channels = 32, 
                out_channels = [32], 
                kernel_sizes = [3], 
                args = self.args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            Multi_Kernel_Conv(
                in_channels = 32, 
                out_channels = [32], 
                kernel_sizes = [3], 
                args = self.args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, self.args.latent_channels, kernel_size=1))
        
        example = self.a(example)
        print(f"UNET a:\t{example.shape}")
        
        self.time_mlp = nn.Sequential(
            nn.Linear(128, 2 * self.args.latent_channels), 
            nn.LeakyReLU(),
            nn.Linear(2 * self.args.latent_channels, 2 * self.args.latent_channels))

        self.apply(init_weights)
        self.to(self.args.device)
        
    def timestep_embed(self, t, dim=128):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        ang = t.float()[:, None] * freqs[None]
        return torch.cat([torch.cos(ang), torch.sin(ang)], dim=1)  # [B, 128]


    def forward(self, latent, t):
        h = self.a(latent)                                 # [B,C,H,W]
        emb = self.timestep_embed(t)                       # [B,128]
        emb = self.time_mlp(emb)                           # [B,2C]
        scale, shift = emb.chunk(2, dim=1)                 # [B,C], [B,C]
        scale = (1 + scale).unsqueeze(-1).unsqueeze(-1)    # [B,C,1,1]
        shift = shift.unsqueeze(-1).unsqueeze(-1)          # [B,C,1,1]
        return h * scale + shift        
                     


# Let's check it out!
if(__name__ == "__main__"):
    args = default_args
    unet = UNET(args)
    print("\n\n")
    print(unet)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(unet, ((args.batch_size, args.latent_channels, 8, 8), (1,))))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
# %%