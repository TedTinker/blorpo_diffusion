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
    SpaceToDepth, DepthToSpace



# Let's make a Latent-Space UNet ε-predictor (UNET)!
class UNET(nn.Module):
    def __init__(self, args = default_args):
        super(UNET, self).__init__()
        
        self.args = args
        
        # This is my kludgey way to see qualities that layers should have.
        example = torch.zeros(self.args.batch_size, self.args.latent_channels, 8, 8)
        example_std = torch.zeros_like(example)

        print(f"\nStart of UNET:\t{example.shape, example_std.shape}")
                
        self.latent = nn.Sequential(
            Multi_Kernel_Conv(
                in_channels = example.shape[1], 
                out_channels = [16, 16], 
                kernel_sizes = [1, 3], 
                pos_sizes = [[8]] * 2,
                args = self.args),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU())
        
        self.std = nn.Sequential(
            Multi_Kernel_Conv(
                in_channels = example_std.shape[1], 
                out_channels = [16, 16], 
                kernel_sizes = [1, 3], 
                pos_sizes = [[8]] * 2,
                args = self.args),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU())
        
        example = self.latent(example)
        example_std = self.std(example_std)
        example = torch.cat([example, example_std], dim = 1)
        print(f"UNET first:\t{example.shape}")
        
        self.a = nn.Sequential(
            Multi_Kernel_Conv(
                in_channels = example.shape[1],
                out_channels = [16, 16], 
                kernel_sizes = [1, 3], 
                pos_sizes = [[8]] * 2,
                args = self.args),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU())
        
        example = self.a(example)
        example_step = example
        
        self.b = nn.Sequential(
            Multi_Kernel_Conv(
                in_channels = 32, 
                out_channels = [16, 16], 
                kernel_sizes = [1, 3], 
                pos_sizes = [[8]] * 2,
                args = self.args),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU())
        
        example = self.b(example)
        example += example_step
        print(f"UNET b:\t{example.shape}")
        
        self.c = nn.Sequential(
            Multi_Kernel_Conv(
                in_channels = example.shape[1], 
                out_channels = [16, 16], 
                kernel_sizes = [1, 3], 
                pos_sizes = [[8]] * 2,
                args = self.args),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(),
            nn.Conv2d(32, self.args.latent_channels, kernel_size=1))
        
        example = self.c(example)
        print(f"UNET c:\t{example.shape}")
        
        self.apply(init_weights)
        self.to(self.args.device)

    def forward(self, latent, std):
        latent = self.latent(latent)
        std = self.std(std)
        x = torch.cat([latent, std], dim = 1)
        a = self.a(x)   
        b = self.b(a)     
        c = self.c(a + b)                 
        return c
                     


# Let's check it out!
if(__name__ == "__main__"):
    args = default_args
    unet = UNET(args)
    print("\n\n")
    print(unet)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(unet, ((args.batch_size, args.latent_channels, 8, 8), (args.batch_size, args.latent_channels, 8, 8))))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
# %%