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
        example = torch.zeros(self.args.batch_size, self.args.latent_channels, 16, 16)
        example_std = torch.zeros_like(example)

        print(f"\nStart of UNET:\t{example.shape, example_std.shape}")
                
        self.latent = nn.Sequential(
            Multi_Kernel_Conv(
                in_channels = example.shape[1], 
                out_channels = [16, 8, 8], 
                kernel_sizes = [3, 5, 7], 
                pos_sizes = [[8]] * 3,
                args = self.args),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU())
        
        self.std = nn.Sequential(
            Multi_Kernel_Conv(
                in_channels = example_std.shape[1], 
                out_channels = [16, 8, 8], 
                kernel_sizes = [3, 5, 7], 
                pos_sizes = [[8]] * 3,
                args = self.args),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU())
        
        example = self.latent(example)
        example_std = self.std(example_std)
        example = torch.cat([example, example_std], dim = 1)
        example_step_1 = example 
        print(f"UNET first:\t{example.shape}")
        
        self.a = nn.Sequential(
            Multi_Kernel_Conv(
                in_channels = example.shape[1],
                out_channels = [16, 32, 16], 
                kernel_sizes = [3, 5, 7], 
                pos_sizes = [[8]] * 3,
                args = self.args),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(),
            Multi_Kernel_Conv(
                in_channels = example.shape[1],
                out_channels = [16, 32, 16], 
                kernel_sizes = [3, 5, 7], 
                pos_sizes = [[8]] * 3,
                args = self.args),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU())
        
        example = self.a(example)
        example_step_a = example
        print(f"UNET a:\t{example.shape}")
        
        self.b = nn.Sequential(
            Multi_Kernel_Conv(
                in_channels = example.shape[1], 
                out_channels = [16, 32, 16], 
                kernel_sizes = [3, 5, 7], 
                pos_sizes = [[8]] * 3,
                args = self.args),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(),
            Multi_Kernel_Conv(
                in_channels = example.shape[1], 
                out_channels = [16, 32, 16], 
                kernel_sizes = [3, 5, 7], 
                pos_sizes = [[8]] * 3,
                args = self.args),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU())
        
        example = self.b(example)
        example_step_b = example
        example += example_step_a
        print(f"UNET b:\t{example.shape}")
        
        self.c = nn.Sequential(
            Multi_Kernel_Conv(
                in_channels = example.shape[1], 
                out_channels = [16, 32, 16], 
                kernel_sizes = [3, 5, 7], 
                pos_sizes = [[8]] * 3,
                args = self.args),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(),
            Multi_Kernel_Conv(
                in_channels = example.shape[1], 
                out_channels = [16, 32, 16], 
                kernel_sizes = [3, 5, 7], 
                pos_sizes = [[8]] * 3,
                args = self.args),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU())
        
        example = self.c(example)
        example_step_c = example
        example += example_step_b
        print(f"UNET c:\t{example.shape}")
        
        self.d = nn.Sequential(
            Multi_Kernel_Conv(
                in_channels = example.shape[1], 
                out_channels = [16, 32, 16], 
                kernel_sizes = [3, 5, 7], 
                pos_sizes = [[8]] * 3,
                args = self.args),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(),
            Multi_Kernel_Conv(
                in_channels = example.shape[1], 
                out_channels = [16, 32, 16], 
                kernel_sizes = [3, 5, 7], 
                pos_sizes = [[8]] * 3,
                args = self.args),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU())
        
        example = self.d(example)
        example += example_step_c
        print(f"UNET d:\t{example.shape}")
        
        self.e = nn.Sequential(
            Multi_Kernel_Conv(
                in_channels = example.shape[1], 
                out_channels = [16, 32, 16], 
                kernel_sizes = [3, 5, 7], 
                pos_sizes = [[8]] * 3,
                args = self.args),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(),
            Multi_Kernel_Conv(
                in_channels = example.shape[1], 
                out_channels = [16, 32, 16], 
                kernel_sizes = [3, 5, 7], 
                pos_sizes = [[8]] * 3,
                args = self.args),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU())
        
        example = self.e(example)
        example += example_step_1
        print(f"UNET e:\t{example.shape}")
        
        self.f = nn.Sequential(
            Multi_Kernel_Conv(
                in_channels = example.shape[1], 
                out_channels = [16, 32, 16], 
                kernel_sizes = [3, 5, 7], 
                pos_sizes = [[8]] * 3,
                args = self.args),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU())
        
        example = self.f(example)
        print(f"UNET f:\t{example.shape}")
        
        self.out = nn.Sequential(
            nn.Conv2d(example.shape[1], self.args.latent_channels, kernel_size=1))
        
        example = self.out(example)
        print(f"UNET out:\t{example.shape}")
        
        self.apply(init_weights)
        self.to(self.args.device)

    def forward(self, latent, std):
        latent = self.latent(latent)
        std = self.std(std)
        x = torch.cat([latent, std], dim = 1)
        a   = self.a(x)   
        b   = self.b(a)     
        c   = self.c(b + a + x)     
        d   = self.d(c + b)          
        e   = self.e(d + c + x)     
        f   = self.f(e + d)   
        out = self.out(f + e + x)
        return out
                     


# Let's check it out!
if(__name__ == "__main__"):
    args = default_args
    unet = UNET(args)
    print("\n\n")
    print(unet)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(unet, ((args.batch_size, args.latent_channels, 16, 16), (args.batch_size, args.latent_channels, 16, 16))))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
# %%