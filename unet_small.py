#%%

import os 
from utils import file_location

os.chdir(file_location)

import torch
import torch.nn as nn
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F

from utils import default_args
from utils_for_torch import init_weights, ConstrainedConv2d, var, sample, Multi_Kernel_CAB, add_position_layers, \
    SpaceToDepth, DepthToSpace, ResBlock, AttnBlock



# Let's make a Latent-Space UNet ε-predictor (UNET)!
class UNET(nn.Module):
    def __init__(self, args = default_args):
        super(UNET, self).__init__()
        
        self.args = args
        
        # This is my kludgey way to see qualities that layers should have.
        example = torch.zeros(self.args.batch_size, 4, 8, 8)
        print(f"\nStart of UNET:\t{example.shape}")
        
        c = 32
        
        self.a = nn.Sequential(
            ConstrainedConv2d(
                in_channels = example.shape[1], 
                out_channels = c,
                kernel_size = 1))
        
        example = self.a(example)
        example_skip = example 
        print(f"UNET a:\t{example.shape}")
        
        self.b = nn.Sequential(
            # 8 by 8
            ResBlock(c),
            ConstrainedConv2d(
                in_channels = c, 
                out_channels = c,
                kernel_size = 1),
            ResBlock(c),
            ConstrainedConv2d(
                in_channels = c, 
                out_channels = c,
                kernel_size = 1),
            AttnBlock(c),
            SpaceToDepth(block_size=2),  
            ConstrainedConv2d(
                in_channels = c * 4, 
                out_channels = c,
                kernel_size = 1),
            # 4 by 4
            ResBlock(c),
            ConstrainedConv2d(
                in_channels = c, 
                out_channels = c,
                kernel_size = 1),
            ResBlock(c),
            ConstrainedConv2d(
                in_channels = c, 
                out_channels = c,
                kernel_size = 1),
            AttnBlock(c),
            DepthToSpace(block_size = 2),
            ConstrainedConv2d(
                in_channels = c // 4, 
                out_channels = c,
                kernel_size = 1),)
            # 8 by 8
            
        example = self.b(example)
        print(f"UNET b:\t{example.shape}")
        example = torch.cat([example, example_skip], dim=1)
        print(f"UNET skip:\t{example.shape}")
        
        self.c = nn.Sequential(
            ConstrainedConv2d(
                in_channels = example.shape[1], 
                out_channels = c,
                kernel_size = 1),
            ResBlock(c),
            ConstrainedConv2d(
                in_channels = c, 
                out_channels = c,
                kernel_size = 1),
            ResBlock(c),
            AttnBlock(c),
            nn.GroupNorm(1, c),  
            nn.SiLU(),           
            nn.Conv2d(c, 4, kernel_size=3, padding=1))
        
        example = self.c(example)
        print(f"UNET c:\t{example.shape}")

        self.apply(init_weights)
        self.to(self.args.device)

    def forward(self, latent):
        x = self.a(latent)                   
        x_skip = x
        x = self.b(x)              
        x = torch.cat([x, x_skip], dim=1)      
        epsilon = self.c(x)
        return epsilon
                     


# Let's check it out!
if(__name__ == "__main__"):
    args = default_args
    unet = UNET(args)
    print("\n\n")
    print(unet)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(unet, (args.batch_size, 4, 8, 8)))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
# %%