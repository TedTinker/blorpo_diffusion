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
from utils_for_torch import init_weights, ConstrainedConv2d, var, sample, Multi_Kernel_Conv, add_position_layers, SpaceToDepth, DepthToSpace



# Let's make a Varational Autoencoder (VAE)!
class VAE(nn.Module):
    def __init__(self, args = default_args):
        super(VAE, self).__init__()
        
        self.args = args
        
        # This is my kludgey way to see qualities that layers should have.
        example = torch.zeros(self.args.batch_size, 3, self.args.image_size, self.args.image_size)
        B = example.shape[0]
        print(f"\nStart of VAE:\t{example.shape}")
                     
        
        
        self.encode = nn.Sequential(
            # 64 by 64
            Multi_Kernel_Conv(
                in_channels = example.shape[1], 
                out_channels = [16, 8, 8], 
                kernel_sizes = [3, 5, 7], 
                args = self.args),
            SpaceToDepth(block_size=2),  
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # 32 by 32
            
            Multi_Kernel_Conv(
                in_channels = 128,
                out_channels = [16, 16], 
                kernel_sizes = [3, 5], 
                args = self.args),
            SpaceToDepth(block_size=2),  
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # 16 by 16
            
            Multi_Kernel_Conv(
                in_channels = 128,
                out_channels = [16, 16], 
                kernel_sizes = [3, 5], 
                args = self.args),
            SpaceToDepth(block_size=2),  
            nn.LeakyReLU())
            # 8 by 8
            
        example = self.encode(example)
        print(f"VAE encode:\t{example.shape}")
            
        self.mu = nn.Sequential(
            nn.Conv2d(
                in_channels = 128, 
                out_channels = self.args.latent_channels,
                kernel_size = 1))
        
        self.logvar = nn.Conv2d(128, self.args.latent_channels, 1)
                
        example_mu = self.mu(example)
        example_logvar = self.logvar(example)
        example = self.reparam(example_mu, example_logvar)
        print(f"VAE latent:\t{example.shape}")

        # Encoded! 
        
        self.decode = nn.Sequential(
            # 8 by 8
            Multi_Kernel_Conv(
                in_channels = self.args.latent_channels,
                out_channels = [32], 
                kernel_sizes = [3], 
                args = self.args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            Multi_Kernel_Conv(
                in_channels = 32,
                out_channels = [32], 
                kernel_sizes = [1], 
                args = self.args),
            DepthToSpace(block_size = 2),
            Multi_Kernel_Conv(
                in_channels = 8,
                out_channels = [32], 
                kernel_sizes = [3], 
                args = self.args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # 16 by 16
            
            Multi_Kernel_Conv(
                in_channels = 32,
                out_channels = [32], 
                kernel_sizes = [1], 
                args = self.args),
            DepthToSpace(block_size = 2),
            Multi_Kernel_Conv(
                in_channels = 8,
                out_channels = [16, 16], 
                kernel_sizes = [3, 5], 
                args = self.args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # 32 by 32
            
            Multi_Kernel_Conv(
                in_channels = 32,
                out_channels = [32], 
                kernel_sizes = [1], 
                args = self.args),
            DepthToSpace(block_size = 2),
            Multi_Kernel_Conv(
                in_channels = 8,
                out_channels = [16, 16], 
                kernel_sizes = [3, 5], 
                args = self.args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # 64 by 64
            
            Multi_Kernel_Conv(
                in_channels = 32,
                out_channels = [16, 8, 8], 
                kernel_sizes = [3, 5, 7], 
                args = self.args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels = 32, 
                out_channels = 3,
                kernel_size = 1),
            nn.Tanh())
            
        example = self.decode(example)
        print(f"VAE decode:\t{example.shape}")
        
        self.apply(init_weights)
        self.to(self.args.device)
        
        
        
    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    

    def forward(self, images, use_logvar = True, clamp_logvar = False):
        
        # Shrink.
        a = self.encode(images)
                    
        # Encode.
        mu = self.mu(a)
        logvar = self.logvar(a)
        if(clamp_logvar):
            logvar = torch.clamp(logvar, min = -10, max = 10)
        if(use_logvar):
            encoded = self.reparam(mu, logvar)
        else:
            encoded = mu
            
        # Decode, grow.
        decoded = (self.decode(encoded) + 1) / 2
        dkl = -0.5*(1 + logvar - mu**2 - torch.exp(logvar)).mean()
        
        return decoded, encoded, mu, logvar, dkl



# Let's check it out!
if(__name__ == "__main__"):
    args = default_args
    vae = VAE(args)
    print("\n\n")
    print(vae)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(vae, (args.batch_size, 3, args.image_size, args.image_size)))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
# %%