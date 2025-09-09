#%%

import os 
from utils import file_location

os.chdir(file_location)

import torch
import torch.nn as nn
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from utils import default_args
from utils_for_torch import init_weights, SelfAttention, get_stats, rgb_to_circular_hsv, calculate_dkl, var, sample, Multi_Kernel_Conv, SpaceToDepth, DepthToSpace



# Let's make a Varational Autoencoder (VAE)!
class VAE(nn.Module):
    def __init__(self, args = default_args):
        super(VAE, self).__init__()
        
        self.args = args
        
        # This is my kludgey way to see qualities that layers should have.
        example_rgb = torch.zeros(self.args.batch_size, 3, self.args.image_size, self.args.image_size)
        example_rgb_stats = get_stats(example_rgb, rgb = True, args = self.args).cpu()
        example_hsv = rgb_to_circular_hsv(example_rgb)
        example_hsv_stats = get_stats(example_hsv, rgb = False, args = self.args).cpu()

        print(f"\nStart of VAE (rgb):\t{example_rgb.shape}")
        print(f"Start of VAE (rgb stats):\t{example_rgb_stats.shape}")
        print(f"Start of VAE (hsv):\t{example_hsv.shape}")
        print(f"Start of VAE (hsv stats):\t{example_hsv_stats.shape}")
        
        self.rgb_in = nn.Sequential(
            # 64 by 64
            Multi_Kernel_Conv(
                in_channels = example_rgb.shape[1], 
                out_channels = [16, 8, 8], 
                kernel_sizes = [3, 5, 7], 
                args = self.args),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU())
        
        self.rgb_stats_in = nn.Sequential(
            # 64 by 64
            Multi_Kernel_Conv(
                in_channels = example_rgb_stats.shape[1], 
                out_channels = [16, 8, 8], 
                kernel_sizes = [3, 5, 7], 
                args = self.args),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU())
        
        self.hsv_in = nn.Sequential(
            # 64 by 64
            Multi_Kernel_Conv(
                in_channels = example_hsv.shape[1], 
                out_channels = [16, 8, 8], 
                kernel_sizes = [3, 5, 7], 
                args = self.args),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU())
        
        self.hsv_stats_in = nn.Sequential(
            # 64 by 64
            Multi_Kernel_Conv(
                in_channels = example_hsv_stats.shape[1], 
                out_channels = [16, 8, 8], 
                kernel_sizes = [3, 5, 7], 
                args = self.args),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU())
        
        example_rgb = self.rgb_in(example_rgb)
        example_rgb_stats = self.rgb_stats_in(example_rgb_stats)
        example_hsv = self.hsv_in(example_hsv)
        example_hsv_stats = self.hsv_stats_in(example_hsv_stats)
        example = torch.cat([example_rgb, example_rgb_stats, example_hsv, example_hsv_stats], dim = 1)     
        print(f"VAE all together:\t{example.shape}")                
        
        self.encode = nn.Sequential(
            # 64 by 64
            Multi_Kernel_Conv(
                in_channels = example.shape[1], 
                out_channels = [16, 8, 8], 
                kernel_sizes = [3, 5, 7], 
                pos_sizes = [[8, 16]] * 3,
                args = self.args),
            SpaceToDepth(block_size=2),  
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),
            
            # 32 by 32
            Multi_Kernel_Conv(
                in_channels = 128,
                out_channels = [16, 16], 
                kernel_sizes = [3, 5], 
                args = self.args),
            SpaceToDepth(block_size=2),  
            nn.Dropout2d(self.args.vae_dropout),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),
            SelfAttention(
                in_channels = 128, 
                kernel_size = 3),
            
            # 16 by 16
            Multi_Kernel_Conv(
                in_channels = 128,
                out_channels = [32], 
                kernel_sizes = [3], 
                args = self.args),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(),
            SelfAttention(
                in_channels = 32, 
                kernel_size = 3))
            
        example = self.encode(example)
        print(f"VAE encode:\t{example.shape}")
            
        self.mu = nn.Sequential(
            nn.Conv2d(
                in_channels = example.shape[1], 
                out_channels = self.args.latent_channels,
                kernel_size = 1))
        
        self.std = nn.Sequential(
            nn.Conv2d(
                in_channels = example.shape[1], 
                out_channels = self.args.latent_channels,
                kernel_size = 1),
            nn.Softplus())
        
        example_mu = self.mu(example)
        example_std = self.std(example)
        example = sample(example_mu, example_std)
        
        print(f"VAE latent:\t{example.shape}")

        # Encoded! 
        
        self.decode = nn.Sequential(
            # 16 by 16
            Multi_Kernel_Conv(
                in_channels = self.args.latent_channels,
                out_channels = [32], 
                kernel_sizes = [3], 
                pos_sizes = [[8]],
                args = self.args),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(),
            SelfAttention(
                in_channels = 32, 
                kernel_size = 3),
                        
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
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(),
            SelfAttention(
                in_channels = 32, 
                kernel_size = 3),
            
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
            nn.Dropout2d(self.args.vae_dropout),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(),
            
            # 64 by 64
            Multi_Kernel_Conv(
                in_channels = 32,
                out_channels = [16, 8, 8], 
                kernel_sizes = [3, 5, 7], 
                args = self.args),
            nn.GroupNorm(8, 32),
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
        
    

    def forward(self, images_rgb, use_std = True):
        
        images_rgb_stats = get_stats(images_rgb, rgb = True, args = self.args)
        images_hsv = rgb_to_circular_hsv(images_rgb)
        images_hsv_stats = get_stats(images_hsv, rgb = False, args = self.args)
        
        images_rgb = self.rgb_in(images_rgb)
        images_rgb_stats = self.rgb_stats_in(images_rgb_stats)
        images_hsv = self.hsv_in(images_hsv)
        images_hsv_stats = self.hsv_stats_in(images_hsv_stats)
        
        images = torch.cat([images_rgb, images_rgb_stats, images_hsv, images_hsv_stats], dim = 1)  
        
        # Shrink.
        a = self.encode(images)
                    
        # Encode.
        mu, std = var(a, self.mu, self.std, self.args)
        if(use_std):    encoded = sample(mu, std)
        else:           encoded = mu
            
        # Decode, grow.
        decoded = (self.decode(encoded) + 1) / 2
        
        # Calculate DKL. (Using both of these is called Jensen-Shannon divergence.)
        dkl_1 = calculate_dkl(mu, std, torch.zeros_like(mu), torch.ones_like(std)).mean()
        dkl_2 = calculate_dkl(torch.zeros_like(mu), torch.ones_like(std), mu, std).mean()
                
        return decoded, encoded, dkl_1, dkl_2, mu, std



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