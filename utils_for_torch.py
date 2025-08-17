import math

import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F
from kornia.color import rgb_to_hsv 

from utils import default_args, args




# For starting neural networks.
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.LayerNorm, nn.InstanceNorm2d)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
            
# How to use mean and standard deviation layers.
def var(x, mu_func, std_func, args):
    mu = mu_func(x)
    std = torch.clamp(std_func(x), min = args.std_min, max = args.std_max)
    return(mu, std)

# How to sample from probability distributions.
def sample(mu, std):
    e = Normal(0, 1).sample(std.shape).to(std.device)
    return(mu + e * std)

def calculate_dkl(mu_1, std_1, mu_2, std_2):
    std_1 = std_1**2
    std_2 = std_2**2
    term_1 = (mu_2 - mu_1)**2 / std_2 
    term_2 = std_1 / std_2 
    term_3 = torch.log(term_2)
    out = (.5 * (term_1 + term_2 - term_3 - 1))
    out = torch.nan_to_num(out)
    return(out)



# Collecting statistics from batch.
def get_stats(x, rgb, args):
    quantiles = args.stat_quantiles
    batch_size, num_channels, height, width = x.size()
    x_flat = x.view(x.size(0), x.size(1), -1)  # (batch, channels, height * width)
    to_cat = []
    
    # Consider removing this for speed. 
    #batch_quantiles = [torch.quantile(x, q, dim=0, keepdim=True) for q in quantiles] # (1, channels, width, height)
    #batch_quantiles_tiled = [q.repeat(batch_size, 1, 1, 1) for q in batch_quantiles]
    #to_cat.extend(batch_quantiles_tiled)
    
    #per_sample_quantiles = [torch.quantile(x_flat, q, dim=2, keepdim=True) for q in quantiles]  # shape: (batch, channels, 1)
    #per_sample_quantiles_tiled = [q.unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3)) for q in per_sample_quantiles]
    #to_cat += per_sample_quantiles_tiled
    
    x_reshaped = x.view(x.size(0), x.size(1), -1)
    #pixel_quantiles = [torch.quantile(x_reshaped, q, dim=2, keepdim=True) for q in quantiles] # (batch, channels, 1)
    #pixel_quantiles_tiled = [q.unsqueeze(-1).repeat(1, 1, height, width) for q in pixel_quantiles]
    #to_cat.extend(pixel_quantiles_tiled)
    
    batch_mean = torch.mean(x, dim=0, keepdim=True)
    batch_mean_tiled = batch_mean.repeat(batch_size, 1, 1, 1)
    to_cat.append(batch_mean_tiled)
    
    per_sample_mean = torch.mean(x, dim=(2, 3), keepdim=True)
    per_sample_mean_tiled = per_sample_mean.repeat(1, 1, height, width)
    to_cat.append(per_sample_mean_tiled)

    pixel_mean = torch.mean(x_reshaped, dim=2, keepdim=True) # (batch, channels, 1)
    pixel_mean = pixel_mean.unsqueeze(-1)
    pixel_mean_tiled = pixel_mean.repeat(1, 1, height, width)
    to_cat.append(pixel_mean_tiled)
    
    batch_std = torch.std(x, dim=0, keepdim=True)
    batch_std_tiled = batch_std.repeat(batch_size, 1, 1, 1)
    to_cat.append(batch_std_tiled)
    
    per_sample_std = torch.std(x, dim=(2, 3), keepdim=True)
    per_sample_std_tiled = per_sample_std.repeat(1, 1, height, width)
    to_cat.append(per_sample_std_tiled)

    pixel_std = torch.std(x_reshaped, dim=2, keepdim=True) # (batch, channels, 1)
    pixel_std = pixel_std.unsqueeze(-1)
    pixel_std_tiled = pixel_std.repeat(1, 1, height, width)
    to_cat.append(pixel_std_tiled)
        
    if(rgb):
        max_rgb, _ = x.max(dim=1, keepdim=True)
        min_rgb, _ = x.min(dim=1, keepdim=True)
        delta = max_rgb - min_rgb
        v = max_rgb
        s = delta / (max_rgb + 1e-7)  # Add a small constant to avoid division by zero
        brightness_threshold_white = 0.9
        brightness_threshold_black = 0.9
        saturation_threshold_white = 0.1  # Low saturation to consider color close to grayscale for white
        saturation_threshold_black = 0.1  # Low saturation to consider color close to grayscale for black
        w = torch.where((v >= brightness_threshold_white) & (s <= saturation_threshold_white), torch.ones_like(v), torch.zeros_like(v))
        b = torch.where((v <= brightness_threshold_black) & (s <= saturation_threshold_black), -torch.ones_like(v), torch.zeros_like(v))
        wb = w + b
        to_cat.append(wb)
                    
        batch_wb_mean = torch.mean(wb, dim=0, keepdim=True) # (1, channels, height, width)
        batch_wb_mean_tiled = batch_wb_mean.repeat(x.shape[0], 1, 1, 1)
        to_cat.append(batch_wb_mean_tiled)
    
    #sobel = sobel_edges(x)
    #to_cat.append(sobel)
    
    to_cat = [stat.to(args.device) for stat in to_cat]
    statistics = torch.cat(to_cat, dim = 1)
    return(statistics)



# Convert RGB images to HSV images.
def rgb_to_circular_hsv(rgb):
    hsv_image = rgb_to_hsv(rgb) 
    hue = hsv_image[:, 0, :, :]
    hue_sin = (torch.sin(hue) + 1) / 2
    hue_cos = (torch.cos(hue) + 1) / 2
    hsv_circular = torch.stack([hue_sin, hue_cos, hsv_image[:, 1, :, :], hsv_image[:, 2, :, :]], dim=1)
    return hsv_circular



# For making smoothly transitioning seeds.
def create_interpolated_tensor():    
    rough_latent_path = [torch.randn((args.latent_channels, 8, 8)) for i in range(args.seeds_used)]
    rough_latent_path.append(rough_latent_path[0])
    latent_path = []
    
    def smooth(t1, t2):
        smooth_path = []
        for j in range(args.seed_duration):
            p1 = 1 - (j / args.seed_duration) 
            p2 = 1 - p1
            smooth_path.append(t1 * p1 + t2 * p2)
        return(smooth_path)
    
    for i in range(args.seeds_used):
        t1 = rough_latent_path[i] 
        latent_path.append(t1)
        t2 = rough_latent_path[i+1]
        smooth_path = smooth(t1, t2)
        latent_path += smooth_path
        
    latent_path = [l.unsqueeze(0) for l in latent_path]
    latent_path = torch.cat(latent_path, dim = 0).to(args.device)
    return(latent_path)



# Pixel shuffling.
def space_to_depth(x, r):
    B, C, H, W = x.shape
    assert H % r == 0 and W % r == 0
    x = x.view(B, C, H//r, r, W//r, r).permute(0,1,3,5,2,4)
    return x.reshape(B, C*r*r, H//r, W//r)

def depth_to_space(x, r):
    B, C, H, W = x.shape
    assert C % (r*r) == 0
    x = x.view(B, C//(r*r), r, r, H, W).permute(0,1,4,2,5,3)
    return x.reshape(B, C//(r*r), H*r, W*r)

class SpaceToDepth(nn.Module):
    def __init__(self, block_size=2):
        super().__init__()
        self.r = block_size
    def forward(self, x):
        return space_to_depth(x, self.r)

class DepthToSpace(nn.Module):
    def __init__(self, block_size=2):
        super().__init__()
        self.r = block_size
    def forward(self, x):
        return depth_to_space(x, self.r)
    
    
    

# CNN with capping (CC2d).
class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(
            input, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups)

    def clamp_weights(self):
        self.weight.data.clamp_(-1.0, 1.0)
        if self.bias is not None:
            self.bias.data.clamp_(-1.0, 1.0)
            
            
            
# Add position layers.
def add_position_layers(x, learned_pos, scale = 1):
    pos = learned_pos.repeat(x.shape[0], 1, 1, 1)
    pos = F.interpolate(pos, scale_factor = scale, mode = "bilinear", align_corners = True)
    x = torch.cat([x, pos], dim = 1)
    return(x)



# Constrained CNN with positional layers 
class ConvPos(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, padding_mode, pos_sizes = []):
        super().__init__()
        
        self.positions = nn.ParameterList()
        for pos_size in pos_sizes:
            self.positions.append(nn.Parameter(torch.randn((1, in_channels, pos_size, pos_size))))
                    
        self.CC2d =  ConstrainedConv2d(
            in_channels = in_channels * (len(pos_sizes) + 1), 
            out_channels = out_channels,
            kernel_size = kernel_size,
            padding = padding,
            padding_mode = "reflect")

    def forward(self, x):
        for pos_layer in self.positions:
            scale = x.shape[-1] // pos_layer.shape[-1]
            x = add_position_layers(x, pos_layer, scale = scale)
        c = self.CC2d(x)
        return c



    
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels, kernel_size = 1):
        super().__init__()
        padding_size = ((kernel_size-1)//2, (kernel_size-1)//2)
        self.query = ConstrainedConv2d(
            in_channels = in_channels, 
            out_channels = in_channels // 8, 
            kernel_size = kernel_size, 
            padding = padding_size, 
            padding_mode = "reflect")
        self.key = ConstrainedConv2d(
            in_channels = in_channels, 
            out_channels = in_channels // 8, 
            kernel_size = kernel_size, 
            padding = padding_size, 
            padding_mode = "reflect")
        self.value = ConstrainedConv2d(
            in_channels = in_channels, 
            out_channels = in_channels, 
            kernel_size = kernel_size, 
            padding = padding_size, 
            padding_mode = "reflect")
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)   
        proj_key   = self.key(x).view(B, -1, H * W)                     
        energy     = torch.bmm(proj_query, proj_key)                  
        attention  = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(B, -1, H * W)                  
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))       
        out = out.view(B, C, H, W)
        return self.gamma * out + x
    
    
    
# Multi-Kernel Conv (MKC).
class Multi_Kernel_Conv(nn.Module):
    
    def __init__(
            self, 
            in_channels = 16, 
            out_channels = [8, 8, 8, 8], 
            kernel_sizes = [3, 3, 3, 3], 
            pos_sizes = [[], [], [], []],
            args = default_args):
        super(Multi_Kernel_Conv, self).__init__()
        
        assert len(kernel_sizes) == len(out_channels) , "kernel_size length should match out_channel length."
        
        self.CABs = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            padding_size = ((kernel_size-1)//2, (kernel_size-1)//2)
            layer = nn.Sequential(
                ConvPos(
                    in_channels = in_channels, 
                    out_channels = out_channels[i],
                    kernel_size = kernel_size,
                    padding = padding_size,
                    padding_mode = "reflect",
                    pos_sizes = pos_sizes[i]))
            self.CABs.append(layer)
                
    def forward(self, x):
        y = []
        for CAB in self.CABs: 
            y.append(CAB(x)) 
        return(torch.cat(y, dim = -3))