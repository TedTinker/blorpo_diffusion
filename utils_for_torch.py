import math

import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F
from kornia.color import rgb_to_hsv 

from utils import default_args




# For starting neural networks.
def init_weights(m):
    try:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
    except: pass

# How to use mean and standard deviation layers.
def var(x, mu_func, std_func, args):
    mu = mu_func(x)
    std = torch.clamp(std_func(x), min = args.std_min, max = args.std_max)
    return(mu, std)

# How to sample from probability distributions.
def sample(mu, std):
    e = Normal(0, 1).sample(std.shape).to(std.device)
    return(mu + e * std)



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
        batch_wb_mean_tiled = batch_wb_mean.repeat(args.batch_size, 1, 1, 1)
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
def make_fourier_loop(num_frames, latent_dim, num_frequencies=4, generator=None):
    t = torch.linspace(0, 2 * math.pi, num_frames, dtype=torch.float32)
    latent_path = torch.zeros(num_frames, latent_dim)
    for k in range(1, num_frequencies + 1):
        a_k = torch.randn(latent_dim, generator=generator) / k
        b_k = torch.randn(latent_dim, generator=generator) / k
        latent_path += torch.sin(k * t[:, None]) * a_k + torch.cos(k * t[:, None]) * b_k
    latent_path = latent_path / latent_path.std()
    return latent_path

def create_interpolated_tensor(args):
    g = torch.Generator()
    g.manual_seed(int(args.init_seed))
    return make_fourier_loop(
        num_frames=args.seeds_used * args.seed_duration,
        latent_dim=args.seed_size,
        num_frequencies=4,
        generator=g)



# Add position layers.
def add_position_layers(x, learned_pos, scale = 1):
    pos = learned_pos.repeat(x.shape[0], 1, 1, 1)
    pos = F.interpolate(pos, scale_factor = scale, mode = "bilinear", align_corners = True)
    x = torch.cat([x, pos], dim = 1)
    return(x)



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
    
    
    

# CNN with capping.
class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(
            input, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups)

    def clamp_weights(self):
        self.weight.data.clamp_(-1.0, 1.0)
    
    
    
# ResBlock. Return the input, plus something.
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConstrainedConv2d(channels, channels, 3, padding=1)
        self.act   = nn.SiLU()
        self.conv2 = ConstrainedConv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return out + x  

    

# Attention layers.
class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=4)

    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten spatial dimensions so each pixel is a "token"
        tokens = x.view(B, C, H*W).permute(2, 0, 1)  # [HW, B, C]
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out = attn_out.permute(1, 2, 0).view(B, C, H, W)
        return attn_out + x  # skip connection
    
    
    
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
    
    
    
# My personal kind of layer. Allows growing, shrinking, and attention.
class CNN_Attention_Blend(nn.Module):
    def __init__(self, 
                 in_shape = (16, 16, 16, 16), 
                 out_channels = 32, 
                 kernel_size = 3, 
                 grow = False,
                 shrink = False, 
                 paying_attention = False, 
                 attention_kernel_size = 1, 
                 args = default_args):
        super(CNN_Attention_Blend, self).__init__()
        
        # This is my kludgey way to make inputs into self-values.
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
        # This is my kludgey way to see qualities that layers should have.
        example = torch.zeros(in_shape)
        print("Start of CAB:", example.shape)
        
        mid_channels = out_channels
        if(self.shrink and paying_attention):
            mid_channels = in_shape[1]
        if(self.grow or (not self.grow and not self.shrink)):
            mid_channels = in_shape[1]
        
        padding_size = ((kernel_size-1)//2, (kernel_size-1)//2)
        
        if(self.grow or (not self.grow and not self.shrink)):
            self.x_in = nn.Sequential(
                ConstrainedConv2d(
                    in_channels = example.shape[1], 
                    out_channels = mid_channels,
                    kernel_size = kernel_size,
                    padding = padding_size,
                    padding_mode = "reflect"),
                nn.BatchNorm2d(mid_channels),
                nn.LeakyReLU())
            
            example_2 = self.x_in(example)
            print("CAB in:", example.shape)
            
        if(self.shrink):
            self.x_in = nn.Sequential(
                ConstrainedConv2d(
                    in_channels = example.shape[1], 
                    out_channels = mid_channels,
                    kernel_size = kernel_size,
                    padding = padding_size,
                    padding_mode = "reflect"),
                SpaceToDepth(block_size=2),  
                nn.BatchNorm2d(mid_channels * 4),
                nn.LeakyReLU())
            
            example_2 = self.x_in(example)
            print("CAB in (shrink):", example.shape)
        
        
        
        if(paying_attention):
            if(self.shrink):
                example = space_to_depth(example, 2)
            self.attention = nn.Sequential(
                SelfAttention(in_channels = example.shape[1], kernel_size = attention_kernel_size))
            example = self.attention(example)
            example_2 = example + example_2
            print("CAB attention:", example_2.shape)
            
            
            
        if(self.shrink or (not self.grow and not self.shrink)):
            self.x_out = nn.Sequential(
                ConstrainedConv2d(
                    in_channels = example_2.shape[1], 
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    padding = padding_size,
                    padding_mode = "reflect"))

            example = self.x_out(example_2)
            print("CAB out:", example.shape)
            
        if(self.grow):
            self.x_out = nn.Sequential(
                ConstrainedConv2d(
                    in_channels = example_2.shape[1],
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    padding = padding_size,
                    padding_mode = "reflect"),
                nn.Upsample(
                    scale_factor = 2,
                    mode = "bilinear",
                    align_corners = True))
            
            example = self.x_out(example_2)
            print("CAB out (grow):", example.shape)
    
    def forward(self, x):
        x_2 = self.x_in(x)
        if(self.paying_attention):
            if(self.shrink):
                x = space_to_depth(x, 2)
            attention = self.attention(x)
            x_2 = x_2 + attention
        x = self.x_out(x_2)
        return x
    
    
    
# Multi-Kernel CAB (MKC).
class Multi_Kernel_CAB(nn.Module):
    
    def __init__(
            self, 
            in_shape = (16, 16, 16, 16), 
            out_channels = [8, 8, 8, 8], 
            kernel_sizes = [3, 3, 3, 3], 
            grow = False,
            shrink = False, 
            paying_attention = False, 
            attention_kernel_sizes = [1, 1, 1, 1], 
            args = default_args):
        super(Multi_Kernel_CAB, self).__init__()
        
        assert len(kernel_sizes) == len(out_channels) , "kernel_size length should match out_channel length."
        if(paying_attention):
            assert len(kernel_sizes) == len(attention_kernel_sizes) , "kernel_sizes length should match attention_kernel_size length."
        else:
            attention_kernel_sizes = kernel_sizes
        
        self.CABs = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            layer = nn.Sequential(
                CNN_Attention_Blend(
                     in_shape = in_shape, 
                     out_channels = out_channels[i], 
                     kernel_size = kernel_sizes[i], 
                     grow = grow,
                     shrink = shrink, 
                     paying_attention = paying_attention, 
                     attention_kernel_size = attention_kernel_sizes[i], 
                     args = default_args))
            self.CABs.append(layer)
                
    def forward(self, x):
        y = []
        for CAB in self.CABs: 
            y.append(CAB(x)) 
        return(torch.cat(y, dim = -3))