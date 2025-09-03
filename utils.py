#%%

import os

# Your file-location here.
file_location = r"C:\Users\Ted\OneDrive\Desktop\blorpo_diffusion"
os.chdir(file_location)

from PIL import Image
import datetime 
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse 
import builtins
from math import exp
import imageio
from statistics import log

import torch 
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n\nDevice: {}.\n\n".format(device))



# Some utilities.
def print(*args, **kwargs):
    kwargs["flush"] = True
    builtins.print(*args, **kwargs)
    
start_time = datetime.datetime.now()

def duration(start_time = start_time):
    change_time = datetime.datetime.now() - start_time
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)

def estimate_total_duration(proportion_completed, start_time=start_time):
    if(proportion_completed != 0): 
        so_far = datetime.datetime.now() - start_time
        estimated_total = so_far / proportion_completed
        estimated_total = estimated_total - datetime.timedelta(microseconds=estimated_total.microseconds)
    else: estimated_total = "?:??:??"
    return(estimated_total)



# Arguments.
parser = argparse.ArgumentParser()

    # Meta
parser.add_argument("--arg_title",                      type=str,       default = "default",
                    help='If using the cluster, an extensive name for these arguments.') 
parser.add_argument("--arg_name",                       type=str,       default = "default",
                    help='If using the cluster, a name for these arguments.')  
parser.add_argument("--agents",                         type=int,       default = 1,
                    help='If using the cluster, the number of agents trained with these arguments.') 
parser.add_argument("--previous_agents",                type=int,       default = 0,
                    help='If using the cluster, the number of agents before this one.') 
parser.add_argument("--comm",                           type=str,       default = "deigo",
                    help='If using the cluster, name of the cluster in use.') 
parser.add_argument("--init_seed",                      type=float,     default = 777,  # I'm not sure this is working. 
                    help='For consistent randomness.') 
parser.add_argument("--device",                         type=str,       default = device,
                    help='Either cpu or cuda.') 

    # Easy options
parser.add_argument("--epochs_for_vae",                 type=int,       default = 10000,
                    help='How many epochs for training?') 
parser.add_argument("--vae_lr",                         type=float,     default = .001,
                    help='Learning rate for generator.') 
parser.add_argument("--vae_dropout",                        type=int,       default = .0,
                    help='How much dropout for the discriminator?') 
parser.add_argument("--vae_beta",                        type=float,    default = .1, #.03,
                    help='Learning rate for discriminator')  

parser.add_argument("--epochs_for_unet",                type=int,       default = 500000,
                    help='How many epochs for training?') 
parser.add_argument("--unet_lr",                        type=float,     default = .001,
                    help='Learning rate for discriminator.')  
parser.add_argument("--unet_dropout",                   type=int,       default = .0,
                    help='How much dropout for the discriminator?') 
parser.add_argument("--min_noise",                      type=float,     default = .3,
                    help='Learning rate for discriminator.')  
parser.add_argument("--max_noise",                      type=float,     default = 7,
                    help='Learning rate for discriminator.')  
parser.add_argument("--change_rate",                    type=float,     default = .001,
                    help='Learning rate for discriminator.')  

parser.add_argument("--batch_size",                     type=int,       default = 64,
                    help='How large are the batches used in epochs?') 
parser.add_argument("--image_size",                     type=int,       default = 64,
                    help='How large are the pictures? (Not used much.)') 
parser.add_argument("--latent_channels",                type=int,       default = 32,
                    help='How large are some linear layers.') 
parser.add_argument('--std_min',                        type=int,       default = exp(-20),
                    help='Minimum value for standard deviation.') 
parser.add_argument('--std_max',                        type=int,       default = exp(2),
                    help='Maximum value for standard deviation.') 
parser.add_argument("--rolling_avg_num",                type=int,       default = 10,
                    help='How many values used in the rolling average?') 
parser.add_argument("--rolling_avg_val",                type=float,     default = .95,
                    help='Max value of discriminator accuracy for training.')  
parser.add_argument("--stat_quantiles",                 type=list,     default = [0.05, .5, 0.95],
                    help='Quantiles for the get_stats function.')  
parser.add_argument("--use_hsv",                        type=bool,     default = True,
                    help='Should the discriminator use the HSV?')  

    # Presentation options
parser.add_argument("--epochs_per_vid",                 type=int,       default = 250,
                    help='How often are pictures and videos saved?') 
parser.add_argument("--seeds_used",                     type=int,       default = 10,
                    help='When making pictures and videos, how many seeds?') 
parser.add_argument("--seed_duration",                  type=int,       default = 10,
                    help='When making pictures and videos, how many steps transationing from one to another?') 

parser.add_argument("--actual_noise_list",              type=list,       default = [999,    2,  1,  .5],
                    help='When making pictures and videos, how many steps transationing from one to another?') 
parser.add_argument("--lied_noise_list",                type=list,       default = [7,      1.75, .75,     .1],
                    help='When making pictures and videos, how many steps transationing from one to another?') 




# Comparing used arguments to default arguments.
try:
    default_args = parser.parse_args([])
    try:    args    = parser.parse_args()
    except: args, _ = parser.parse_known_args()
except:
    import sys ; sys.argv=[''] ; del sys           
    default_args = parser.parse_args([])
    try:    args    = parser.parse_args()
    except: args, _ = parser.parse_known_args()
    
    
        
# Making a title for these arguments.
args_not_in_title = ["arg_title", "init_seed"]
def get_args_title(default_args, args):
    if(args.arg_title[:3] == "___"): return(args.arg_title)
    name = "" ; first = True
    arg_list = list(vars(default_args).keys())
    arg_list.insert(0, arg_list.pop(arg_list.index("arg_name")))
    for arg in arg_list:
        if(arg in args_not_in_title): pass 
        else: 
            default, this_time = getattr(default_args, arg), getattr(args, arg)
            if(this_time == default): pass
            elif(arg == "arg_name"):
                name += "{} (".format(this_time)
            else: 
                if first: first = False
                else: name += ", "
                name += "{}: {}".format(arg, this_time)
    if(name == ""): name = "default" 
    else:           name += ")"
    if(name.endswith(" ()")): name = name[:-3]
    parts = name.split(',')
    name = "" ; line = ""
    for i, part in enumerate(parts):
        if(len(line) > 50 and len(part) > 2): name += line + "\n" ; line = ""
        line += part
        if(i+1 != len(parts)): line += ","
    name += line
    return(name)

args.arg_title = get_args_title(default_args, args)



# Use random seed.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(int(args.init_seed))



# Collecting pictures.
transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor()])

image_files = [f for f in os.listdir("real_images") if os.path.isfile(os.path.join("real_images", f))]
image_files = [f for f in image_files if f != "original.png"]
image_files.sort()
images = []
for file_name in image_files:
    image_path = os.path.join("real_images", file_name)
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image_tensor = transform(image)
    images.append(image_tensor)
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_image_tensor = transform(flipped_image)
    images.append(flipped_image_tensor)
all_images_tensor = torch.stack(images).to(device)
these_pokemon = [1, 4, 7, 25, 38, 53, 63, 69, 83, 140]
these_pokemon = [(i-1)*2 for i in these_pokemon]
pokemon_sample = all_images_tensor[these_pokemon]



# For batch collection.
def get_random_batch(all_images_tensor = all_images_tensor, batch_size=64):
    num_images = all_images_tensor.size(0)
    indices = random.sample(range(num_images), batch_size)
    batch_tensor = all_images_tensor[indices]
    return batch_tensor



def save_vae_comparison_grid(real_pokemon, vae_pokemon, noisy_pokemon, unet_pokemon, filename, std):
    # Move to CPU for plotting
    
    N = real_pokemon.size(0)
    # Build a 2 x N grid: top = originals, bottom = reconstructions
    fig, axes = plt.subplots(5, N, figsize=(2*N, 4), squeeze=False)
    for i in range(N):
        # Originals
        ax = axes[0, i]
        ax.imshow(real_pokemon[i].cpu().permute(1, 2, 0).numpy())
        ax.axis("off")
        if i == 0:
            ax.set_title("original", fontsize=10)

        # Reconstructions
        ax = axes[1, i]
        ax.imshow(vae_pokemon[i].cpu().permute(1, 2, 0).numpy())
        ax.axis("off")
        if i == 0:
            ax.set_title("reconstruction", fontsize=10)
            
        # Reconstructions after unet
        ax = axes[2, i]
        ax.imshow(noisy_pokemon[i].cpu().permute(1, 2, 0).numpy())
        ax.axis("off")
        if i == 0:
            ax.set_title("reconstruction of noisy", fontsize=10)
            
        # Reconstructions after unet
        ax = axes[3, i]
        ax.imshow(unet_pokemon[i].cpu().permute(1, 2, 0).numpy())
        ax.axis("off")
        if i == 0:
            ax.set_title("reconstruction w/ unet", fontsize=10)
            
        ax = axes[4, i]
        ax.text(0.5, 0.5, f"{round(std[i].item(), 2)}")
        ax.axis("off")
        if i == 0:
            ax.set_title("noise", fontsize=10)
        
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)



# Make pictures, then make gif transitioning between them.
def show_images_from_tensor(image_tensor, save_path='output_folder', fps=10):
    os.makedirs(save_path, exist_ok=True)

    image_tensor = image_tensor.detach()
    if image_tensor.dim() == 5:
        N, T, C, H, W = image_tensor.shape
        animate = True
    elif image_tensor.dim() == 4:
        N, C, H, W = image_tensor.shape
        T = 1
        animate = False
    else:
        raise ValueError("Unexpected tensor shape")

    frames = []
    frame_index = 1
    for t in range(T):
        for i in range(N):
            img = image_tensor[i, t] if animate else image_tensor[i]
            img = img.permute(1, 2, 0).to("cpu").numpy()

            # Normalize the image to be between 0 and 1
            img = (img - img.min()) / (img.max() - img.min())

            # Convert numpy array to PIL image directly
            pil_image = Image.fromarray((img * 255).astype(np.uint8))  # Assuming image is in [0, 1] range

            # Save the image as a PNG file in the specified folder
            image_filename = os.path.join(save_path, f'{frame_index}.png')
            pil_image.save(image_filename)

            # Append image to frames list for GIF creation
            frames.append(pil_image)
            frame_index += 1

    # Create and save the GIF
    gif_path = os.path.join(save_path, 'animation.gif')
    resized_frames = [frame.resize((frame.width * 20, frame.height * 20), Image.NEAREST) for frame in frames]
    resized_frames[0].save(gif_path, save_all=True, append_images=resized_frames[1:], loop=0, duration=1000//fps)
    
    
    
# Make gifs for epoch-to-epoch progress.
def make_animation(save_dir, image_name='1.png', output_name='animation_1.gif'):
    print(f"Animating {image_name}.")
    # Get list of epoch folders sorted by epoch number
    folders = sorted(
        [f for f in os.listdir(save_dir) if f.startswith('epoch_')],
        key=lambda x: int(x.split('_')[1]))

    images = []
    for folder in folders:
        print(f"Folder {folder}...", end = " ")
        path = os.path.join(save_dir, folder, image_name)
        if os.path.exists(path):
            images.append(imageio.imread(path))
    
    output_path = os.path.join(save_dir, output_name)
    imageio.mimsave(output_path, images, fps=5)
    print(f"Saved animation to {output_path}")
    
    
    
# MUST BE CHANGED FOR STABLE DIFFUSION
# Plotting losses, entropy, curiosity, etc.
def plot_vals(plot_vals_dict, save_path='losses.png', fontsize=7):

    vae_epochs = [i for i in range(len(plot_vals_dict["vae_loss"]))]
    unet_epochs = [i for i in range(len(plot_vals_dict["unet_loss"]))]
    epsilon = 1e-8
    
    # Plotting
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(vae_epochs, plot_vals_dict["vae_mu"], 'red', label="VAE MU", alpha=0.8)
    plt.plot(vae_epochs, plot_vals_dict["vae_std"], 'green', label="VAE STD", alpha=0.8)
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    #plt.ylim(-1, 25)
    ymin, ymax = plt.ylim()   # get current y-axis range
    #yticks = np.arange(np.floor(ymin*10)/10, np.ceil(ymax*10)/10 + 0.1, 0.1)
    #plt.yticks(yticks)
    plt.title("VAE MU and STD")
    plt.legend(fontsize=fontsize)
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(vae_epochs, [l + epsilon for l in plot_vals_dict["vae_loss"]], 'red', label="VAE Loss", alpha=0.8)
    plt.plot(vae_epochs, [l + epsilon for l in plot_vals_dict["dkl_1_loss"]], 'green', label="DKL 1 Loss", alpha=0.8)
    plt.plot(vae_epochs, [l + epsilon for l in plot_vals_dict["dkl_2_loss"]], 'blue', label="DKL 2 Loss", alpha=0.8)
    plt.plot(vae_epochs, [l_1 + l_2 + epsilon for l_1, l_2 in zip(plot_vals_dict["dkl_1_loss"], plot_vals_dict["dkl_2_loss"])], 'black', label="Total DKL Loss", alpha=0.8)

    plt.xlabel("Epochs")
    plt.yscale("log")
    plt.ylabel("VAE Loss")
    #plt.ylim(-1, 25)
    ymin, ymax = plt.ylim()   # get current y-axis range
    #yticks = np.arange(np.floor(ymin*10)/10, np.ceil(ymax*10)/10 + 0.1, 0.1)
    #plt.yticks(yticks)
    plt.title("VAE Losses")
    plt.legend(fontsize=fontsize)
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(unet_epochs, [l + epsilon for l in plot_vals_dict["unet_loss"]], 'red', label="UNET Loss", alpha=0.8)
    plt.xlabel("UNET Epochs")
    plt.yscale("log")
    plt.ylabel("UNET Loss")
    #plt.ylim(-1, 25)
    ymin, ymax = plt.ylim()   # get current y-axis range
    #yticks = np.arange(np.floor(ymin*10)/10, np.ceil(ymax*10)/10 + 0.1, 0.1)
    #plt.yticks(yticks)
    plt.title("UNET Losses")
    plt.legend(fontsize=fontsize)
    plt.grid(True)

    name_part_1 = 'VAE' if len(unet_epochs) == 0 else 'UNET'
    name_part_2 = len(vae_epochs) if len(unet_epochs) == 0 else len(unet_epochs)
    plt.tight_layout()
    plt.savefig(save_path + f"/losses_{name_part_1}.png")
    plt.close()
    
    

# Quick example.
if(__name__ == "__main__"):
    print("Shape of all data:", all_images_tensor.shape)
    
    # Compute the average image
    avg_image = all_images_tensor.mean(dim=0)  # shape: (C, H, W)

    # Convert to numpy format for plotting
    avg_image_np = avg_image.permute(1, 2, 0).cpu().numpy()
    avg_image_np = (avg_image_np - avg_image_np.min()) / (avg_image_np.max() - avg_image_np.min())  # normalize to [0,1]

    # Plot and show the image
    plt.figure(figsize=(4, 4))
    plt.imshow(avg_image_np)
    plt.axis('off')
    plt.title("Average Image")
    plt.tight_layout()
    plt.savefig("average_of_all_images.png")
# %%