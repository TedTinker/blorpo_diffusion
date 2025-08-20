#%% 
import os
from utils import file_location
os.chdir(file_location)

from collections import Counter
import datetime
import numpy as np
import operator
import pickle

import torch 
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Normal

from utils import default_args, pokemon_sample, plot_vals, get_random_batch, print, duration, show_images_from_tensor, save_vae_comparison_grid
from utils_for_torch import sample, create_interpolated_tensor, ConstrainedConv2d
from vae import VAE
#from unet import UNET
from unet_small import UNET



# Let's put all this together!
class SD:
    def __init__(self, args = default_args):
        self.args = args
        
        # Folder to save in.
        self.gen_location = file_location + f"/generated_images/{self.args.arg_name}"
        if not os.path.exists(self.gen_location):
            os.makedirs(self.gen_location)
        
        self.epochs_for_vae = 1
        self.epochs_for_unet = 1
        self.session_start_time = datetime.datetime.now()
        
        self.vae = VAE()
        try:
            self.vae.load_state_dict(torch.load(self.gen_location + "/vae.py", weights_only=True))
            self.args.epochs_for_vae = 0
        except:
            pass
        self.vae_opt = Adam(self.vae.parameters(), args.vae_lr, weight_decay=.0001)
        self.vae.train()
        
        self.unet = UNET()
        self.unet_opt = Adam(self.unet.parameters(), args.unet_lr, weight_decay=.0001)
        self.unet.train()
        
        self.current_noise = self.args.min_noise
        self.plot_vals_dict = {
            "vae_loss" : [],
            "dkl_loss" : [],
            "unet_loss" : []}
        self.pokemon = pokemon_sample
        self.loop = create_interpolated_tensor()
                
        std = [i * self.args.max_noise / 10 for i in range(10)]
        self.std = torch.tensor(std).to(self.args.device)
        
        
        
    def save_vae(self):
        torch.save(self.vae.state_dict(), self.gen_location + "/vae.py")




    
    # Interaction between vae and unet.
    def vae_vs_unet(
            self,
            vae_train = False,
            unet_train = False,
            new_batch = True):
        
        if(vae_train):  self.vae.train()
        else:           self.vae.eval()
        if(unet_train): self.unet.train()
        else:           self.unet.eval()
        if(new_batch):  real_imgs = get_random_batch(batch_size=self.args.batch_size)
        else:           real_imgs = self.pokemon
            
        if(unet_train or not new_batch):
            with torch.no_grad():                          
                decoded, encoded, dkl = self.vae(real_imgs, use_std=False)
        else:
            decoded, encoded, dkl = self.vae(real_imgs)
            
        if(new_batch):  
            std = torch.rand(size=(encoded.size(0),), device=self.args.device) * self.current_noise
            std = std.clamp_(self.args.min_noise, self.args.max_noise)
        else:           
            std = self.std
            encoded = encoded[0].repeat(10, 1, 1, 1)
        
        std = std.view((std.shape[0], 1, 1, 1)).repeat(1, encoded.shape[1], encoded.shape[2], encoded.shape[3])
        epsilon = Normal(0, 1).sample(std.shape).to(std.device)    
        noisy_encoded = encoded + std * epsilon
        
        predicted_epsilon = self.unet(noisy_encoded, std)
        predicted_encoded = noisy_encoded - predicted_epsilon * std
        
        with torch.no_grad():   
            self.vae.eval()
            noisy_imgs = (self.vae.decode(noisy_encoded) + 1) / 2
            predicted_imgs = (self.vae.decode(predicted_encoded) + 1) / 2
            
        return {
            "real_imgs" : real_imgs,
            "encoded" : encoded,
            "decoded" : decoded,
            "dkl" : dkl,
            "std" : std,
            "epsilon" : epsilon,
            "predicted_epsilon" : predicted_epsilon,
            "predicted_encoded" : predicted_encoded,
            "noisy_imgs" : noisy_imgs,
            "predicted_imgs" : predicted_imgs
            }
    
    
        
    # Train VAE.
    def epoch_for_vae(self):
        
        real_imgs, decoded, dkl = operator.itemgetter(
            "real_imgs", "decoded", "dkl")(
                self.vae_vs_unet(vae_train = True))
        
        recon_loss = F.l1_loss(decoded, real_imgs)   
        dkl_loss = self.args.vae_beta * dkl            
        vae_loss = recon_loss + dkl_loss
    
        self.vae_opt.zero_grad(set_to_none=True)
        vae_loss.backward()
        self.vae_opt.step()
        
        self.plot_vals_dict["vae_loss"].append(vae_loss.detach().cpu())
        self.plot_vals_dict["dkl_loss"].append(dkl_loss.detach().cpu())
            
        for module in self.vae.modules():
            if isinstance(module, ConstrainedConv2d):
                module.clamp_weights()
                
        if self.epochs_for_vae % self.args.epochs_per_vid == 0:
            print("Saving VAE example...")
            self.save_examples(
                grid_save_pos = self.gen_location + f"/VAE_epoch_{self.epochs_for_vae}.png",
                val_save_pos = self.gen_location)
        torch.cuda.empty_cache()
                
        self.epochs_for_vae += 1
        #print(self.epochs_for_vae, end = "... ")
        
        
        
    # Train UNET.
    def epoch_for_unet(self):
        
        epsilon, predicted_epsilon = operator.itemgetter(
            "epsilon", "predicted_epsilon")(
                self.vae_vs_unet(unet_train = True))
        unet_loss = F.mse_loss(predicted_epsilon, epsilon)
    
        self.unet_opt.zero_grad(set_to_none=True)
        unet_loss.backward()
        self.unet_opt.step()
        
        self.plot_vals_dict["unet_loss"].append(unet_loss.detach().cpu())

        #for module in self.unet.modules():
        #    if isinstance(module, ConstrainedConv2d):
        #        module.clamp_weights()
    
        if self.epochs_for_unet % self.args.epochs_per_vid == 0:
            print(f"Saving UNET examples... (Current Noise: {round(self.current_noise, 2)})")
            self.save_examples(
                grid_save_pos = self.gen_location + f"/UNET_epoch_{self.epochs_for_unet}/UNET_epoch_{self.epochs_for_unet}.png",
                val_save_pos = self.gen_location + f"/UNET_epoch_{self.epochs_for_unet}")
            for rate in [.4, .5, .6]:
                imgs = self.unet_loop(rate)
                save_rel = file_location + f"/generated_images/{self.args.arg_name}/UNET_epoch_{self.epochs_for_unet}/{rate}"
                show_images_from_tensor(imgs, save_path=save_rel, fps=10)  
        torch.cuda.empty_cache()
        
        if(self.current_noise < self.args.max_noise):
            self.current_noise += self.args.change_rate
        else:
            self.current_noise = self.args.max_noise
        self.epochs_for_unet += 1
        #print(self.epochs_for_unet, end = "... ")
        
        
    
    # Let's do this!
    def training(self):
        
        print("\nTRAINING VAE:")
        for epoch in range(self.args.epochs_for_vae):
            self.epoch_for_vae()
            if(epoch % 25 == 0):
                percent_done = round(100 * self.epochs_for_vae / self.args.epochs_for_vae, 2)
                print(f"{percent_done}%", end = "... ")
                
        self.save_vae()
                
                
        print("\n\nTRAINING UNET:")
        for epoch in range(self.args.epochs_for_unet):
            self.epoch_for_unet()
            if(epoch % 25 == 0):
                percent_done = round(100 * self.epochs_for_unet / self.args.epochs_for_unet, 2)
                print(f"{percent_done}%", end = "... ")
                
                

        
    def save_examples(self, grid_save_pos, val_save_pos):
        with torch.no_grad():
            decoded, noisy_imgs, predicted_imgs = operator.itemgetter(
                "decoded", "noisy_imgs", "predicted_imgs")(
                    self.vae_vs_unet(new_batch = False))
        save_vae_comparison_grid(self.pokemon, decoded, noisy_imgs, predicted_imgs, grid_save_pos)
        plot_vals(self.plot_vals_dict, val_save_pos)
        
    
    
    @torch.no_grad()
    def unet_loop(self, rate):
        self.vae.eval() 
        self.unet.eval()
        img = self.loop.clone()
        
        current_noise = torch.tensor(float(self.args.max_noise)).to(self.args.device)

        while current_noise > .1:
            _, encoded, _ = self.vae(img, use_std = False)
            std = current_noise.view(1,1,1,1).expand_as(encoded)
            epsilon = Normal(0, 1).sample(std.shape).to(std.device)    
            noisy_encoded = encoded + std * epsilon
            eps_hat = self.unet(noisy_encoded, std) * std    
            encoded = encoded - eps_hat
            img = (self.vae.decode(encoded) + 1) / 2
            current_noise *= rate
            
        return img

            
        
        
# :D
if(__name__ == "__main__"):
    sd = SD()
    sd.training()
    
    


    