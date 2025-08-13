#%% 
import os
from utils import file_location
os.chdir(file_location)

from collections import Counter
import datetime
import numpy as np

import torch 
from torch.optim import Adam
import torch.nn.functional as F

from utils import default_args, pokemon_sample, get_random_batch, print, duration, show_images_from_tensor, save_vae_comparison_grid
from utils_for_torch import create_interpolated_tensor, ConstrainedConv2d
from vae import VAE
#from unet import UNET
from unet_small import UNET



# Let's put all this together!
class SD:
    def __init__(self, args = default_args):
        self.args = args
        
        # Folder to save in.
        folder_name = file_location + "/generated_images/" + str(self.args.arg_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        self.epochs_for_vae = 1
        self.epochs_for_unet = 1
        self.session_start_time = datetime.datetime.now()
        
        self.vae = VAE()
        self.vae_opt = Adam(self.vae.parameters(), args.vae_lr)
        self.vae.train()
        
        self.unet = UNET()
        self.unet_opt = Adam(self.unet.parameters(), args.unet_lr)
        self.unet.train()
        
        self.plot_vals_dict = {}
        self.pokemon = pokemon_sample
        self.loop = create_interpolated_tensor(self.args).to(self.args.device)
        
        self.T = 1000
        self.alpha_bar = torch.cumprod(1 - torch.linspace(1e-4, 2e-2, self.T, device=self.args.device), dim=0)  
        
        
    
    @torch.no_grad()
    def vae_test(self):
        self.vae.eval()
        imgs, _, _, _, _ = self.vae(self.pokemon, use_logvar=False)
        return imgs



    @torch.no_grad()
    def unet_loop(self):
        self.vae.eval()
        self.unet.eval()
        predicted_epsilon = self.unet(self.loop, torch.tensor((3 * self.T / 10,)).to(self.args.device))
        new_loop = self.loop - predicted_epsilon
        imgs = (self.vae.decode(new_loop) + 1) / 2
        return imgs



    @torch.no_grad()
    def unet_real(self, predicted_encodings):
        self.vae.eval()
        self.unet.eval()
        imgs = (self.vae.decode(predicted_encodings) + 1) / 2
        return imgs

        
        
    # Train VAE.
    def epoch_for_vae(self):
        self.vae.train()
        imgs = get_random_batch(batch_size=self.args.batch_size)
        decoded, encoded, mu, logvar, dkl = self.vae(imgs)  # returns all five. :contentReference[oaicite:3]{index=3}
        recon_loss = F.l1_loss(decoded, imgs)               # L1 works well at 64×64
        vae_beta = 0.05                                     # small DKL pressure
        vae_loss = recon_loss + vae_beta * dkl
    
        self.vae_opt.zero_grad(set_to_none=True)
        vae_loss.backward()
        self.vae_opt.step()
            
        for module in self.vae.modules():
            if isinstance(module, ConstrainedConv2d):
                module.clamp_weights()
                
        if self.epochs_for_vae % self.args.epochs_per_vid == 0:
            print("Saving VAE example...")
            with torch.no_grad():
                imgs = self.vae_test()
            save_rel = file_location + f"/generated_images/{self.args.arg_name}/VAE_epoch_{self.epochs_for_vae}.png"
            save_vae_comparison_grid(self.pokemon, imgs, save_rel)
        torch.cuda.empty_cache()
                
        self.epochs_for_vae += 1
        #print(self.epochs_for_vae, end = "... ")
        
        
        
    # Train UNET.
    def epoch_for_unet(self):
        self.vae.eval()
        self.unet.train()
        real_imgs = get_random_batch(batch_size=self.args.batch_size)
    
        with torch.no_grad():                          
            _, encoded, _, _, _ = self.vae(real_imgs, use_logvar=False)
    
        t = torch.randint(low=0, high=self.T, size=(encoded.size(0),), device=self.args.device) 
        alpha_t = self.alpha_bar[t].sqrt().view(encoded.size(0), 1, 1, 1)
        sigma_t = (1.0 - self.alpha_bar[t]).sqrt().view(encoded.size(0), 1, 1, 1)
        epsilon = torch.randn_like(encoded)
        z_t = alpha_t * encoded + sigma_t * epsilon
    
        predicted_epsilon = self.unet(z_t, t)
        unet_loss = F.mse_loss(predicted_epsilon, epsilon)
    
        self.unet_opt.zero_grad(set_to_none=True)
        unet_loss.backward()
        self.unet_opt.step()
        
        for module in self.unet.modules():
            if isinstance(module, ConstrainedConv2d):
                module.clamp_weights()
    
        if self.epochs_for_unet % self.args.epochs_per_vid == 0:
            print("Saving UNET examples...")
            with torch.no_grad():
                imgs = self.unet_loop()
            save_rel = f"/{self.args.arg_name}/UNET_epoch_{self.epochs_for_unet}"
            show_images_from_tensor(imgs, save_path=save_rel, fps=10)  
            
            with torch.no_grad():
                _, encoded, _, _, _ = self.vae(self.pokemon, use_logvar=False)
                
                T = self.T
                ts = [T-1, int(8*T/10), int(7*T/10), int(6*T/10), int(5*T/10), int(4*T/10), int(3*T/10), int(2*T/10), int(T/10), 0]
                t = torch.tensor(ts).to(self.args.device)
                alpha_t = self.alpha_bar[t].sqrt().view(encoded.size(0), 1, 1, 1)
                sigma_t = (1.0 - self.alpha_bar[t]).sqrt().view(encoded.size(0), 1, 1, 1)
                epsilon = torch.randn_like(encoded)
                z_t = alpha_t * encoded + sigma_t * epsilon
                predicted_epsilon = self.unet(z_t, t)
                predicted_encodings = z_t - predicted_epsilon
                imgs = (self.vae.decode(predicted_encodings) + 1) / 2
            save_rel = file_location + f"/generated_images/{self.args.arg_name}/UNET_real_epoch_{self.epochs_for_unet}.png"
            save_vae_comparison_grid(self.pokemon, imgs, save_rel) 
        torch.cuda.empty_cache()
        
        self.epochs_for_unet += 1
        #print(self.epochs_for_unet, end = "... ")
        
        
    
    # Let's do this!
    def training(self):
        
        with torch.no_grad():
            imgs = self.vae_test()
        save_rel = file_location + f"/generated_images/{self.args.arg_name}/VAE_epoch_1.png"
        save_vae_comparison_grid(self.pokemon, imgs, save_rel)
        
        print("\nTRAINING VAE:")
        for epoch in range(self.args.epochs_for_vae):
            self.epoch_for_vae()
            if(epoch % 25 == 0):
                percent_done = round(100 * self.epochs_for_vae / self.args.epochs_for_vae, 2)
                print(f"{percent_done}%", end = "... ")
                
        with torch.no_grad():
            imgs = self.unet_loop()
        save_rel = f"/{self.args.arg_name}/UNET_epoch_1"
        show_images_from_tensor(imgs, save_path=save_rel, fps=10)
                
        print("\n\nTRAINING UNET:")
        for epoch in range(self.args.epochs_for_unet):
            self.epoch_for_unet()
            if(epoch % 25 == 0):
                percent_done = round(100 * self.epochs_for_unet / self.args.epochs_for_unet, 2)
                print(f"{percent_done}%", end = "... ")
            
                
        
        
# :D
if(__name__ == "__main__"):
    sd = SD()
    sd.training()
    
    


    