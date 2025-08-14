#%% 
import os
from utils import file_location
os.chdir(file_location)

from collections import Counter
import datetime
import numpy as np
import operator

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
        
        T = 1000
        ts = [0, int(T/10), int(2*T/10), int(3*T/10), int(4*T/10), int(5*T/10), 
              int(6*T/10), int(7*T/10), int(8*T/10), int(9*T/10)]
        self.t = torch.tensor(ts).to(self.args.device)
        self.T = T
        self.alpha_bar = torch.cumprod(1 - torch.linspace(1e-4, 2e-2, self.T, device=self.args.device), dim=0)  



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
    
    
    
    # Interaction between vae and unet.
    def vae_vs_unet(
            self,
            vae_train = False,
            unet_train = False,
            new_batch = True,
            t = None):
        
        if(vae_train):
            self.vae.train()
        else:
            self.vae.eval()
            
        if(unet_train):
            self.unet.train()
        else:
            self.unet.eval()
    
        if(new_batch):
            real_imgs = get_random_batch(batch_size=self.args.batch_size)
        else:
            real_imgs = self.pokemon
            
        if(unet_train):
            with torch.no_grad():                          
                decoded, encoded, mu, logvar, dkl = self.vae(real_imgs, use_logvar=False)
        else:
            decoded, encoded, mu, logvar, dkl = self.vae(real_imgs)
            
        if(t == None):
            t = torch.randint(low=0, high=self.T, size=(encoded.size(0),), device=self.args.device) 
        alpha_t = self.alpha_bar[t].sqrt().view(encoded.size(0), 1, 1, 1)
        sigma_t = (1.0 - self.alpha_bar[t]).sqrt().view(encoded.size(0), 1, 1, 1)
        epsilon = torch.randn_like(encoded)
        if(not new_batch):
            encoded = encoded[0].repeat(10, 1, 1, 1)
        z_t = alpha_t * encoded + sigma_t * epsilon
        predicted_epsilon = self.unet(z_t, t)
        
        predicted_encodings = z_t - predicted_epsilon
        with torch.no_grad():   
            self.vae.eval()
            predicted_imgs = (self.vae.decode(predicted_encodings) + 1) / 2
            
        return {
            "real_imgs" : real_imgs,
            "decoded" : decoded,
            "encoded" : encoded,
            "mu" : mu,
            "logvar" : logvar,
            "dkl" : dkl,
            "epsilon" : epsilon,
            "z_t" : z_t,
            "predicted_epsilon" : predicted_epsilon,
            "predicted_imgs" : predicted_imgs
            }
    
    
        
    # Train VAE.
    def epoch_for_vae(self):
        
        real_imgs, decoded, dkl = operator.itemgetter(
            "real_imgs", "decoded", "dkl")(
                self.vae_vs_unet(vae_train = True))
        
        recon_loss = F.l1_loss(decoded, real_imgs)               
        vae_loss = recon_loss + self.args.vae_beta * dkl
    
        self.vae_opt.zero_grad(set_to_none=True)
        vae_loss.backward()
        self.vae_opt.step()
            
        for module in self.vae.modules():
            if isinstance(module, ConstrainedConv2d):
                module.clamp_weights()
                
        if self.epochs_for_vae % self.args.epochs_per_vid == 0:
            print("Saving VAE example...")
            with torch.no_grad():
                decoded, predicted_imgs = operator.itemgetter(
                    "decoded", "predicted_imgs")(
                        self.vae_vs_unet(new_batch = False, t = self.t))
            save_rel = file_location + f"/generated_images/{self.args.arg_name}/VAE_epoch_{self.epochs_for_vae}.png"
            save_vae_comparison_grid(self.pokemon, decoded, predicted_imgs, save_rel)
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
                
                vae_imgs, predicted_imgs = operator.itemgetter(
                    "decoded", "predicted_imgs")(
                        self.vae_vs_unet(new_batch = False, t = self.t))
            save_rel = file_location + f"/generated_images/{self.args.arg_name}/UNET_real_epoch_{self.epochs_for_unet}.png"
            save_vae_comparison_grid(self.pokemon, vae_imgs, predicted_imgs, save_rel) 
        torch.cuda.empty_cache()
        
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
    
    


    