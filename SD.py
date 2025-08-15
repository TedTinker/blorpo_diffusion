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

from utils import default_args, pokemon_sample, plot_vals, get_random_batch, print, duration, show_images_from_tensor, save_vae_comparison_grid
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
        
        self.plot_vals_dict = {
            "vae_loss" : [],
            "dkl_loss" : [],
            "unet_loss" : []}
        self.pokemon = pokemon_sample
        self.loop = create_interpolated_tensor(self.args).to(self.args.device)
        
        T = 1000
        ts = [0, int(T/10), int(2*T/10), int(3*T/10), int(4*T/10), int(5*T/10), 
              int(6*T/10), int(7*T/10), int(8*T/10), int(9*T/10)]
        self.t = torch.tensor(ts).to(self.args.device)
        self.T = T
        self.alpha_bar = torch.cumprod(1 - torch.linspace(1e-4, 2e-2, self.T, device=self.args.device), dim=0)  

    
        
    @torch.no_grad()
    def unet_loop(self, t_start_frac: float = .6, eta: float = 0.0, match_stats: bool = True):
        """
        Generate images by:
          1) Interpreting self.loop as proposed clean latents x0~ (shape [F,C,8,8])
          2) (Optional) Matching mean/std to VAE latent stats
          3) Forward-diffusing to t_start
          4) DDIM reverse to t=0, consistent with training z_t construction
    
        t_start_frac in (0,1]: 0.7 means start at t ≈ 0.7*T (moderate noise)
        eta=0 => deterministic DDIM; >0 adds stochasticity.
        """
        self.vae.eval()
        self.unet.eval()
    
        # ---- helpers for schedule (match training) ----
        def abar(idx): return self.alpha_bar[idx].view(1,1,1,1)
        def a(idx):    return abar(idx).sqrt()
        def s(idx):    return (1.0 - abar(idx)).sqrt()
    
        # ---- 1) proposed clean latents ----
        x0 = self.loop.clone()  # [F, C, 8, 8]
    
        # ---- 2) (optional) match mean/std of VAE latents ----
        if match_stats:
            # Use your fixed Pokémon batch to estimate latent stats (mu) once
            _, _, mu, _, _ = self.vae(self.pokemon, use_logvar=False)    # mu is the clean latent proxy
            mu_mean = mu.mean(dim=(0,2,3), keepdim=True)                 # [1,C,1,1]
            mu_std  = mu.std(dim=(0,2,3), keepdim=True).clamp_min(1e-6)
            # self.loop is whitened to ~N(0,1) by make_fourier_loop:contentReference[oaicite:1]{index=1};
            # re-scale it to match the VAE latent distribution
            x0 = x0 * mu_std + mu_mean
    
        # ---- 3) forward diffuse to a valid z_t_start ----
        t_start = max(0, min(self.T-1, int(self.T * t_start_frac)))
        alpha_t  = a(t_start)
        sigma_t  = s(t_start)
        eps0     = torch.randn_like(x0) if eta > 0 else torch.zeros_like(x0)
        z = alpha_t * x0 + sigma_t * eps0   # EXACT formula used in training to make z_t:contentReference[oaicite:2]{index=2}
    
        # build the descending step list from t_start -> 0
        steps = torch.arange(t_start, -1, -1, device=self.args.device)
    
        # ---- 4) reverse (DDIM) ----
        for i, t_idx in enumerate(steps):
            t_tensor = t_idx.view(1).to(self.args.device)  # [1]; your UNet expects (B,)
            eps_hat  = self.unet(z, t_tensor)              # ε̂(z_t, t):contentReference[oaicite:3]{index=3}
    
            alpha_t  = a(t_idx)
            sigma_t  = s(t_idx)
            x0_hat   = (z - sigma_t * eps_hat) / alpha_t   # consistent x̂₀
    
            if i + 1 < steps.numel():
                t_prev   = steps[i + 1]
                alpha_p  = a(t_prev)
                sigma_p  = s(t_prev)
    
                # DDIM step with optional stochasticity
                if eta == 0.0:
                    z = alpha_p * x0_hat + sigma_p * eps_hat
                else:
                    # stochastic term size chosen to keep variance correct across timesteps
                    var = (sigma_p**2 - (alpha_p/alpha_t * sigma_t)**2).clamp_min(0.0)
                    z = alpha_p * x0_hat + (var.sqrt()) * torch.randn_like(z) + (alpha_p/alpha_t * sigma_t) * eps_hat
            else:
                z = x0_hat  # final clean latent
    
        imgs = (self.vae.decode(z) + 1) / 2
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
        predicted_encodings = (z_t - sigma_t * predicted_epsilon) / alpha_t
        
        with torch.no_grad():   
            self.vae.eval()
            noisy_imgs = (self.vae.decode(z_t) + 1) / 2
            predicted_imgs = (self.vae.decode(predicted_encodings) + 1) / 2
            
        return {
            "real_imgs" : real_imgs,
            "decoded" : decoded,
            "encoded" : encoded,
            "mu" : mu,
            "logvar" : logvar,
            "dkl" : dkl,
            "epsilon" : epsilon,
            "noisy_imgs" : noisy_imgs,
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
        
        self.plot_vals_dict["vae_loss"].append(vae_loss.detach().cpu())
        self.plot_vals_dict["dkl_loss"].append(self.args.vae_beta * dkl.detach().cpu())
            
        for module in self.vae.modules():
            if isinstance(module, ConstrainedConv2d):
                module.clamp_weights()
                
        if self.epochs_for_vae % self.args.epochs_per_vid == 0:
            print("Saving VAE example...")
            with torch.no_grad():
                decoded, predicted_imgs, noisy_imgs = operator.itemgetter(
                    "decoded", "noisy_imgs", "predicted_imgs")(
                        self.vae_vs_unet(new_batch = False, t = self.t))
            save_rel = file_location + f"/generated_images/{self.args.arg_name}/VAE_epoch_{self.epochs_for_vae}.png"
            save_vae_comparison_grid(self.pokemon, decoded, noisy_imgs, predicted_imgs, save_rel)
            plot_vals(self.plot_vals_dict, file_location + f"/generated_images/{self.args.arg_name}")
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

        
        for module in self.unet.modules():
            if isinstance(module, ConstrainedConv2d):
                module.clamp_weights()
    
        if self.epochs_for_unet % self.args.epochs_per_vid == 0:
            print("Saving UNET examples...")
            with torch.no_grad():
                imgs = self.unet_loop()
            save_rel = file_location + f"/generated_images/{self.args.arg_name}/UNET_epoch_{self.epochs_for_unet}"
            show_images_from_tensor(imgs, save_path=save_rel, fps=10)  
            
            with torch.no_grad():
                _, encoded, _, _, _ = self.vae(self.pokemon, use_logvar=False)
                
                vae_imgs, noisy_imgs, predicted_imgs = operator.itemgetter(
                    "decoded", "noisy_imgs", "predicted_imgs")(
                        self.vae_vs_unet(new_batch = False, t = self.t))
            save_rel = file_location + f"/generated_images/{self.args.arg_name}/UNET_epoch_{self.epochs_for_unet}/UNET_epoch_{self.epochs_for_unet}.png"
            save_vae_comparison_grid(self.pokemon, vae_imgs, noisy_imgs, predicted_imgs, save_rel) 
            plot_vals(self.plot_vals_dict, file_location + f"/generated_images/{self.args.arg_name}/UNET_epoch_{self.epochs_for_unet}")
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
    
    


    