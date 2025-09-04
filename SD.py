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
from unet import UNET



# Let's put all this together!
class SD:
    def __init__(self, args = default_args):
        self.args = args
        self.session_start_time = datetime.datetime.now()
        
        self.plot_vals_dict = {
            "vae_loss" : [],
            "dkl_1_loss" : [],
            "dkl_2_loss" : [],
            "vae_mu" : [],
            "vae_std" : [],
            "unet_loss" : []}
        self.pokemon = pokemon_sample
        self.loop = create_interpolated_tensor()
                
        std = [i * self.args.max_noise_unet / 10 for i in range(10)]
        self.std = torch.tensor(std).to(self.args.device)
        
        # Folder to save in.
        self.gen_location = file_location + f"/generated_images/{self.args.arg_name}"
        if not os.path.exists(self.gen_location):
            os.makedirs(self.gen_location)
        
        
        
        self.epochs_for_vae = 1
        self.current_noise_vae = self.args.min_noise_vae
        self.vae = VAE()
        try:
            self.vae.load_state_dict(torch.load(self.gen_location + "/vae.py", weights_only=True))
            self.args.epochs_for_vae = 0
        except:
            pass
        self.vae_opt = Adam(self.vae.parameters(), args.vae_lr, weight_decay=.0001)
        self.vae.train()
        
        
        
        self.epochs_for_unet = 1
        self.current_noise_unet = self.args.min_noise_unet
        self.unet = UNET()
        try:
            self.unet.load_state_dict(torch.load(self.gen_location + "/unet.py", weights_only=True))
            self.args.epochs_for_unet = 0
        except:
            pass
        self.unet_opt = Adam(self.unet.parameters(), args.unet_lr, weight_decay=.0001)
        self.unet.train()
        
        
        
    def save_vae(self):
        torch.save(self.vae.state_dict(), self.gen_location + "/vae.py")
        
    def save_unet(self):
        torch.save(self.unet.state_dict(), self.gen_location + "/unet.py")




    
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
        
        b = real_imgs.size(0)
        std = torch.rand(b, 1, 1, 1, device=self.args.device) * self.current_noise_vae
        eps = torch.randn_like(real_imgs)
        real_imgs_noisy = (real_imgs + std * eps)
        real_imgs_noisy_for_plotting = real_imgs_noisy.clamp_(0, 1)
            
        if(unet_train or not new_batch):
            with torch.no_grad():                          
                decoded, encoded, dkl_1, dkl_2, vae_mu, vae_std = self.vae(real_imgs_noisy, use_std=False)
        else:
            decoded, encoded, dkl_1, dkl_2, vae_mu, vae_std = self.vae(real_imgs_noisy)
            
        if(new_batch):  
            std = torch.rand(size=(encoded.size(0),), device=self.args.device) * self.current_noise_unet
            std = std.clamp_(self.args.min_noise_unet, self.args.max_noise_unet)
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
            "real_imgs_noisy" : real_imgs_noisy_for_plotting,
            "encoded" : encoded,
            "decoded" : decoded,
            "dkl_1" : dkl_1,
            "dkl_2" : dkl_2,
            "vae_mu" : vae_mu,
            "vae_std" : vae_std,
            "std" : std,
            "epsilon" : epsilon,
            "predicted_epsilon" : predicted_epsilon,
            "predicted_encoded" : predicted_encoded,
            "noisy_imgs" : noisy_imgs,
            "predicted_imgs" : predicted_imgs
            }
    
    
        
    # Train VAE.
    def epoch_for_vae(self):
        
        real_imgs, decoded, dkl_1, dkl_2, vae_mu, vae_std = operator.itemgetter(
            "real_imgs", "decoded", "dkl_1", "dkl_2", "vae_mu", "vae_std")(
                self.vae_vs_unet(vae_train = True))
        
        recon_loss = F.l1_loss(decoded, real_imgs)   
        dkl_1_loss = self.args.vae_beta * dkl_1
        dkl_2_loss = self.args.vae_beta * dkl_2
        vae_loss = recon_loss + (dkl_1_loss + dkl_2_loss)/2
    
        self.vae_opt.zero_grad(set_to_none=True)
        vae_loss.backward()
        self.vae_opt.step()
        
        self.plot_vals_dict["vae_loss"].append(vae_loss.detach().cpu())
        self.plot_vals_dict["dkl_1_loss"].append(dkl_1_loss.detach().cpu())
        self.plot_vals_dict["dkl_2_loss"].append(dkl_2_loss.detach().cpu())
        self.plot_vals_dict["vae_mu"].append(vae_mu.mean().detach().cpu())
        self.plot_vals_dict["vae_std"].append(vae_std.mean().detach().cpu())
            
        for module in self.vae.modules():
            if isinstance(module, ConstrainedConv2d):
                module.clamp_weights()
                
        if self.epochs_for_vae % self.args.epochs_per_vid == 0:
            print(f"Saving VAE example... (Current Noise: {round(self.current_noise_vae, 2)})")
            self.save_vae()
            self.save_examples(
                grid_save_pos = self.gen_location + f"/VAE_example_{self.epochs_for_vae}.png",
                val_save_pos = self.gen_location)
        torch.cuda.empty_cache()
                
        if(self.current_noise_vae < self.args.max_noise_vae):
            self.current_noise_vae += self.args.change_rate_vae
        else:
            self.current_noise_vae = self.args.max_noise_vae
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
            print(f"Saving UNET examples... (Current Noise: {round(self.current_noise_unet, 2)})")
            self.save_unet()
            self.save_examples(
                grid_save_pos = self.gen_location + f"/UNET_epoch_{self.epochs_for_unet}/UNET_epoch_{self.epochs_for_unet}.png",
                val_save_pos = self.gen_location + f"/UNET_epoch_{self.epochs_for_unet}")
            
            imgs = self.unet_loop(
                actual_noise_list = self.args.actual_noise_list,       
                lied_noise_list = self.args.lied_noise_list)
            save_rel = file_location + f"/generated_images/{self.args.arg_name}/UNET_epoch_{self.epochs_for_unet}/loop"
            show_images_from_tensor(imgs, save_path=save_rel, fps=10) 
        torch.cuda.empty_cache()
        
        if(self.current_noise_unet < self.args.max_noise_unet):
            self.current_noise_unet += self.args.change_rate_unet
        else:
            self.current_noise_unet = self.args.max_noise_unet
        self.epochs_for_unet += 1
        #print(self.epochs_for_unet, end = "... ")
        
        
    
    # Let's do this!
    def training(self):
        print("\nTRAINING VAE:")
        vae_start = datetime.datetime.now()
        for epoch in range(self.args.epochs_for_vae):
            self.epoch_for_vae()
            if(epoch % 25 == 0):
                percent_done = round(100 * self.epochs_for_vae / self.args.epochs_for_vae, 2)
                print(f"{percent_done}%, {duration(vae_start)}", end = "... ")
                
        self.save_vae()
                 
        print("\n\nTRAINING UNET:")
        unet_start = datetime.datetime.now()
        for epoch in range(self.args.epochs_for_unet):
            self.epoch_for_unet()
            if(epoch % 25 == 0):
                percent_done = round(100 * self.epochs_for_unet / self.args.epochs_for_unet, 2)
                print(f"{percent_done}%, {duration(unet_start)}", end = "... ")
                
        self.save_unet()
            
        imgs = self.unet_loop(
            actual_noise_list = self.args.actual_noise_list,     
            lied_noise_list = self.args.lied_noise_list)
        save_rel = file_location + f"/generated_images/{self.args.arg_name}/loop"
        show_images_from_tensor(imgs, save_path=save_rel, fps=10)  
                
                

    def save_examples(self, grid_save_pos, val_save_pos):
        with torch.no_grad():
            real_imgs_noisy, decoded, noisy_imgs, predicted_imgs = operator.itemgetter(
                "real_imgs_noisy", "decoded", "noisy_imgs", "predicted_imgs")(
                    self.vae_vs_unet(new_batch = False))
        save_vae_comparison_grid(real_imgs_noisy, decoded, noisy_imgs, predicted_imgs, grid_save_pos, self.std)
        plot_vals(self.plot_vals_dict, val_save_pos)
        
    
    
    @torch.no_grad()
    def unet_loop(self, actual_noise_list, lied_noise_list):
        self.vae.eval() 
        self.unet.eval()
        img = self.loop.clone()
        
        actual_noise_list = [(torch.tensor(float(n)).to(self.args.device)) for n in actual_noise_list]
        lied_noise_list = [(torch.tensor(float(n)).to(self.args.device)) for n in lied_noise_list]

        for actual_noise, lied_noise in zip(actual_noise_list, lied_noise_list):
            _, encoded, _, _, _, _ = self.vae(img, use_std = False)     # Is it better to NOT re-encode each time?
            actual_std = actual_noise.view(1,1,1,1).expand_as(encoded)
            actual_epsilon = Normal(0, 1).sample(actual_std.shape).to(actual_std.device)    
            lied_std = lied_noise.view(1,1,1,1).expand_as(encoded)
            noisy_encoded = encoded + actual_std * actual_epsilon
            eps_hat = self.unet(noisy_encoded, lied_std) * lied_std    
            encoded = noisy_encoded - eps_hat
            img = (self.vae.decode(encoded) + 1) / 2
            
        return img
        
    
        
# :D
if(__name__ == "__main__"):
    sd = SD()
    sd.training()
    
    


    