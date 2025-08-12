#%% 
import os
from utils import file_location
os.chdir(file_location)

from collections import Counter
import datetime

import torch 
from torch.optim import Adam
import torch.nn.functional as F

from utils import default_args, get_random_batch, print, duration, show_images_from_tensor
from utils_for_torch import create_interpolated_tensor, ConstrainedConv2d
from vae import VAE
from unet import UNET



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
        
        # Dictionary for plotting loss-information, etc.
        self.plot_vals_dict = {}
        
        self.T = 1000
        self.alpha_bar = torch.cumprod(1 - torch.linspace(1e-4, 2e-2, self.T, device=self.args.device), dim=0)  
        
        
    
    @torch.no_grad()
    def _sample(self, n=100, steps=50, cond=None):
        self.vae.eval()
        self.unet.eval()
        # Deterministic DDIM-style preview (eta=0)
        z = torch.randn(n, 4, 8, 8, device=self.args.device)
        ts = torch.linspace(self.T-1, 0, steps, dtype=torch.long, device=self.args.device)
    
        for i, t in enumerate(ts):
            ab_t = self.alpha_bar[t].sqrt().view(1,1,1,1)
            sig_t = (1.0 - self.alpha_bar[t]).sqrt().view(1,1,1,1)
            predicted_epsilon = self.unet(z)  # <- later: pass t if you add time-cond
            x0_hat = (z - sig_t * predicted_epsilon) / (ab_t + 1e-8)
    
            if i+1 < len(ts):
                t_prev = ts[i+1]
                ab_prev = self.alpha_bar[t_prev].sqrt().view(1,1,1,1)
                sig_prev = (1.0 - self.alpha_bar[t_prev]).sqrt().view(1,1,1,1)
                z = ab_prev * x0_hat + sig_prev * predicted_epsilon
            else:
                z = predicted_epsilon  # final latent
    
        # Decode with your VAE decoder path; your forward rescales tanh-> [0,1]
        imgs = (self.vae.b(z) + 1) / 2
        return imgs.clamp(0,1)

        
        
    # One step of training.
    def epoch_for_vae(self):
        self.vae.train()
        imgs = get_random_batch(batch_size=self.args.batch_size)
    
        # ---- 2) VAE step (reconstruction + KL) ----
        decoded, encoded, mu, logvar, kl = self.vae(imgs)  # returns all five. :contentReference[oaicite:3]{index=3}
        recon_loss = F.l1_loss(decoded, imgs)               # L1 works well at 64×64
        vae_beta = 0.05                                     # small KL pressure
        vae_loss = recon_loss + vae_beta * kl
    
        self.vae_opt.zero_grad(set_to_none=True)
        vae_loss.backward()
        self.vae_opt.step()
            
        for module in self.vae.modules():
            if isinstance(module, ConstrainedConv2d):
                module.clamp_weights()
                
        self.epochs_for_vae += 1
        #print(self.epochs_for_vae, end = "... ")
        
        
        
    # One step of training.
    def epoch_for_unet(self):
        self.vae.eval()
        self.unet.train()
        imgs = get_random_batch(batch_size=self.args.batch_size)
    
        # ---- 3) Diffusion step (train UNet on ε) ----
        with torch.no_grad():                               # encode latents without updating VAE
            _, encoded, _, _, _ = self.vae(imgs, use_logvar=False)
    
        t = torch.randint(low=0, high=self.T, size=(encoded.size(0),), device=self.args.device)  # pick timesteps
        alpha_t = self.alpha_bar[t].sqrt().view(encoded.size(0), 1, 1, 1)
        sigma_t = (1.0 - self.alpha_bar[t]).sqrt().view(encoded.size(0), 1, 1, 1)
    
        epsilon = torch.randn_like(encoded)
        z_t = alpha_t * encoded + sigma_t * epsilon
    
        predicted_epsilon = self.unet(z_t)            # later: pass t if you add time-cond
        unet_loss = F.mse_loss(predicted_epsilon, epsilon)
    
        self.unet_opt.zero_grad(set_to_none=True)
        unet_loss.backward()
        self.unet_opt.step()
        
        for module in self.unet.modules():
            if isinstance(module, ConstrainedConv2d):
                module.clamp_weights()
    
        # ---- 4) Occasionally: save previews ----
        if self.epochs_for_unet % self.args.epochs_per_vid == 0:
            with torch.no_grad():
                samples = self._sample(n=8, steps=50)
            # save to: generated_images/{arg_name}/epoch_{N}/
            save_rel = f"{self.args.arg_name}/epoch_{self.epochs_for_unet}"
            show_images_from_tensor(samples, save_path=save_rel, fps=10)   # saves PNGs+GIF. :contentReference[oaicite:4]{index=4}
        torch.cuda.empty_cache()
        
        self.epochs_for_unet += 1
        #print(self.epochs_for_unet, end = "... ")
        
        
    
    # Let's do this!
    def training(self):
        for epoch in range(self.args.epochs_for_vae):
            self.epoch_for_vae()
            if(epoch % 25 == 0):
                percent_done = 100 * self.epochs_for_vae / self.args.epochs_for_vae
                print(f"{percent_done}%")
        for epoch in range(self.args.epochs_for_unet):
            self.epoch_for_unet()
            if(epoch % 25 == 0):
                percent_done = 100 * self.epochs_for_unet / self.args.epochs_for_unet
                print(f"{percent_done}%")
            
                
        
        
# :D
if(__name__ == "__main__"):
    sd = SD()
    sd.training()
    
    


    