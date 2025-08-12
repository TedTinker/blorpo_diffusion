#%%

# I use this if I'm using the cluster. 

import torch, random
import numpy as np

from gan import GAN
from utils import args, duration, print



print("\nname:\n{}".format(args.arg_name))
print("\ntitle:\n{}".format(args.arg_title))



if __name__ == '__main__':    
    seed = args.init_seed 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
            
    gan = GAN(args = args)
    gan.training()
    
    print("\nDuration: {}. Done!".format(duration()))
    # %%
