import random
import numpy as np
import torch
import pickle

# Set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def derive_split_seed(base_seed, split_index):
    rng = random.Random(int(base_seed))
    seed = int(base_seed)
    for _ in range(int(split_index) + 1):
        seed = rng.randint(0, 2**31 - 1)
    return int(seed)

# Save model state and configuration
def save_model(state_dict, gnn_config, dir):
    pickle.dump(gnn_config, open('{}.config'.format(dir), 'wb'))
    torch.save({k:v.cpu() for k, v in state_dict.items()}, '{}.ckpt'.format(dir))
