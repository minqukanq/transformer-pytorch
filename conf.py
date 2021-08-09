import torch

# model setting
embed_size        = 512
num_layers        = 6
forward_expansion = 4
heads             = 8
dropout           = 0.1
max_length        = 512
batch_size        = 1024

# optimizer setting
weight_decay      = 5e-4
factor            = 0.9
patience          = 10
init_lr           = 1e-5
adam_eps          = 5e-9
N_EPOCHS          = 1000
CLIP              = 1
best_valid_loss = float('inf')

device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

