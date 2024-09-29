import os
import torch
import sys

from optimization.run_optimization import main_batch, main
from argparse import Namespace

prompts = ['A photo of an old face.','A photo of a sad face.','A photo of a smiling face.','A photo of an angry face.','A photo of a face with curly hair.']
output_dirs = ['old', 'sad', 'smile', 'angry', 'curly']

experiment_type = 'edit' #@param ['edit', 'free_generation']

description = prompts[int(sys.argv[1])] #@param {type:"string"}
print(description)
results_dir = output_dirs[int(sys.argv[1])]
print(results_dir)

# latent_path = 'latents.pt' #@param {type:"string"}
latent_path = None

optimization_steps = 40 #@param {type:"number"}

l2_lambda = 0.008 #@param {type:"number"}

id_lambda = 0.005 #@param {type:"number"}

stylespace = False #@param {type:"boolean"}

create_video = False #@param {type:"boolean"}

use_seed = True #@param {type:"boolean"}

seed = 1 #@param {type: "number"}

#@title Additional Arguments
args = {
    "description": description,
    "ckpt": "stylegan2-ffhq-config-f.pt",
    "stylegan_size": 1024,
    "lr_rampup": 0.05,
    "lr": 0.1,
    "step": optimization_steps,
    "mode": experiment_type,
    "l2_lambda": l2_lambda,
    "id_lambda": id_lambda,
    'work_in_stylespace': stylespace,
    "latent_path": latent_path,
    "truncation": 0.7,
    "save_intermediate_image_every": 1 if create_video else 20,
    "results_dir": results_dir,
    "ir_se50_weights": "model_ir_se50.pth"
}

if __name__ == '__main__':
    torch.manual_seed(seed)

    result = main_batch(Namespace(**args))
