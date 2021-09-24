import Multi_optim.Methods as Methods
import numpy as np
import torch
from torch.autograd import Variable
import Multi_optim.Transform as Transform
import importlib
from tqdm import tqdm


def molecular_generation(params, mlp_file_name, property_name, vae_file_name):
    importlib.reload(Methods)
    importlib.reload(Transform)
    optim_result, optim_latent_space_code = Methods.gradient_descent_optimize(params, mlp_file_name, property_name)
    optim_latent_space_code = optim_latent_space_code[:, :, 2:]
    optim_latent_space_code = Variable(torch.tensor(optim_latent_space_code)).cuda()
    optim_smiles = np.zeros((optim_latent_space_code.shape[0], optim_latent_space_code.shape[1], 2)).tolist()
    point_index = 0
    for point in optim_latent_space_code:
        optim_step_code_index = 0
        for optim_step_code in point:
            anion_smiles, cation_smiles = Transform.latent_code_to_smiles(optim_step_code, vae_file_name)
            optim_smiles[point_index][optim_step_code_index][0] = anion_smiles
            optim_smiles[point_index][optim_step_code_index][1] = cation_smiles
            optim_step_code_index += 1
        point_index += 1
    return optim_latent_space_code.cpu().detach().numpy(), optim_smiles, optim_result


def molecular_generation_cat(params, mlp_file_name, vae_file_name):
    importlib.reload(Methods)
    importlib.reload(Transform)
    optim_result, optim_latent_space_code = Methods.gradient_descent_optimize_cat(params, mlp_file_name)
    optim_latent_space_code = optim_latent_space_code[:, :, 2:]
    optim_latent_space_code = Variable(torch.tensor(optim_latent_space_code)).cuda()
    optim_smiles = np.zeros((optim_latent_space_code.shape[0], optim_latent_space_code.shape[1])).tolist()
    point_index = 0
    for point in tqdm(optim_latent_space_code):
        optim_step_code_index = 0
        for optim_step_code in point:
            smiles = Transform.latent_code_to_smiles_cat(optim_step_code, vae_file_name)
            optim_smiles[point_index][optim_step_code_index] = smiles
            optim_step_code_index += 1
        point_index += 1
    return optim_latent_space_code.cpu().detach().numpy(), optim_smiles, optim_result
