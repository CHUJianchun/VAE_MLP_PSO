import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
import re


class MLP(nn.Module):
    def __init__(self, vae_file_name, hidden_1, hidden_2):
        super(MLP, self).__init__()
        # vae_file_name form: VAE_??.pkl
        latent_num = int(re.sub('\D', '', vae_file_name))

        self.viscosity_mlp = nn.Sequential(
            torch.nn.Linear(latent_num + 2, hidden_1),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_1, hidden_2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_2, 1)
        ).cuda()
        for p in self.viscosity_mlp.parameters():
            torch.nn.init.normal_(p, mean=0, std=0.1)
        torch.nn.init.constant_(self.viscosity_mlp[0].bias, val=0.)
        torch.nn.init.constant_(self.viscosity_mlp[2].bias, val=0.)
        torch.nn.init.constant_(self.viscosity_mlp[4].bias, val=0.)

        self.thermal_conductivity_mlp = nn.Sequential(
            torch.nn.Linear(latent_num + 2, hidden_1),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_1, hidden_2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_2, 1)
        ).cuda()
        for p in self.thermal_conductivity_mlp.parameters():
            torch.nn.init.normal_(p, mean=0, std=0.1)
        torch.nn.init.constant_(self.thermal_conductivity_mlp[0].bias, val=0.)
        torch.nn.init.constant_(self.thermal_conductivity_mlp[2].bias, val=0.)
        torch.nn.init.constant_(self.thermal_conductivity_mlp[4].bias, val=0.)

        self.heat_capacity_mlp = nn.Sequential(
            torch.nn.Linear(latent_num + 2, hidden_1),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_1, hidden_2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_2, 1)
        ).cuda()
        for p in self.heat_capacity_mlp.parameters():
            torch.nn.init.normal_(p, mean=0, std=0.1)
        torch.nn.init.constant_(self.heat_capacity_mlp[0].bias, val=0.)
        torch.nn.init.constant_(self.heat_capacity_mlp[2].bias, val=0.)
        torch.nn.init.constant_(self.heat_capacity_mlp[4].bias, val=0.)

    def viscosity_forward(self, input_):
        return self.viscosity_mlp(input_)

    def thermal_conductivity_forward(self, input_):
        return self.thermal_conductivity_mlp(input_)

    def heat_capacity_forward(self, input_):
        return self.heat_capacity_mlp(input_)


def train_mlp(latent_code_num, params, device, hidden_1, hidden_2):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    mlp = MLP('VAE_' + str(latent_code_num) + '.pkl', hidden_1, hidden_2).cuda()

    viscosity_train_loader = torch.load(
        'Data/viscosity_mlp_train_loader_VAE_' + str(latent_code_num) + '.pkl')
    thermal_conductivity_train_loader = torch.load(
        'Data/thermal_conductivity_mlp_train_loader_VAE_' + str(latent_code_num) + '.pkl')
    heat_capacity_train_loader = torch.load(
        'Data/heat_capacity_mlp_train_loader_VAE_' + str(latent_code_num) + '.pkl')

    viscosity_optimizer = optim.Adam(
        mlp.viscosity_mlp.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    thermal_conductivity_optimizer = optim.Adam(
        mlp.thermal_conductivity_mlp.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    heat_capacity_optimizer = optim.Adam(
        mlp.heat_capacity_mlp.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    total_loss_list = np.ones(params['MLP_epoch_num'] + 10)
    total_loss_list *= 5000000
    model_file_name = \
        'Model_pkl/MLP_latent_' + str(latent_code_num) + '_structure_' + str(hidden_1) + '_' + str(hidden_2) + '.pkl'

    def train(property_, epoch):
        if property_ == 'viscosity':
            train_loader = viscosity_train_loader
            optimizer = viscosity_optimizer
        elif property_ == 'thermal_conductivity':
            train_loader = thermal_conductivity_train_loader
            optimizer = thermal_conductivity_optimizer
        else:
            train_loader = heat_capacity_train_loader
            optimizer = heat_capacity_optimizer
        mlp.train()
        total_loss = 0
        for i, (input_, label_) in enumerate(tqdm(train_loader), 0):
            optimizer.zero_grad()
            input_ = input_.to(torch.float32).clone().detach().cuda()
            label_ = label_.to(torch.float32).clone().detach().cuda()
            local_value = locals()
            exec('prediction = mlp.' + property_ + '_mlp(input_)')
            prediction = local_value['prediction']
            prediction = prediction[:, 0]
            mse = torch.nn.MSELoss()
            loss = mse(prediction, label_)
            total_loss += loss.data.item()
            loss.backward()
            optimizer.step()
        print('====> Epoch: {} Average loss: {:.5f}'.format(epoch, total_loss / len(train_loader.dataset)))
        total_loss_list[epoch] = total_loss / len(train_loader.dataset)
        if np.argmin(total_loss_list) == epoch:
            torch.save(mlp, model_file_name)
            print('best result, saving the model to ' + model_file_name)

    for epoch_ in range(params['MLP_epoch_num']):
        train('viscosity', epoch_)

    for epoch_ in range(params['MLP_epoch_num']):
        train('thermal_conductivity', epoch_)

    for epoch_ in range(params['MLP_epoch_num']):
        train('heat_capacity', epoch_)
