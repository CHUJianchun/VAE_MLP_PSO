import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
import re


class MLP_CAT(nn.Module):
    def __init__(self, vae_file_name, hidden):
        super(MLP_CAT, self).__init__()
        latent_num = int(re.sub('\D', '', vae_file_name))

        self.mlp = nn.Sequential(
            torch.nn.Linear(latent_num + 1, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(hidden, 1)
        ).cuda()
        for p in self.mlp.parameters():
            torch.nn.init.normal_(p, mean=0, std=0.1)
        torch.nn.init.constant_(self.mlp[0].bias, val=0.)
        torch.nn.init.constant_(self.mlp[2].bias, val=0.)

    def forward(self, input_):
        return self.mlp(input_)


def train_mlp(latent_code_num, params, device, hidden):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    viscosity_mlp = MLP_CAT('VAE_CAT' + str(latent_code_num) + '.pkl', hidden).cuda()
    heat_capacity_mlp = MLP_CAT('VAE_CAT' + str(latent_code_num) + '.pkl', hidden).cuda()
    thermal_conductivity_mlp = MLP_CAT('VAE_CAT' + str(latent_code_num) + '.pkl', hidden).cuda()

    viscosity_train_loader = torch.load(
        'Data/viscosity_mlp_cat_train_loader_VAE_CAT_' + str(latent_code_num) + '.pkl')
    thermal_conductivity_train_loader = torch.load(
        'Data/thermal_conductivity_mlp_cat_train_loader_VAE_CAT_' + str(latent_code_num) + '.pkl')
    heat_capacity_train_loader = torch.load(
        'Data/heat_capacity_mlp_cat_train_loader_VAE_CAT_' + str(latent_code_num) + '.pkl')

    viscosity_optimizer = optim.Adam(
        viscosity_mlp.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    thermal_conductivity_optimizer = optim.Adam(
        thermal_conductivity_mlp.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    heat_capacity_optimizer = optim.Adam(
        heat_capacity_mlp.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    viscosity_total_loss_list = np.ones(params['MLP_epoch_num'] + 10)
    thermal_conductivity_total_loss_list = np.ones(params['MLP_epoch_num'] + 10)
    heat_capacity_total_loss_list = np.ones(params['MLP_epoch_num'] + 10)

    viscosity_total_loss_list *= 5000000
    thermal_conductivity_total_loss_list *= 5000000
    heat_capacity_total_loss_list *= 5000000

    viscosity_model_file_name = \
        'Model_pkl/MLP_CAT_viscosity_latent_' + str(latent_code_num) + \
        '_structure_' + str(hidden) + '.pkl'
    thermal_conductivity_model_file_name = \
        'Model_pkl/MLP_CAT_thermal_conductivity_latent_' + str(latent_code_num) + \
        '_structure_' + str(hidden) + '.pkl'
    heat_capacity_model_file_name = \
        'Model_pkl/MLP_CAT_heat_capacity_latent_' + str(latent_code_num) + \
        '_structure_' + str(hidden) + '.pkl'
    mse = torch.nn.MSELoss()

    for epoch in range(params['MLP_epoch_num']):
        total_loss = 0
        for i, (input_, label_) in enumerate(tqdm(viscosity_train_loader), 0):
            viscosity_optimizer.zero_grad()
            input_ = input_.to(torch.float32).clone().detach().cuda()
            label_ = label_.to(torch.float32).clone().detach().cuda()
            prediction = viscosity_mlp(input_)
            prediction = prediction[:, 0]
            loss = mse(prediction, label_)
            total_loss += loss.data.item()
            loss.backward()
            viscosity_optimizer.step()
        print('====> Epoch: {} Average loss: {:.5f}'
              .format(epoch, total_loss / len(viscosity_train_loader.dataset)))
        viscosity_total_loss_list[epoch] = total_loss / len(viscosity_train_loader.dataset)
        if np.argmin(viscosity_total_loss_list) == epoch:
            torch.save(viscosity_mlp, viscosity_model_file_name)
            print('best result, saving the model to ' + viscosity_model_file_name)
        elif np.argmin(viscosity_total_loss_list) == epoch - 25:
            print('Finish: Training process over due to useless training')
            break

    for epoch in range(params['MLP_epoch_num']):
        total_loss = 0
        for i, (input_, label_) in enumerate(tqdm(heat_capacity_train_loader), 0):
            heat_capacity_optimizer.zero_grad()
            input_ = input_.to(torch.float32).clone().detach().cuda()
            label_ = label_.to(torch.float32).clone().detach().cuda()
            prediction = heat_capacity_mlp(input_)
            prediction = prediction[:, 0]
            loss = mse(prediction, label_)
            total_loss += loss.data.item()
            loss.backward()
            heat_capacity_optimizer.step()
        print('====> Epoch: {} Average loss: {:.5f}'
              .format(epoch, total_loss / len(heat_capacity_train_loader.dataset)))
        heat_capacity_total_loss_list[epoch] = total_loss / len(heat_capacity_train_loader.dataset)
        if np.argmin(heat_capacity_total_loss_list) == epoch:
            torch.save(heat_capacity_mlp, heat_capacity_model_file_name)
            print('best result, saving the model to ' + heat_capacity_model_file_name)
        elif np.argmin(heat_capacity_total_loss_list) == epoch - 25:
            print('Finish: Training process over due to useless training')
            break

    for epoch in range(params['MLP_epoch_num']):
        total_loss = 0
        for i, (input_, label_) in enumerate(tqdm(thermal_conductivity_train_loader), 0):
            thermal_conductivity_optimizer.zero_grad()
            input_ = input_.to(torch.float32).clone().detach().cuda()
            label_ = label_.to(torch.float32).clone().detach().cuda()
            prediction = thermal_conductivity_mlp(input_)
            prediction = prediction[:, 0]
            loss = mse(prediction, label_)
            total_loss += loss.data.item()
            loss.backward()
            thermal_conductivity_optimizer.step()
        print('====> Epoch: {} Average loss: {:.5f}'
              .format(epoch, total_loss / len(thermal_conductivity_train_loader.dataset)))
        thermal_conductivity_total_loss_list[epoch] = total_loss / len(thermal_conductivity_train_loader.dataset)
        if np.argmin(thermal_conductivity_total_loss_list) == epoch:
            torch.save(thermal_conductivity_mlp, thermal_conductivity_model_file_name)
            print('best result, saving the model to ' + thermal_conductivity_model_file_name)
        elif np.argmin(thermal_conductivity_total_loss_list) == epoch - 25:
            print('Finish: Training process over due to useless training')
            break
