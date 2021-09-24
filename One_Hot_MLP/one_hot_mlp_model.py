import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.autograd import Variable
import os
import numpy as np
from tqdm import tqdm


class ONE_HOT_MLP(nn.Module):
    def __init__(self, hidden):
        super(ONE_HOT_MLP, self).__init__()
        self.cnn = nn.Sequential(
            # 1, 124, 32
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 32, 62, 16
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64, 15, 15
            nn.Conv2d(64, 128, kernel_size=5, stride=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128, 20, 5
        )

        self.mlp = nn.Sequential(nn.Linear(128 * 10 * 5 * 2 + 1, hidden),
                                 nn.Tanh(),
                                 nn.Linear(hidden, 1))

        for p in self.mlp.parameters():
            torch.nn.init.normal_(p, mean=0, std=0.1)
        torch.nn.init.constant_(self.mlp[0].bias, val=0.)
        torch.nn.init.constant_(self.mlp[2].bias, val=0.)

    def forward(self, x, t):
        mid = self.cnn(x)
        return self.mlp(torch.cat((t.reshape(t.shape[0], -1), mid.reshape(mid.shape[0], -1)), dim=1))


def train_one_hot_mlp(params, hidden, device):
    mse = torch.nn.MSELoss()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    thermal_conductivity_train_loader = torch.load('Data/thermal_conductivity_vae_mlp_train_loader.pkl')
    heat_capacity_train_loader = torch.load('Data/heat_capacity_vae_mlp_train_loader.pkl')

    heat_capacity_one_hot_mlp = ONE_HOT_MLP(hidden).cuda()
    thermal_conductivity_one_hot_mlp = ONE_HOT_MLP(hidden).cuda()

    thermal_conductivity_optimizer = optim.Adam(
        thermal_conductivity_one_hot_mlp.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    heat_capacity_optimizer = optim.Adam(
        heat_capacity_one_hot_mlp.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    thermal_conductivity_total_loss_list = np.ones(params['VAE_epoch_num'] + 10)
    heat_capacity_total_loss_list = np.ones(params['VAE_epoch_num'] + 10)

    thermal_conductivity_total_loss_list *= 5000000
    heat_capacity_total_loss_list *= 5000000

    thermal_conductivity_model_file_name = \
        'Model_pkl/ONE_HOT_MLP_thermal_conductivity_hidden_' + str(hidden) + '.pkl'
    heat_capacity_model_file_name = \
        'Model_pkl/ONE_HOT_MLP_heat_capacity_hidden_' + str(hidden) + '.pkl'

    for epoch in range(params['VAE_epoch_num']):
        total_loss = 0
        thermal_conductivity_one_hot_mlp.train()
        for i, data in enumerate(tqdm(thermal_conductivity_train_loader, 0)):
            one_hot = torch.cat((data[0], data[1]), dim=1)
            one_hot = one_hot.reshape(one_hot.shape[0], 1, one_hot.shape[1], one_hot.shape[2])
            one_hot = Variable(one_hot).cuda().type(torch.cuda.FloatTensor)
            thermal_conductivity_optimizer.zero_grad()
            t = data[2].cuda().reshape(data[2].shape[0], 1).type(torch.cuda.FloatTensor)
            label = data[3].cuda().reshape(data[3].shape[0], 1).type(torch.cuda.FloatTensor)
            prediction = thermal_conductivity_one_hot_mlp(one_hot, t)
            loss = mse(prediction, label)
            loss.backward()
            total_loss += loss.data.item() / 1000
            thermal_conductivity_optimizer.step()
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, total_loss / len(thermal_conductivity_train_loader.dataset)))
        thermal_conductivity_total_loss_list[epoch] = total_loss / len(thermal_conductivity_train_loader.dataset)
        if np.argmin(thermal_conductivity_total_loss_list) == epoch:
            torch.save(thermal_conductivity_one_hot_mlp, thermal_conductivity_model_file_name)
            print('best result, saving the model to ' + thermal_conductivity_model_file_name)
        elif np.argmin(thermal_conductivity_total_loss_list) == epoch - 25:
            print('Finish: Training process over due to useless training')
            break

    for epoch in range(params['VAE_epoch_num']):
        total_loss = 0
        heat_capacity_one_hot_mlp.train()
        for i, data in enumerate(tqdm(heat_capacity_train_loader, 0)):
            one_hot = torch.cat((data[0], data[1]), dim=1)
            one_hot = one_hot.reshape(one_hot.shape[0], 1, one_hot.shape[1], one_hot.shape[2])
            one_hot = Variable(one_hot).cuda().type(torch.cuda.FloatTensor)
            heat_capacity_optimizer.zero_grad()
            t = data[2].cuda().reshape(data[2].shape[0], 1).type(torch.cuda.FloatTensor)
            label = data[3].cuda().reshape(data[3].shape[0], 1).type(torch.cuda.FloatTensor)
            prediction = heat_capacity_one_hot_mlp(one_hot, t)
            loss = mse(prediction, label)
            loss.backward()
            total_loss += loss.data.item() / 1000
            heat_capacity_optimizer.step()
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, total_loss / len(heat_capacity_train_loader.dataset)))
        heat_capacity_total_loss_list[epoch] = total_loss / len(heat_capacity_train_loader.dataset)
        if np.argmin(heat_capacity_total_loss_list) == epoch:
            torch.save(heat_capacity_one_hot_mlp, heat_capacity_model_file_name)
            print('best result, saving the model to ' + heat_capacity_model_file_name)
        elif np.argmin(heat_capacity_total_loss_list) == epoch - 25:
            print('Finish: Training process over due to useless training')
            break