import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.autograd import Variable
import os
import numpy as np
from tqdm import tqdm


def reparameterize(mu, logvar):
    eps = Variable(torch.randn(mu.size(0), mu.size(1))).cuda()
    z = mu + eps * torch.exp(logvar / 2)
    return z


class VAE_MLP_CAT(nn.Module):
    def __init__(self, latent_code_num, hidden):
        super(VAE_MLP_CAT, self).__init__()
        self.encoder = nn.Sequential(
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

        self.fc11 = nn.Linear(128 * 10 * 5 * 2, latent_code_num)
        self.fc12 = nn.Linear(128 * 10 * 5 * 2, latent_code_num)

        self.mlp = nn.Sequential(
            torch.nn.Linear(latent_code_num + 1, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1)
        )

        for p in self.mlp.parameters():
            torch.nn.init.normal_(p, mean=0, std=0.1)
        torch.nn.init.constant_(self.mlp[0].bias, val=0.)
        torch.nn.init.constant_(self.mlp[2].bias, val=0.)

        self.fc2 = nn.Linear(latent_code_num, 128 * 10 * 5 * 2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=6, stride=3, padding=1),
            nn.Sigmoid()
        )

    def get_reparameterized_code(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)  # batch_s, 8, 7, 7
        mu = self.fc11(out1.view(out1.size(0), -1))  # batch_s, latent
        logvar = self.fc12(out2.view(out2.size(0), -1))  # batch_s, latent
        z = self.reparameterize(mu, logvar)  # batch_s, latent
        return z

    def forward(self, x, t):
        out1, out2 = self.encoder(x), self.encoder(x)  # batch_s, 8, 7, 7
        mu = self.fc11(out1.view(out1.size(0), -1))  # batch_s, latent
        logvar = self.fc12(out2.view(out2.size(0), -1))  # batch_s, latent
        pre = self.mlp(torch.cat((t, mu), dim=1))
        z = reparameterize(mu, logvar)  # batch_s, latent
        out3 = self.fc2(z).view(z.size(0), 128, 20, 5)  # batch_s, 8, 7, 7
        return self.decoder(out3), mu, logvar, pre

    def predict(self, x, t):
        out1, out2 = self.encoder(x), self.encoder(x)  # batch_s, 8, 7, 7
        mu = self.fc11(out1.view(out1.size(0), -1))  # batch_s, latent
        pre = self.mlp(torch.cat((t, mu), dim=1))
        return pre

    def get_mid(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)
        mu = self.fc11(out1.view(out1.size(0), -1))
        return mu

    def decode(self, z):
        out3 = self.fc2(z).view(1, 128, 20, 5)
        return self.decoder(out3)


def loss_func(recon_x, x, mu, logvar, pre_, label_):
    mse = torch.nn.MSELoss()
    binary_cross_entropy = f.binary_cross_entropy(recon_x, x, size_average=False)
    k_l_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    mse_loss = mse(pre_, label_)
    return binary_cross_entropy + k_l_divergence + mse_loss


def train_vae_mlp(latent_code_num, hidden, params, device):
    print('Totally ' + str(params['VAE_epoch_num']) + ' epochs to train')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    thermal_conductivity_train_loader = torch.load('Data/thermal_conductivity_vae_mlp_train_loader.pkl')
    heat_capacity_train_loader = torch.load('Data/heat_capacity_vae_mlp_train_loader.pkl')

    heat_capacity_vae_mlp = VAE_MLP_CAT(latent_code_num, hidden).cuda()
    thermal_conductivity_vae_mlp = VAE_MLP_CAT(latent_code_num, hidden).cuda()

    thermal_conductivity_optimizer = optim.Adam(
        thermal_conductivity_vae_mlp.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    heat_capacity_optimizer = optim.Adam(
        heat_capacity_vae_mlp.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    thermal_conductivity_total_loss_list = np.ones(params['VAE_epoch_num'] + 10)
    heat_capacity_total_loss_list = np.ones(params['VAE_epoch_num'] + 10)

    thermal_conductivity_total_loss_list *= 5000000
    heat_capacity_total_loss_list *= 5000000

    thermal_conductivity_model_file_name = \
        'Model_pkl/VAE_MLP_CAT_thermal_conductivity_latent_' + str(latent_code_num) + \
        '_structure_' + str(hidden) + '.pkl'
    heat_capacity_model_file_name = \
        'Model_pkl/VAE_MLP_CAT_heat_capacity_latent_' + str(latent_code_num) + \
        '_structure_' + str(hidden) + '.pkl'

    for epoch in range(params['VAE_epoch_num']):
        total_loss = 0
        thermal_conductivity_vae_mlp.train()
        for i, data in enumerate(tqdm(thermal_conductivity_train_loader, 0)):
            one_hot = torch.cat((data[0], data[1]), dim=1)
            one_hot = one_hot.reshape(one_hot.shape[0], 1, one_hot.shape[1], one_hot.shape[2])
            one_hot = Variable(one_hot).cuda().type(torch.cuda.FloatTensor)
            thermal_conductivity_optimizer.zero_grad()
            t = data[2].cuda().reshape(data[2].shape[0], 1).type(torch.cuda.FloatTensor)
            label = data[3].cuda().reshape(data[3].shape[0], 1).type(torch.cuda.FloatTensor)
            recon_x, mu, logvar, pre = thermal_conductivity_vae_mlp.forward(one_hot, t)
            recon_x = recon_x[:, :, :one_hot.shape[2], :one_hot.shape[3]]
            loss = loss_func(recon_x, one_hot, mu, logvar, pre, label)
            loss.backward()
            total_loss += loss.data.item() / 1000
            thermal_conductivity_optimizer.step()
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, total_loss / len(thermal_conductivity_train_loader.dataset)))
        thermal_conductivity_total_loss_list[epoch] = total_loss / len(thermal_conductivity_train_loader.dataset)
        if np.argmin(thermal_conductivity_total_loss_list) == epoch:
            torch.save(heat_capacity_vae_mlp, thermal_conductivity_model_file_name)
            print('best result, saving the model to ' + thermal_conductivity_model_file_name)
        elif np.argmin(thermal_conductivity_total_loss_list) == epoch - 25:
            print('Finish: Training process over due to useless training')
            break

    for epoch in range(params['VAE_epoch_num']):
        total_loss = 0
        heat_capacity_vae_mlp.train()
        for i, data in enumerate(tqdm(heat_capacity_train_loader, 0)):
            one_hot = torch.cat((data[0], data[1]), dim=1)
            one_hot = one_hot.reshape(one_hot.shape[0], 1, one_hot.shape[1], one_hot.shape[2])
            one_hot = Variable(one_hot).cuda().type(torch.cuda.FloatTensor)
            heat_capacity_optimizer.zero_grad()
            t = data[2].cuda().reshape(data[2].shape[0], 1).type(torch.cuda.FloatTensor)
            recon_x, mu, logvar, pre = heat_capacity_vae_mlp.forward(one_hot, t)
            recon_x = recon_x[:, :, :one_hot.shape[2], :one_hot.shape[3]]
            loss = loss_func(recon_x, one_hot, mu, logvar, pre, t)
            loss.backward()
            total_loss += loss.data.item() / 1000
            heat_capacity_optimizer.step()
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, total_loss / len(heat_capacity_train_loader.dataset)))
        heat_capacity_total_loss_list[epoch] = total_loss / len(heat_capacity_train_loader.dataset)
        if np.argmin(heat_capacity_total_loss_list) == epoch:
            torch.save(heat_capacity_vae_mlp, heat_capacity_model_file_name)
            print('best result, saving the model to ' + heat_capacity_model_file_name)
        elif np.argmin(thermal_conductivity_total_loss_list) == epoch - 25:
            print('Finish: Training process over due to useless training')
            break
