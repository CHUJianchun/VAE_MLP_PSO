import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.autograd import Variable
import os
import numpy as np
from tqdm import tqdm
import sys

def reparameterize(mu, logvar):
    eps = Variable(torch.randn(mu.size(0), mu.size(1))).cuda()
    z = mu + eps * torch.exp(logvar / 2)
    return z


class VAE_CAT(nn.Module):
    def __init__(self, latent_code_num):
        super(VAE_CAT, self).__init__()
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

    def forward(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)  # batch_s, 8, 7, 7
        mu = self.fc11(out1.view(out1.size(0), -1))  # batch_s, latent
        logvar = self.fc12(out2.view(out2.size(0), -1))  # batch_s, latent
        z = reparameterize(mu, logvar)  # batch_s, latent
        out3 = self.fc2(z).view(z.size(0), 128, 20, 5)  # batch_s, 8, 7, 7
        return self.decoder(out3), mu, logvar

    def get_mid(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)
        mu = self.fc11(out1.view(out1.size(0), -1))
        return mu

    def decode(self, z):
        out3 = self.fc2(z).view(1, 128, 20, 5)
        return self.decoder(out3)


def loss_func(recon_x, x, mu, logvar):
    mse = f.mse_loss(recon_x, x, size_average=False)
    k_l_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + k_l_divergence


def train_vae_cat(latent_code_num, params, device):
    print('Totally ' + str(params['VAE_epoch_num']) + ' epochs to train')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    train_loader = torch.load('Data/vae_train_loader.pkl')
    vae = VAE_CAT(latent_code_num).cuda()
    optimizer = optim.Adam(vae.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    total_loss_list = np.ones(params['VAE_epoch_num'] + 10)
    total_loss_list *= 5000000
    model_file_name = 'Model_pkl/VAE_CAT_' + str(latent_code_num) + '.pkl'

    def train(epoch):
        vae.train()
        total_loss = 0
        for i, data in enumerate(tqdm(train_loader, 0)):
            data = torch.cat((data[0], data[1]), dim=1)
            data = data.reshape(data.shape[0], 1, data.shape[1], data.shape[2]).cuda()
            data = Variable(data).cuda().type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            recon_x, mu, logvar = vae.forward(data)
            recon_x = recon_x[:, :, :data.shape[2], :data.shape[3]]
            loss = loss_func(recon_x, data, mu, logvar)
            loss.backward()
            total_loss += loss.data.item()
            optimizer.step()
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, total_loss / len(train_loader.dataset)))
        total_loss_list[epoch] = total_loss / len(train_loader.dataset)
        if np.argmin(total_loss_list) == epoch:
            torch.save(vae, model_file_name)
            print('best result, saving the model to ' + model_file_name)
            fail = 0
        elif np.argmin(total_loss_list) == epoch - 15:
            print('Finish: Training process over due to useless training')
            sys.exit()

    for epoch_ in range(params['VAE_epoch_num']):
        train(epoch_)
