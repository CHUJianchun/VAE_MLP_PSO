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


class VAE(nn.Module):
    def __init__(self, latent_code_num):
        super(VAE, self).__init__()
        self.anion_encoder = nn.Sequential(
            # 1, 62, 32
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 32, 31, 16
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64, 30, 15
            nn.Conv2d(64, 128, kernel_size=5, stride=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128, 10, 5
        )
        self.cation_encoder = nn.Sequential(
            # 1, 62, 32
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 32, 31, 16
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64, 30, 15
            nn.Conv2d(64, 128, kernel_size=5, stride=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128, 10, 5
        )

        self.fc11 = nn.Linear(128 * 10 * 5 * 2, latent_code_num)
        self.fc12 = nn.Linear(128 * 10 * 5 * 2, latent_code_num)
        self.fc2 = nn.Linear(latent_code_num, 128 * 10 * 5 * 2)

        self.anion_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=3, padding=1),
            nn.Sigmoid()
        )

        self.cation_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=3, padding=1),
            nn.Sigmoid()
        )

    def get_reparameterized_code(self, anion, cation):
        anion_out1, anion_out2 = self.anion_encoder(anion), self.anion_encoder(anion)
        cation_out1, cation_out2 = self.cation_encoder(cation), self.cation_encoder(cation)
        # 128, 50 + 50 = 100
        out1 = torch.cat((anion_out1.view(anion_out1.size(0), -1), cation_out1.view(cation_out1.size(0), -1)), dim=1)
        out2 = torch.cat((anion_out2.view(anion_out2.size(0), -1), cation_out2.view(cation_out2.size(0), -1)), dim=1)
        # latent_code_num
        mu = self.fc11(out1)
        logvar = self.fc12(out2)
        z = self.reparameterize(mu, logvar)
        return z

    def get_mid(self, anion, cation):
        anion_out1 = self.anion_encoder(anion.type(torch.cuda.FloatTensor))
        cation_out1 = self.cation_encoder(cation.type(torch.cuda.FloatTensor))
        out1 = torch.cat((anion_out1.view(anion_out1.size(0), -1), cation_out1.view(cation_out1.size(0), -1)), dim=1)
        mu = self.fc11(out1)
        return mu

    def forward(self, anion, cation):
        # 128, 10, 5
        anion_out1, anion_out2 = self.anion_encoder(anion), self.anion_encoder(anion)
        cation_out1, cation_out2 = self.cation_encoder(cation), self.cation_encoder(cation)
        # 128, 50 + 50 = 100
        out1 = torch.cat((anion_out1.view(anion_out1.size(0), -1), cation_out1.view(cation_out1.size(0), -1)), dim=1)
        out2 = torch.cat((anion_out2.view(anion_out2.size(0), -1), cation_out2.view(cation_out2.size(0), -1)), dim=1)
        # latent_code_num
        mu = self.fc11(out1)
        logvar = self.fc12(out2)
        z = reparameterize(mu, logvar)
        out3 = self.fc2(z).view(2, z.size(0), 128, 10, 5)
        out3_anion = out3[0]
        out3_cation = out3[1]
        return self.anion_decoder(out3_anion), self.cation_decoder(out3_cation), mu, logvar

    def decode(self, z):
        out3 = self.fc2(z).view(2, -1, 128, 10, 5)
        out3_anion = out3[0]
        out3_cation = out3[1]
        return self.anion_decoder(out3_anion), self.cation_decoder(out3_cation)


def loss_func(recon_x, x, mu, logvar):
    binary_cross_entropy = f.binary_cross_entropy(recon_x, x, size_average=False)
    k_l_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return binary_cross_entropy + k_l_divergence


def train_vae(latent_code_num, params, device):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    train_loader = torch.load('Data/vae_train_loader.pkl')
    vae = VAE(latent_code_num).cuda()
    optimizer = optim.Adam(vae.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    total_loss_list = np.ones(params['VAE_epoch_num'] + 10)
    total_loss_list *= 5000000
    model_file_name = 'Model_pkl/VAE_' + str(latent_code_num) + '.pkl'

    def train(epoch):
        vae.train()
        total_loss = 0
        for i, data_ in enumerate(tqdm(train_loader), 0):
            anion = data_[0]
            cation = data_[1]
            anion = anion.reshape(anion.shape[0], 1, anion.shape[1], anion.shape[2]).cuda()
            cation = cation.reshape(cation.shape[0], 1, cation.shape[1], cation.shape[2]).cuda()
            anion = Variable(anion).cuda().type(torch.cuda.FloatTensor)
            cation = Variable(cation).cuda().type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            new_anion, new_cation, mu, logvar = vae.forward(anion, cation)
            new_anion = new_anion[:anion.shape[0], :anion.shape[1], :anion.shape[2], :anion.shape[3]]
            new_cation = new_cation[:anion.shape[0], :anion.shape[1], :anion.shape[2], :anion.shape[3]]
            loss = loss_func(torch.cat((new_anion, new_cation), dim=1), torch.cat((anion, cation), dim=1), mu, logvar)
            loss.backward()
            total_loss += loss.data.item()
            optimizer.step()
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, total_loss / len(train_loader.dataset)))
        total_loss_list[epoch] = total_loss / len(train_loader.dataset)
        if np.argmin(total_loss_list) == epoch:
            torch.save(vae, model_file_name)
            print('best result, saving the model to ' + model_file_name)

    for epoch_ in range(params['VAE_epoch_num']):
        train(epoch_)
