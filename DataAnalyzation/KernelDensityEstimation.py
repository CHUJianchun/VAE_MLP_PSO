import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from hyperparameters import load_params
from tqdm import tqdm
import torch
import os


def plot_kde(params, vae_name, device):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    vae = torch.load('Model_pkl/' + vae_name)
    dataset = torch.load('Data/vae_dataset.pkl')
    index = torch.utils.data.sampler.SubsetRandomSampler(np.random.choice(range(len(dataset)), params['KDE_samples']))
    sample = []
    for i in index:
        sample.append(torch.cat((dataset[i][0], dataset[i][1]), dim=0).numpy())
    sample = torch.tensor(np.array(sample))
    sample = sample.reshape(sample.shape[0], 1, sample.shape[1], sample.shape[2]).type(torch.cuda.FloatTensor).cuda()
    del dataset
    latent_space_matrix = vae.get_mid(sample).detach().cpu().numpy()
    latent_space_matrix = np.sort(latent_space_matrix, axis=0)
    del sample, vae
    plt.figure(figsize=(9, 7), dpi=400)
    fontdict = {'family': 'Times New Roman', 'size': '25'}
    print('start drawing')
    for i in tqdm(range(latent_space_matrix.shape[1])):
        kde = KernelDensity(kernel='gaussian', bandwidth=1.2).fit(latent_space_matrix[:, i].reshape(-1, 1))
        plt.plot(latent_space_matrix[:, 0], np.exp(kde.score_samples(latent_space_matrix[:, i].reshape(-1, 1))), '-')
    # plt.title('KDE of each dimension of the latent space',
    #           fontdict={'family': 'Times New Roman', 'size': 25})
    plt.xlabel('Unstandardized z', fontdict=fontdict)
    plt.ylabel('Normalized Frequency', fontdict=fontdict)
    plt.xticks(np.arange(-0.2, 0.2, 0.1), fontproperties='Times New Roman', size=25)
    plt.yticks(np.arange(0, 0.5, 0.1), fontproperties='Times New Roman', size=25)
    plt.savefig('Graph/' + vae_name + '_kde.jpg')
    plt.show()
