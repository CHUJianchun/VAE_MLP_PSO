import torch
from torch.autograd import Variable
import sys
from tqdm import tqdm
import numpy as np
import pickle

def vae_cat_mean(latent_code):
    try:
        vae = torch.load('Model_pkl/VAE_CAT_' + str(latent_code) + '.pkl').cuda()
    except FileNotFoundError:
        print('Error: No such file found, please train it first')
        sys.exit()

    try:
        dataset = torch.load('Data/vae_dataset.pkl')
    except FileNotFoundError:
        print('Error: vae_dataset.pkl not found')
        sys.exit()

    print('Start: Analysing mean of Model_pkl/VAE_CAT_' + str(latent_code) + '.pkl')
    mid_list = []
    for i, data_ in enumerate(tqdm(dataset), 0):
        data_ = torch.cat((data_[0], data_[1]), dim=1)
        data_ = data_.reshape(1, 1, data_.shape[0], data_.shape[1]).cuda()
        data_ = Variable(data_).cuda().type(torch.cuda.FloatTensor)
        mid = vae.get_mid(data_)
        mid_list.append(mid.detach().cpu().numpy())
    mid_list = np.array(mid_list)
    mid_average = np.average(mid_list, axis=0)
    return mid_average
