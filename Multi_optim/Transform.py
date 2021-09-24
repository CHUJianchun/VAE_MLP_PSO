import pickle
import sys
import torch
import numpy as np


def latent_code_to_smiles(latent_code, vae_file_name):
    try:
        with open('Data/one_hot_dictionary.data', 'rb') as f_one_hot_dictionary:
            one_hot_dictionary = pickle.load(f_one_hot_dictionary)
    except IOError:
        print('Error: Data/one_hot_dictionary.data not found, please finish previous steps first')
        sys.exit()

    try:
        vae = torch.load('Model_pkl/' + vae_file_name)
    except FileNotFoundError:
        print('Error: ' + 'Model_pkl/' + vae_file_name + ' No such model found, please train one')
        sys.exit()

    def one_hot_to_smiles(one_hot):
        def get_atom(a):
            for item in one_hot_dictionary.items():
                if item[1] == a:
                    return item[0]
        one_hot = np.squeeze(one_hot)
        smiles = ''
        for i in range(one_hot.shape[0]):
            if np.max(one_hot[i]) > 0.0:
                j = np.argmax(one_hot[i])
                string = get_atom(j)
                if string == 'A':
                    string = 'Br'
                elif string == 'D':
                    string = 'Na'
                elif string == 'E':
                    string = 'Cl'
                elif string == 'G':
                    string = 'Al'
                elif string == 'J':
                    string = 'NH3'
                elif string == 'K':
                    string = 'NH2'
                elif string == 'L':
                    string = 'NH'
                elif string is None:
                    string = ''
            else:
                string = ' '
            smiles += string
        return smiles

    anion_one_hot = vae.decode(latent_code.to(torch.float32))[0].cpu().detach().numpy()
    cation_one_hot = vae.decode(latent_code.to(torch.float32))[1].cpu().detach().numpy()
    anion_smiles = one_hot_to_smiles(anion_one_hot)
    cation_smiles = one_hot_to_smiles(cation_one_hot)
    return anion_smiles, cation_smiles


def latent_code_to_smiles_cat(latent_code, vae_file_name):
    try:
        with open('Data/one_hot_dictionary.data', 'rb') as f_one_hot_dictionary:
            one_hot_dictionary = pickle.load(f_one_hot_dictionary)
    except IOError:
        print('Error: Data/one_hot_dictionary.data not found, please finish previous steps first')
        sys.exit()

    try:
        vae = torch.load('Model_pkl/' + vae_file_name)
    except FileNotFoundError:
        print('Error: ' + 'Model_pkl/' + vae_file_name + ' No such model found, please train one')
        sys.exit()

    def one_hot_to_smiles(one_hot_):
        def get_atom(a):
            for item in one_hot_dictionary.items():
                if item[1] == a:
                    return item[0]
        one_hot_ = np.squeeze(one_hot_)
        smiles_ = ''
        for i in range(one_hot_.shape[0]):
            if np.max(one_hot_[i]) > 0.5:
                j = np.argmax(one_hot_[i])
                string = get_atom(j)
                if string == 'A':
                    string = 'Br'
                elif string == 'D':
                    string = 'Na'
                elif string == 'E':
                    string = 'Cl'
                elif string == 'G':
                    string = 'Al'
                elif string == 'J':
                    string = 'NH3'
                elif string == 'K':
                    string = 'NH2'
                elif string == 'L':
                    string = 'NH'
                elif string is None:
                    string = ''
            else:
                string = ' '
            smiles_ += string
        return smiles_

    one_hot = vae.decode(latent_code.to(torch.float32)).cpu().detach().numpy()

    smiles = one_hot_to_smiles(one_hot)

    return smiles
