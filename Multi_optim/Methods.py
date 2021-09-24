import torch
from torch.autograd import Variable
import numpy as np
import sys
import pickle
import DataAnalyzation.Mean as Mean
from tqdm import tqdm
from rdkit import Chem
import Multi_optim.Transform as Transform
from sko.PSO import PSO
import rdkit
from rdkit.Chem import Draw
from rdkit.ML.Descriptors import MoleculeDescriptors
import os


def gradient_descent_optimize(params, mlp_file_name, property_name):
    latent_code_num = int(mlp_file_name.split('_')[2])
    data_init = torch.randn(params['Start_point_num'], latent_code_num + 2)
    data_init = torch.clamp(data_init, -0.4, 0.4)
    try:
        with open('Data/vae_cat_mean_' + str(latent_code_num) + '.data') as f_mean:
            vae_mean = pickle.load(f_mean)
    except IOError:
        print('Warning: Data/vae_cat_mean_' + str(latent_code_num) + '.data not found')
        Mean.vae_cat_mean(latent_code_num)
        with open('Data/vae_cat_mean_' + str(latent_code_num) + '.data') as f_mean:
            vae_mean = pickle.load(f_mean)
    for i in range(params['Start_point_num']):
        data_init[i][2:] += vae_mean
    data_init[:, 0] = 0.5
    data_init[:, 1] = 0.5
    data_init = Variable(data_init).cuda()
    try:
        mlp = torch.load('Model_pkl/' + mlp_file_name).cuda()
    except FileNotFoundError:
        print('Error: ' + 'Model_pkl/' + mlp_file_name + 'No such model found, please train one')
        sys.exit()
    net = mlp.viscosity_mlp
    exec('net = mlp.' + property_name + '_mlp')
    average = (torch.ones(latent_code_num + 2) * 0.5).cuda()
    optim_result = np.zeros((data_init.shape[0], params['OPTIM_epoch_num']))
    optim_latent_space_code = np.zeros((data_init.shape[0], params['OPTIM_epoch_num'], data_init.shape[1]))
    for epoch in range(params['OPTIM_epoch_num']):
        for i in range(params['Start_point_num']):
            optim_point = data_init[i]
            optim_point.requires_grad_(True)
            optimizer = torch.optim.Adam([optim_point], lr=1e-2)
            optimizer.zero_grad()
            property_init = net(optim_point)
            difference = torch.sum((optim_point - average) ** 2)
            loss = -property_init + difference * params['Radical_factor']
            loss.backward()
            optimizer.step()
            data_init[i] = optim_point.detach()
            optim_result[i][epoch] = net(optim_point).cpu().detach().numpy()
            optim_latent_space_code[i][epoch] = optim_point.cpu().detach().numpy()
    return optim_result, optim_latent_space_code


def gradient_descent_optimize_cat(params, mlp_file_name):
    try:
        latent_code_num = int(mlp_file_name.split('_')[5])
    except ValueError:
        latent_code_num = int(mlp_file_name.split('_')[6])
    data_init = torch.randn(params['Start_point_num'], latent_code_num + 2)
    data_init = torch.clamp(data_init, -1.1, 1.1)

    try:
        with open('Data/vae_cat_mean_' + str(latent_code_num) + '.data', 'rb') as f_mean:
            vae_mean = pickle.load(f_mean)
    except IOError:
        print('Warning: Data/vae_cat_mean_' + str(latent_code_num) + '.data not found')
        Mean.vae_cat_mean(latent_code_num)
        with open('Data/vae_cat_mean_' + str(latent_code_num) + '.data') as f_mean:
            vae_mean = pickle.load(f_mean)

    for i in range(params['Start_point_num']):
        data_init[i][2:] += vae_mean
    data_init[:, 0] = 0.5
    data_init[:, 1] = 0.5
    data_init = Variable(data_init).cuda()
    try:
        mlp = torch.load('Model_pkl/' + mlp_file_name).cuda()
    except FileNotFoundError:
        print('Error: ' + 'Model_pkl/' + mlp_file_name + 'No such model found, please train one')
        sys.exit()

    for p in mlp.parameters():
        p.requires_grad_(False)

    average = torch.tensor(vae_mean).to(torch.float32).cuda()
    optim_result = np.zeros((data_init.shape[0], params['OPTIM_epoch_num']))
    optim_latent_space_code = np.zeros((data_init.shape[0], params['OPTIM_epoch_num'], latent_code_num + 2))
    for epoch in tqdm(range(params['OPTIM_epoch_num'])):
        for i in range(params['Start_point_num']):
            optim_point = data_init[i]
            optim_point.requires_grad_(True)
            optimizer = torch.optim.Adam([optim_point], lr=1e-1)
            optimizer.zero_grad()
            property_ = mlp(optim_point)
            difference = torch.sum((optim_point[2:] - average) ** 2)
            loss = - property_ + difference * params['Radical_factor']
            loss.backward()
            optimizer.step()
            data_init[i] = optim_point.detach()
            data_init[i, 0] = 0.5
            data_init[i, 1] = 0.5
            optim_result[i][epoch] = mlp(optim_point).cpu().detach().numpy()
            optim_latent_space_code[i][epoch] = optim_point.cpu().detach().numpy()
    return optim_result, optim_latent_space_code


def particle_swarm_optimization_cat(mlp_file_name, temperature):
    # graph_file_name = 'default.png'
    for it in range(1, 100000):
        if not os.path.exists('Graph/' + str(it) + '.png'):
            graph_file_name = str(it) + '.png'
            break

    try:
        latent_code_num = int(mlp_file_name.split('_')[5])
    except ValueError:
        latent_code_num = int(mlp_file_name.split('_')[6])

    try:
        mlp = torch.load('Model_pkl/' + mlp_file_name).cuda()
    except FileNotFoundError:
        print('Error: ' + 'Model_pkl/' + mlp_file_name + 'No such model found, please train one')
        sys.exit()

    for p in mlp.parameters():
        p.requires_grad_(False)
    vae_file_name = 'VAE_CAT_' + str(latent_code_num) + '.pkl'

    def available_test(smiles):
        available = 0
        write_ = False
        ionic_ = ''
        ionic_list_ = []
        for character_ in smiles:
            if character_ == '0' and write_ is not True:
                ionic_ = ''
                write_ = True
            if not character_ == '0' and not character_ == '9' and write_ is True:
                ionic_ += character_
            if character_ == '9':
                write_ = False
                if len(ionic_) > 0:
                    ionic_list_.append(ionic_)
        if len(ionic_list_) > 1:
            first = max(ionic_list_, key=len)
            ionic_list_.remove(first)
            second = max(ionic_list_, key=len)
            if 'I' not in first and 'I' not in second:
                try:
                    mol_1 = Chem.MolFromSmiles(first)
                    mol_2 = Chem.MolFromSmiles(second)
                except ValueError:
                    pass
                else:
                    if type(mol_1) == rdkit.Chem.rdchem.Mol and type(mol_2) == rdkit.Chem.rdchem.Mol:
                        if '-' in first and '+' in second and ' ' not in first and ' ' not in second:
                            MolWeight = MoleculeDescriptors.MolecularDescriptorCalculator(['MolWt'])
                            count_1 = MolWeight.CalcDescriptors(mol_1)[0]
                            count_2 = MolWeight.CalcDescriptors(mol_2)[0]
                            if count_1 > 80 and count_2 > 80:
                                available = -3000
                        elif '+' in first and '-' in second and ' ' not in first and ' ' not in second:
                            MolWeight = MoleculeDescriptors.MolecularDescriptorCalculator(['MolWt'])
                            count_1 = MolWeight.CalcDescriptors(mol_1)[0]
                            count_2 = MolWeight.CalcDescriptors(mol_2)[0]
                            if count_1 > 80 and count_2 > 80:
                                available = -3000
        return available

    def optim_func(latent_code):
        latent_code = Variable(torch.FloatTensor(latent_code)).cuda()
        property_ = -mlp(latent_code)
        smiles = Transform.latent_code_to_smiles_cat(latent_code[2:], vae_file_name)
        available = available_test(smiles)
        return property_ + available

    low_boundary = (np.ones(latent_code_num + 1) * (-1)).tolist()
    up_boundary = (np.ones(latent_code_num + 1)).tolist()

    t_max = 663.1
    t_min = 0

    temperature = (temperature - t_min) / (t_max - t_min)

    low_boundary[0] = temperature  # T
    up_boundary[0] = temperature + 0.000001

    pso = PSO(func=optim_func, n_dim=latent_code_num + 1, pop=300, max_iter=200,
              lb=low_boundary, ub=up_boundary, w=0.8, c1=0.5, c2=0.5)
    pso.run()
    print(pso.best_y)
    latent_code = pso.best_x
    smiles_final = Transform.latent_code_to_smiles_cat(torch.FloatTensor(pso.best_x[2:]).cuda(), vae_file_name)
    ionic_list = []
    write = False
    ionic = ''
    for character in smiles_final:
        if character == '0' and write is not True:
            ionic = ''
            write = True
        if not character == '0' and not character == '9' and write is True:
            ionic += character
        if character == '9':
            write = False
            if len(ionic) > 0:
                ionic_list.append(ionic)
    if len(ionic_list) > 1:
        first = max(ionic_list, key=len)
        ionic_list.remove(first)
        second = max(ionic_list, key=len)
        try:
            mol_1 = Chem.MolFromSmiles(first)
            mol_2 = Chem.MolFromSmiles(second)
        except ValueError:
            print('Error: SMILES illegal')
        else:
            if type(mol_1) == rdkit.Chem.rdchem.Mol and type(mol_2) == rdkit.Chem.rdchem.Mol:
                if '-' in first and '+' in second and ' ' not in first and ' ' not in second:
                    mols = [mol_1, mol_2]
                    img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(400, 400),
                                               legends=['' for x in mols])
                    img.save('Graph/' + graph_file_name)
                elif '+' in first and '-' in second and ' ' not in first and ' ' not in second:
                    mols = [mol_1, mol_2]
                    img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(400, 400),
                                               legends=['' for x in mols])
                    img.save('Graph/' + graph_file_name)


def multi_particle_swarm_optimization_cat(mlp_file_name_h, mlp_file_name_t, temperature):
    # graph_file_name = 'default.png'
    for it in range(1, 100000):
        if not os.path.exists('Graph/' + str(it) + '.png'):
            graph_file_name = str(it) + '.png'
            break

    try:
        latent_code_num = int(mlp_file_name_t.split('_')[5])
    except ValueError:
        latent_code_num = int(mlp_file_name_t.split('_')[6])

    try:
        mlp_t = torch.load('Model_pkl/' + mlp_file_name_t).cuda()
        mlp_h = torch.load('Model_pkl/' + mlp_file_name_h).cuda()
    except FileNotFoundError:
        print('Error: No such model found, please train one')
        sys.exit()

    for p in mlp_t.parameters():
        p.requires_grad_(False)
    for p in mlp_h.parameters():
        p.requires_grad_(False)
    vae_file_name = 'VAE_CAT_' + str(latent_code_num) + '.pkl'

    def available_test(smiles):
        available = 0
        write_ = False
        ionic_ = ''
        ionic_list_ = []
        for character_ in smiles:
            if character_ == '0' and write_ is not True:
                ionic_ = ''
                write_ = True
            if not character_ == '0' and not character_ == '9' and write_ is True:
                ionic_ += character_
            if character_ == '9':
                write_ = False
                if len(ionic_) > 0:
                    ionic_list_.append(ionic_)
        if len(ionic_list_) > 1:
            first = max(ionic_list_, key=len)
            ionic_list_.remove(first)
            second = max(ionic_list_, key=len)
            if 'I' not in first and 'I' not in second:
                try:
                    mol_1 = Chem.MolFromSmiles(first)
                    mol_2 = Chem.MolFromSmiles(second)
                except ValueError:
                    pass
                else:
                    if type(mol_1) == rdkit.Chem.rdchem.Mol and type(mol_2) == rdkit.Chem.rdchem.Mol:
                        if '-' in first and '+' in second and ' ' not in first and ' ' not in second:
                            MolWeight = MoleculeDescriptors.MolecularDescriptorCalculator(['MolWt'])
                            count_1 = MolWeight.CalcDescriptors(mol_1)[0]
                            count_2 = MolWeight.CalcDescriptors(mol_2)[0]
                            if count_1 > 80 and count_2 > 80:
                                available = -1
                        elif '+' in first and '-' in second and ' ' not in first and ' ' not in second:
                            MolWeight = MoleculeDescriptors.MolecularDescriptorCalculator(['MolWt'])
                            count_1 = MolWeight.CalcDescriptors(mol_1)[0]
                            count_2 = MolWeight.CalcDescriptors(mol_2)[0]
                            if count_1 > 80 and count_2 > 80:
                                available = -1
        return available

    def optim_func(latent_code_):
        latent_code_ = Variable(torch.FloatTensor(latent_code_)).cuda()
        property_t = mlp_t(latent_code_)
        property_h = mlp_h(latent_code_)
        smiles = Transform.latent_code_to_smiles_cat(latent_code_[1:], vae_file_name)
        available = available_test(smiles)
        return -property_t * 0.18 - property_h * 0.00002 + available

    low_boundary = (np.ones(latent_code_num + 1) * (-1)).tolist()
    up_boundary = (np.ones(latent_code_num + 1)).tolist()

    t_max = 663.1
    t_min = 0

    temperature = (temperature - t_min) / (t_max - t_min)

    low_boundary[0] = temperature  # T
    up_boundary[0] = temperature + 0.000001

    pso = PSO(func=optim_func, n_dim=latent_code_num + 1, pop=300, max_iter=200,
              lb=low_boundary, ub=up_boundary, w=0.8, c1=0.5, c2=0.5)
    pso.run()
    print(pso.best_y)
    latent_code = pso.best_x
    smiles_final = Transform.latent_code_to_smiles_cat(torch.FloatTensor(pso.best_x[1:]).cuda(), vae_file_name)
    ionic_list = []
    write = False
    ionic = ''
    for character in smiles_final:
        if character == '0' and write is not True:
            ionic = ''
            write = True
        if not character == '0' and not character == '9' and write is True:
            ionic += character
        if character == '9':
            write = False
            if len(ionic) > 0:
                ionic_list.append(ionic)
    if len(ionic_list) > 1:
        first = max(ionic_list, key=len)
        ionic_list.remove(first)
        second = max(ionic_list, key=len)
        try:
            mol_1 = Chem.MolFromSmiles(first)
            mol_2 = Chem.MolFromSmiles(second)
        except ValueError:
            print('Error: SMILES illegal')
        else:
            if type(mol_1) == rdkit.Chem.rdchem.Mol and type(mol_2) == rdkit.Chem.rdchem.Mol:
                if '-' in first and '+' in second and ' ' not in first and ' ' not in second:
                    mols = [mol_1, mol_2]
                    img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(400, 400),
                                               legends=['' for x in mols])
                    img.save('Graph/' + graph_file_name)
                elif '+' in first and '-' in second and ' ' not in first and ' ' not in second:
                    mols = [mol_1, mol_2]
                    img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(400, 400),
                                               legends=['' for x in mols])
                    img.save('Graph/' + graph_file_name)
