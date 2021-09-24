from VariationalAutoEncoder.VAE_model import VAE
from MultiLayerPerceptron.MLP_model import MLP
import torch
from torch.autograd import Variable
import sys
from tqdm import tqdm
import numpy as np
import importlib
from hyperparameters import load_params
import os


def check_vae_accuracy(latent_code):
    # importlib.reload(VAE)
    # importlib.reload(MLP)
    try:
        vae = torch.load('Model_pkl/VAE_' + str(latent_code) + '.pkl').cuda()
    except FileNotFoundError:
        print('Error: No such file found, please train it first')
        sys.exit()

    try:
        dataset = torch.load('Data/vae_dataset.pkl')
    except FileNotFoundError:
        print('Error: vae_dataset.pkl not found')
        sys.exit()

    print('Start: Analysing accuracy of Model_pkl/VAE_' + str(latent_code) + '.pkl')
    accuracy_list = []
    for i, data_ in enumerate(tqdm(dataset), 0):
        anion = data_[0].reshape((1, 1, data_[0].shape[0], data_[0].shape[1]))
        cation = data_[1].reshape((1, 1, data_[1].shape[0], data_[1].shape[1]))
        anion = Variable(anion).cuda().type(torch.cuda.FloatTensor)
        cation = Variable(cation).cuda().type(torch.cuda.FloatTensor)
        # new_anion, new_cation, _, __ = vae.forward(anion, cation)
        new_anion, new_cation = vae.decode(vae.get_mid(anion, cation))
        anion = torch.squeeze(anion)
        cation = torch.squeeze(cation)
        new_anion = torch.squeeze(new_anion)
        new_cation = torch.squeeze(new_cation)
        cols = cation.shape[0]
        error = 0
        for bit in range(cols):
            anion_decode_value = torch.argmax(new_anion[bit])
            anion_input_value = torch.argmax(anion[bit])
            cation_decode_value = torch.argmax(new_cation[bit])
            cation_input_value = torch.argmax(cation[bit])
            if not anion_input_value == anion_decode_value:
                error += 1
            if not cation_input_value == cation_decode_value:
                error += 1
        accuracy = 1 - error / (cols * 2)
        accuracy_list.append(accuracy)
    accuracy_list = np.array(accuracy_list)
    accuracy = np.average(accuracy_list)
    return accuracy


def check_mlp_accuracy(mlp_file_name):
    # importlib.reload(VAE)
    # importlib.reload(MLP)

    latent_code_num = int(mlp_file_name.split('_')[2])

    try:
        mlp = torch.load('Model_pkl/' + mlp_file_name)
    except FileNotFoundError:
        print('Error: No such file found, please train it first')
        sys.exit()

    try:
        v_dataset = torch.load('Data/viscosity_mlp_dataset_VAE_' + str(latent_code_num) + '.pkl')
        t_dataset = torch.load('Data/thermal_conductivity_mlp_dataset_VAE_' + str(latent_code_num) + '.pkl')
        h_dataset = torch.load('Data/heat_capacity_mlp_dataset_VAE_' + str(latent_code_num) + '.pkl')
    except FileNotFoundError:
        print('Error: No such dataset found, please create first')
        sys.exit()

    def property_accuracy(property_):
        if property_ == 'viscosity':
            dataset = v_dataset
            net = mlp.viscosity_mlp
        elif property_ == 'heat_capacity':
            dataset = h_dataset
            net = mlp.heat_capacity_mlp
        else:
            dataset = t_dataset
            net = mlp.thermal_conductivity_mlp
        AARD = []
        for i, (input_, label_) in enumerate(tqdm(dataset), 0):
            input_ = Variable(input_).cuda().type(torch.cuda.FloatTensor)
            prediction = net(input_.cuda()).detach().cpu().numpy()
            label_ = label_.detach().cpu().numpy()
            AARD.append(abs((prediction - label_) / label_))
        AARD = np.average(np.array(AARD))
        return AARD
    viscosity_aard = property_accuracy('viscosity')
    thermal_conductivity_aard = property_accuracy('thermal_conductivity')
    heat_capacity_aard = property_accuracy('heat_capacity')
    return viscosity_aard, thermal_conductivity_aard, heat_capacity_aard


def check_vae_cat_accuracy(latent_code, form, device):
    params = load_params()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    if form == 'CNN':
        try:
            vae = torch.load('Model_pkl/VAE_CAT_' + str(latent_code) + '.pkl').cuda()
            print('Start: Analysing accuracy of Model_pkl/VAE_CAT_' + str(latent_code) + '.pkl')
        except FileNotFoundError:
            print('Error: No such file found, please train it first')
            sys.exit()
    elif form == 'LSTM':
        try:
            vae = torch.load('Model_pkl/VAE_LSTM_CAT_' + str(latent_code) + '.pkl').cuda()
            print('Start: Analysing accuracy of Model_pkl/VAE_LSTM_CAT_' + str(latent_code) + '.pkl')
        except FileNotFoundError:
            print('Error: No such file found, please train it first')
            sys.exit()
    try:
        dataset = torch.load('Data/vae_dataset.pkl')
    except FileNotFoundError:
        print('Error: vae_dataset.pkl not found')
        sys.exit()

    accuracy_decode_list = []
    accuracy_reform_list = []
    for i, data_ in enumerate(tqdm(dataset), 0):
        data_ = torch.cat((data_[0], data_[1]), dim=0)
        data_ = data_.reshape(1, 1, data_.shape[0], data_.shape[1]).cuda()
        data_ = Variable(data_).cuda().type(torch.cuda.FloatTensor)

        if form == 'CNN':
            new_data, _, __ = vae.forward(data_)
            reform_data = vae.decoder(vae.fc2(vae.get_mid(data_)).view(1, 128, 20, 5))
        elif form == 'LSTM':
            new_data = vae.decode(vae.get_reparameterized_code(data_))
            reform_data = vae.decode(vae.get_mid(data_))
            new_data = new_data.permute(1, 0, 2)
            reform_data = reform_data.permute(1, 0, 2)
        data_ = torch.squeeze(data_)
        new_data = torch.squeeze(new_data)
        reform_data = torch.squeeze(reform_data)
        cols = data_.shape[0]
        error_decode = 0
        error_reform = 0
        for bit in range(cols):
            decode_value = torch.argmax(new_data[bit])
            reform_value = torch.argmax(reform_data[bit])
            input_value = torch.argmax(data_[bit])
            if not input_value == decode_value:
                error_decode += 1
            if not input_value == reform_value:
                error_reform += 1

        accuracy_decode = 1 - error_decode / cols
        accuracy_reform = 1 - error_reform / cols
        accuracy_decode_list.append(accuracy_decode)
        accuracy_reform_list.append(accuracy_reform)
    accuracy_decode_list = np.array(accuracy_decode_list)
    accuracy_reform_list = np.array(accuracy_reform_list)
    accuracy_decode = np.average(accuracy_decode_list)
    accuracy_reform = np.average(accuracy_reform_list)
    return accuracy_decode, accuracy_reform


def check_mlp_cat_accuracy(mlp_cat_file_name):
    # importlib.reload(VAE)
    # importlib.reload(MLP)

    latent_code_num = int(mlp_cat_file_name.split('_')[3])
    v_file_name = mlp_cat_file_name[:8] + 'viscosity' + mlp_cat_file_name[7:]
    t_file_name = mlp_cat_file_name[:8] + 'thermal_conductivity' + mlp_cat_file_name[7:]
    h_file_name = mlp_cat_file_name[:8] + 'heat_capacity' + mlp_cat_file_name[7:]

    try:
        v_mlp = torch.load('Model_pkl/' + v_file_name)
        t_mlp = torch.load('Model_pkl/' + t_file_name)
        h_mlp = torch.load('Model_pkl/' + h_file_name)
    except FileNotFoundError:
        print('Error: No such file found, please train it first')
        sys.exit()

    try:
        v_dataset = torch.load('Data/viscosity_mlp_cat_dataset_VAE_CAT_' + str(latent_code_num) + '.pkl')
        t_dataset = torch.load('Data/thermal_conductivity_mlp_cat_dataset_VAE_CAT_' + str(latent_code_num) + '.pkl')
        h_dataset = torch.load('Data/heat_capacity_mlp_cat_dataset_VAE_CAT_' + str(latent_code_num) + '.pkl')
    except FileNotFoundError:
        print('Error: No such dataset found, please create first')
        sys.exit()

    def property_accuracy(property_):
        if property_ == 'viscosity':
            dataset = v_dataset
            net = v_mlp
        elif property_ == 'heat_capacity':
            dataset = h_dataset
            net = h_mlp
        else:
            dataset = t_dataset
            net = t_mlp
        AARD = []
        for i, (input_, label_) in enumerate(tqdm(dataset), 0):
            input_ = Variable(input_).cuda().type(torch.cuda.FloatTensor)
            prediction = net(input_.cuda()).detach().cpu().numpy()
            label_ = label_.detach().cpu().numpy()
            AARD.append(abs((prediction - label_) / label_))
        AARD = np.average(np.array(AARD))
        return AARD

    viscosity_aard = property_accuracy('viscosity')
    thermal_conductivity_aard = property_accuracy('thermal_conductivity')
    heat_capacity_aard = property_accuracy('heat_capacity')
    return viscosity_aard, thermal_conductivity_aard, heat_capacity_aard
