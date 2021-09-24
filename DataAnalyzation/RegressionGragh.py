import numpy as np
import matplotlib.pyplot as plt
from VariationalAutoEncoder.VAE_model import VAE
from MultiLayerPerceptron.MLP_model import MLP
import torch
from torch.autograd import Variable
import sys
from tqdm import tqdm
import importlib
from hyperparameters import load_params
import os


def plot_regression_graph(mlp_cat_file_name, property):
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

    if property == 'viscosity':
        dataset = v_dataset
        net = v_mlp
    elif property == 'heat_capacity':
        dataset = h_dataset
        net = h_mlp
    else:
        dataset = t_dataset
        net = t_mlp
    prediction_list = []
    label_list = []
    input_list = []
    for i, (input_, label_) in enumerate(tqdm(dataset), 0):
        input_ = Variable(input_).cuda().type(torch.cuda.FloatTensor)
        input_list.append(input_.detach().cpu().numpy())
        prediction_list.append(net(input_.cuda()).detach().cpu().numpy())
        label_list.append(label_.detach().cpu().numpy())
    prediction = np.squeeze(np.array(prediction_list))
    label = np.array(label_list)
    input_list = np.array(input_list)
    plt.scatter(label, prediction)
    # plt.title(str(latent_code_num) + property)
    plt.xlabel('X')
    plt.ylabel('Y')
    if property == 'viscosity':
        xmax = 50
        ymax = 50
        xmin = 0
        ymin = 0
    elif property == 'heat_capacity':
        xmax = 2000
        ymax = 2000
        xmin = 0
        ymin = 0
    else:
        xmax = 0.25
        ymax = 0.25
        xmin = 0.1
        ymin = 0.1

    plt.xlim(xmax=xmax, xmin=xmin)
    plt.ylim(ymax=ymax, ymin=ymin)
    plt.show()
    return input_list, prediction, label


def error_vae_mlp(t_model_name, h_model_name):
    t_dataset = torch.load('Data/thermal_conductivity_vae_mlp_dataset.pkl')
    h_dataset = torch.load('Data/heat_capacity_vae_mlp_dataset.pkl')

    t_vae_mlp = torch.load('Model_pkl/' + t_model_name).cuda()
    h_vae_mlp = torch.load('Model_pkl/' + h_model_name).cuda()

    t_pre_list = []
    t_label_list = []
    h_pre_list = []
    h_label_list = []

    for i, (anion, cation, temperature, label) in enumerate(tqdm(t_dataset), 0):
        one_hot = torch.cat((anion, cation), dim=1)
        one_hot = one_hot.reshape(1, 1, one_hot.shape[0], one_hot.shape[1])
        one_hot = Variable(one_hot).cuda().type(torch.cuda.FloatTensor)
        t = temperature.cuda().reshape(1, 1).type(torch.cuda.FloatTensor)
        t_pre_list.append(t_vae_mlp.predict(one_hot, t).detach().cpu().numpy())
        t_label_list.append(label.numpy())

    for i, (anion, cation, temperature, label) in enumerate(tqdm(h_dataset), 0):
        one_hot = torch.cat((anion, cation), dim=1)
        one_hot = one_hot.reshape(1, 1, one_hot.shape[0], one_hot.shape[1])
        one_hot = Variable(one_hot).cuda().type(torch.cuda.FloatTensor)
        t = temperature.cuda().reshape(1, 1).type(torch.cuda.FloatTensor)
        h_pre_list.append(h_vae_mlp.predict(one_hot, t).detach().cpu().numpy())
        h_label_list.append(label.numpy())

    t_pre_list = np.squeeze(np.array(t_pre_list))
    h_pre_list = np.squeeze(np.array(h_pre_list))
    t_label_list = np.squeeze(np.array(t_label_list))
    h_label_list = np.squeeze(np.array(h_label_list))

    t_aard = np.average(np.abs(t_pre_list - t_label_list) / t_label_list) * 100
    h_aard = np.average(np.abs(h_pre_list - h_label_list) / h_label_list) * 100

    return t_aard, h_aard


def error_one_hot_mlp(t_model_name, h_model_name):
    t_dataset = torch.load('Data/thermal_conductivity_vae_mlp_dataset.pkl')
    h_dataset = torch.load('Data/heat_capacity_vae_mlp_dataset.pkl')

    t_one_hot_mlp = torch.load('Model_pkl/' + t_model_name).cuda()
    h_one_hot_mlp = torch.load('Model_pkl/' + h_model_name).cuda()

    t_pre_list = []
    t_label_list = []
    h_pre_list = []
    h_label_list = []

    for i, (anion, cation, temperature, label) in enumerate(tqdm(t_dataset), 0):
        one_hot = torch.cat((anion, cation), dim=1)
        one_hot = one_hot.reshape(1, 1, one_hot.shape[0], one_hot.shape[1])
        one_hot = Variable(one_hot).cuda().type(torch.cuda.FloatTensor)
        t = temperature.cuda().reshape(1, 1).type(torch.cuda.FloatTensor)
        t_pre_list.append(t_one_hot_mlp(one_hot, t).detach().cpu().numpy())
        t_label_list.append(label.numpyone_hot)

    for i, (anion, cation, temperature, label) in enumerate(tqdm(h_dataset), 0):
        one_hot = torch.cat((anion, cation), dim=1)
        one_hot = one_hot.reshape(1, 1, one_hot.shape[0], one_hot.shape[1])
        one_hot = Variable(one_hot).cuda().type(torch.cuda.FloatTensor)
        t = temperature.cuda().reshape(1, 1).type(torch.cuda.FloatTensor)
        h_pre_list.append(h_one_hot_mlp(one_hot, t).detach().cpu().numpy())
        h_label_list.append(label.numpy())

    t_pre_list = np.squeeze(np.array(t_pre_list))
    h_pre_list = np.squeeze(np.array(h_pre_list))
    t_label_list = np.squeeze(np.array(t_label_list))
    h_label_list = np.squeeze(np.array(h_label_list))

    t_aard = np.average(np.abs(t_pre_list - t_label_list) / t_label_list) * 100
    h_aard = np.average(np.abs(h_pre_list - h_label_list) / h_label_list) * 100

    return t_aard, h_aard
