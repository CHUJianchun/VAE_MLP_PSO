import pickle
from DataPreparation.IonicLiquid import *
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
import sys
from DataPreparation.Datapoint import *


class VaeMlpDataset(Dataset):
    def __init__(self, anion, cation, temperature, label):
        super(VaeMlpDataset, self).__init__()
        self.anion = anion
        self.cation = cation
        self.temperature = temperature
        self.label = label

    def __getitem__(self, idx):
        x = self.anion[idx]
        y = self.cation[idx]
        t = self.temperature[idx]
        label = self.label[idx]
        return x, y, t, label

    def __len__(self):
        return len(self.anion)


def create_vae_mlp_dataset(params):
    try:
        with open('Data/ionic_liquid_list.data', 'rb') as f_ionic_liquid_list:
            IonicLiquid.list_of_all_Ionic_Liquids = pickle.load(f_ionic_liquid_list)
    except IOError:
        print('Warning: File Data/ionic_liquid_list.data not found, creating')
        create_ionic_liquid()
        with open('Data/ionic_liquid_list.data', 'rb') as f_ionic_liquid_list:
            IonicLiquid.list_of_all_Ionic_Liquids = pickle.load(f_ionic_liquid_list)
    print('Start: MLP dataset and train loader is creating')

    try:
        with open('Data/data_points_list.data', 'rb') as f_data_points_list:
            data_points_list = pickle.load(f_data_points_list)
    except IOError:
        print('Warning: File Data/data_points_list.data not found, creating')
        create_data_point()
        with open('Data/data_points_list.data', 'rb') as f_data_points_list:
            data_points_list = pickle.load(f_data_points_list)
    print('----Start: Normalize the temperature and pressure')

    def normalization(data_points_list_):
        temperature_max = 0.
        temperature_min = 0.
        pressure_max = 0.
        pressure_min = 0.
        print('--------Process 1 : Finding maximum and minimum')
        for data_point_ in tqdm(data_points_list_):
            if float(data_point_.temperature) > temperature_max:
                temperature_max = float(data_point_.temperature)
            if float(data_point_.temperature) < temperature_min:
                temperature_min = float(data_point_.temperature)
            if float(data_point_.pressure) > pressure_max:
                pressure_max = float(data_point_.pressure)
            if float(data_point_.pressure) < pressure_min:
                pressure_min = float(data_point_.pressure)
        print('--------Process 2 : Normalizing')
        for data_point_ in tqdm(data_points_list_):
            data_point_.temperature = float(data_point_.temperature) / (temperature_max - temperature_min)
            data_point_.pressure = float(data_point_.pressure) / (pressure_max - pressure_min)
        return data_points_list_

    data_points_list = normalization(data_points_list)
    print('----Finish: Normalize the temperature and pressure')
    viscosity_data_point_list = []
    thermal_conductivity_data_point_list = []
    heat_capacity_data_point_list = []

    for data_point in tqdm(data_points_list):
        if data_point.property == 'viscosity':
            viscosity_data_point_list.append(data_point)
        elif data_point.property == 'thermal_conductivity':
            thermal_conductivity_data_point_list.append(data_point)
        elif data_point.property == 'heat_capacity':
            heat_capacity_data_point_list.append(data_point)
        else:
            print('Error: At least one data point in data_points_list.data has incorrect property attribute')
            sys.exit()

    def compose_t_p_encode(data_point_list):
        dataset_anion = []
        dataset_cation = []
        dataset_temperature = []
        dataset_label = []
        for data_point_ in data_point_list:
            dataset_anion.append(data_point.ionic_liquid.anion_one_hot)
            dataset_cation.append(data_point_.ionic_liquid.cation_one_hot)
            dataset_temperature.append(data_point_.temperature)
            dataset_label.append(float(data_point_.value))
        dataset_anion = Variable(torch.from_numpy(np.array(dataset_anion)))
        dataset_cation = Variable(torch.from_numpy(np.array(dataset_cation)))
        dataset_temperature = Variable(torch.from_numpy(np.array(dataset_temperature)))
        dataset_label = Variable(torch.from_numpy(np.array(dataset_label)))
        return dataset_anion, dataset_cation, dataset_temperature, dataset_label

    v_dataset_anion, v_dataset_cation, v_dataset_temperature, v_dataset_label = compose_t_p_encode(
        viscosity_data_point_list)
    t_dataset_anion, t_dataset_cation, t_dataset_temperature, t_dataset_label = compose_t_p_encode(
        thermal_conductivity_data_point_list)
    h_dataset_anion, h_dataset_cation, h_dataset_temperature, h_dataset_label = compose_t_p_encode(
        heat_capacity_data_point_list)

    viscosity_vaemlp_dataset = VaeMlpDataset(
        v_dataset_anion, v_dataset_cation, v_dataset_temperature, v_dataset_label)
    thermal_conductivity_vaemlp_dataset = VaeMlpDataset(
        t_dataset_anion, t_dataset_cation, t_dataset_temperature, t_dataset_label)
    heat_capacity_vaemlp_dataset = VaeMlpDataset(
        h_dataset_anion, h_dataset_cation, h_dataset_temperature, h_dataset_label)

    viscosity_vaemlp_train_loader = DataLoader(viscosity_vaemlp_dataset,
                                               batch_size=params['MLP_batch_size'],
                                               drop_last=True, shuffle=True)
    thermal_conductivity_vaemlp_train_loader = DataLoader(thermal_conductivity_vaemlp_dataset,
                                                          batch_size=params['MLP_batch_size'],
                                                          drop_last=True, shuffle=True)
    heat_capacity_vaemlp_train_loader = DataLoader(heat_capacity_vaemlp_dataset,
                                                   batch_size=params['MLP_batch_size'],
                                                   drop_last=True, shuffle=True)

    torch.save(viscosity_vaemlp_dataset,
               'Data/viscosity_vae_mlp_dataset.pkl')
    torch.save(viscosity_vaemlp_train_loader,
               'Data/viscosity_vae_mlp_train_loader.pkl')
    torch.save(thermal_conductivity_vaemlp_dataset,
               'Data/thermal_conductivity_vae_mlp_dataset.pkl')
    torch.save(thermal_conductivity_vaemlp_train_loader,
               'Data/thermal_conductivity_vae_mlp_train_loader.pkl')
    torch.save(heat_capacity_vaemlp_dataset,
               'Data/heat_capacity_vae_mlp_dataset.pkl')
    torch.save(heat_capacity_vaemlp_train_loader,
               'Data/heat_capacity_vae_mlp_train_loader.pkl')