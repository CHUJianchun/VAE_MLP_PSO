import pickle
from DataPreparation.IonicLiquid import *
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
from DataPreparation.Datapoint import *
import sys
from VariationalAutoEncoder.VAE_cat_model import VAE_CAT


class MlpCatDataset(Dataset):
    def __init__(self, input_, label_):
        super(MlpCatDataset, self).__init__()
        self.input = input_
        self.label = label_

    def __getitem__(self, idx):
        return self.input[idx], self.label[idx]

    def __len__(self):
        return len(self.label)


def create_mlp_cat_dataset(params, vae_file_name):
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
    vae = torch.load('Model_pkl/' + vae_file_name).cuda()

    viscosity_data_point_list = []
    thermal_conductivity_data_point_list = []
    heat_capacity_data_point_list = []

    for data_point in tqdm(data_points_list):
        anion_variable = Variable(
            torch.from_numpy(data_point.ionic_liquid.anion_one_hot)).cuda()
        cation_variable = Variable(
            torch.from_numpy(data_point.ionic_liquid.cation_one_hot)).cuda()
        one_hot_input = torch.cat((anion_variable, cation_variable), dim=0)
        one_hot_input = one_hot_input.reshape(1, 1, one_hot_input.shape[0], one_hot_input.shape[1])\
            .type(torch.cuda.FloatTensor)
        data_point.encode = vae.get_mid(one_hot_input).detach().cpu()

        data_point.temperature = Variable(torch.from_numpy(np.array(float(data_point.temperature))))
        data_point.pressure = Variable(torch.from_numpy(np.array(float(data_point.pressure))))
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
        dataset_input = []
        dataset_output = []
        for data_point_ in data_point_list:
            input_data = torch.cat([data_point_.temperature.reshape(-1),
                                    torch.squeeze(data_point_.encode)], dim=0).numpy()
            output_data = Variable(torch.from_numpy(np.array(float(data_point_.value)))).numpy()
            dataset_input.append(input_data)
            dataset_output.append(output_data)
        dataset_input = Variable(torch.from_numpy(np.array(dataset_input)))
        dataset_output = Variable(torch.from_numpy(np.array(dataset_output)))
        return dataset_input, dataset_output

    print('----Process: Getting encode for data')
    viscosity_dataset_input, viscosity_dataset_output = compose_t_p_encode(viscosity_data_point_list)
    thermal_conductivity_input, thermal_conductivity_output = compose_t_p_encode(thermal_conductivity_data_point_list)
    heat_capacity_input, heat_capacity_output = compose_t_p_encode(heat_capacity_data_point_list)

    viscosity_mlp_dataset = MlpCatDataset(viscosity_dataset_input, viscosity_dataset_output)
    thermal_conductivity_mlp_dataset = MlpCatDataset(thermal_conductivity_input, thermal_conductivity_output)
    heat_capacity_mlp_dataset = MlpCatDataset(heat_capacity_input, heat_capacity_output)

    viscosity_mlp_train_loader = DataLoader(viscosity_mlp_dataset,
                                            batch_size=params['MLP_batch_size'],
                                            drop_last=True, shuffle=True)
    thermal_conductivity_mlp_train_loader = DataLoader(thermal_conductivity_mlp_dataset,
                                                       batch_size=params['MLP_batch_size'],
                                                       drop_last=True, shuffle=True)
    heat_capacity_mlp_train_loader = DataLoader(heat_capacity_mlp_dataset,
                                                batch_size=params['MLP_batch_size'],
                                                drop_last=True, shuffle=True)
    vae_name = vae_file_name.replace('.pkl', '')
    torch.save(viscosity_mlp_dataset,
               'Data/viscosity_mlp_cat_dataset_' + vae_name + '.pkl')
    torch.save(viscosity_mlp_train_loader,
               'Data/viscosity_mlp_cat_train_loader_' + vae_name + '.pkl')
    torch.save(thermal_conductivity_mlp_dataset,
               'Data/thermal_conductivity_mlp_cat_dataset_' + vae_name + '.pkl')
    torch.save(thermal_conductivity_mlp_train_loader,
               'Data/thermal_conductivity_mlp_cat_train_loader_' + vae_name + '.pkl')
    torch.save(heat_capacity_mlp_dataset,
               'Data/heat_capacity_mlp_cat_dataset_' + vae_name + '.pkl')
    torch.save(heat_capacity_mlp_train_loader,
               'Data/heat_capacity_mlp_cat_train_loader_' + vae_name + '.pkl')
    print('Finish: mlp_dataset and mlp_train_loader are saved to Data/***.pkl')
