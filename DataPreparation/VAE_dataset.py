# 注意：读取的是所有离子液体的列表，不是数据集的列表
import pickle
from DataPreparation.IonicLiquid import *
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
from hyperparameters import load_params


class VaeDataset(Dataset):

    def __init__(self, anion, cation):
        super(VaeDataset, self).__init__()
        self.dataset_input_anion = anion
        self.dataset_input_cation = cation
        self.dataset_label_anion = anion
        self.dataset_label_cation = cation

    def __getitem__(self, idx):
        inputs_x = self.dataset_input_anion[idx]
        inputs_y = self.dataset_input_cation[idx]
        label_x = self.dataset_label_anion[idx]
        label_y = self.dataset_label_cation[idx]
        return inputs_x, inputs_y, label_x, label_y

    def __len__(self):
        return len(self.dataset_label_anion)


def create_vae_dataset(params):
    try:
        with open('Data/ionic_liquid_list.data', 'rb') as f_ionic_liquid_list:
            IonicLiquid.list_of_all_Ionic_Liquids = pickle.load(f_ionic_liquid_list)
    except IOError:
        print('Warning: File Data/ionic_liquid_list.data not found, creating')
        create_ionic_liquid()
        IonicLiquid.list_of_all_Ionic_Liquids = pickle.load(f_ionic_liquid_list)
    print('Start: VAE dataset and train loader is creating')

    anion_one_hot_matrix = []
    cation_one_hot_matrix = []
    for ionic_liquid in IonicLiquid.list_of_all_Ionic_Liquids:
        anion_one_hot_matrix.append(ionic_liquid.anion_one_hot)
        cation_one_hot_matrix.append(ionic_liquid.cation_one_hot)
    anion_one_hot_matrix = np.unique(np.array(anion_one_hot_matrix), axis=0)
    cation_one_hot_matrix = np.unique(np.array(cation_one_hot_matrix), axis=0)

    anion_part_dataset = []
    cation_part_dataset = []

    for i in range(anion_one_hot_matrix.shape[0]):
        for j in range(cation_one_hot_matrix.shape[0]):
            anion_part_dataset.append(anion_one_hot_matrix[i])
            cation_part_dataset.append(cation_one_hot_matrix[j])

    anion_part_dataset = Variable(torch.from_numpy(np.array(anion_part_dataset)))
    cation_part_dataset = Variable(torch.from_numpy(np.array(cation_part_dataset)))
    vae_dataset = VaeDataset(anion_part_dataset, cation_part_dataset)
    vae_train_loader = DataLoader(vae_dataset, batch_size=params['VAE_batch_size'], drop_last=True, shuffle=True)
    torch.save(vae_dataset, 'Data/vae_dataset.pkl')
    torch.save(vae_train_loader, 'Data/vae_train_loader.pkl')
    print('Finish: vae_dataset and vae_train_loader are saved to Data/***.pkl')
