import DataPreparation.Read_data as Read_data
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
import importlib


class IonicLiquid:
    list_of_all_Ionic_Liquids = []

    def __init__(self, name_, smiles_, anion_, cation_):
        self.name = name_
        self.smiles = smiles_
        self.anion_smiles = anion_
        self.cation_smiles = cation_
        IonicLiquid.list_of_all_Ionic_Liquids.append(self)


def create_ionic_liquid():
    importlib.reload(Read_data)

    if not Path('Data/one_hot_dictionary.data').is_file():
        print('Warning: Data/one_hot_dictionary.data not found, creating')
        Read_data.create_one_hot_dict()
    if not Path('Data/component_smiles_list.data').is_file():
        print('Warning: Data/component_smiles_list.data not found, creating')
        Read_data.create_component_smiles_list()

    print('Start: Creating ionic liquid list')
    with open('Data/component_smiles_list.data', 'rb') as f_component_smiles_list:
        component_smiles_list = pickle.load(f_component_smiles_list)
    with open('Data/one_hot_dictionary.data', 'rb') as f_one_hot_dictionary:
        one_hot_dictionary = pickle.load(f_one_hot_dictionary)

    dictionary_length = len(one_hot_dictionary)

    anion_string_length_list = []
    cation_string_length_list = []

    for item in component_smiles_list:
        anion_string_length_list.append(len(list(item[2])))
        cation_string_length_list.append(len(list(item[3])))
    anion_length_max = max(anion_string_length_list)
    cation_length_max = max(cation_string_length_list)
    one_hot_length = max(anion_length_max, cation_length_max)
    no_ = 0
    for item in component_smiles_list:
        name_item = 'IL_' + str(no_)
        exec(name_item + ' = IonicLiquid(name_ = item[0], smiles_ = item[1], anion_ = item[2], cation_ = item[3])')
        no_ += 1

    def get_one_hot(i_l):
        anion_string_list = list(i_l.anion_smiles)
        cation_string_list = list(i_l.cation_smiles)
        anion_one_hot_length = len(anion_string_list)
        cation_one_hot_length = len(cation_string_list)

        i_l.anion_one_hot = np.zeros((one_hot_length, dictionary_length + 1))
        i_l.cation_one_hot = np.zeros((one_hot_length, dictionary_length + 1))

        for i in range(anion_one_hot_length):
            location = one_hot_dictionary[anion_string_list[i]]
            i_l.anion_one_hot[i][location] = 1

        for j in range(cation_one_hot_length):
            location = one_hot_dictionary[cation_string_list[j]]
            i_l.cation_one_hot[j][location] = 1
        return i_l

    for k in tqdm(range(len(IonicLiquid.list_of_all_Ionic_Liquids))):
        IonicLiquid.list_of_all_Ionic_Liquids[k] = get_one_hot(IonicLiquid.list_of_all_Ionic_Liquids[k])
    with open('Data/ionic_liquid_list.data', 'wb') as f_ionic_liquid_list:
        pickle.dump(IonicLiquid.list_of_all_Ionic_Liquids, f_ionic_liquid_list)
    print('Finish: Creating the ionic liquid list and Saved to Data/ionic_liquid_list.data')
    return IonicLiquid.list_of_all_Ionic_Liquids
