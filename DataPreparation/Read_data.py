import json
import sys
import pickle
import os
from pathlib import Path
from tqdm import tqdm


def load_data_origin():
    print('Start: Saving origin data from Data/data.txt')
    try:
        with open('Data/data.txt') as f:
            data_list = json.loads(f.read())
    except IOError:
        print('Error: File Data/data.txt not found')
        sys.exit()
    else:
        with open('Data/origin_data_list.data', 'wb') as f:
            pickle.dump(data_list, f)
            print('Finish: Saving origin data from Data/data.txt')


def load_data_classified():
    def read():
        with open('Data/origin_data_list.data', 'rb') as f:
            data_list_ = pickle.load(f)
        return data_list_

    print('Start: Loading origin data from Data/origin_data_list.data')
    try:
        data_list = read()
    except IOError:
        print('Warning: File origin_data_list.data not found, reinitializing')
        load_data_origin()
        data_list = read()
    print('Finish: Loading origin data from Data/origin_data_list.data')
    print('Start: Classifying origin data')

    data_viscosity = []
    data_conductivity = []
    data_capacity = []

    str_viscosity = 'Transport properties: Viscosity'
    str_conductivity = 'Transport properties: Thermal conductivity'
    str_capacity = 'Heat capacity and derived properties: Heat capacity at constant pressure'

    for data in data_list:
        if len(data[1]['components']) == 1 and data[1]['solvent'] is None:
            if data[1]['title'] == str_viscosity:
                data_viscosity.append(data[1])
            elif data[1]['title'] == str_conductivity:
                data_conductivity.append(data[1])
            elif data[1]['title'] == str_capacity:
                data_capacity.append(data[1])

    with open('Data/data_viscosity.data', 'wb') as f_viscosity:
        with open('Data/data_conductivity.data', 'wb') as f_conductivity:
            with open('Data/data_capacity.data', 'wb') as f_capacity:
                pickle.dump(data_viscosity, f_viscosity)
                pickle.dump(data_conductivity, f_conductivity)
                pickle.dump(data_capacity, f_capacity)
    print('Finish: Classifying origin data and save to Data/data_viscosity.data, data_conductivity.data, '
          'Data/data_capacity.data')


def read_3_properties_data_files():
    with open('Data/data_viscosity.data', 'rb') as f_viscosity:
        with open('Data/data_conductivity.data', 'rb') as f_conductivity:
            with open('Data/data_capacity.data', 'rb') as f_capacity:
                data_viscosity_ = pickle.load(f_viscosity)
                data_conductivity_ = pickle.load(f_conductivity)
                data_capacity_ = pickle.load(f_capacity)
    return data_viscosity_, data_conductivity_, data_capacity_


def smiles_parser():
    def create_list_for_exist_components(list_):
        component_list_ = []
        for dictionary in list_:
            if len(dictionary['components']) == 1:
                name = dictionary['components'][0]['name']
                component_list_.append(name)
        component_list_ = list(set(component_list_))
        return component_list_

    print('Start: Loading Classified data from Data/data_viscosity.data, data_conductivity.data, '
          'Data/data_capacity.data')
    try:
        data_viscosity, data_conductivity, data_capacity = read_3_properties_data_files()
    except IOError:
        print('Warning: File data specially for one property not found, reinitializing')
        load_data_classified()
        data_viscosity, data_conductivity, data_capacity = read_3_properties_data_files()
    print('Finish: Loading Classified data from Data/data_viscosity.data, data_conductivity.data, '
          'Data/data_capacity.data')

    print('Start: Creating Component list')
    viscosity_component_list = create_list_for_exist_components(data_viscosity)
    conductivity_component_list = create_list_for_exist_components(data_conductivity)
    capacity_component_list = create_list_for_exist_components(data_capacity)
    component_list = viscosity_component_list + conductivity_component_list + capacity_component_list
    with open('Data/component_list.txt', 'w') as txt_component_list:
        for item in component_list:
            txt_component_list.write(item + '\n')
    print('Finish: Saving Component list to Data/component_list.data')


def get_smiles_by_opsin():
    if Path('Data/component_list.txt').is_file():
        print('Start: Getting SMILES for components in component_list.txt')
        os.system('java -jar DataPreparation/opsin.jar -osmi Data/component_list.txt Data/component_smiles.txt')
        print('Finish: Save SMILES to component_smiles.txt')
    else:
        print('Warning: File component_list.txt not found, generating')
        smiles_parser()


def create_component_smiles_list():
    if not Path('Data/component_list.txt').is_file():
        print('Warning: component_list.txt not found, generating')
        smiles_parser()
    if not Path('Data/component_smiles.txt').is_file():
        print('Warning: component_smiles.txt not found, generating')

    with open('Data/component_list.txt', 'r') as txt_component_list:
        component_list = txt_component_list.readlines()

    with open('Data/component_smiles.txt', 'r') as txt_component_smiles:
        component_smiles = txt_component_smiles.readlines()

    print('Start: Creating component anions and cations smiles list')
    for i in tqdm(range(len(component_smiles))):
        component_smiles[i] = component_smiles[i].replace('Br', 'A')
        component_smiles[i] = component_smiles[i].replace('Na', 'D')
        component_smiles[i] = component_smiles[i].replace('Cl', 'E')
        component_smiles[i] = component_smiles[i].replace('Al', 'G')
        component_smiles[i] = component_smiles[i].replace('NH3', 'J')
        component_smiles[i] = component_smiles[i].replace('NH2', 'K')
        component_smiles[i] = component_smiles[i].replace('NH', 'L')

    component_smiles_list_ = []
    anion_smiles_list_ = []
    cation_smiles_list_ = []
    error_statistic = 0
    for i in tqdm(range(len(component_list))):
        smiles = component_smiles[i]
        if smiles == '\n' or '-' not in smiles or '.' not in smiles or '+' not in smiles or smiles.count('.') > 1:
            error_statistic += 1
            continue
        else:
            split_smiles = smiles.replace('\n', '').split('.', 1)
            if '-' in split_smiles[0] and '+' in split_smiles[1]:
                anion_smiles_list_.append(split_smiles[0])
                cation_smiles_list_.append(split_smiles[1])
            elif '+' in split_smiles[0] and '-' in split_smiles[1]:
                anion_smiles_list_.append(split_smiles[1])
                cation_smiles_list_.append(split_smiles[0])
            component_smiles_list_.append([component_list[i].replace('\n', ''), component_smiles[i].replace('\n', ''),
                                           anion_smiles_list_[-1], cation_smiles_list_[-1]])
    component_smiles_list_copy = []
    for item in component_smiles_list_:
        if item not in component_smiles_list_copy:
            component_smiles_list_copy.append(item)
    component_smiles_list_ = component_smiles_list_copy
    start_str = '0'
    end_str = '9'
    for item in component_smiles_list_:
        item[1] = start_str + item[1] + end_str
        item[2] = start_str + item[2] + end_str
        item[3] = start_str + item[3] + end_str
    with open('Data/component_smiles_list.data', 'wb') as f_component_smiles_list:
        pickle.dump(component_smiles_list_, f_component_smiles_list)
    print(
        f"Finish: Creating component anions and cations smiles list and saved to component_smiles_list.data,\n \
         Totally {len(component_smiles_list_):d} ILs created successfully and {error_statistic:d} failed")


def create_one_hot_dict():
    if not Path('Data/component_smiles_list.data').is_file():
        print('Warning: component_smiles_list.data not found, generating')
        create_component_smiles_list()
    print('Start: creating character dictionary')
    list_string = ''
    with open('Data/component_smiles_list.data', 'rb') as f_component_smiles_list:
        component_smiles_list = pickle.load(f_component_smiles_list)
        for item in component_smiles_list:
            list_string += item[1]
            list_string = ''.join(set(list_string))
    list_string = list(list_string)
    list_string.remove('.')
    dict_character = {}
    key_i = 0
    for character in list_string:
        dict_character[character] = key_i
        key_i += 1
    with open('Data/one_hot_dictionary.data', 'wb') as f_one_hot_dictionary:
        pickle.dump(dict_character, f_one_hot_dictionary)
    print('The length of dictionary of characters is : ' + len(dict_character))
    print('Finish: Saving character dictionary to one_hot_dictionary.data.')


if __name__ == '__main__':
    pass
