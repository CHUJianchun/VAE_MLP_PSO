import numpy as np
import os.path
import shutil
import os


# uniqueness = 531 / 661 = 0.803
def novelty():
    with open('Data/component_smiles.txt', 'r') as f:
        dataset_smiles = f.readlines()
    dataset_smiles_ = []
    for item in dataset_smiles:
        if item != '\n':
            dataset_smiles_.append(item.replace('\n', ''))
    dataset_smiles = dataset_smiles_
    filename_list = os.listdir('Graph')
    t_list = np.zeros(len(filename_list))
    anion_list_303 = []
    cation_list_303 = []
    anion_list_363 = []
    cation_list_363 = []
    for i in range(len(filename_list)):
        if float(filename_list[i].replace('.png', '').split('_')[0]) < 0.5:
            t_list[i] = 303
        else:
            t_list[i] = 363
        filename_list[i] = filename_list[i].replace('.png', '').split('_')[1]
    filename_list_ = []
    filename_list = list(set(filename_list))
    for i in filename_list:
        if i not in filename_list_:
            filename_list_.append(i)
    filename_list = filename_list_
    for i in range(len(filename_list)):
        (s1, s2) = filename_list[i].split('.')
        if t_list[i] == 303:
            if '-' in s1:
                anion_list_303.append(s1)
                cation_list_303.append(s2)
            else:
                anion_list_303.append(s2)
                cation_list_303.append(s1)
        else:
            if '-' in s1:
                anion_list_363.append(s1)
                cation_list_363.append(s2)
            else:
                anion_list_363.append(s2)
                cation_list_363.append(s1)

    il_sum = len(anion_list_363) + len(anion_list_303)

    il_in_dataset = 0

    for i in range(len(anion_list_303)):
        if anion_list_303[i] + '.' + cation_list_303[i] in dataset_smiles:
            il_in_dataset += 1

    for i in range(len(anion_list_363)):
        if anion_list_363[i] + '.' + cation_list_363[i] in dataset_smiles:
            il_in_dataset += 1

    novel = 1 - il_in_dataset / il_sum