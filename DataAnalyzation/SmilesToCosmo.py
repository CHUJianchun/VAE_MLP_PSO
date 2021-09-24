import os
from rdkit.Chem import MolFromSmiles
from tqdm import trange
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Conformer
from rdkit.Chem import rdMolDescriptors
import numpy as np
import os.path
import shutil
from rdkit.Chem.Descriptors import ExactMolWt

'''
def smiles_to_xyz():
    def get_fragments(mol, name):
        fragment_names = []
        fragments = Chem.GetMolFrags(mol, asMols=True)
        labels = ["A", "B", "C"]
        for label, fragment in zip(labels, fragments):
            fragment_names.append(name + label)
        return fragments, fragment_names

    def generate_conformations(fragments, max_confs=20):
        for fragment in fragments:
            rot_bond = rdMolDescriptors.CalcNumRotatableBonds(fragment)
            confs = min(3 + 3 * rot_bond, max_confs)
            AllChem.EmbedMultipleConfs(fragment, numConfs=confs)

        return fragments

    def write_xtb_input_file(fragment, fragment_name):
        number_of_atoms = fragment.GetNumAtoms()
        charge = Chem.GetFormalCharge(fragment)
        symbols = [a.GetSymbol() for a in fragment.GetAtoms()]
        for i_, conf in enumerate(fragment.GetConformers()):
            file_name_ = "xyz_file/" + fragment_name + "+" + str(i_) + ".xyz"
            with open(file_name_, "w") as file:
                file.write(str(number_of_atoms) + "\n")
                file.write("title\n")
                for atom, symbol in enumerate(symbols):
                    p = conf.GetAtomPosition(atom)
                    line = " ".join((symbol, str(p.x), str(p.y), str(p.z), "\n"))
                    file.write(line)
                if charge != 0:
                    file.write("$set\n")
                    file.write("chrg " + str(charge) + "\n")
                    file.write("$end")

    def write_input_files(mol, name):
        mol = Chem.AddHs(mol)
        fragments, fragment_names = get_fragments(mol, name)
        fragments = generate_conformations(fragments)
        for fragment, fragment_name in zip(fragments, fragment_names):
            write_xtb_input_file(fragment, fragment_name)

    filename_list = os.listdir('Graph')
    anion_list_303 = []
    cation_list_303 = []
    anion_list_363 = []
    cation_list_363 = []
    for i in range(len(filename_list)):
        file_name = filename_list[i].replace('.png', '')
        (t, s) = file_name.split('_')
        (s1, s2) = s.split('.')
        if float(t) < 0.5:
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

    for i in trange(len(anion_list_303)):
        anion = MolFromSmiles(anion_list_303[i])
        cation = MolFromSmiles(cation_list_303[i])
        write_input_files(anion, str(i))
        write_input_files(cation, str(i))

    for i in trange(len(anion_list_363)):
        anion = MolFromSmiles(anion_list_363[i])
        cation = MolFromSmiles(cation_list_363[i])
        write_input_files(anion, str(i))
        write_input_files(cation, str(i))
'''


def smiles_to_xyz():

    def write_input_files(mol, name, i_):
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=1)
        number_of_atoms = mol.GetNumAtoms()
        charge = Chem.GetFormalCharge(mol)
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        for i__, conf in enumerate(mol.GetConformers()):
            file_name_ = "Cosmo_file/xyz_file/" + name + "_" + str(i_) + ".xyz"
            with open(file_name_, "w") as file:
                file.write(str(number_of_atoms) + "\n")
                file.write("title\n")
                for atom, symbol in enumerate(symbols):
                    p = conf.GetAtomPosition(atom)
                    line = " ".join((symbol, str(p.x), str(p.y), str(p.z), "\n"))
                    file.write(line)
                if charge != 0:
                    file.write("$set\n")
                    file.write("chrg " + str(charge) + "\n")
                    file.write("$end")

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

    for i in trange(len(anion_list_303)):
        anion = MolFromSmiles(anion_list_303[i])
        cation = MolFromSmiles(cation_list_303[i])
        write_input_files(anion, 'anion_303', str(i))
        write_input_files(cation, 'cation_303', str(i))

    for i in trange(len(anion_list_363)):
        anion = MolFromSmiles(anion_list_363[i])
        cation = MolFromSmiles(cation_list_363[i])
        write_input_files(anion, 'anion_363', str(i))
        write_input_files(cation, 'cation_363', str(i))


def collect_cosmo():
    for _, sub_dir, __ in os.walk('Cosmo_file/a_xyz2cosmo'):
        for dir_name in sub_dir:
            try:
                shutil.copy('Cosmo_file/a_xyz2cosmo/' + dir_name + '/' + dir_name + '.cosmo', 'Cosmo_file/anion_cosmo')
            except FileNotFoundError:
                print(dir_name)

    for _, sub_dir, __ in os.walk('Cosmo_file/c_xyz2cosmo'):
        for dir_name in sub_dir:
            try:
                shutil.copy('Cosmo_file/c_xyz2cosmo/' + dir_name + '/' + dir_name + '.cosmo', 'Cosmo_file/cation_cosmo')
            except FileNotFoundError:
                print(dir_name)


def create_input_file(temperature, min_, max_):
    input_string = ''
    no = 0
    for i in range(min_, max_ + 1):
        input_string += '<DbuCompoundCombination anion="anion_' + str(temperature) + '_' + str(i) +\
                            '.cosmo" cation="cation_' + str(temperature) + '_' + str(i) +\
                            '.cosmo">' + str(no) + '</DbuCompoundCombination>\n'
        no += 1
    return input_string


def mw_sort():
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

    # filename_list = list(set(filename_list))
    filename_list_ = []
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
    mw_303 = np.zeros((len(anion_list_303), 2))
    mw_363 = np.zeros((len(anion_list_363), 2))
    for i in range(len(anion_list_303)):
        anion = Chem.AddHs(MolFromSmiles(anion_list_303[i]))
        cation = Chem.AddHs(MolFromSmiles(cation_list_303[i]))
        molecular_weight = ExactMolWt(anion) + ExactMolWt(cation)
        mw_303[i, 0] = i
        mw_303[i, 1] = molecular_weight

    for i in range(len(anion_list_363)):
        anion = Chem.AddHs(MolFromSmiles(anion_list_303[i]))
        cation = Chem.AddHs(MolFromSmiles(cation_list_303[i]))
        molecular_weight = ExactMolWt(anion) + ExactMolWt(cation)
        mw_363[i, 0] = i
        mw_363[i, 1] = molecular_weight
        pd_data_ = pd.DataFrame(mw_303)
        pd_data_.to_csv('mw_303.csv')
        pd_data_ = pd.DataFrame(mw_363)
        pd_data_.to_csv('mw_363.csv')
    return mw_303, mw_363


if __name__ == '__main__':
    mw_303_, mw_363_ = mw_sort()
