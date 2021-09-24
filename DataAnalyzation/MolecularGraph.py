from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm import tqdm
import numpy as np
import rdkit
import os
import shutil


def get_valid_il(smiles_list):
    if not os.path.exists('Graph/'):
        os.mkdir('Graph/')
    else:
        shutil.rmtree('Graph/')
        os.mkdir('Graph/')
    molecular_list = np.zeros((len(smiles_list), len(smiles_list[0]), 2)).tolist()
    x = 0
    for molecular in tqdm(smiles_list):
        y = 0
        for process in molecular:
            ionic_list = []
            write = False
            ionic = ''
            for character in process:
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
                    pass
                else:
                    if type(mol_1) == rdkit.Chem.rdchem.Mol and type(mol_2) == rdkit.Chem.rdchem.Mol:
                        if '-' in first and '+' in second and ' ' not in first and ' ' not in second:
                            molecular_list[x][y][0] = mol_1
                            molecular_list[x][y][1] = mol_2
                            mols = [mol_1, mol_2]
                            img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(400, 400),
                                                       legends=['' for x in mols])
                            img.save('Graph/' + str(x) + '_' + str(y) + '.png')
                        elif '+' in first and '-' in second and ' ' not in first and ' ' not in second:
                            molecular_list[x][y][0] = mol_1
                            molecular_list[x][y][1] = mol_2
                            mols = [mol_1, mol_2]
                            img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(400, 400),
                                                       legends=['' for x in mols])
                            img.save('Graph/' + str(x) + '_' + str(y) + '.png')
            y += 1
        x += 1
    return molecular_list


if __name__ == '__main__':
    mol = Chem.MolFromSmiles('C1CCC[N+]1C')
    # img = Draw.MolToImage(mol)
    # img.show()
    MolWeight = MoleculeDescriptors.MolecularDescriptorCalculator(['MolWt'])
    count = MolWeight.CalcDescriptors(mol)[0]
