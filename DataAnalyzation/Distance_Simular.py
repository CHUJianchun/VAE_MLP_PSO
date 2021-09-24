import pickle
import torch
import numpy as np
from DataPreparation.IonicLiquid import *
from torch.autograd import Variable
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import DataStructs


class Ionic:

    def __init__(self, smiles_, one_hot_):
        self.smiles = smiles_
        self.one_hot = one_hot_


with open('Data/ionic_liquid_list.data', 'rb') as f_ionic_liquid_list:
    list_of_all_Ionic_Liquids = pickle.load(f_ionic_liquid_list)
vae = torch.load('Model_pkl/VAE_CAT_32.pkl')

anion_list = []
anion_one_hot_list = []
cation_list = []
cation_one_hot_list = []
for ionic_liquid in tqdm(list_of_all_Ionic_Liquids):
    anion_repeat = False
    cation_repeat = False
    for take_anion in anion_one_hot_list:
        if (take_anion == ionic_liquid.anion_one_hot).all():
            anion_repeat = True
            break
    for take_cation in cation_one_hot_list:
        if (take_cation == ionic_liquid.cation_one_hot).all():
            cation_repeat = True
            break

    if not anion_repeat:
        anion_list.append(Ionic(ionic_liquid.anion_smiles, ionic_liquid.anion_one_hot))
        anion_one_hot_list.append(ionic_liquid.anion_one_hot)
    if not cation_repeat:
        cation_list.append(Ionic(ionic_liquid.cation_smiles, ionic_liquid.cation_one_hot))
        cation_one_hot_list.append(ionic_liquid.cation_one_hot)

list_of_all_Ionic_Liquids = []

for anion in tqdm(anion_list):
    for cation in cation_list:
        list_of_all_Ionic_Liquids.append([anion, cation])

coordinate = []
for ionic_liquid in tqdm(list_of_all_Ionic_Liquids):
    anion_variable = Variable(torch.from_numpy(ionic_liquid[0].one_hot)).cuda()
    cation_variable = Variable(torch.from_numpy(ionic_liquid[1].one_hot)).cuda()
    one_hot_input = torch.cat((anion_variable, cation_variable), dim=0)
    one_hot_input = one_hot_input.reshape(1, 1, one_hot_input.shape[0], one_hot_input.shape[1]) \
        .type(torch.cuda.FloatTensor)
    coordinate.append(vae.get_mid(one_hot_input).detach().cpu().numpy())

distance = []
'''
for co in coordinate:
    distance.append(np.sum(co ** 2))

mid_ionic_liquid = list_of_all_Ionic_Liquids[np.argmin(distance)]
'''

tf2n_smiles = '0[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F9'
bmim_smiles = '0C(CCC)[N+]1=CN(C=C1)C9'
start_mol = Chem.MolFromSmiles(tf2n_smiles[1: -1] + '.' + bmim_smiles[1: -1])
tf2n_mol = Chem.MolFromSmiles(tf2n_smiles[1: -1])
bmim_mol = Chem.MolFromSmiles(bmim_smiles[1: -1])
for i in range(len(list_of_all_Ionic_Liquids)):
    if list_of_all_Ionic_Liquids[i][0].smiles == tf2n_smiles and list_of_all_Ionic_Liquids[i][1].smiles == bmim_smiles:
        break
for co in coordinate:
    distance.append(np.sum((co - coordinate[i]) ** 2) ** 0.5)

for i in range(len(list_of_all_Ionic_Liquids)):
    list_of_all_Ionic_Liquids[i].append(distance[i])

list_of_all_Ionic_Liquids = np.array(list_of_all_Ionic_Liquids)

index = np.argsort(list_of_all_Ionic_Liquids[:, 2])
list_sort = list_of_all_Ionic_Liquids[index, :]

start = 10000
end = start + 5
for i in range(start, end):
    anion_smiles = list_sort[i][0].smiles[1: -1]
    cation_smiles = list_sort[i][1].smiles[1: -1]
    smiles = anion_smiles + '.' + cation_smiles
    smiles = smiles.replace('A', 'Br')
    smiles = smiles.replace('D', 'Na')
    smiles = smiles.replace('E', 'Cl')
    smiles = smiles.replace('G', 'Al')
    smiles = smiles.replace('J', 'NH3')
    smiles = smiles.replace('K', 'NH2')
    smiles = smiles.replace('L', 'NH')
    mol = Chem.MolFromSmiles(smiles)
    anion_mol, cation_mol = Chem.MolFromSmiles(smiles.split('.')[0]), Chem.MolFromSmiles(smiles.split('.')[1])
    img = Draw.MolToFile(mol, 'Mol_img/' + str(list_sort[i][2]) + '.png', size=(700, 700))

    d = Draw.MolDraw2DSVG(400, 400)
    d.ClearDrawing()
    anion_similarity_figure, anion_similarity = SimilarityMaps.GetSimilarityMapForFingerprint(
        tf2n_mol, mol, lambda m_, index_: SimilarityMaps.GetMorganFingerprint(
            m_, index_, radius=2, fpType='bv'),
        draw2d=d)
    d.FinishDrawing()
    with open('Mol_img/' + str(list_sort[i][2]) + 'anion.svg', 'w+') as f:
        f.write(d.GetDrawingText())

    d = Draw.MolDraw2DSVG(400, 400)
    d.ClearDrawing()
    cation_similarity_figure, cation_similarity = SimilarityMaps.GetSimilarityMapForFingerprint(
        bmim_mol, mol, lambda m_, index_: SimilarityMaps.GetMorganFingerprint(
            m_, index_, radius=2, fpType='bv'),
        draw2d=d)
    d.FinishDrawing()
    with open('Mol_img/' + str(list_sort[i][2]) + 'cation.svg', 'w+') as f:
        f.write(d.GetDrawingText())
