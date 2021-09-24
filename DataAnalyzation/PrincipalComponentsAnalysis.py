import numpy as np
from sklearn.decomposition import PCA
import torch
import sys
from matplotlib import pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm


def vae_principal_components_analysis(vae_name, mlp_name, colorbar):
    try:
        vae = torch.load('Model_pkl/' + vae_name).cuda()
    except FileNotFoundError:
        print('Error: File ' + vae_name + ' not found')
        sys.exit()
    try:
        mlp = torch.load('Model_pkl/' + mlp_name).cuda()
    except FileNotFoundError:
        print('Error: File ' + mlp_name + ' not found')
        sys.exit()
    try:
        dataset = torch.load('Data/vae_dataset.pkl')
    except FileNotFoundError:
        print('Error: Dataset not found')
        sys.exit()

    dataset = torch.cat((dataset.dataset_input_anion, dataset.dataset_input_cation), dim=1)
    dataset = Variable(dataset).cuda().type(torch.cuda.FloatTensor)
    mu_set = []
    for i in tqdm(range(dataset.shape[0])):
        mu_set.append(vae.get_mid(dataset[i].reshape(1, 1, dataset.shape[1], dataset.shape[2])).detach().cpu().numpy())
    del dataset
    mu_set = Variable(torch.FloatTensor(np.array(mu_set))).cuda().type(torch.cuda.FloatTensor)
    t = torch.ones((mu_set.shape[0], 1)) * 0.3

    data_input = torch.cat((t.cuda(), torch.squeeze(mu_set)), dim=1).cuda()
    del mu_set
    data_label = []
    for i in tqdm(range(data_input.shape[0])):
        label = mlp(data_input[i])
        data_label.append(label.detach().cpu().numpy())
    pca = PCA(n_components=2)
    data_input = data_input.detach().cpu().numpy()
    data_label = np.array(data_label)
    pca.fit(data_input)
    pc_input = pca.fit_transform(data_input)
    print(pca.explained_variance_ratio_)

    cm = plt.cm.get_cmap(colorbar)
    sc = plt.scatter(pc_input[:, 0], pc_input[:, 1], c=data_label,
                     vmin=np.min(data_label), vmax=np.max(data_label), s=5, cmap=cm, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cb = plt.colorbar(sc)
    cb.ax.tick_params(labelsize=16)
    plt.savefig('Graph/' + mlp_name + '.png')
    plt.show()


def vae_mlp_principal_components_analysis(vae_mlp_name, colorbar):
    try:
        vae_mlp = torch.load('Model_pkl/' + vae_mlp_name).cuda()
    except FileNotFoundError:
        print('Error: File ' + vae_mlp_name + ' not found')
        sys.exit()
    try:
        dataset = torch.load('Data/vae_dataset.pkl')
    except FileNotFoundError:
        print('Error: Dataset not found')
        sys.exit()

    dataset = torch.cat((dataset.dataset_input_anion, dataset.dataset_input_cation), dim=1)
    dataset = Variable(dataset).type(torch.cuda.FloatTensor)
    t = Variable(torch.ones((dataset.shape[0], 1)) * 0.3).type(torch.cuda.FloatTensor)
    data_label = []
    data_input = []
    for i in tqdm(range(dataset.shape[0])):
        _, mu, __, label = vae_mlp.forward(dataset[i].reshape(1, 1, dataset.shape[1], dataset.shape[2]).cuda(),
                                           t[i].reshape(1, 1).cuda())
        data_label.append(label.detach().cpu().numpy())
        data_input.append(mu.detach().cpu().numpy())

    pca = PCA(n_components=2)
    data_input = np.squeeze(np.array(data_input))
    data_label = np.squeeze(np.array(data_label))
    for i in range(data_label.shape[0]):
        if data_label[i] < 0:
            data_label[i] = 0
    for i in range(data_label.shape[0]):
        data_label[i] += 0.4 * (np.max(data_label) - data_label[i])
    pca.fit(data_input)
    pc_input = pca.fit_transform(data_input)
    print(pca.explained_variance_ratio_)

    cm = plt.cm.get_cmap(colorbar)
    sc = plt.scatter(pc_input[:, 0], pc_input[:, 1], c=data_label,
                     vmin=np.min(data_label), vmax=np.max(data_label), s=5, cmap=cm, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cb = plt.colorbar(sc)
    cb.ax.tick_params(labelsize=16)
    plt.savefig('Graph/' + vae_mlp_name + '.png')
    plt.show()


if __name__ == '__main__':
    pass
