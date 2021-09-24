import importlib

import hyperparameters as param

import DataPreparation.Read_data as Read
import DataPreparation.IonicLiquid as IL
import DataPreparation.Datapoint as Data

import DataPreparation.VAE_dataset as VAEDataset
import DataPreparation.MLP_dataset as MLPDataset
import DataPreparation.MLP_CAT_dataset as MLPCATDataset
import DataPreparation.VAE_MLP_dataset as VAEMLPDataset

import VariationalAutoEncoder.VAE_model as Vae
import MultiLayerPerceptron.MLP_model as Mlp
import VariationalAutoEncoder.VAE_cat_model as Vae_Cat
import MultiLayerPerceptron.MLP_cat_model as Mlp_Cat
import VariationalAutoEncoder.VAE_LSTM_cat_model as Vae_Lstm_Cat
import VAE_MLP.VAE_MLP_cat_model as Vae_Mlp

import DataAnalyzation.Accuracy as Accuracy
import DataAnalyzation.Mean as Mean
import DataAnalyzation.KernelDensityEstimation as KDE
import DataAnalyzation.PrincipalComponentsAnalysis as PCA
import DataAnalyzation.SmilesToCosmo as Xyz
import Multi_optim.Generate_SMILES as Generate
import DataAnalyzation.MolecularGraph as Graph
import Multi_optim.Methods as Methods
import DataAnalyzation.RegressionGragh as Regress
from tqdm import tqdm

if __name__ == '__main__':
    importlib.reload(Read)
    importlib.reload(IL)
    importlib.reload(Data)

    importlib.reload(VAEDataset)
    importlib.reload(MLPDataset)
    importlib.reload(MLPCATDataset)
    importlib.reload(VAEMLPDataset)
    importlib.reload(Methods)
    importlib.reload(Vae)
    importlib.reload(Mlp)
    importlib.reload(Vae_Cat)
    importlib.reload(Vae_Lstm_Cat)
    importlib.reload(Mlp_Cat)
    importlib.reload(Vae_Mlp)
    # importlib.reload(Generate)
    importlib.reload(Accuracy)
    importlib.reload(Mean)
    importlib.reload(Graph)
    importlib.reload(param)
    importlib.reload(KDE)
    importlib.reload(PCA)
    importlib.reload(Regress)
    importlib.reload(Xyz)
    params = param.load_params()

    # Read.load_data_origin()
    # Read.load_data_classified()
    # Read.smiles_parser()
    # Read.get_smiles_by_opsin()
    # Read.create_component_smiles_list()
    # Read.create_one_hot_dict()
    # IL.create_ionic_liquid()
    # Data.create_data_point()
    # VAEDataset.create_vae_dataset(params)

    # Vae_Cat.train_vae_cat(latent_code_num=32, params=params, device=0)
    # Vae_Cat.train_vae_cat(latent_code_num=64, params=params, device=1)
    # Vae_Cat.train_vae_cat(latent_code_num=128, params=params, device=0)
    # Vae_Cat.train_vae_cat(latent_code_num=256, params=params, device=1)

    # MLPCATDataset.create_mlp_cat_dataset(params, 'VAE_CAT_32.pkl')
    # MLPCATDataset.create_mlp_cat_dataset(params, 'VAE_CAT_64.pkl')
    # MLPCATDataset.create_mlp_cat_dataset(params, 'VAE_CAT_128.pkl')
    # MLPCATDataset.create_mlp_cat_dataset(params, 'VAE_CAT_256.pkl')

    # VAEMLPDataset.create_vae_mlp_dataset(params)

    # Mlp_Cat.train_mlp(latent_code_num=32, params=params, device=0, hidden=32)
    # Mlp_Cat.train_mlp(latent_code_num=64, params=params, device=1, hidden=64)
    # Mlp_Cat.train_mlp(latent_code_num=128, params=params, device=0, hidden=128)
    # Mlp_Cat.train_mlp(latent_code_num=256, params=params, device=1, hidden=256)
    # Vae_Mlp.train_vae_mlp(32, 16, params, 0)
    # Vae_Mlp.train_vae_mlp(64, 32, params, 1)
    # Vae_Mlp.train_vae_mlp(128, 64, params, 1)
    # Vae_Mlp.train_vae_mlp(256, 128, params, 0)
    # vae_cat_accuracy_decode_32, vae_cat_accuracy_reform_32 = Accuracy.check_vae_cat_accuracy(32, 'CNN', 0)
    # vae_cat_accuracy_decode_64, vae_cat_accuracy_reform_64 = Accuracy.check_vae_cat_accuracy(64, 'CNN', 1)
    # vae_cat_accuracy_decode_128, vae_cat_accuracy_reform_128 = Accuracy.check_vae_cat_accuracy(128, 'CNN', 0)
    # vae_cat_accuracy_decode_256, vae_cat_accuracy_reform_256 = Accuracy.check_vae_cat_accuracy(256, 'CNN', 1)

    # v_aard, t_aard, h_aard = Accuracy.check_mlp_cat_accuracy('MLP_CAT_latent_32_structure_4.pkl')
    # v_aard, t_aard, h_aard = Accuracy.check_mlp_cat_accuracy('MLP_CAT_latent_64_structure_16.pkl')
    # v_aard, t_aard, h_aard = Accuracy.check_mlp_cat_accuracy('MLP_CAT_latent_128_structure_32.pkl')
    # v_aard, t_aard, h_aard = Accuracy.check_mlp_cat_accuracy('MLP_CAT_latent_256_structure_64.pkl')

    # vae_cat_mean_32 = Mean.vae_cat_mean(32)
    # vae_cat_mean_64 = Mean.vae_cat_mean(64)
    # vae_cat_mean_128 = Mean.vae_cat_mean(128)
    # vae_cat_mean_256 = Mean.vae_cat_mean(256)

    # code, smiles, result = Generate.molecular_generation(
    #     params, 'MLP_latent_32_structure_16_4.pkl', 'thermal_conductivity', 'VAE_32.pkl')
    # code, smiles, result = Generate.molecular_generation_cat(
    #     params=params, mlp_file_name='MLP_CAT_heat_capacity_latent_256_structure_128_32.pkl',
    #     vae_file_name='VAE_CAT_256.pkl')
    # smiles_list = Graph.get_valid_il(smiles)

    # for times in tqdm(range(1000)):
    #     Methods.multi_particle_swarm_optimization_cat('MLP_CAT_heat_capacity_latent_64_structure_16.pkl',
    #                                                   'MLP_CAT_thermal_conductivity_latent_64_structure_16.pkl', 303)

    # KDE.plot_kde(params, 'VAE_CAT_32.pkl', 0)
    # KDE.plot_kde(params, 'VAE_CAT_64.pkl', 1)
    # KDE.plot_kde(params, 'VAE_CAT_128.pkl', 1)
    # KDE.plot_kde(params, 'VAE_CAT_256.pkl', 1)

    # input_h, pre_h, label_h = Regress.plot_regression_graph('MLP_CAT_latent_32_structure_32.pkl', 'heat_capacity')
    # input_t, pre_t, label_t = Regress.plot_regression_graph('MLP_CAT_latent_32_structure_32.pkl', 'thermal_conductivity')
    # input_v, pre_v, label_v = Regress.plot_regression_graph('MLP_CAT_latent_32_structure_32.pkl', 'viscosity')

    # input_h, pre_h, label_h = Regress.plot_regression_graph('MLP_CAT_latent_64_structure_64.pkl', 'heat_capacity')
    # input_t, pre_t, label_t = Regress.plot_regression_graph('MLP_CAT_latent_64_structure_64.pkl', 'thermal_conductivity')
    # input_v, pre_v, label_v = Regress.plot_regression_graph('MLP_CAT_latent_64_structure_64.pkl', 'viscosity')

    # input_h, pre_h, label_h = Regress.plot_regression_graph('MLP_CAT_latent_128_structure_128.pkl', 'heat_capacity')
    # input_t, pre_t, label_t = Regress.plot_regression_graph('MLP_CAT_latent_128_structure_128.pkl', 'thermal_conductivity')
    # input_v, pre_v, label_v = Regress.plot_regression_graph('MLP_CAT_latent_128_structure_128.pkl', 'viscosity')

    # input_h, pre_h, label_h = Regress.plot_regression_graph('MLP_CAT_latent_256_structure_256.pkl', 'heat_capacity')
    # input_t, pre_t, label_t = Regress.plot_regression_graph('MLP_CAT_latent_256_structure_256.pkl', 'thermal_conductivity')
    # input_v, pre_v, label_v = Regress.plot_regression_graph('MLP_CAT_latent_256_structure_256.pkl', 'viscosity')

    # t_aard_32, h_aard_32 = Regress.error_vae_mlp('VAE_MLP_CAT_thermal_conductivity_latent_32_structure_16.pkl',
    #                                              'VAE_MLP_CAT_heat_capacity_latent_32_structure_16.pkl')
    # t_aard_64, h_aard_64 = Regress.error_vae_mlp('VAE_MLP_CAT_thermal_conductivity_latent_64_structure_32.pkl',
    #                                        'VAE_MLP_CAT_heat_capacity_latent_64_structure_32.pkl')
    # t_aard_128, h_aard_128 = Regress.error_vae_mlp('VAE_MLP_CAT_thermal_conductivity_latent_128_structure_64.pkl',
    #                                        'VAE_MLP_CAT_heat_capacity_latent_128_structure_64.pkl')
    # t_aard_256, h_aard_256 = Regress.error_vae_mlp('VAE_MLP_CAT_thermal_conductivity_latent_256_structure_128.pkl',
    #                                        'VAE_MLP_CAT_heat_capacity_latent_256_structure_128.pkl')

    # PCA.vae_principal_components_analysis('VAE_CAT_32.pkl', 'MLP_CAT_heat_capacity_latent_32_structure_4.pkl',
    #                                       'cool')
    # PCA.vae_principal_components_analysis('VAE_CAT_64.pkl', 'MLP_CAT_heat_capacity_latent_64_structure_16.pkl',
    #                                       'cool')
    # PCA.vae_principal_components_analysis('VAE_CAT_128.pkl', 'MLP_CAT_heat_capacity_latent_128_structure_32.pkl',
    #                                       'cool')
    # PCA.vae_principal_components_analysis('VAE_CAT_256.pkl', 'MLP_CAT_heat_capacity_latent_256_structure_64.pkl',
    #                                       'cool')

    # PCA.vae_principal_components_analysis('VAE_CAT_32.pkl', 'MLP_CAT_thermal_conductivity_latent_32_structure_4.pkl',
    #                                       'hot')
    # PCA.vae_principal_components_analysis('VAE_CAT_64.pkl', 'MLP_CAT_thermal_conductivity_latent_64_structure_16.pkl',
    #                                       'hot')
    # PCA.vae_principal_components_analysis('VAE_CAT_128.pkl', 'MLP_CAT_thermal_conductivity_latent_128_structure_32.pkl',
    #                                       'hot')
    # PCA.vae_principal_components_analysis('VAE_CAT_256.pkl', 'MLP_CAT_thermal_conductivity_latent_256_structure_64.pkl',
    #                                       'hot')

    # PCA.vae_mlp_principal_components_analysis('VAE_MLP_CAT_heat_capacity_latent_32_structure_16.pkl', 'cool')
    # PCA.vae_mlp_principal_components_analysis('VAE_MLP_CAT_heat_capacity_latent_64_structure_32.pkl', 'cool')
    # PCA.vae_mlp_principal_components_analysis('VAE_MLP_CAT_heat_capacity_latent_128_structure_64.pkl', 'cool')
    # PCA.vae_mlp_principal_components_analysis('VAE_MLP_CAT_heat_capacity_latent_256_structure_128.pkl', 'cool')
    # PCA.vae_mlp_principal_components_analysis('VAE_MLP_CAT_thermal_conductivity_latent_32_structure_16.pkl', 'hot')
    # PCA.vae_mlp_principal_components_analysis('VAE_MLP_CAT_thermal_conductivity_latent_64_structure_32.pkl', 'hot')
    # PCA.vae_mlp_principal_components_analysis('VAE_MLP_CAT_thermal_conductivity_latent_128_structure_64.pkl', 'hot')
    # PCA.vae_mlp_principal_components_analysis('VAE_MLP_CAT_thermal_conductivity_latent_256_structure_128.pkl', 'hot')

    # t_aard_32, h_aard_32 = Regress.error_vae_mlp('ONE_HOT_MLP_thermal_conductivity_hidden_512.pkl',
    #                                              'ONE_HOT_MLP_heat_capacity_hidden_512.pkl')
    # t_aard_64, h_aard_64 = Regress.error_vae_mlp('ONE_HOT_MLP_thermal_conductivity_hidden_1024.pkl',
    #                                              'ONE_HOT_MLP_heat_capacity_hidden_1024.pkl')
    # t_aard_128, h_aard_128 = Regress.error_vae_mlp('ONE_HOT_MLP_thermal_conductivity_hidden_2048.pkl',
    #                                                'ONE_HOT_MLP_heat_capacity_hidden_2048.pkl')
    # t_aard_256, h_aard_256 = Regress.error_vae_mlp('ONE_HOT_MLP_thermal_conductivity_hidden_4096.pkl',
    #                                                'ONE_HOT_MLP_heat_capacity_hidden_4096.pkl')
    Xyz.smiles_to_xyz()
