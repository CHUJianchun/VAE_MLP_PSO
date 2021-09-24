def load_params():
    parameters = {
        'VAE_batch_size': 16,
        'MLP_batch_size': 4,
        'VAE_epoch_num': 200,
        'MLP_epoch_num': 300,
        'Start_point_num': 100,
        'OPTIM_epoch_num': 50,
        'Radical_factor': 5,
        'KDE_samples': 5000
    }
    return parameters
