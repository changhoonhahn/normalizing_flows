import os
import numpy as np
import torch
import torch.utils.data


def load_data():
    version = '0.0'
    props, mags, sigs, zreds = [], [], [], []
    for seed in range(11): 
        _prop = np.load(os.path.join(dat_dir,
            'train.v%s.%i.props.prune_cnf.npy' % (version, seed)))
        _mags = np.load(os.path.join(dat_dir,
            'train.v%s.%i.mags.noise_cnf_flux.prune_cnf.npy' % (version, seed)))
        _sigs = np.load(os.path.join(dat_dir,
            'train.v%s.%i.sigma_mags.noise_cnf_flux.prune_cnf.npy' % (version, seed)))
        _zred = np.load(os.path.join(dat_dir,
            'train.v%s.%i.redshifts.prune_cnf.npy' % (version, seed)))

        props.append(_prop)
        mags.append(_mags)
        sigs.append(_sigs)
        zreds.append(_zred)
    
    data_x = np.concatenate(props)
    data_y = np.concatenate([
        np.concatenate(mags), 
        np.concatenate(sigs), 
        np.atleast_2d(np.concatenate(zreds)).T], 
        axis=1)
    N_validate = int(0.1 * data_x.shape[0]) 

    data_x_train = data_x[:-N_validate]
    data_y_train = data_y[:-N_validate]

    data_x_valid = data_x[-N_validate:]
    data_y_valid = data_y[-N_validate:]

    # load test data
    for seed in [101]: 
        _prop = np.load(os.path.join(dat_dir,
            'train.v%s.%i.props.prune_cnf.npy' % (version, seed)))
        _mags = np.load(os.path.join(dat_dir,
            'train.v%s.%i.mags.noise_cnf_flux.prune_cnf.npy' % (version, seed)))
        _sigs = np.load(os.path.join(dat_dir,
            'train.v%s.%i.sigma_mags.noise_cnf_flux.prune_cnf.npy' % (version, seed)))
        _zred = np.load(os.path.join(dat_dir,
            'train.v%s.%i.redshifts.prune_cnf.npy' % (version, seed)))
    data_x_test = _prop
    data_y_test = np.concatenate([_mags, _sigs, np.atleast_2d(_zred).T], axis=1)

    return data_x_train, data_y_train, data_x_valid, data_y_valid, data_x_test, data_y_test


def load_dataloaders(batch_size, device): 
    x_train, y_train, x_valid, y_valid, _, _ = load_data()

    print('Ntrain = %i; Nvalid = %i' % (x_train.shape[0], x_valid.shape[0]))
    train_tensor    = torch.from_numpy(x_train.astype(np.float32))
    train_cond      = torch.from_numpy(y_train.astype(np.float32))
    train_dataset   = torch.utils.data.TensorDataset(train_tensor, train_cond)

    valid_tensor    = torch.from_numpy(x_valid.astype(np.float32))
    valid_cond      = torch.from_numpy(y_valid.astype(np.float32))
    valid_dataset   = torch.utils.data.TensorDataset(valid_tensor, valid_cond)

    # number of conditional inputs
    num_inputs      = x_train.shape[1]
    num_cond_inputs = y_train.shape[1]

    # set up loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=batch_size, shuffle=True, **kwargs).to(device)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                    batch_size=test_batch_size, shuffle=False, drop_last=False,
                    **kwargs).to(device) 
    return train_loader, valid_loader 
