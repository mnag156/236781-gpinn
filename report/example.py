import sys

sys.path.append('..')
import torch
import numpy as np
import scipy.io
from gpinn.burgers import train, Generator, Encoder, Reconstructor, Discriminator
from pathlib import Path
from datetime import datetime


if __name__ == '__main__':
    torch.manual_seed(1234)
    np.random.seed(5678)

    # Where to save the output.
    save_path = Path(f'./example/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    save_path.mkdir(parents=True, exist_ok=True)

    # Save the results every N iterations.
    save_freq = 10000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print(device)

    # Load the data - initial conditions and exact solutions (chebfun output).
    data = scipy.io.loadmat('.data/cheb.nu0.03_ni1.mat')
    nu = data['nu'][0, 0]
    x0 = data['x0']
    t0 = data['t']
    idx_train = data['idx_train'].flatten()
    n_i = idx_train.shape[-1]
    u0 = torch.tensor(data['u0'][idx_train, :], dtype=torch.float)

    # Define the time domain
    t_dom = (data['t'][0, 0], data['t'][-1, 0])
    n_t = 200
    t = torch.linspace(t_dom[0], t_dom[-1], n_t)

    # Define the spatial domain.
    x_dom = (data['x0'][0, 0], data['x0'][-1, 0])
    x = torch.tensor(data['x0'].flatten(), dtype=torch.float)
    n_x = x.numel()

    # Number of collocation points.
    n_col = (200, 500)

    # Train for N epochs.
    epochs = 40000

    # Batch sizes: initial conditions, collocation points, boundary conditions, reconstructor.
    init_bs = 100
    col_bs = 1000
    bc_bs = 50
    r_bs = init_bs

    # Loss parameters.
    alpha = 30
    beta = 100
    gamma = 30
    delta = 30

    # Label smoothing and noise.
    label_noise = 0.2
    label_true = 0.8

    t = t.to(device)
    x = x.to(device)
    u0 = u0.to(device)

    # Model definition.
    z_dim = 32
    v_dim = 32
    layers_gen = [2 + v_dim, 256, 256, 256, 256, 256, 1]
    layers_rec = [3, 256, 256, 256, v_dim]
    layers_enc = [n_x + z_dim, 256, 256, 256, v_dim]
    layers_dis = [3, 256, 256, 256, 1]

    gen = Generator(layers_gen).to(device)
    rec = Reconstructor(layers_rec).to(device)
    enc = Encoder(layers_enc).to(device)
    dis = Discriminator(layers_dis).to(device)

    # Learning rate.
    lr_gen = 0.0001
    lr_dis = 0.0002

    # Optimizers.
    optimizer_gen = torch.optim.Adam([{'params': gen.parameters(), 'lr': lr_gen},
                                      {'params': rec.parameters(), 'lr': lr_gen},
                                      {'params': enc.parameters(), 'lr': lr_gen}])
    optimizer_dis = torch.optim.Adam([{'params': dis.parameters(), 'lr': lr_dis}])

    # Save the following parameters to a .mat file.
    params_to_save = {'layers_dec': layers_gen, 'layers_enc': layers_enc,
                      'layers_rec': layers_rec, 'layers_dis': layers_dis,
                      'z_dim': z_dim, 'v_dim': v_dim,
                      'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta,
                      'label_noise': label_noise, 'label_true': label_true,
                      'lr_dec': lr_gen, 'lr_dis': lr_dis, 'epochs_num': epochs,
                      't_dom': t_dom, 'x_dom': x_dom, 'n_x': n_x, 'n_i': n_i,
                      'n_col': n_col, 'col_bs': col_bs, 'init_bs': init_bs, 'bc_bs': bc_bs, 'n_t': n_t,
                      'nu': nu}
    scipy.io.savemat(save_path / f'.params.mat', params_to_save)

    # Training parameters.
    params = {'col_bs': col_bs, 'n_col': n_col, 'init_bs': init_bs, 'r_bs': r_bs, 'n_i': n_i, 'z_dim': z_dim,
              'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta,
              'label_noise': label_noise, 'label_true': label_true, 'bc_bs': bc_bs, 'n_t': n_t,
              'save_path': save_path, 'save_freq': save_freq,
              'nu': nu}

    # Train the network.
    train(epochs, gen, enc, rec, dis, optimizer_gen, optimizer_dis, t, x, u0, params, device)
