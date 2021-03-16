import abc
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from torch.utils.tensorboard import SummaryWriter


class Generator(nn.Module, abc.ABC):
    def __init__(self, layers):
        """
        The generator network.

        :param layers: layers configuration (example: [2 + v_dim, 256, 256, 256, 256, 256, 1]).
        """
        super().__init__()
        modules = [nn.Linear(layers[0], layers[1])]
        for i in range(1, len(layers) - 1):
            modules += [nn.Tanh(), nn.Linear(layers[i], layers[i + 1])]
        self.net = nn.Sequential(*modules)

    def forward(self, *x):
        return self.net(torch.cat(x, 1))

    def predict(self, *x):
        with torch.no_grad():
            out = self.forward(*x)
        return out.to('cpu')


class Reconstructor(nn.Module, abc.ABC):
    def __init__(self, layers):
        """
        The reconstructor network.

        :param layers: layers configuration (example: [3, 256, 256, 256, v_dim])
        """
        super().__init__()
        modules = [nn.Linear(layers[0], layers[1])]
        for i in range(1, len(layers) - 1):
            modules += [nn.Tanh(), nn.Linear(layers[i], layers[i + 1])]
        self.net = nn.Sequential(*modules)

    def forward(self, *x):
        return self.net(torch.cat(x, 1))


class Encoder(nn.Module, abc.ABC):
    """
    The encoder network.

    :param layers: layers configuration (example: [n_x + z_dim, 256, 256, 256, v_dim]).
    """
    def __init__(self, layers):
        super().__init__()
        modules = [nn.Linear(layers[0], layers[1])]
        for i in range(1, len(layers) - 1):
            modules += [nn.ReLU(), nn.Linear(layers[i], layers[i + 1])]
        self.net = nn.Sequential(*modules)

    def forward(self, x, z):
        out = self.net(torch.cat((x, z), 1))
        return out


class Discriminator(nn.Module, abc.ABC):
    """
    The encoder network.

    :param layers: layers configuration (example: [3, 256, 256, 256, 1]).
    """
    def __init__(self, layers):
        super().__init__()
        modules = [nn.Linear(layers[0], layers[1])]
        for i in range(1, len(layers) - 1):
            modules += [nn.ReLU(), nn.Linear(layers[i], layers[i + 1])]
        self.net = nn.Sequential(*modules)

    def forward(self, *x):
        return self.net(torch.cat(x, 1))


class PINNDataset(Dataset):
    """
    The dataset definition

    Example: PINNDataset(t_bc, xlb, xub, device=device, gradient=True)
    """
    def __init__(self, *tensors, device='cpu', gradient=False):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.device = device
        self.grad = gradient
        self.tensors = tensors

    def __getitem__(self, index):
        return [tensor[index].to(self.device, non_blocking=True).requires_grad_(self.grad)
                for tensor in self.tensors]

    def __len__(self):
        return self.tensors[0].size(0)


def collate_pinn(batch):
    """
    The collate function for the dataloader.
    """
    return [x for x in batch[0]]


def gen_data(t, x, n_t, n_col, col_bs, init_bs, bc_bs, device='cpu'):
    """
    Generates the dataloaders for the trainer.

    :param t: time domain limits ([t0, tf]).
    :param x: spatial coordinates.
    :param n_t: number of time domain points.
    :param n_col: number of collocation points.
    :param col_bs: collocations points batch size.
    :param init_bs: initial conditions batch size.
    :param bc_bs: boundary conditions batch size.
    :return: dataloaders: (collocation points, boundary conditions, initial conditions).
    """
    idx_init = torch.arange(x.numel(), device=device).view(-1, 1)
    ds_init = BatchSampler(RandomSampler(range(x.numel()), replacement=False),
                           batch_size=init_bs, drop_last=True)
    dl_init = DataLoader(PINNDataset(x.view(-1, 1), idx_init, device=device),
                         sampler=ds_init, collate_fn=collate_pinn)

    t_bc = torch.linspace(t[0], t[-1], n_t, device=device).view(-1, 1)
    xlb = x[0] * torch.ones((n_t, 1), device=device, requires_grad=True)
    xub = x[-1] * torch.ones((n_t, 1), device=device, requires_grad=True)
    ds_bc = BatchSampler(RandomSampler(range(t_bc.numel()), replacement=False), batch_size=bc_bs, drop_last=True)
    dl_bc = DataLoader(PINNDataset(t_bc, xlb, xub, device=device, gradient=True), sampler=ds_bc,
                       collate_fn=collate_pinn)

    t_col = torch.linspace(t[0], t[-1], n_col[0], device=device)
    x_col = torch.linspace(x[0], x[-1], n_col[1], device=device)
    t_col, x_col = torch.meshgrid([t_col, x_col])
    t_col = t_col.flatten().view(-1, 1)
    x_col = x_col.flatten().view(-1, 1)
    ds_col = BatchSampler(RandomSampler(range(x_col.numel()), replacement=False), batch_size=col_bs, drop_last=True)
    dl_col = DataLoader(PINNDataset(t_col, x_col, device=device, gradient=True), sampler=ds_col,
                        collate_fn=collate_pinn)

    return dl_col, dl_init, dl_bc


def grad(phi, x):
    """
    Calculates the gradient of phi with respect to x
    """
    return torch.autograd.grad(phi, x, torch.ones_like(phi), create_graph=True, allow_unused=True)[0]


def time_step(t, dt, n_t, dn_t, x, n_col, col_bs, init_bs, bc_bs, device='cpu'):
    """
    Function for adaptive time-stepping.

    :param t: time (vector).
    :param dt: time step.
    :param n_t: number of time domain points.
    :param dn_t: how many time domain point to add when extending the time domain.
    :param x: spatial coordinates.
    :param n_col: number of collocation points.
    :param col_bs: collocation points batch size.
    :param init_bs: initial conditions batch size.
    :param bc_bs: boundary conditions batch size.
    :return: updated dataloaders, time vector and number of collocation points
    """
    t = torch.tensor((torch.min(t).item(), torch.max(t).item() + dt), device=device)
    n_col = (n_col[0] + dn_t, n_col[1])
    n_t = n_t + dn_t
    dl_col, dl_init, dl_bc = gen_data(t, x, n_t, n_col, col_bs, init_bs, bc_bs, device=device)

    return dl_col, dl_init, dl_bc, t, n_col


def train(epochs, gen, enc, rec, dis, optimizer_gen, optimizer_dis,
          t, x, u0, params, device):
    """
    Trains the model to solve the 1D Burger's equation.
    :param epochs: number of epochs.
    :param gen: generator model.
    :param enc: encoder model.
    :param rec: reconstructor model.
    :param dis: discriminator model.
    :param optimizer_gen: generator optimizer.
    :param optimizer_dis: discriminator optimizer.
    :param t: time (vector).
    :param x: spatial coordinates.
    :param u0: initial conditions.
    :param params: training parameters: save_path, save_freq, col_bs, init_bs, bc_bs, r_bs, n_col, n_t,
    n_i, z_dim, alpha, beta, gamma, delta, label_noise, label_true (smoothing), nu (for Burger's),
    optional: dt (for adaptive time-stepping).
    """
    # Define training parameters:
    save_path = params['save_path']
    save_freq = params['save_freq']
    col_bs = params['col_bs']
    init_bs = params['init_bs']
    bc_bs = params['bc_bs']
    r_bs = params['r_bs']
    n_col = params['n_col']
    n_t = params['n_t']
    n_i = params['n_i']
    z_dim = params['z_dim']
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    delta = params['delta']
    label_noise = params['label_noise']
    label_true = params['label_true']
    nu = params['nu']

    # Prepare the trainer for adaptive time-stepping, if relevant:
    if 'dt' in params and params['dt']:
        dt = params['dt']
        dn_t = params['dn_t']
        t_steps_tot = params['t_steps_tot']
        t_step_loss = params['t_step_loss']
        iters_max = int(epochs * np.floor((n_col[0] + dn_t * t_steps_tot) * n_col[1] / col_bs))
    else:
        dt = None
        dn_t = None
        t_steps_tot = None
        t_step_loss = None
        iters_max = int(epochs * np.floor(np.prod(n_col) / col_bs))

    # Define the loss function:
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

    # Define the tensorboard summary writer and tqdm progress bar:
    writer = SummaryWriter(log_dir=save_path / 'tb')
    pbar = tqdm.tqdm(total=iters_max, bar_format='{desc}{percentage:3.0f}%|{bar}{r_bar}', position=0, leave=True)

    # Define the losses dictionary:
    losses = {'G': torch.empty(iters_max, device=device), 'PDE': torch.empty(iters_max, device=device),
              'IC': torch.empty(iters_max, device=device), 'BC': torch.empty(iters_max, device=device),
              'R': torch.empty(iters_max, device=device), 'PINN': torch.empty(iters_max, device=device),
              'D': torch.empty(iters_max, device=device), 'DT': torch.empty(iters_max, device=device),
              'DF': torch.empty(iters_max, device=device), 'iters_done': 0, 'time_steps_done': 0}

    # The training loop:
    t0_init = torch.zeros((init_bs, 1), device=device)
    dl_col, dl_init, dl_bc = gen_data(t, x, n_t, n_col, col_bs, init_bs, bc_bs, device=device)
    tot_iter = 0
    t_step = 0
    while tot_iter < iters_max:
        dl_col_iter = iter(dl_col)
        for t_col, x_col in dl_col_iter:
            x_init, idx_init = next(iter(dl_init))
            t_bc, xlb, xub = next(iter(dl_bc))

            # Calculate the generator loss:
            j = torch.randint(n_i, (1,), device=device)
            u0_j = u0[j].repeat(col_bs + 2 * bc_bs + init_bs, 1)

            z = torch.randn(col_bs + 2 * bc_bs + init_bs, z_dim, device=device)
            x_cat = torch.cat((x_col, xlb, xub, x_init), 0)
            t_cat = torch.cat((t_col, t_bc, t_bc, t0_init), 0)

            v_cat = enc.forward(u0_j, z)
            u_cat_pred = gen.forward(t_cat, x_cat, v_cat)
            dis_fake = dis.forward(t0_init, x_init, u_cat_pred[col_bs + 2 * bc_bs:])
            real_labels = torch.zeros((init_bs, 1), device=device).uniform_(1 - label_noise, 1 + label_noise)
            loss_g = loss_fn(dis_fake, real_labels)

            # Calculate the pde, bc, ic and reconstructor losses:
            u_t = grad(u_cat_pred, t_cat)
            u_x = grad(u_cat_pred, x_cat)
            u_xx = grad(u_x, x_cat)

            loss_pde = torch.mean((u_t[:col_bs] + u_cat_pred[:col_bs] * u_x[:col_bs] - nu * u_xx[:col_bs]) ** 2)

            loss_bc = F.mse_loss(u_cat_pred[col_bs:col_bs + bc_bs], u_cat_pred[col_bs + bc_bs:col_bs + 2 * bc_bs]) + \
                      F.mse_loss(u_x[col_bs:col_bs + bc_bs], u_x[col_bs + bc_bs:col_bs + 2 * bc_bs])

            loss_ic = F.mse_loss(u_cat_pred[col_bs + 2 * bc_bs:], u0_j[0, idx_init].view(-1, 1), reduction='mean')

            idx_r = torch.randperm(col_bs, device=device)[:r_bs]
            v_recon = rec.forward(t_col[idx_r], x_col[idx_r], u_cat_pred[:col_bs][idx_r])
            loss_r = F.mse_loss(v_cat[:init_bs, :], v_recon)

            # The PINN loss:
            loss_pinn = loss_g + alpha * loss_pde + beta * loss_ic + gamma * loss_bc + delta * loss_r
            optimizer_gen.zero_grad()
            loss_pinn.backward(retain_graph=True)
            optimizer_gen.step()

            # Calculate the discriminator loss:
            real_labels = torch.zeros((init_bs, 1), device=device).uniform_(label_true - label_noise,
                                                                            label_true + label_noise)
            fake_labels = torch.zeros((init_bs, 1), device=device).uniform_(-1 * label_noise, label_noise)

            dis_real, dis_fake = dis.forward(t0_init, x_init, u0_j[0, idx_init].view(-1, 1)), \
                                 dis.forward(t0_init, x_init, u_cat_pred[col_bs + 2 * bc_bs:].detach())

            loss_dt, loss_df = loss_fn(dis_real, real_labels), loss_fn(dis_fake, fake_labels)
            loss_dis = loss_dt + loss_df

            optimizer_dis.zero_grad()
            loss_dis.backward(retain_graph=False)
            optimizer_dis.step()

            # Update the losses dictionary (for saving/printing):
            losses['DT'][tot_iter] = loss_dt.item()
            losses['DF'][tot_iter] = loss_df.item()
            losses['D'][tot_iter] = loss_dis.item()
            losses['G'][tot_iter] = loss_g.item()
            losses['R'][tot_iter] = loss_r.item()
            losses['PDE'][tot_iter] = loss_pde.item()
            losses['IC'][tot_iter] = loss_ic.item()
            losses['BC'][tot_iter] = loss_bc.item()
            losses['PINN'][tot_iter] = loss_pinn.item()

            # Update TensorBoard and the progress bar every 50 iterations:
            tot_iter += 1
            if tot_iter % 50 == 0:
                writer.add_scalar('L(PINN)/Total', loss_pinn.item(), tot_iter)
                writer.add_scalar('L(PINN)/G', loss_g.item(), tot_iter)
                writer.add_scalar('L(PINN)/R', loss_r.item(), tot_iter)
                writer.add_scalar('L(PINN)/PDE', loss_pde.item(), tot_iter)
                writer.add_scalar('L(PINN)/IC', loss_ic.item(), tot_iter)
                writer.add_scalar('L(PINN)/BC', loss_bc.item(), tot_iter)
                writer.add_scalar('L(D)/Total', loss_dis.item(), tot_iter)
                writer.add_scalar('L(D)/Relative', loss_dt.item() / (loss_df.item() + 1e-8), tot_iter)

                pbar.set_description(f'L(PINN): {loss_pinn.item():.3e}, L(G): {loss_g.item():.3e}, '
                                     f'L(R): {loss_r.item():.3e}, L(PDE): {loss_pde.item():.3e}, '
                                     f'L(IC): {loss_ic.item():.3e}, L(BC): {loss_bc.item():.3e}, '
                                     f'L(D): {loss_dis.item():.3e},'
                                     f' L(DT)/L(DF): {loss_dt.item() / (loss_df.item() + 1e-8):.3f}')
                pbar.update(50)

            # Save the progress:
            if (save_freq > 0) and (tot_iter % save_freq == 0):
                torch.save(gen.state_dict(), save_path.joinpath(f'{tot_iter:010d}.gen.pt'))
                torch.save(enc.state_dict(), save_path.joinpath(f'{tot_iter:010d}.enc.pt'))
                torch.save(rec.state_dict(), save_path.joinpath(f'{tot_iter:010d}.rec.pt'))
                torch.save(dis.state_dict(), save_path.joinpath(f'{tot_iter:010d}.dis.pt'))

                losses_save = {'iters_done': tot_iter, 'time_steps_done': t_step}
                for key in ('PINN', 'PDE', 'IC', 'BC', 'R', 'G', 'D', 'DT', 'DF'):
                    losses_save.update({f'{key}': losses[key][:tot_iter].cpu().numpy()})
                scipy.io.savemat(save_path.joinpath(f'{tot_iter:010d}.losses.mat'), losses_save)

            # If relevent: adaptive time-stepping
            if dt and (tot_iter >= 10000) and (tot_iter % 500 == 0) and (t_step < t_steps_tot):
                pde_loss_mean = torch.mean(losses['IC'][tot_iter - 500:tot_iter] +
                                           losses['BC'][tot_iter - 500:tot_iter] +
                                           losses['PDE'][tot_iter - 500:tot_iter]).item()
                if pde_loss_mean < t_step_loss:
                    dl_col, dl_init, dl_bc, t, n_col = time_step(t, dt, n_t, dn_t, x, n_col, col_bs, init_bs, bc_bs,
                                                                 device='cpu')

                    t_step += 1
                    print(f'\n{pde_loss_mean:.3e}, Time step: {t_step}\n')
