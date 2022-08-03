import torch
import torch.nn as nn
import numpy as np
from src.ndmd import NDMD
from src.config import *
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


# parameters for Net
Nx, Ny = 192, 384
args = ConfigNDMD(
    input_datasize=[Nx, Ny],
    latent_dim=2,
    encoder=[1, 8, 16, 32, 64, 64, 64],
    encoder_mlp=[2048],
    latent=[2048],
    decoder=[512, 512, 512, 256, 128, 64, 1],
    decoder_mlp=[2048],
    batch_normalization=True,
    activation='GELU'
)

# construct Net
net = NDMD(args).cuda()
input_data = optimizer = optim.Adam(net.parameters(), lr=0.0005)

# dataset
data = np.fromfile('../dataset/periodic.dat').reshape([1000, 1, Nx, Ny])
data_x = data[:-1]
data_x_shift = data[1:]
data_x = data_x.reshape([-1, 1, Nx, Ny])
data_x_shift = data_x_shift.reshape([-1, 1, Nx, Ny])

data_x = torch.from_numpy(data_x).float()
data_x_shift = torch.from_numpy(data_x_shift).float()

database = TensorDataset(data_x, data_x_shift)
train_loader = DataLoader(
    dataset=database,
    batch_size=10,
    shuffle=True,
    drop_last=True
)

# Net Training
_loss, loss_base, epoch = 1.0, 0.1, -1
while epoch < 2000:
    _BCE, _KLD = 0, 0
    _loss, _loss1, _loss2 = 0, 0, 0
    epoch += 1
    i = 0
    for i, data_cut in enumerate(train_loader):
        input_data = Variable(data_cut[0]).cuda()
        output_data = Variable(data_cut[1]).cuda()

        optimizer.zero_grad()
        x_reconst_1, x_reconst_2, z2, z2_from_shift = net(input_data, output_data)
        loss, loss1, loss2 = net.loss_func(x_reconst_1, input_data, x_reconst_2, output_data, z2, z2_from_shift)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        _loss += loss.cpu().detach().numpy()
        _loss1 += loss1.cpu().detach().numpy()
        _loss2 += loss2.cpu().detach().numpy()

    _loss /= i + 1
    _loss1 /= i + 1
    _loss2 /= i + 1

    if epoch % 1 == 0:
        print(epoch, _loss, _loss1, _loss2)

    if _loss < loss_base:
        loss_base = _loss
        torch.save(net, 'NDMD.net')
