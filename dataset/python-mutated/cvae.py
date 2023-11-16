from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO

class Encoder(nn.Module):

    def __init__(self, z_dim, hidden_1, hidden_2):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc31 = nn.Linear(hidden_2, z_dim)
        self.fc32 = nn.Linear(hidden_2, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        xc = x.clone()
        xc[x == -1] = y[x == -1]
        xc = xc.view(-1, 784)
        hidden = self.relu(self.fc1(xc))
        hidden = self.relu(self.fc2(hidden))
        z_loc = self.fc31(hidden)
        z_scale = torch.exp(self.fc32(hidden))
        return (z_loc, z_scale)

class Decoder(nn.Module):

    def __init__(self, z_dim, hidden_1, hidden_2):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 784)
        self.relu = nn.ReLU()

    def forward(self, z):
        if False:
            return 10
        y = self.relu(self.fc1(z))
        y = self.relu(self.fc2(y))
        y = torch.sigmoid(self.fc3(y))
        return y

class CVAE(nn.Module):

    def __init__(self, z_dim, hidden_1, hidden_2, pre_trained_baseline_net):
        if False:
            while True:
                i = 10
        super().__init__()
        self.baseline_net = pre_trained_baseline_net
        self.prior_net = Encoder(z_dim, hidden_1, hidden_2)
        self.generation_net = Decoder(z_dim, hidden_1, hidden_2)
        self.recognition_net = Encoder(z_dim, hidden_1, hidden_2)

    def model(self, xs, ys=None):
        if False:
            for i in range(10):
                print('nop')
        pyro.module('generation_net', self)
        batch_size = xs.shape[0]
        with pyro.plate('data'):
            with torch.no_grad():
                y_hat = self.baseline_net(xs).view(xs.shape)
            (prior_loc, prior_scale) = self.prior_net(xs, y_hat)
            zs = pyro.sample('z', dist.Normal(prior_loc, prior_scale).to_event(1))
            loc = self.generation_net(zs)
            if ys is not None:
                mask_loc = loc[(xs == -1).view(-1, 784)].view(batch_size, -1)
                mask_ys = ys[xs == -1].view(batch_size, -1)
                pyro.sample('y', dist.Bernoulli(mask_loc, validate_args=False).to_event(1), obs=mask_ys)
            else:
                pyro.deterministic('y', loc.detach())
            return loc

    def guide(self, xs, ys=None):
        if False:
            for i in range(10):
                print('nop')
        with pyro.plate('data'):
            if ys is None:
                y_hat = self.baseline_net(xs).view(xs.shape)
                (loc, scale) = self.prior_net(xs, y_hat)
            else:
                (loc, scale) = self.recognition_net(xs, ys)
            pyro.sample('z', dist.Normal(loc, scale).to_event(1))

def train(device, dataloaders, dataset_sizes, learning_rate, num_epochs, early_stop_patience, model_path, pre_trained_baseline_net):
    if False:
        i = 10
        return i + 15
    pyro.clear_param_store()
    cvae_net = CVAE(200, 500, 500, pre_trained_baseline_net)
    cvae_net.to(device)
    optimizer = pyro.optim.Adam({'lr': learning_rate})
    svi = SVI(cvae_net.model, cvae_net.guide, optimizer, loss=Trace_ELBO())
    best_loss = np.inf
    early_stop_count = 0
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            running_loss = 0.0
            num_preds = 0
            bar = tqdm(dataloaders[phase], desc='CVAE Epoch {} {}'.format(epoch, phase).ljust(20))
            for (i, batch) in enumerate(bar):
                inputs = batch['input'].to(device)
                outputs = batch['output'].to(device)
                if phase == 'train':
                    loss = svi.step(inputs, outputs)
                else:
                    loss = svi.evaluate_loss(inputs, outputs)
                running_loss += loss / inputs.size(0)
                num_preds += 1
                if i % 10 == 0:
                    bar.set_postfix(loss='{:.2f}'.format(running_loss / num_preds), early_stop_count=early_stop_count)
            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(cvae_net.state_dict(), model_path)
                    early_stop_count = 0
                else:
                    early_stop_count += 1
        if early_stop_count >= early_stop_patience:
            break
    cvae_net.load_state_dict(torch.load(model_path))
    cvae_net.eval()
    return cvae_net