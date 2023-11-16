"""
We use a semi-supervised deep generative model of transcriptomics data to propagate labels
from a small set of labeled cells to a larger set of unlabeled cells. In particular we
use a dataset of peripheral blood mononuclear cells (PBMC) from 10x Genomics and
(approximately) reproduce Figure 6 in reference [1].

Note that for simplicity we do not reproduce every aspect of the scANVI pipeline. For
example, we do not use dropout in our neural network encoders/decoders, nor do we include
batch/dataset annotations in our model.

References:
[1] "Harmonization and Annotation of Single-cell Transcriptomics data with Deep Generative Models,"
    Chenling Xu, Romain Lopez, Edouard Mehlman, Jeffrey Regier, Michael I. Jordan, Nir Yosef.
[2] https://github.com/YosefLab/scvi-tutorials/blob/50dd3269abfe0c375ec47114f2c20725a016736f/seed_labeling.ipynb
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Patch
from torch.distributions import constraints
from torch.nn.functional import softmax, softplus
from torch.optim import Adam
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.examples.scanvi_data import get_data
from pyro.distributions.util import broadcast_shape
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import MultiStepLR

def make_fc(dims):
    if False:
        print('Hello World!')
    layers = []
    for (in_dim, out_dim) in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers[:-1])

def split_in_half(t):
    if False:
        while True:
            i = 10
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)

def broadcast_inputs(input_args):
    if False:
        while True:
            i = 10
    shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
    input_args = [s.expand(shape) for s in input_args]
    return input_args

class Z2Decoder(nn.Module):

    def __init__(self, z1_dim, y_dim, z2_dim, hidden_dims):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        dims = [z1_dim + y_dim] + hidden_dims + [2 * z2_dim]
        self.fc = make_fc(dims)

    def forward(self, z1, y):
        if False:
            print('Hello World!')
        z1_y = torch.cat([z1, y], dim=-1)
        _z1_y = z1_y.reshape(-1, z1_y.size(-1))
        hidden = self.fc(_z1_y)
        hidden = hidden.reshape(z1_y.shape[:-1] + hidden.shape[-1:])
        (loc, scale) = split_in_half(hidden)
        scale = softplus(scale)
        return (loc, scale)

class XDecoder(nn.Module):

    def __init__(self, num_genes, z2_dim, hidden_dims):
        if False:
            print('Hello World!')
        super().__init__()
        dims = [z2_dim] + hidden_dims + [2 * num_genes]
        self.fc = make_fc(dims)

    def forward(self, z2):
        if False:
            return 10
        (gate_logits, mu) = split_in_half(self.fc(z2))
        mu = softmax(mu, dim=-1)
        return (gate_logits, mu)

class Z2LEncoder(nn.Module):

    def __init__(self, num_genes, z2_dim, hidden_dims):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        dims = [num_genes] + hidden_dims + [2 * z2_dim + 2]
        self.fc = make_fc(dims)

    def forward(self, x):
        if False:
            print('Hello World!')
        x = torch.log(1 + x)
        (h1, h2) = split_in_half(self.fc(x))
        (z2_loc, z2_scale) = (h1[..., :-1], softplus(h2[..., :-1]))
        (l_loc, l_scale) = (h1[..., -1:], softplus(h2[..., -1:]))
        return (z2_loc, z2_scale, l_loc, l_scale)

class Z1Encoder(nn.Module):

    def __init__(self, num_labels, z1_dim, z2_dim, hidden_dims):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        dims = [num_labels + z2_dim] + hidden_dims + [2 * z1_dim]
        self.fc = make_fc(dims)

    def forward(self, z2, y):
        if False:
            i = 10
            return i + 15
        z2_y = broadcast_inputs([z2, y])
        z2_y = torch.cat(z2_y, dim=-1)
        _z2_y = z2_y.reshape(-1, z2_y.size(-1))
        hidden = self.fc(_z2_y)
        hidden = hidden.reshape(z2_y.shape[:-1] + hidden.shape[-1:])
        (loc, scale) = split_in_half(hidden)
        scale = softplus(scale)
        return (loc, scale)

class Classifier(nn.Module):

    def __init__(self, z2_dim, hidden_dims, num_labels):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        dims = [z2_dim] + hidden_dims + [num_labels]
        self.fc = make_fc(dims)

    def forward(self, x):
        if False:
            return 10
        logits = self.fc(x)
        return logits

class SCANVI(nn.Module):

    def __init__(self, num_genes, num_labels, l_loc, l_scale, latent_dim=10, alpha=0.01, scale_factor=1.0):
        if False:
            while True:
                i = 10
        assert isinstance(num_genes, int)
        self.num_genes = num_genes
        assert isinstance(num_labels, int) and num_labels > 1
        self.num_labels = num_labels
        assert isinstance(latent_dim, int) and latent_dim > 0
        self.latent_dim = latent_dim
        assert isinstance(l_loc, float)
        self.l_loc = l_loc
        assert isinstance(l_scale, float) and l_scale > 0
        self.l_scale = l_scale
        assert isinstance(alpha, float) and alpha > 0
        self.alpha = alpha
        assert isinstance(scale_factor, float) and scale_factor > 0
        self.scale_factor = scale_factor
        super().__init__()
        self.z2_decoder = Z2Decoder(z1_dim=self.latent_dim, y_dim=self.num_labels, z2_dim=self.latent_dim, hidden_dims=[50])
        self.x_decoder = XDecoder(num_genes=num_genes, hidden_dims=[100], z2_dim=self.latent_dim)
        self.z2l_encoder = Z2LEncoder(num_genes=num_genes, z2_dim=self.latent_dim, hidden_dims=[100])
        self.classifier = Classifier(z2_dim=self.latent_dim, hidden_dims=[50], num_labels=num_labels)
        self.z1_encoder = Z1Encoder(num_labels=num_labels, z1_dim=self.latent_dim, z2_dim=self.latent_dim, hidden_dims=[50])
        self.epsilon = 0.005

    def model(self, x, y=None):
        if False:
            while True:
                i = 10
        pyro.module('scanvi', self)
        theta = pyro.param('inverse_dispersion', 10.0 * x.new_ones(self.num_genes), constraint=constraints.positive)
        with pyro.plate('batch', len(x)), poutine.scale(scale=self.scale_factor):
            z1 = pyro.sample('z1', dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1))
            y = pyro.sample('y', dist.OneHotCategorical(logits=x.new_zeros(self.num_labels)), obs=y)
            (z2_loc, z2_scale) = self.z2_decoder(z1, y)
            z2 = pyro.sample('z2', dist.Normal(z2_loc, z2_scale).to_event(1))
            l_scale = self.l_scale * x.new_ones(1)
            l = pyro.sample('l', dist.LogNormal(self.l_loc, l_scale).to_event(1))
            (gate_logits, mu) = self.x_decoder(z2)
            nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()
            x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta, logits=nb_logits)
            pyro.sample('x', x_dist.to_event(1), obs=x)

    def guide(self, x, y=None):
        if False:
            i = 10
            return i + 15
        pyro.module('scanvi', self)
        with pyro.plate('batch', len(x)), poutine.scale(scale=self.scale_factor):
            (z2_loc, z2_scale, l_loc, l_scale) = self.z2l_encoder(x)
            pyro.sample('l', dist.LogNormal(l_loc, l_scale).to_event(1))
            z2 = pyro.sample('z2', dist.Normal(z2_loc, z2_scale).to_event(1))
            y_logits = self.classifier(z2)
            y_dist = dist.OneHotCategorical(logits=y_logits)
            if y is None:
                y = pyro.sample('y', y_dist)
            else:
                classification_loss = y_dist.log_prob(y)
                pyro.factor('classification_loss', -self.alpha * classification_loss, has_rsample=False)
            (z1_loc, z1_scale) = self.z1_encoder(z2, y)
            pyro.sample('z1', dist.Normal(z1_loc, z1_scale).to_event(1))

def main(args):
    if False:
        i = 10
        return i + 15
    pyro.util.set_rng_seed(args.seed)
    (dataloader, num_genes, l_mean, l_scale, anndata) = get_data(dataset=args.dataset, batch_size=args.batch_size, cuda=args.cuda)
    scanvi = SCANVI(num_genes=num_genes, num_labels=4, l_loc=l_mean, l_scale=l_scale, scale_factor=1.0 / (args.batch_size * num_genes))
    if args.cuda:
        scanvi.cuda()
    scheduler = MultiStepLR({'optimizer': Adam, 'optim_args': {'lr': args.learning_rate}, 'milestones': [20], 'gamma': 0.2})
    guide = config_enumerate(scanvi.guide, 'parallel', expand=True)
    elbo = TraceEnum_ELBO(strict_enumeration_warning=False)
    svi = SVI(scanvi.model, guide, scheduler, elbo)
    for epoch in range(args.num_epochs):
        losses = []
        for (x, y) in dataloader:
            if y is not None:
                y = y.type_as(x)
            loss = svi.step(x, y)
            losses.append(loss)
        scheduler.step()
        print('[Epoch %04d]  Loss: %.5f' % (epoch, np.mean(losses)))
    scanvi.eval()
    if args.plot and args.dataset == 'pbmc':
        import scanpy as sc
        latent_rep = scanvi.z2l_encoder(dataloader.data_x)[0]
        y_logits = scanvi.classifier(latent_rep)
        y_probs = softmax(y_logits, dim=-1).data.cpu().numpy()
        anndata.obsm['X_scANVI'] = latent_rep.data.cpu().numpy()
        sc.pp.neighbors(anndata, use_rep='X_scANVI')
        sc.tl.umap(anndata)
        (umap1, umap2) = (anndata.obsm['X_umap'][:, 0], anndata.obsm['X_umap'][:, 1])
        (fig, axes) = plt.subplots(3, 2)
        seed_marker_sizes = anndata.obs['seed_marker_sizes']
        axes[0, 0].scatter(umap1, umap2, s=seed_marker_sizes, c=anndata.obs['seed_colors'], marker='.', alpha=0.7)
        axes[0, 0].set_title('Hand-Curated Seed Labels')
        patch1 = Patch(color='lightcoral', label='CD8-Naive')
        patch2 = Patch(color='limegreen', label='CD4-Naive')
        patch3 = Patch(color='deepskyblue', label='CD4-Memory')
        patch4 = Patch(color='mediumorchid', label='CD4-Regulatory')
        axes[0, 1].legend(loc='center left', handles=[patch1, patch2, patch3, patch4])
        axes[0, 1].get_xaxis().set_visible(False)
        axes[0, 1].get_yaxis().set_visible(False)
        axes[0, 1].set_frame_on(False)
        s10 = axes[1, 0].scatter(umap1, umap2, s=1, c=y_probs[:, 0], marker='.', alpha=0.7)
        axes[1, 0].set_title('Inferred CD8-Naive probability')
        fig.colorbar(s10, ax=axes[1, 0])
        s11 = axes[1, 1].scatter(umap1, umap2, s=1, c=y_probs[:, 1], marker='.', alpha=0.7)
        axes[1, 1].set_title('Inferred CD4-Naive probability')
        fig.colorbar(s11, ax=axes[1, 1])
        s20 = axes[2, 0].scatter(umap1, umap2, s=1, c=y_probs[:, 2], marker='.', alpha=0.7)
        axes[2, 0].set_title('Inferred CD4-Memory probability')
        fig.colorbar(s20, ax=axes[2, 0])
        s21 = axes[2, 1].scatter(umap1, umap2, s=1, c=y_probs[:, 3], marker='.', alpha=0.7)
        axes[2, 1].set_title('Inferred CD4-Regulatory probability')
        fig.colorbar(s21, ax=axes[2, 1])
        fig.tight_layout()
        plt.savefig('scanvi.pdf')
if __name__ == '__main__':
    assert pyro.__version__.startswith('1.8.6')
    parser = argparse.ArgumentParser(description='single-cell ANnotation using Variational Inference')
    parser.add_argument('-s', '--seed', default=0, type=int, help='rng seed')
    parser.add_argument('-n', '--num-epochs', default=60, type=int, help='number of training epochs')
    parser.add_argument('-d', '--dataset', default='pbmc', type=str, help='which dataset to use', choices=['pbmc', 'mock'])
    parser.add_argument('-bs', '--batch-size', default=100, type=int, help='mini-batch size')
    parser.add_argument('-lr', '--learning-rate', default=0.005, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('--plot', action='store_true', default=False, help='whether to make a plot')
    args = parser.parse_args()
    main(args)