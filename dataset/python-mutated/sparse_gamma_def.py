import argparse
import errno
import os
import numpy as np
import torch
import wget
from torch.nn.functional import softplus
import pyro
import pyro.optim as optim
from pyro.contrib.easyguide import EasyGuide
from pyro.contrib.examples.util import get_data_directory
from pyro.distributions import Gamma, Normal, Poisson
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, init_to_feasible
torch.set_default_dtype(torch.float32)
pyro.util.set_rng_seed(0)

def rand_tensor(shape, mean, sigma):
    if False:
        i = 10
        return i + 15
    return mean * torch.ones(shape) + sigma * torch.randn(shape)

class SparseGammaDEF:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.top_width = 100
        self.mid_width = 40
        self.bottom_width = 15
        self.image_size = 64 * 64
        self.alpha_z = torch.tensor(0.1)
        self.beta_z = torch.tensor(0.1)
        self.alpha_w = torch.tensor(0.1)
        self.beta_w = torch.tensor(0.3)
        self.alpha_init = 0.5
        self.mean_init = 0.0
        self.sigma_init = 0.1

    def model(self, x):
        if False:
            while True:
                i = 10
        x_size = x.size(0)
        with pyro.plate('w_top_plate', self.top_width * self.mid_width):
            w_top = pyro.sample('w_top', Gamma(self.alpha_w, self.beta_w))
        with pyro.plate('w_mid_plate', self.mid_width * self.bottom_width):
            w_mid = pyro.sample('w_mid', Gamma(self.alpha_w, self.beta_w))
        with pyro.plate('w_bottom_plate', self.bottom_width * self.image_size):
            w_bottom = pyro.sample('w_bottom', Gamma(self.alpha_w, self.beta_w))
        with pyro.plate('data', x_size):
            z_top = pyro.sample('z_top', Gamma(self.alpha_z, self.beta_z).expand([self.top_width]).to_event(1))
            w_top = w_top.reshape(self.top_width, self.mid_width) if w_top.dim() == 1 else w_top.reshape(-1, self.top_width, self.mid_width)
            mean_mid = torch.matmul(z_top, w_top)
            z_mid = pyro.sample('z_mid', Gamma(self.alpha_z, self.beta_z / mean_mid).to_event(1))
            w_mid = w_mid.reshape(self.mid_width, self.bottom_width) if w_mid.dim() == 1 else w_mid.reshape(-1, self.mid_width, self.bottom_width)
            mean_bottom = torch.matmul(z_mid, w_mid)
            z_bottom = pyro.sample('z_bottom', Gamma(self.alpha_z, self.beta_z / mean_bottom).to_event(1))
            w_bottom = w_bottom.reshape(self.bottom_width, self.image_size) if w_bottom.dim() == 1 else w_bottom.reshape(-1, self.bottom_width, self.image_size)
            mean_obs = torch.matmul(z_bottom, w_bottom)
            pyro.sample('obs', Poisson(mean_obs).to_event(1), obs=x)

    def guide(self, x):
        if False:
            for i in range(10):
                print('nop')
        x_size = x.size(0)

        def sample_zs(name, width):
            if False:
                return 10
            alpha_z_q = pyro.param('alpha_z_q_%s' % name, lambda : rand_tensor((x_size, width), self.alpha_init, self.sigma_init))
            mean_z_q = pyro.param('mean_z_q_%s' % name, lambda : rand_tensor((x_size, width), self.mean_init, self.sigma_init))
            (alpha_z_q, mean_z_q) = (softplus(alpha_z_q), softplus(mean_z_q))
            pyro.sample('z_%s' % name, Gamma(alpha_z_q, alpha_z_q / mean_z_q).to_event(1))

        def sample_ws(name, width):
            if False:
                for i in range(10):
                    print('nop')
            alpha_w_q = pyro.param('alpha_w_q_%s' % name, lambda : rand_tensor(width, self.alpha_init, self.sigma_init))
            mean_w_q = pyro.param('mean_w_q_%s' % name, lambda : rand_tensor(width, self.mean_init, self.sigma_init))
            (alpha_w_q, mean_w_q) = (softplus(alpha_w_q), softplus(mean_w_q))
            pyro.sample('w_%s' % name, Gamma(alpha_w_q, alpha_w_q / mean_w_q))
        with pyro.plate('w_top_plate', self.top_width * self.mid_width):
            sample_ws('top', self.top_width * self.mid_width)
        with pyro.plate('w_mid_plate', self.mid_width * self.bottom_width):
            sample_ws('mid', self.mid_width * self.bottom_width)
        with pyro.plate('w_bottom_plate', self.bottom_width * self.image_size):
            sample_ws('bottom', self.bottom_width * self.image_size)
        with pyro.plate('data', x_size):
            sample_zs('top', self.top_width)
            sample_zs('mid', self.mid_width)
            sample_zs('bottom', self.bottom_width)

def clip_params():
    if False:
        return 10
    for (param, clip) in zip(('alpha', 'mean'), (-2.5, -4.5)):
        for layer in ['_q_top', '_q_mid', '_q_bottom']:
            for wz in ['_w', '_z']:
                pyro.param(param + wz + layer).data.clamp_(min=clip)

class MyEasyGuide(EasyGuide):

    def guide(self, x):
        if False:
            while True:
                i = 10
        global_group = self.group(match='w_.*')
        global_mean = pyro.param('w_mean', lambda : rand_tensor(global_group.event_shape, 0.5, 0.1))
        global_scale = softplus(pyro.param('w_scale', lambda : rand_tensor(global_group.event_shape, 0.0, 0.1)))
        global_group.sample('ws', Normal(global_mean, global_scale).to_event(1))
        local_group = self.group(match='z_.*')
        x_shape = x.shape[:1] + local_group.event_shape
        with self.plate('data', x.size(0)):
            local_mean = pyro.param('z_mean', lambda : rand_tensor(x_shape, 0.5, 0.1))
            local_scale = softplus(pyro.param('z_scale', lambda : rand_tensor(x_shape, 0.0, 0.1)))
            local_group.sample('zs', Normal(local_mean, local_scale).to_event(1))

def main(args):
    if False:
        while True:
            i = 10
    print('loading training data...')
    dataset_directory = get_data_directory(__file__)
    dataset_path = os.path.join(dataset_directory, 'faces_training.csv')
    if not os.path.exists(dataset_path):
        try:
            os.makedirs(dataset_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            pass
        wget.download('https://d2hg8soec8ck9v.cloudfront.net/datasets/faces_training.csv', dataset_path)
    data = torch.tensor(np.loadtxt(dataset_path, delimiter=',')).float()
    sparse_gamma_def = SparseGammaDEF()
    learning_rate = 0.2 if args.guide in ['auto', 'easy'] else 4.5
    momentum = 0.05 if args.guide in ['auto', 'easy'] else 0.1
    opt = optim.AdagradRMSProp({'eta': learning_rate, 't': momentum})
    if args.guide == 'auto':
        guide = AutoDiagonalNormal(sparse_gamma_def.model, init_loc_fn=init_to_feasible)
    elif args.guide == 'easy':
        guide = MyEasyGuide(sparse_gamma_def.model)
    else:
        guide = sparse_gamma_def.guide
    svi = SVI(sparse_gamma_def.model, guide, opt, loss=TraceMeanField_ELBO())
    svi_eval = SVI(sparse_gamma_def.model, guide, opt, loss=TraceMeanField_ELBO(num_particles=args.eval_particles, vectorize_particles=True))
    print('\nbeginning training with %s guide...' % args.guide)
    for k in range(args.num_epochs):
        loss = svi.step(data)
        if args.guide == 'custom':
            clip_params()
        if k % args.eval_frequency == 0 and k > 0 or k == args.num_epochs - 1:
            loss = svi_eval.evaluate_loss(data)
            print('[epoch %04d] training elbo: %.4g' % (k, -loss))
if __name__ == '__main__':
    assert pyro.__version__.startswith('1.8.6')
    parser = argparse.ArgumentParser(description='parse args')
    parser.add_argument('-n', '--num-epochs', default=1500, type=int, help='number of training epochs')
    parser.add_argument('-ef', '--eval-frequency', default=25, type=int, help='how often to evaluate elbo (number of epochs)')
    parser.add_argument('-ep', '--eval-particles', default=20, type=int, help='number of samples/particles to use during evaluation')
    parser.add_argument('--guide', default='custom', type=str, help='use a custom, auto, or easy guide')
    args = parser.parse_args()
    assert args.guide in ['custom', 'auto', 'easy']
    main(args)