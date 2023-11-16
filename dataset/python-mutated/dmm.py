"""
An implementation of a Deep Markov Model in Pyro based on reference [1].
This is essentially the DKS variant outlined in the paper. The primary difference
between this implementation and theirs is that in our version any KL divergence terms
in the ELBO are estimated via sampling, while they make use of the analytic formulae.
We also illustrate the use of normalizing flows in the variational distribution (in which
case analytic formulae for the KL divergences are in any case unavailable).

Reference:

[1] Structured Inference Networks for Nonlinear State Space Models [arXiv:1609.09869]
    Rahul G. Krishnan, Uri Shalit, David Sontag
"""
import argparse
import logging
import time
from os.path import exists
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.contrib.examples.polyphonic_data_loader as poly
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceEnum_ELBO, TraceTMC_ELBO, config_enumerate
from pyro.optim import ClippedAdam

class Emitter(nn.Module):
    """
    Parameterizes the bernoulli observation likelihood `p(x_t | z_t)`
    """

    def __init__(self, input_dim, z_dim, emission_dim):
        if False:
            return 10
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, z_t):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given the latent z at a particular time step t we return the vector of\n        probabilities `ps` that parameterizes the bernoulli distribution `p(x_t|z_t)`\n        '
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        ps = torch.sigmoid(self.lin_hidden_to_input(h2))
        return ps

class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.
    """

    def __init__(self, z_dim, transition_dim):
        if False:
            while True:
                i = 10
        super().__init__()
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        if False:
            while True:
                i = 10
        '\n        Given the latent `z_{t-1}` corresponding to the time step t-1\n        we return the mean and scale vectors that parameterize the\n        (diagonal) gaussian distribution `p(z_t | z_{t-1})`\n        '
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        return (loc, scale)

class Combiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)
    """

    def __init__(self, z_dim, rnn_dim):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        if False:
            while True:
                i = 10
        '\n        Given the latent z at at a particular time step t-1 as well as the hidden\n        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that\n        parameterize the (diagonal) gaussian distribution `q(z_t | z_{t-1}, x_{t:T})`\n        '
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        loc = self.lin_hidden_to_loc(h_combined)
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        return (loc, scale)

class DMM(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    """

    def __init__(self, input_dim=88, z_dim=100, emission_dim=100, transition_dim=200, rnn_dim=600, num_layers=1, rnn_dropout_rate=0.0, num_iafs=0, iaf_dim=50, use_cuda=False):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.emitter = Emitter(input_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, transition_dim)
        self.combiner = Combiner(z_dim, rnn_dim)
        rnn_dropout_rate = 0.0 if num_layers == 1 else rnn_dropout_rate
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim, nonlinearity='relu', batch_first=True, bidirectional=False, num_layers=num_layers, dropout=rnn_dropout_rate)
        self.iafs = [affine_autoregressive(z_dim, hidden_dims=[iaf_dim]) for _ in range(num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def model(self, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, annealing_factor=1.0):
        if False:
            print('Hello World!')
        T_max = mini_batch.size(1)
        pyro.module('dmm', self)
        z_prev = self.z_0.expand(mini_batch.size(0), self.z_0.size(0))
        with pyro.plate('z_minibatch', len(mini_batch)):
            for t in pyro.markov(range(1, T_max + 1)):
                (z_loc, z_scale) = self.trans(z_prev)
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample('z_%d' % t, dist.Normal(z_loc, z_scale).mask(mini_batch_mask[:, t - 1:t]).to_event(1))
                emission_probs_t = self.emitter(z_t)
                pyro.sample('obs_x_%d' % t, dist.Bernoulli(emission_probs_t).mask(mini_batch_mask[:, t - 1:t]).to_event(1), obs=mini_batch[:, t - 1, :])
                z_prev = z_t

    def guide(self, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, annealing_factor=1.0):
        if False:
            while True:
                i = 10
        T_max = mini_batch.size(1)
        pyro.module('dmm', self)
        h_0_contig = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()
        (rnn_output, _) = self.rnn(mini_batch_reversed, h_0_contig)
        rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)
        z_prev = self.z_q_0.expand(mini_batch.size(0), self.z_q_0.size(0))
        with pyro.plate('z_minibatch', len(mini_batch)):
            for t in pyro.markov(range(1, T_max + 1)):
                (z_loc, z_scale) = self.combiner(z_prev, rnn_output[:, t - 1, :])
                if len(self.iafs) > 0:
                    z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
                    assert z_dist.event_shape == (self.z_q_0.size(0),)
                    assert z_dist.batch_shape[-1:] == (len(mini_batch),)
                else:
                    z_dist = dist.Normal(z_loc, z_scale)
                    assert z_dist.event_shape == ()
                    assert z_dist.batch_shape[-2:] == (len(mini_batch), self.z_q_0.size(0))
                with pyro.poutine.scale(scale=annealing_factor):
                    if len(self.iafs) > 0:
                        z_t = pyro.sample('z_%d' % t, z_dist.mask(mini_batch_mask[:, t - 1]))
                    else:
                        z_t = pyro.sample('z_%d' % t, z_dist.mask(mini_batch_mask[:, t - 1:t]).to_event(1))
                z_prev = z_t

def main(args):
    if False:
        i = 10
        return i + 15
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=args.log, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logging.info(args)
    data = poly.load_data(poly.JSB_CHORALES)
    training_seq_lengths = data['train']['sequence_lengths']
    training_data_sequences = data['train']['sequences']
    test_seq_lengths = data['test']['sequence_lengths']
    test_data_sequences = data['test']['sequences']
    val_seq_lengths = data['valid']['sequence_lengths']
    val_data_sequences = data['valid']['sequences']
    N_train_data = len(training_seq_lengths)
    N_train_time_slices = float(torch.sum(training_seq_lengths))
    N_mini_batches = int(N_train_data / args.mini_batch_size + int(N_train_data % args.mini_batch_size > 0))
    logging.info('N_train_data: %d     avg. training seq. length: %.2f    N_mini_batches: %d' % (N_train_data, training_seq_lengths.float().mean(), N_mini_batches))
    val_test_frequency = 50
    n_eval_samples = 1

    def rep(x):
        if False:
            for i in range(10):
                print('nop')
        rep_shape = torch.Size([x.size(0) * n_eval_samples]) + x.size()[1:]
        repeat_dims = [1] * len(x.size())
        repeat_dims[0] = n_eval_samples
        return x.repeat(repeat_dims).reshape(n_eval_samples, -1).transpose(1, 0).reshape(rep_shape)
    val_seq_lengths = rep(val_seq_lengths)
    test_seq_lengths = rep(test_seq_lengths)
    (val_batch, val_batch_reversed, val_batch_mask, val_seq_lengths) = poly.get_mini_batch(torch.arange(n_eval_samples * val_data_sequences.shape[0]), rep(val_data_sequences), val_seq_lengths, cuda=args.cuda)
    (test_batch, test_batch_reversed, test_batch_mask, test_seq_lengths) = poly.get_mini_batch(torch.arange(n_eval_samples * test_data_sequences.shape[0]), rep(test_data_sequences), test_seq_lengths, cuda=args.cuda)
    dmm = DMM(rnn_dropout_rate=args.rnn_dropout_rate, num_iafs=args.num_iafs, iaf_dim=args.iaf_dim, use_cuda=args.cuda)
    adam_params = {'lr': args.learning_rate, 'betas': (args.beta1, args.beta2), 'clip_norm': args.clip_norm, 'lrd': args.lr_decay, 'weight_decay': args.weight_decay}
    adam = ClippedAdam(adam_params)
    if args.tmc:
        if args.jit:
            raise NotImplementedError('no JIT support yet for TMC')
        tmc_loss = TraceTMC_ELBO()
        dmm_guide = config_enumerate(dmm.guide, default='parallel', num_samples=args.tmc_num_samples, expand=False)
        svi = SVI(dmm.model, dmm_guide, adam, loss=tmc_loss)
    elif args.tmcelbo:
        if args.jit:
            raise NotImplementedError('no JIT support yet for TMC ELBO')
        elbo = TraceEnum_ELBO()
        dmm_guide = config_enumerate(dmm.guide, default='parallel', num_samples=args.tmc_num_samples, expand=False)
        svi = SVI(dmm.model, dmm_guide, adam, loss=elbo)
    else:
        elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
        svi = SVI(dmm.model, dmm.guide, adam, loss=elbo)

    def save_checkpoint():
        if False:
            print('Hello World!')
        logging.info('saving model to %s...' % args.save_model)
        torch.save(dmm.state_dict(), args.save_model)
        logging.info('saving optimizer states to %s...' % args.save_opt)
        adam.save(args.save_opt)
        logging.info('done saving model and optimizer checkpoints to disk.')

    def load_checkpoint():
        if False:
            i = 10
            return i + 15
        assert exists(args.load_opt) and exists(args.load_model), '--load-model and/or --load-opt misspecified'
        logging.info('loading model from %s...' % args.load_model)
        dmm.load_state_dict(torch.load(args.load_model))
        logging.info('loading optimizer states from %s...' % args.load_opt)
        adam.load(args.load_opt)
        logging.info('done loading model and optimizer states.')

    def process_minibatch(epoch, which_mini_batch, shuffled_indices):
        if False:
            while True:
                i = 10
        if args.annealing_epochs > 0 and epoch < args.annealing_epochs:
            min_af = args.minimum_annealing_factor
            annealing_factor = min_af + (1.0 - min_af) * (float(which_mini_batch + epoch * N_mini_batches + 1) / float(args.annealing_epochs * N_mini_batches))
        else:
            annealing_factor = 1.0
        mini_batch_start = which_mini_batch * args.mini_batch_size
        mini_batch_end = np.min([(which_mini_batch + 1) * args.mini_batch_size, N_train_data])
        mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
        (mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths) = poly.get_mini_batch(mini_batch_indices, training_data_sequences, training_seq_lengths, cuda=args.cuda)
        loss = svi.step(mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, annealing_factor)
        return loss

    def do_evaluation():
        if False:
            print('Hello World!')
        dmm.rnn.eval()
        val_nll = svi.evaluate_loss(val_batch, val_batch_reversed, val_batch_mask, val_seq_lengths) / float(torch.sum(val_seq_lengths))
        test_nll = svi.evaluate_loss(test_batch, test_batch_reversed, test_batch_mask, test_seq_lengths) / float(torch.sum(test_seq_lengths))
        dmm.rnn.train()
        return (val_nll, test_nll)
    if args.load_opt != '' and args.load_model != '':
        load_checkpoint()
    times = [time.time()]
    for epoch in range(args.num_epochs):
        if args.checkpoint_freq > 0 and epoch > 0 and (epoch % args.checkpoint_freq == 0):
            save_checkpoint()
        epoch_nll = 0.0
        shuffled_indices = torch.randperm(N_train_data)
        for which_mini_batch in range(N_mini_batches):
            epoch_nll += process_minibatch(epoch, which_mini_batch, shuffled_indices)
        times.append(time.time())
        epoch_time = times[-1] - times[-2]
        logging.info('[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)' % (epoch, epoch_nll / N_train_time_slices, epoch_time))
        if val_test_frequency > 0 and epoch > 0 and (epoch % val_test_frequency == 0):
            (val_nll, test_nll) = do_evaluation()
            logging.info('[val/test epoch %04d]  %.4f  %.4f' % (epoch, val_nll, test_nll))
if __name__ == '__main__':
    assert pyro.__version__.startswith('1.8.6')
    parser = argparse.ArgumentParser(description='parse args')
    parser.add_argument('-n', '--num-epochs', type=int, default=5000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0003)
    parser.add_argument('-b1', '--beta1', type=float, default=0.96)
    parser.add_argument('-b2', '--beta2', type=float, default=0.999)
    parser.add_argument('-cn', '--clip-norm', type=float, default=10.0)
    parser.add_argument('-lrd', '--lr-decay', type=float, default=0.99996)
    parser.add_argument('-wd', '--weight-decay', type=float, default=2.0)
    parser.add_argument('-mbs', '--mini-batch-size', type=int, default=20)
    parser.add_argument('-ae', '--annealing-epochs', type=int, default=1000)
    parser.add_argument('-maf', '--minimum-annealing-factor', type=float, default=0.2)
    parser.add_argument('-rdr', '--rnn-dropout-rate', type=float, default=0.1)
    parser.add_argument('-iafs', '--num-iafs', type=int, default=0)
    parser.add_argument('-id', '--iaf-dim', type=int, default=100)
    parser.add_argument('-cf', '--checkpoint-freq', type=int, default=0)
    parser.add_argument('-lopt', '--load-opt', type=str, default='')
    parser.add_argument('-lmod', '--load-model', type=str, default='')
    parser.add_argument('-sopt', '--save-opt', type=str, default='')
    parser.add_argument('-smod', '--save-model', type=str, default='')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--tmc', action='store_true')
    parser.add_argument('--tmcelbo', action='store_true')
    parser.add_argument('--tmc-num-samples', default=10, type=int)
    parser.add_argument('-l', '--log', type=str, default='dmm.log')
    args = parser.parse_args()
    main(args)