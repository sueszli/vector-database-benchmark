import argparse
from os.path import exists
from urllib.request import urlopen
import numpy as np
import torch
import pyro
from pyro.contrib.timeseries import IndependentMaternGP, LinearlyCoupledMaternGP

def download_data():
    if False:
        while True:
            i = 10
    if not exists('eeg.dat'):
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff'
        with open('eeg.dat', 'wb') as f:
            f.write(urlopen(url).read())

def main(args):
    if False:
        return 10
    if not args.test:
        download_data()
        T_forecast = 349
        data = np.loadtxt('eeg.dat', delimiter=',', skiprows=19)
        print('[raw data shape] {}'.format(data.shape))
        data = torch.tensor(data[::20, :-1]).double()
        print('[data shape after thinning] {}'.format(data.shape))
    else:
        data = torch.randn(20, 3).double()
        T_forecast = 10
    (T, obs_dim) = data.shape
    T_train = T - T_forecast
    data_mean = data[0:T_train, :].mean(0)
    data -= data_mean
    data_std = data[0:T_train, :].std(0)
    data /= data_std
    torch.manual_seed(args.seed)
    if args.model == 'imgp':
        gp = IndependentMaternGP(nu=1.5, obs_dim=obs_dim, length_scale_init=1.5 * torch.ones(obs_dim)).double()
    elif args.model == 'lcmgp':
        num_gps = 9
        gp = LinearlyCoupledMaternGP(nu=1.5, obs_dim=obs_dim, num_gps=num_gps, length_scale_init=1.5 * torch.ones(num_gps)).double()
    adam = torch.optim.Adam(gp.parameters(), lr=args.init_learning_rate, betas=(args.beta1, 0.999), amsgrad=True)
    gamma = (args.final_learning_rate / args.init_learning_rate) ** (1.0 / args.num_steps)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(adam, gamma=gamma)
    report_frequency = 10
    for step in range(args.num_steps):
        loss = -gp.log_prob(data[0:T_train, :]).sum() / T_train
        loss.backward()
        adam.step()
        scheduler.step()
        if step % report_frequency == 0 or step == args.num_steps - 1:
            print('[step %03d]  loss: %.3f' % (step, loss.item()))
    if args.plot:
        assert not args.test
        T_multistep = 49
        T_onestep = T_forecast - T_multistep
        print('doing one-step-ahead forecasting...')
        (onestep_means, onestep_stds) = (np.zeros((T_onestep, obs_dim)), np.zeros((T_onestep, obs_dim)))
        for t in range(T_onestep):
            dts = torch.tensor([1.0]).double()
            pred_dist = gp.forecast(data[0:T_train + t, :], dts)
            onestep_means[t, :] = pred_dist.loc.data.numpy()
            if args.model == 'imgp':
                onestep_stds[t, :] = pred_dist.scale.data.numpy()
            elif args.model == 'lcmgp':
                onestep_stds[t, :] = pred_dist.covariance_matrix.diagonal(dim1=-1, dim2=-2).data.numpy()
        print('doing multi-step forecasting...')
        dts = (1 + torch.arange(T_multistep)).double()
        pred_dist = gp.forecast(data[0:T_train + T_onestep, :], dts)
        multistep_means = pred_dist.loc.data.numpy()
        if args.model == 'imgp':
            multistep_stds = pred_dist.scale.data.numpy()
        elif args.model == 'lcmgp':
            multistep_stds = pred_dist.covariance_matrix.diagonal(dim1=-1, dim2=-2).data.numpy()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        (f, axes) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        T = data.size(0)
        to_seconds = 117.0 / T
        for (k, ax) in enumerate(axes):
            which = [0, 4, 10][k]
            ax.plot(to_seconds * np.arange(T), data[:, which], 'ko', markersize=2, label='Data')
            ax.plot(to_seconds * (T_train + np.arange(T_onestep)), onestep_means[:, which], ls='solid', color='b', label='One-step')
            ax.fill_between(to_seconds * (T_train + np.arange(T_onestep)), onestep_means[:, which] - 1.645 * onestep_stds[:, which], onestep_means[:, which] + 1.645 * onestep_stds[:, which], color='b', alpha=0.2)
            ax.plot(to_seconds * (T_train + T_onestep + np.arange(T_multistep)), multistep_means[:, which], ls='solid', color='r', label='Multi-step')
            ax.fill_between(to_seconds * (T_train + T_onestep + np.arange(T_multistep)), multistep_means[:, which] - 1.645 * multistep_stds[:, which], multistep_means[:, which] + 1.645 * multistep_stds[:, which], color='r', alpha=0.2)
            ax.set_ylabel('$y_{%d}$' % (which + 1), fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=14)
            if k == 1:
                ax.legend(loc='upper left', fontsize=16)
        plt.tight_layout(pad=0.7)
        plt.savefig('eeg.{}.pdf'.format(args.model))
if __name__ == '__main__':
    assert pyro.__version__.startswith('1.8.6')
    parser = argparse.ArgumentParser(description='contrib.timeseries example usage')
    parser.add_argument('-n', '--num-steps', default=300, type=int)
    parser.add_argument('-s', '--seed', default=0, type=int)
    parser.add_argument('-m', '--model', default='imgp', type=str, choices=['imgp', 'lcmgp'])
    parser.add_argument('-ilr', '--init-learning-rate', default=0.01, type=float)
    parser.add_argument('-flr', '--final-learning-rate', default=0.0003, type=float)
    parser.add_argument('-b1', '--beta1', default=0.5, type=float)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    main(args)