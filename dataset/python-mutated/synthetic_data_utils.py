from __future__ import print_function
import h5py
import numpy as np
import os
from utils import write_datasets
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal

def generate_rnn(rng, N, g, tau, dt, max_firing_rate):
    if False:
        for i in range(10):
            print('nop')
    'Create a (vanilla) RNN with a bunch of hyper parameters for generating\nchaotic data.\n  Args:\n    rng: numpy random number generator\n    N: number of hidden units\n    g: scaling of recurrent weight matrix in g W, with W ~ N(0,1/N)\n    tau: time scale of individual unit dynamics\n    dt: time step for equation updates\n    max_firing_rate: how to resecale the -1,1 firing rates\n  Returns:\n    the dictionary of these parameters, plus some others.\n'
    rnn = {}
    rnn['N'] = N
    rnn['W'] = rng.randn(N, N) / np.sqrt(N)
    rnn['Bin'] = rng.randn(N) / np.sqrt(1.0)
    rnn['Bin2'] = rng.randn(N) / np.sqrt(1.0)
    rnn['b'] = np.zeros(N)
    rnn['g'] = g
    rnn['tau'] = tau
    rnn['dt'] = dt
    rnn['max_firing_rate'] = max_firing_rate
    mfr = rnn['max_firing_rate']
    nbins_per_sec = 1.0 / rnn['dt']
    rnn['conversion_factor'] = mfr / nbins_per_sec
    return rnn

def generate_data(rnn, T, E, x0s=None, P_sxn=None, input_magnitude=0.0, input_times=None):
    if False:
        for i in range(10):
            print('nop')
    " Generates data from an randomly initialized RNN.\n  Args:\n    rnn: the rnn\n    T: Time in seconds to run (divided by rnn['dt'] to get steps, rounded down.\n    E: total number of examples\n    S: number of samples (subsampling N)\n  Returns:\n    A list of length E of NxT tensors of the network being run.\n  "
    N = rnn['N']

    def run_rnn(rnn, x0, ntime_steps, input_time=None):
        if False:
            for i in range(10):
                print('nop')
        rs = np.zeros([N, ntime_steps])
        x_tm1 = x0
        r_tm1 = np.tanh(x0)
        tau = rnn['tau']
        dt = rnn['dt']
        alpha = 1.0 - dt / tau
        W = dt / tau * rnn['W'] * rnn['g']
        Bin = dt / tau * rnn['Bin']
        Bin2 = dt / tau * rnn['Bin2']
        b = dt / tau * rnn['b']
        us = np.zeros([1, ntime_steps])
        for t in range(ntime_steps):
            x_t = alpha * x_tm1 + np.dot(W, r_tm1) + b
            if input_time is not None and t == input_time:
                us[0, t] = input_magnitude
                x_t += Bin * us[0, t]
            r_t = np.tanh(x_t)
            x_tm1 = x_t
            r_tm1 = r_t
            rs[:, t] = r_t
        return (rs, us)
    if P_sxn is None:
        P_sxn = np.eye(N)
    ntime_steps = int(T / rnn['dt'])
    data_e = []
    inputs_e = []
    for e in range(E):
        input_time = input_times[e] if input_times is not None else None
        (r_nxt, u_uxt) = run_rnn(rnn, x0s[:, e], ntime_steps, input_time)
        r_sxt = np.dot(P_sxn, r_nxt)
        inputs_e.append(u_uxt)
        data_e.append(r_sxt)
    S = P_sxn.shape[0]
    data_e = normalize_rates(data_e, E, S)
    return (data_e, x0s, inputs_e)

def normalize_rates(data_e, E, S):
    if False:
        for i in range(10):
            print('nop')
    for e in range(E):
        r_sxt = data_e[e]
        for i in range(S):
            rmin = np.min(r_sxt[i, :])
            rmax = np.max(r_sxt[i, :])
            assert rmax - rmin != 0, 'Something wrong'
            r_sxt[i, :] = (r_sxt[i, :] - rmin) / (rmax - rmin)
        data_e[e] = r_sxt
    return data_e

def spikify_data(data_e, rng, dt=1.0, max_firing_rate=100):
    if False:
        i = 10
        return i + 15
    ' Apply spikes to a continuous dataset whose values are between 0.0 and 1.0\n  Args:\n    data_e: nexamples length list of NxT trials\n    dt: how often the data are sampled\n    max_firing_rate: the firing rate that is associated with a value of 1.0\n  Returns:\n    spikified_e: a list of length b of the data represented as spikes,\n    sampled from the underlying poisson process.\n    '
    E = len(data_e)
    spikes_e = []
    for e in range(E):
        data = data_e[e]
        (N, T) = data.shape
        data_s = np.zeros([N, T]).astype(np.int)
        for n in range(N):
            f = data[n, :]
            s = rng.poisson(f * max_firing_rate * dt, size=T)
            data_s[n, :] = s
        spikes_e.append(data_s)
    return spikes_e

def gaussify_data(data_e, rng, dt=1.0, max_firing_rate=100):
    if False:
        print('Hello World!')
    ' Apply gaussian noise to a continuous dataset whose values are between\n  0.0 and 1.0\n\n  Args:\n    data_e: nexamples length list of NxT trials\n    dt: how often the data are sampled\n    max_firing_rate: the firing rate that is associated with a value of 1.0\n  Returns:\n    gauss_e: a list of length b of the data with noise.\n    '
    E = len(data_e)
    mfr = max_firing_rate
    gauss_e = []
    for e in range(E):
        data = data_e[e]
        (N, T) = data.shape
        noisy_data = data * mfr + np.random.randn(N, T) * (5.0 * mfr) * np.sqrt(dt)
        gauss_e.append(noisy_data)
    return gauss_e

def get_train_n_valid_inds(num_trials, train_fraction, nreplications):
    if False:
        return 10
    'Split the numbers between 0 and num_trials-1 into two portions for\n  training and validation, based on the train fraction.\n  Args:\n    num_trials: the number of trials\n    train_fraction: (e.g. .80)\n    nreplications: the number of spiking trials per initial condition\n  Returns:\n    a 2-tuple of two lists: the training indices and validation indices\n    '
    train_inds = []
    valid_inds = []
    for i in range(num_trials):
        if i % nreplications + 1 > train_fraction * nreplications:
            valid_inds.append(i)
        else:
            train_inds.append(i)
    return (train_inds, valid_inds)

def split_list_by_inds(data, inds1, inds2):
    if False:
        return 10
    'Take the data, a list, and split it up based on the indices in inds1 and\n  inds2.\n  Args:\n    data: the list of data to split\n    inds1, the first list of indices\n    inds2, the second list of indices\n  Returns: a 2-tuple of two lists.\n  '
    if data is None or len(data) == 0:
        return ([], [])
    else:
        dout1 = [data[i] for i in inds1]
        dout2 = [data[i] for i in inds2]
        return (dout1, dout2)

def nparray_and_transpose(data_a_b_c):
    if False:
        print('Hello World!')
    'Convert the list of items in data to a numpy array, and transpose it\n  Args:\n    data: data_asbsc: a nested, nested list of length a, with sublist length\n      b, with sublist length c.\n  Returns:\n    a numpy 3-tensor with dimensions a x c x b\n'
    data_axbxc = np.array([datum_b_c for datum_b_c in data_a_b_c])
    data_axcxb = np.transpose(data_axbxc, axes=[0, 2, 1])
    return data_axcxb

def add_alignment_projections(datasets, npcs, ntime=None, nsamples=None):
    if False:
        return 10
    'Create a matrix that aligns the datasets a bit, under\n  the assumption that each dataset is observing the same underlying dynamical\n  system.\n\n  Args:\n    datasets: The dictionary of dataset structures.\n    npcs:  The number of pcs for each, basically like lfads factors.\n    nsamples (optional): Number of samples to take for each dataset.\n    ntime (optional): Number of time steps to take in each sample.\n\n  Returns:\n    The dataset structures, with the field alignment_matrix_cxf added.\n    This is # channels x npcs dimension\n'
    nchannels_all = 0
    channel_idxs = {}
    conditions_all = {}
    nconditions_all = 0
    for (name, dataset) in datasets.items():
        cidxs = np.where(dataset['P_sxn'])[1]
        channel_idxs[name] = [cidxs[0], cidxs[-1] + 1]
        nchannels_all += cidxs[-1] + 1 - cidxs[0]
        conditions_all[name] = np.unique(dataset['condition_labels_train'])
    all_conditions_list = np.unique(np.ndarray.flatten(np.array(conditions_all.values())))
    nconditions_all = all_conditions_list.shape[0]
    if ntime is None:
        ntime = dataset['train_data'].shape[1]
    if nsamples is None:
        nsamples = dataset['train_data'].shape[0]
    avg_data_all = {}
    for (name, conditions) in conditions_all.items():
        dataset = datasets[name]
        avg_data_all[name] = {}
        for cname in conditions:
            td_idxs = np.argwhere(np.array(dataset['condition_labels_train']) == cname)
            data = np.squeeze(dataset['train_data'][td_idxs, :, :], axis=1)
            avg_data = np.mean(data, axis=0)
            avg_data_all[name][cname] = avg_data
    all_data_nxtc = np.zeros([nchannels_all, ntime * nconditions_all])
    for (name, dataset) in datasets.items():
        cidx_s = channel_idxs[name][0]
        cidx_f = channel_idxs[name][1]
        for cname in conditions_all[name]:
            cidxs = np.argwhere(all_conditions_list == cname)
            if cidxs.shape[0] > 0:
                cidx = cidxs[0][0]
                all_tidxs = np.arange(0, ntime + 1) + cidx * ntime
                all_data_nxtc[cidx_s:cidx_f, all_tidxs[0]:all_tidxs[-1]] = avg_data_all[name][cname].T
    filt_len = 6
    bc_filt = np.ones([filt_len]) / float(filt_len)
    for c in range(nchannels_all):
        all_data_nxtc[c, :] = scipy.signal.filtfilt(bc_filt, [1.0], all_data_nxtc[c, :])
    all_data_mean_nx1 = np.mean(all_data_nxtc, axis=1, keepdims=True)
    all_data_zm_nxtc = all_data_nxtc - all_data_mean_nx1
    corr_mat_nxn = np.dot(all_data_zm_nxtc, all_data_zm_nxtc.T)
    (evals_n, evecs_nxn) = np.linalg.eigh(corr_mat_nxn)
    sidxs = np.flipud(np.argsort(evals_n))
    evals_n = evals_n[sidxs]
    evecs_nxn = evecs_nxn[:, sidxs]
    all_data_pca_pxtc = np.dot(evecs_nxn[:, 0:npcs].T, all_data_zm_nxtc)
    for (name, dataset) in datasets.items():
        cidx_s = channel_idxs[name][0]
        cidx_f = channel_idxs[name][1]
        all_data_zm_chxtc = all_data_zm_nxtc[cidx_s:cidx_f, :]
        (W_chxp, _, _, _) = np.linalg.lstsq(all_data_zm_chxtc.T, all_data_pca_pxtc.T)
        dataset['alignment_matrix_cxf'] = W_chxp
        alignment_bias_cx1 = all_data_mean_nx1[cidx_s:cidx_f]
        dataset['alignment_bias_c'] = np.squeeze(alignment_bias_cx1, axis=1)
    do_debug_plot = False
    if do_debug_plot:
        pc_vecs = evecs_nxn[:, 0:npcs]
        ntoplot = 400
        plt.figure()
        plt.plot(np.log10(evals_n), '-x')
        plt.figure()
        plt.subplot(311)
        plt.imshow(all_data_pca_pxtc)
        plt.colorbar()
        plt.subplot(312)
        plt.imshow(np.dot(W_chxp.T, all_data_zm_chxtc))
        plt.colorbar()
        plt.subplot(313)
        plt.imshow(np.dot(all_data_zm_chxtc.T, W_chxp).T - all_data_pca_pxtc)
        plt.colorbar()
        import pdb
        pdb.set_trace()
    return datasets