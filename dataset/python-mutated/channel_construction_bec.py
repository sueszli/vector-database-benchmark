import numpy as np
from . import helper_functions as hf

def bec_channel(eta):
    if False:
        print('Hello World!')
    '\n    binary erasure channel (BEC)\n    for each y e Y\n    W(y|0) * W(y|1) = 0 or W(y|0) = W(y|1)\n    transitions are 1 -> 1 or 0 -> 0 or {0, 1} -> ? (erased symbol)\n    '
    w = np.array((1 - eta, eta, 1 - eta), dtype=float)
    return w

def odd_rec(iwn):
    if False:
        for i in range(10):
            print('nop')
    return iwn ** 2

def even_rec(iwn):
    if False:
        return 10
    return 2 * iwn - iwn ** 2

def calc_one_recursion(iw0):
    if False:
        for i in range(10):
            print('nop')
    iw1 = np.zeros(2 * len(iw0))
    for i in range(len(iw0)):
        iw1[2 * i] = odd_rec(iw0[i])
        iw1[2 * i + 1] = even_rec(iw0[i])
    return iw1

def calculate_bec_channel_capacities_loop(initial_channel, block_power):
    if False:
        while True:
            i = 10
    iw = np.array([initial_channel], dtype=float)
    for i in range(block_power):
        iw = calc_one_recursion(iw)
    return iw

def calc_vector_capacities_one_recursion(iw0):
    if False:
        print('Hello World!')
    degraded = odd_rec(iw0)
    upgraded = even_rec(iw0)
    iw1 = np.empty(2 * len(iw0), dtype=degraded.dtype)
    iw1[0::2] = degraded
    iw1[1::2] = upgraded
    return iw1

def calculate_bec_channel_capacities_vector(initial_channel, block_power):
    if False:
        i = 10
        return i + 15
    iw = np.array([initial_channel], dtype=float)
    for i in range(block_power):
        iw = calc_vector_capacities_one_recursion(iw)
    return iw

def calculate_bec_channel_capacities(eta, block_size):
    if False:
        while True:
            i = 10
    iw = 1 - eta
    lw = hf.power_of_2_int(block_size)
    return calculate_bec_channel_capacities_vector(iw, lw)

def calculate_z_parameters_one_recursion(z_params):
    if False:
        print('Hello World!')
    z_next = np.empty(2 * z_params.size, dtype=z_params.dtype)
    z_sq = z_params ** 2
    z_low = 2 * z_params - z_sq
    z_next[0::2] = z_low
    z_next[1::2] = z_sq
    return z_next

def calculate_bec_channel_z_parameters(eta, block_size):
    if False:
        i = 10
        return i + 15
    block_power = hf.power_of_2_int(block_size)
    z_params = np.array([eta], dtype=float)
    for block_size in range(block_power):
        z_params = calculate_z_parameters_one_recursion(z_params)
    return z_params

def design_snr_to_bec_eta(design_snr):
    if False:
        for i in range(10):
            print('nop')
    s = 10.0 ** (design_snr / 10.0)
    return np.exp(-s)

def bhattacharyya_bounds(design_snr, block_size):
    if False:
        for i in range(10):
            print('nop')
    "\n    Harish Vangala, Emanuele Viterbo, Yi Hong: 'A Comparative Study of Polar Code Constructions for the AWGN Channel', 2015\n    In this paper it is called Bhattacharyya bounds channel construction and is abbreviated PCC-0\n    Best design SNR for block_size = 2048, R = 0.5, is 0dB.\n    Compare with Arikan: 'Channel Polarization: A Method for Constructing Capacity-Achieving Codes for Symmetric Binary-Input Memoryless Channels.\n    Proposition 5. inequalities turn into equalities for BEC channel. Otherwise they represent an upper bound.\n    Also compare [0, Arikan] eq. 6 and 38\n    For BEC that translates to capacity(i) = 1 - bhattacharyya(i)\n    :return Z-parameters in natural bit-order. Choose according to desired rate.\n    "
    eta = design_snr_to_bec_eta(design_snr)
    return calculate_bec_channel_z_parameters(eta, block_size)

def plot_channel_capacities(capacity, save_file=None):
    if False:
        for i in range(10):
            print('nop')
    block_size = len(capacity)
    try:
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('figure', autolayout=True)
        plt.plot(capacity)
        plt.xlim([0, block_size])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('synthetic channel number')
        plt.ylabel('channel capacity')
        plt.grid()
        plt.gcf().set_size_inches(plt.gcf().get_size_inches() * 0.5)
        if save_file:
            plt.savefig(save_file)
        plt.show()
    except ImportError:
        pass

def plot_average_channel_distance(save_file=None):
    if False:
        while True:
            i = 10
    eta = 0.5
    powers = np.arange(4, 26)
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('figure', autolayout=True)
        dist = []
        medians = []
        initial_channel = 1 - eta
        for p in powers:
            bs = int(2 ** p)
            capacities = calculate_bec_channel_capacities(eta, bs)
            avg_capacity = np.repeat(initial_channel, len(capacities))
            averages = np.abs(capacities - avg_capacity)
            avg_distance = np.sum(averages) / float(len(capacities))
            dist.append(avg_distance)
            variance = np.std(averages)
            medians.append(variance)
        plt.errorbar(powers, dist, yerr=medians)
        plt.grid()
        plt.xlabel('block size $N$')
        plt.ylabel('$\\frac{1}{N} \\sum_i |I(W_N^{(i)}) - 0.5|$')
        axes = plt.axes()
        tick_values = np.array(axes.get_xticks().tolist())
        tick_labels = np.array(tick_values, dtype=int)
        tick_labels = ['$2^{' + str(i) + '}$' for i in tick_labels]
        plt.xticks(tick_values, tick_labels)
        plt.xlim((powers[0], powers[-1]))
        plt.ylim((0.2, 0.5001))
        plt.gcf().set_size_inches(plt.gcf().get_size_inches() * 0.5)
        if save_file:
            plt.savefig(save_file)
        plt.show()
    except ImportError:
        pass

def plot_capacity_histogram(design_snr, save_file=None):
    if False:
        for i in range(10):
            print('nop')
    eta = design_snr_to_bec_eta(design_snr)
    try:
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('figure', autolayout=True)
        block_sizes = [32, 128, 512]
        for b in block_sizes:
            capacities = calculate_bec_channel_capacities(eta, b)
            w = 1.0 / float(len(capacities))
            weights = [w] * b
            plt.hist(capacities, bins=b, weights=weights, range=(0.95, 1.0))
        plt.grid()
        plt.xlabel('synthetic channel capacity')
        plt.ylabel('normalized item count')
        print(plt.gcf().get_size_inches())
        plt.gcf().set_size_inches(plt.gcf().get_size_inches() * 0.5)
        if save_file:
            plt.savefig(save_file)
        plt.show()
    except ImportError:
        pass

def main():
    if False:
        return 10
    print('channel construction main')
    n = 11
    block_size = int(2 ** n)
    design_snr = -1.59
    eta = design_snr_to_bec_eta(design_snr)
    calculate_bec_channel_z_parameters(eta, block_size)
if __name__ == '__main__':
    main()