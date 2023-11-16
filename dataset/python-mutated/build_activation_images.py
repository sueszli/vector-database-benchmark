"""
This script will generate input-out plots for all of the activation
functions. These are for use in the documentation, and potentially in
online tutorials.
"""
from pathlib import Path
import matplotlib
import torch
from matplotlib import pyplot as plt
matplotlib.use('Agg')
ACTIVATION_IMAGE_PATH = Path(__file__).parent / 'activation_images'
if not ACTIVATION_IMAGE_PATH.exists():
    ACTIVATION_IMAGE_PATH.mkdir()
functions = [torch.nn.ELU(), torch.nn.Hardshrink(), torch.nn.Hardtanh(), torch.nn.Hardsigmoid(), torch.nn.Hardswish(), torch.nn.LeakyReLU(negative_slope=0.1), torch.nn.LogSigmoid(), torch.nn.PReLU(), torch.nn.ReLU(), torch.nn.ReLU6(), torch.nn.RReLU(), torch.nn.SELU(), torch.nn.SiLU(), torch.nn.Mish(), torch.nn.CELU(), torch.nn.GELU(), torch.nn.Sigmoid(), torch.nn.Softplus(), torch.nn.Softshrink(), torch.nn.Softsign(), torch.nn.Tanh(), torch.nn.Tanhshrink()]

def plot_function(function, **args):
    if False:
        while True:
            i = 10
    '\n    Plot a function on the current plot. The additional arguments may\n    be used to specify color, alpha, etc.\n    '
    xrange = torch.arange(-7.0, 7.0, 0.01)
    plt.plot(xrange.numpy(), function(xrange).detach().numpy(), **args)
for function in functions:
    function_name = function._get_name()
    plot_path = ACTIVATION_IMAGE_PATH / f'{function_name}.png'
    if not plot_path.exists():
        plt.clf()
        plt.grid(color='k', alpha=0.2, linestyle='--')
        plot_function(function)
        plt.title(function)
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.xlim([-7, 7])
        plt.ylim([-7, 7])
        plt.savefig(plot_path)
        print(f'Saved activation image for {function_name} at {plot_path}')