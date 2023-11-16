import warnings
from chainer import configuration

def nondeterministic(f_name):
    if False:
        while True:
            i = 10
    'Function to warn non-deterministic functions\n\n    If `config.warn_nondeterministic` is True, this function will give a\n    warning that this functions contains a non-deterministic function, such\n    as atomicAdd.\n    '
    if configuration.config.warn_nondeterministic:
        warnings.warn('Potentially non-deterministic code is being executed while config.warn_nondeterministic set. Source: ' + f_name)