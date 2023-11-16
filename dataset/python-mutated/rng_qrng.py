import numpy as np
import scipy.stats as stats
_future_warn = 'Passing `None` as the seed currently return the NumPy singleton RandomState\n(np.random.mtrand._rand). After release 0.13 this will change to using the\ndefault generator provided by NumPy (np.random.default_rng()). If you need\nreproducible draws, you should pass a seeded np.random.Generator, e.g.,\n\nimport numpy as np\nseed = 32839283923801\nrng = np.random.default_rng(seed)"\n'

def check_random_state(seed=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Turn `seed` into a random number generator.\n\n    Parameters\n    ----------\n    seed : {None, int, array_like[ints], `numpy.random.Generator`,\n            `numpy.random.RandomState`, `scipy.stats.qmc.QMCEngine`}, optional\n\n        If `seed` is None fresh, unpredictable entropy will be pulled\n        from the OS and `numpy.random.Generator` is used.\n        If `seed` is an int or ``array_like[ints]``, a new ``Generator``\n        instance is used, seeded with `seed`.\n        If `seed` is already a ``Generator``, ``RandomState`` or\n        `scipy.stats.qmc.QMCEngine` instance then\n        that instance is used.\n\n        `scipy.stats.qmc.QMCEngine` requires SciPy >=1.7. It also means\n        that the generator only have the method ``random``.\n\n    Returns\n    -------\n    seed : {`numpy.random.Generator`, `numpy.random.RandomState`,\n            `scipy.stats.qmc.QMCEngine`}\n\n        Random number generator.\n    '
    if hasattr(stats, 'qmc') and isinstance(seed, stats.qmc.QMCEngine):
        return seed
    elif isinstance(seed, np.random.RandomState):
        return seed
    elif isinstance(seed, np.random.Generator):
        return seed
    elif seed is not None:
        return np.random.default_rng(seed)
    else:
        import warnings
        warnings.warn(_future_warn, FutureWarning)
        return np.random.mtrand._rand