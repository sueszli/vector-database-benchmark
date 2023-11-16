import numpy as np

class DatasetNormalizer:

    def __init__(self, dataset: np.ndarray, normalizer: str, path_lengths: int=None):
        if False:
            print('Hello World!')
        dataset = flatten(dataset, path_lengths)
        self.observation_dim = dataset['observations'].shape[1]
        self.action_dim = dataset['actions'].shape[1]
        if type(normalizer) == str:
            normalizer = eval(normalizer)
        self.normalizers = {}
        for (key, val) in dataset.items():
            try:
                self.normalizers[key] = normalizer(val)
            except:
                print(f'[ utils/normalization ] Skipping {key} | {normalizer}')

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        string = ''
        for (key, normalizer) in self.normalizers.items():
            string += f'{key}: {normalizer}]\n'
        return string

    def normalize(self, x, key):
        if False:
            while True:
                i = 10
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        if False:
            while True:
                i = 10
        return self.normalizers[key].unnormalize(x)

def flatten(dataset, path_lengths):
    if False:
        return 10
    '\n        flattens dataset of { key: [ n_episodes x max_path_lenth x dim ] }\n            to { key : [ (n_episodes * sum(path_lengths)) x dim ]}\n    '
    flattened = {}
    for (key, xs) in dataset.items():
        assert len(xs) == len(path_lengths)
        if key == 'path_lengths':
            continue
        flattened[key] = np.concatenate([x[:length] for (x, length) in zip(xs, path_lengths)], axis=0)
    return flattened

class Normalizer:
    """
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    """

    def __init__(self, X):
        if False:
            i = 10
            return i + 15
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'[ Normalizer ] dim: {self.mins.size}\n    -: {np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'

    def normalize(self, *args, **kwargs):
        if False:
            return 10
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

class GaussianNormalizer(Normalizer):
    """
        normalizes to zero mean and unit variance
    """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.means = self.X.mean(axis=0)
        self.stds = self.X.std(axis=0)
        self.z = 1

    def __repr__(self):
        if False:
            return 10
        return f'[ Normalizer ] dim: {self.mins.size}\n    means: {np.round(self.means, 2)}\n    stds: {np.round(self.z * self.stds, 2)}\n'

    def normalize(self, x):
        if False:
            for i in range(10):
                print('nop')
        return (x - self.means) / self.stds

    def unnormalize(self, x):
        if False:
            print('Hello World!')
        return x * self.stds + self.means

class CDFNormalizer(Normalizer):
    """
        makes training data uniform (over each dimension) by transforming it with marginal CDFs
    """

    def __init__(self, X):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(atleast_2d(X))
        self.dim = self.X.shape[1]
        self.cdfs = [CDFNormalizer1d(self.X[:, i]) for i in range(self.dim)]

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'[ CDFNormalizer ] dim: {self.mins.size}\n' + '    |    '.join((f'{i:3d}: {cdf}' for (i, cdf) in enumerate(self.cdfs)))

    def wrap(self, fn_name, x):
        if False:
            while True:
                i = 10
        shape = x.shape
        x = x.reshape(-1, self.dim)
        out = np.zeros_like(x)
        for (i, cdf) in enumerate(self.cdfs):
            fn = getattr(cdf, fn_name)
            out[:, i] = fn(x[:, i])
        return out.reshape(shape)

    def normalize(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self.wrap('normalize', x)

    def unnormalize(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self.wrap('unnormalize', x)

class CDFNormalizer1d:
    """
        CDF normalizer for a single dimension
    """

    def __init__(self, X):
        if False:
            return 10
        import scipy.interpolate as interpolate
        assert X.ndim == 1
        self.X = X.astype(np.float32)
        if self.X.max() == self.X.min():
            self.constant = True
        else:
            self.constant = False
            (quantiles, cumprob) = empirical_cdf(self.X)
            self.fn = interpolate.interp1d(quantiles, cumprob)
            self.inv = interpolate.interp1d(cumprob, quantiles)
            (self.xmin, self.xmax) = (quantiles.min(), quantiles.max())
            (self.ymin, self.ymax) = (cumprob.min(), cumprob.max())

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'[{np.round(self.xmin, 2):.4f}, {np.round(self.xmax, 2):.4f}'

    def normalize(self, x):
        if False:
            return 10
        if self.constant:
            return x
        x = np.clip(x, self.xmin, self.xmax)
        y = self.fn(x)
        y = 2 * y - 1
        return y

    def unnormalize(self, x, eps=0.0001):
        if False:
            while True:
                i = 10
        '\n             X : [ -1, 1 ]\n        '
        if self.constant:
            return x
        x = (x + 1) / 2.0
        if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
            print(f'[ dataset/normalization ] Warning: out of range in unnormalize: [{x.min()}, {x.max()}] | x : [{self.xmin}, {self.xmax}] | y: [{self.ymin}, {self.ymax}]')
        x = np.clip(x, self.ymin, self.ymax)
        y = self.inv(x)
        return y

def empirical_cdf(sample):
    if False:
        print('Hello World!')
    (quantiles, counts) = np.unique(sample, return_counts=True)
    cumprob = np.cumsum(counts).astype(np.double) / sample.size
    return (quantiles, cumprob)

def atleast_2d(x):
    if False:
        i = 10
        return i + 15
    if x.ndim < 2:
        x = x[:, None]
    return x

class LimitsNormalizer(Normalizer):
    """
        maps [ xmin, xmax ] to [ -1, 1 ]
    """

    def normalize(self, x):
        if False:
            return 10
        x = (x - self.mins) / (self.maxs - self.mins)
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=0.0001):
        if False:
            for i in range(10):
                print('nop')
        '\n            x : [ -1, 1 ]\n        '
        if x.max() > 1 + eps or x.min() < -1 - eps:
            x = np.clip(x, -1, 1)
        x = (x + 1) / 2.0
        return x * (self.maxs - self.mins) + self.mins