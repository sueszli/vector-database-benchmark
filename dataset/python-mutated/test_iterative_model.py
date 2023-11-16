import numpy as np
from mlxtend._base import _IterativeModel

class BlankModel(_IterativeModel):

    def __init__(self, print_progress=0, random_seed=1):
        if False:
            print('Hello World!')
        self.print_progress = print_progress
        self.random_seed = random_seed
        np.random.seed(random_seed)

def test_init():
    if False:
        while True:
            i = 10
    est = BlankModel(print_progress=0, random_seed=1)
    assert hasattr(est, 'print_progress')
    assert hasattr(est, 'random_seed')

def test_shuffle():
    if False:
        return 10
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    est = BlankModel(print_progress=0, random_seed=1)
    (X_sh, y_sh) = est._shuffle_arrays(arrays=[X, np.array(y)])
    np.testing.assert_equal(X_sh, np.array([[1], [3], [2]]))
    np.testing.assert_equal(y_sh, np.array([1, 3, 2]))

def test_init_params():
    if False:
        for i in range(10):
            print('nop')
    est = BlankModel(print_progress=0, random_seed=1)
    (b, w) = est._init_params(weights_shape=(3, 3), bias_shape=(1,), random_seed=0)
    assert b == np.array([0.0]), b
    expect_w = np.array([[0.018, 0.004, 0.01], [0.022, 0.019, -0.01], [0.01, -0.002, -0.001]])
    np.testing.assert_almost_equal(w, expect_w, decimal=3)

def test_minibatches_divisible():
    if False:
        i = 10
        return i + 15
    ary = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    est = BlankModel(print_progress=0, random_seed=1)
    rgen = np.random.RandomState(1)
    gen_arys = est._yield_minibatches_idx(rgen=rgen, n_batches=2, data_ary=ary)
    arys = list(gen_arys)
    assert (arys[0] == np.array([7, 2, 1, 6])).all()
    assert (arys[1] == np.array([0, 4, 3, 5])).all()

def test_minibatches_remainder():
    if False:
        return 10
    ary = np.array([1, 2, 3, 4, 5, 6, 7])
    est = BlankModel(print_progress=0, random_seed=1)
    rgen = np.random.RandomState(1)
    gen_arys = est._yield_minibatches_idx(rgen=rgen, n_batches=2, data_ary=ary)
    arys = list(gen_arys)
    assert len(arys) == 2
    assert (arys[0] == np.array([6, 2, 1])).all()
    assert (arys[1] == np.array([0, 4, 3, 5])).all()

def test_minibatch_1sample():
    if False:
        while True:
            i = 10
    ary = np.array([1, 2, 3, 4, 5, 6, 7])
    est = BlankModel(print_progress=0, random_seed=1)
    rgen = np.random.RandomState(1)
    gen_arys = est._yield_minibatches_idx(rgen=rgen, n_batches=7, data_ary=ary)
    arys = list(gen_arys)
    assert len(arys) == 7
    assert arys[0] == np.array([6]), arys[0]

def test_minibatch_allsample():
    if False:
        for i in range(10):
            print('nop')
    ary = np.array([1, 2, 3, 4, 5, 6, 7])
    est = BlankModel(print_progress=0, random_seed=1)
    rgen = np.random.RandomState(1)
    gen_arys = est._yield_minibatches_idx(rgen=rgen, n_batches=1, data_ary=ary, shuffle=False)
    arys = list(gen_arys)[0]
    assert (arys == np.array([0, 1, 2, 3, 4, 5, 6])).all()