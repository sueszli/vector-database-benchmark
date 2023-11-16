import math
import torch
from pyro.distributions import Uniform
N_SAMPLES = 100

def test_add():
    if False:
        return 10
    X = Uniform(0, 1).rv
    X = X + 1
    X = 1 + X
    X += 1
    x = X.dist.sample([N_SAMPLES])
    assert ((3 <= x) & (x <= 4)).all().item()

def test_subtract():
    if False:
        print('Hello World!')
    X = Uniform(0, 1).rv
    X = 1 - X
    X = X - 1
    X -= 1
    x = X.dist.sample([N_SAMPLES])
    assert ((-2 <= x) & (x <= -1)).all().item()

def test_multiply_divide():
    if False:
        for i in range(10):
            print('nop')
    X = Uniform(0, 1).rv
    X *= 4
    X /= 2
    x = X.dist.sample([N_SAMPLES])
    assert ((0 <= x) & (x <= 2)).all().item()

def test_abs():
    if False:
        return 10
    X = Uniform(0, 1).rv
    X = 2 * (X - 0.5)
    X = abs(X)
    x = X.dist.sample([N_SAMPLES])
    assert ((0 <= x) & (x <= 1)).all().item()

def test_neg():
    if False:
        while True:
            i = 10
    X = Uniform(0, 1).rv
    X = -X
    x = X.dist.sample([N_SAMPLES])
    assert ((-1 <= x) & (x <= 0)).all().item()

def test_pow():
    if False:
        for i in range(10):
            print('nop')
    X = Uniform(0, 1).rv
    X = X ** 2
    x = X.dist.sample([N_SAMPLES])
    assert ((0 <= x) & (x <= 1)).all().item()

def test_tensor_ops():
    if False:
        print('Hello World!')
    pi = 3.141592654
    X = Uniform(0, 1).expand([5, 5]).rv
    a = torch.tensor([[1, 2, 3, 4, 5]])
    b = a.T
    X = abs(pi * (-X + a - 3 * b))
    x = X.dist.sample()
    assert x.shape == (5, 5)
    assert (x >= 0).all().item()

def test_chaining():
    if False:
        for i in range(10):
            print('nop')
    X = Uniform(0, 1).rv.add(1).pow(2).mul(2).sub(5).tanh().exp()
    x = X.dist.sample([N_SAMPLES])
    assert ((1 / math.e <= x) & (x <= math.e)).all().item()