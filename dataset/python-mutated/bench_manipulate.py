from .common import Benchmark, get_squares_, TYPES1, DLPACK_TYPES
import numpy as np
from collections import deque

class BroadcastArrays(Benchmark):
    params = [[(16, 32), (128, 256), (512, 1024)], TYPES1]
    param_names = ['shape', 'ndtype']
    timeout = 10

    def setup(self, shape, ndtype):
        if False:
            print('Hello World!')
        self.xarg = np.random.ranf(shape[0] * shape[1]).reshape(shape)
        self.xarg = self.xarg.astype(ndtype)
        if ndtype.startswith('complex'):
            self.xarg += np.random.ranf(1) * 1j

    def time_broadcast_arrays(self, shape, ndtype):
        if False:
            for i in range(10):
                print('nop')
        np.broadcast_arrays(self.xarg, np.ones(1))

class BroadcastArraysTo(Benchmark):
    params = [[16, 64, 512], TYPES1]
    param_names = ['size', 'ndtype']
    timeout = 10

    def setup(self, size, ndtype):
        if False:
            print('Hello World!')
        self.rng = np.random.default_rng()
        self.xarg = self.rng.random(size)
        self.xarg = self.xarg.astype(ndtype)
        if ndtype.startswith('complex'):
            self.xarg += self.rng.random(1) * 1j

    def time_broadcast_to(self, size, ndtype):
        if False:
            return 10
        np.broadcast_to(self.xarg, (size, size))

class ConcatenateStackArrays(Benchmark):
    params = [[(16, 32), (32, 64)], [2, 5], TYPES1]
    param_names = ['shape', 'narrays', 'ndtype']
    timeout = 10

    def setup(self, shape, narrays, ndtype):
        if False:
            i = 10
            return i + 15
        self.xarg = [np.random.ranf(shape[0] * shape[1]).reshape(shape) for x in range(narrays)]
        self.xarg = [x.astype(ndtype) for x in self.xarg]
        if ndtype.startswith('complex'):
            [x + np.random.ranf(1) * 1j for x in self.xarg]

    def time_concatenate_ax0(self, size, narrays, ndtype):
        if False:
            i = 10
            return i + 15
        np.concatenate(self.xarg, axis=0)

    def time_concatenate_ax1(self, size, narrays, ndtype):
        if False:
            while True:
                i = 10
        np.concatenate(self.xarg, axis=1)

    def time_stack_ax0(self, size, narrays, ndtype):
        if False:
            return 10
        np.stack(self.xarg, axis=0)

    def time_stack_ax1(self, size, narrays, ndtype):
        if False:
            print('Hello World!')
        np.stack(self.xarg, axis=1)

class ConcatenateNestedArrays(ConcatenateStackArrays):
    params = [[(1, 1)], [1000, 100000], TYPES1]

class DimsManipulations(Benchmark):
    params = [[(2, 1, 4), (2, 1), (5, 2, 3, 1)]]
    param_names = ['shape']
    timeout = 10

    def setup(self, shape):
        if False:
            i = 10
            return i + 15
        self.xarg = np.ones(shape=shape)
        self.reshaped = deque(shape)
        self.reshaped.rotate(1)
        self.reshaped = tuple(self.reshaped)

    def time_expand_dims(self, shape):
        if False:
            while True:
                i = 10
        np.expand_dims(self.xarg, axis=1)

    def time_expand_dims_neg(self, shape):
        if False:
            for i in range(10):
                print('nop')
        np.expand_dims(self.xarg, axis=-1)

    def time_squeeze_dims(self, shape):
        if False:
            for i in range(10):
                print('nop')
        np.squeeze(self.xarg)

    def time_flip_all(self, shape):
        if False:
            i = 10
            return i + 15
        np.flip(self.xarg, axis=None)

    def time_flip_one(self, shape):
        if False:
            print('Hello World!')
        np.flip(self.xarg, axis=1)

    def time_flip_neg(self, shape):
        if False:
            while True:
                i = 10
        np.flip(self.xarg, axis=-1)

    def time_moveaxis(self, shape):
        if False:
            for i in range(10):
                print('nop')
        np.moveaxis(self.xarg, [0, 1], [-1, -2])

    def time_roll(self, shape):
        if False:
            print('Hello World!')
        np.roll(self.xarg, 3)

    def time_reshape(self, shape):
        if False:
            i = 10
            return i + 15
        np.reshape(self.xarg, self.reshaped)