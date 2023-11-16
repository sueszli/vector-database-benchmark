"""
Backends in `einops` are organized to meet the following requirements
- backends are not imported unless those are actually needed, because
    - backends may not be installed
    - importing all available backends will drive to significant memory footprint
    - backends may by present but installed with errors (but never used),
      importing may drive to crashes
- backend should be either symbolic or imperative (tensorflow is for both, but that causes problems)
    - this determines which methods (from_numpy/to_numpy or create_symbol/eval_symbol) should be defined
- if backend can't (temporarily) provide symbols for shape dimensions, UnknownSize objects are used
"""
import sys
import warnings
__author__ = 'Alex Rogozhnikov, RuiYang Liu'
_backends = {}
_debug_importing = False

def get_backend(tensor) -> 'AbstractBackend':
    if False:
        i = 10
        return i + 15
    '\n    Takes a correct backend (e.g. numpy backend if tensor is numpy.ndarray) for a tensor.\n    If needed, imports package and creates backend\n    '
    for (framework_name, backend) in _backends.items():
        if backend.is_appropriate_type(tensor):
            return backend
    backend_subclasses = []
    backends = AbstractBackend.__subclasses__()
    while backends:
        backend = backends.pop()
        backends += backend.__subclasses__()
        backend_subclasses.append(backend)
    for BackendSubclass in backend_subclasses:
        if _debug_importing:
            print('Testing for subclass of ', BackendSubclass)
        if BackendSubclass.framework_name not in _backends:
            if BackendSubclass.framework_name in sys.modules:
                if _debug_importing:
                    print('Imported backend for ', BackendSubclass.framework_name)
                backend = BackendSubclass()
                _backends[backend.framework_name] = backend
                if backend.is_appropriate_type(tensor):
                    return backend
    raise RuntimeError('Tensor type unknown to einops {}'.format(type(tensor)))

class AbstractBackend:
    """ Base backend class, major part of methods are only for debugging purposes. """
    framework_name = None

    def is_appropriate_type(self, tensor):
        if False:
            i = 10
            return i + 15
        ' helper method should recognize tensors it can handle '
        raise NotImplementedError()

    def from_numpy(self, x):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError("framework doesn't support imperative execution")

    def to_numpy(self, x):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError("framework doesn't support imperative execution")

    def create_symbol(self, shape):
        if False:
            return 10
        raise NotImplementedError("framework doesn't support symbolic computations")

    def eval_symbol(self, symbol, input_dict):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError("framework doesn't support symbolic computations")

    def arange(self, start, stop):
        if False:
            while True:
                i = 10
        raise NotImplementedError("framework doesn't implement arange")

    def shape(self, x):
        if False:
            for i in range(10):
                print('nop')
        'shape should return a tuple with integers or "shape symbols" (which will evaluate to actual size)'
        return x.shape

    def reshape(self, x, shape):
        if False:
            i = 10
            return i + 15
        return x.reshape(shape)

    def transpose(self, x, axes):
        if False:
            return 10
        return x.transpose(axes)

    def reduce(self, x, operation, axes):
        if False:
            while True:
                i = 10
        return getattr(x, operation)(axis=axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def add_axis(self, x, new_position):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def add_axes(self, x, n_axes, pos2len):
        if False:
            while True:
                i = 10
        repeats = [1] * n_axes
        for (axis_position, axis_length) in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return self.tile(x, tuple(repeats))

    def tile(self, x, repeats):
        if False:
            print('Hello World!')
        'repeats is a number of  '
        raise NotImplementedError()

    def is_float_type(self, x):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def layers(self):
        if False:
            return 10
        raise NotImplementedError('backend does not provide layers')

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<einops backend for {}>'.format(self.framework_name)

    def einsum(self, pattern, *x):
        if False:
            return 10
        raise NotImplementedError('backend does not support einsum')

class UnknownSize:
    """ pseudo-symbol for symbolic frameworks which do not provide symbols for shape elements """

    def __floordiv__(self, other):
        if False:
            print('Hello World!')
        return self

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return True

    def __mul__(self, other):
        if False:
            return 10
        return self

    def __rmul__(self, other):
        if False:
            while True:
                i = 10
        return self

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return None.__hash__()

class NumpyBackend(AbstractBackend):
    framework_name = 'numpy'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        import numpy
        self.np = numpy

    def is_appropriate_type(self, tensor):
        if False:
            while True:
                i = 10
        return isinstance(tensor, self.np.ndarray)

    def from_numpy(self, x):
        if False:
            while True:
                i = 10
        return x

    def to_numpy(self, x):
        if False:
            print('Hello World!')
        return x

    def arange(self, start, stop):
        if False:
            while True:
                i = 10
        return self.np.arange(start, stop)

    def stack_on_zeroth_dimension(self, tensors: list):
        if False:
            while True:
                i = 10
        return self.np.stack(tensors)

    def tile(self, x, repeats):
        if False:
            return 10
        return self.np.tile(x, repeats)

    def is_float_type(self, x):
        if False:
            print('Hello World!')
        return x.dtype in ('float16', 'float32', 'float64', 'float128', 'bfloat16')

    def add_axis(self, x, new_position):
        if False:
            while True:
                i = 10
        return self.np.expand_dims(x, new_position)

    def einsum(self, pattern, *x):
        if False:
            i = 10
            return i + 15
        return self.np.einsum(pattern, *x)

class HashableTuple:
    """Overcomes non-hashability of symbolic elements"""

    def __init__(self, elements: tuple):
        if False:
            print('Hello World!')
        self.elements = elements

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        for x in self.elements:
            yield x

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.elements)

    def __getitem__(self, item):
        if False:
            print('Hello World!')
        return self.elements[item]

class JittorBackend(AbstractBackend):
    framework_name = 'jittor'

    def __init__(self):
        if False:
            return 10
        import jittor
        self.jittor = jittor

    def is_appropriate_type(self, tensor):
        if False:
            return 10
        return isinstance(tensor, self.jittor.Var)

    def from_numpy(self, x):
        if False:
            for i in range(10):
                print('nop')
        variable = self.jittor.array(x)
        return variable

    def to_numpy(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x.detach().numpy()

    def arange(self, start, stop):
        if False:
            return 10
        return self.jittor.arange(start, stop, dtype='int64')

    def shape(self, x):
        if False:
            return 10
        return tuple(x.shape)

    def reshape(self, x, shape):
        if False:
            return 10
        if len(shape) == 0:
            return x
        return self.jittor.reshape(x, shape)

    def reduce(self, x, operation, reduced_axes):
        if False:
            i = 10
            return i + 15
        if operation == 'prod':
            return x.prod(reduced_axes)
        for axis in sorted(reduced_axes, reverse=True):
            if operation == 'min':
                x = x.min(dim=axis)
            elif operation == 'max':
                x = x.max(dim=axis)
            elif operation in ['sum', 'mean']:
                x = getattr(x, operation)(dim=axis)
            else:
                raise NotImplementedError('Unknown reduction ', operation)
        return x

    def transpose(self, x, axes):
        if False:
            for i in range(10):
                print('nop')
        return x.permute(axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        if False:
            for i in range(10):
                print('nop')
        return self.jittor.stack(tensors)

    def add_axes(self, x, n_axes, pos2len):
        if False:
            while True:
                i = 10
        repeats = [-1] * n_axes
        for (axis_position, axis_length) in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return x.expand(repeats)

    def tile(self, x, repeats):
        if False:
            for i in range(10):
                print('nop')
        return x.repeat(repeats)

    def add_axis(self, x, new_position):
        if False:
            while True:
                i = 10
        return self.jittor.unsqueeze(x, new_position)

    def is_float_type(self, x):
        if False:
            while True:
                i = 10
        return x.dtype in ['float16', 'bfloat16', 'float32', 'float64']

    def layers(self):
        if False:
            return 10
        from jittor.einops.layers import jittor
        return jittor

    def einsum(self, pattern, *x):
        if False:
            return 10
        return self.jittor.linalg.einsum(pattern, *x)