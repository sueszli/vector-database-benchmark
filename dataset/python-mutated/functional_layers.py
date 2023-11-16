from ...tensor import linalg, manipulation, math
from ..layer.layers import Layer
__all__ = []

class FloatFunctionalLayer(Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()

class add(FloatFunctionalLayer):

    def __init__(self):
        if False:
            return 10
        super().__init__()

    def forward(self, x, y, name=None):
        if False:
            return 10
        return math.add(x, y, name)

class subtract(FloatFunctionalLayer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def forward(self, x, y, name=None):
        if False:
            for i in range(10):
                print('nop')
        return math.subtract(x, y, name)

class multiply(FloatFunctionalLayer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def forward(self, x, y, name=None):
        if False:
            i = 10
            return i + 15
        return math.multiply(x, y, name)

class divide(FloatFunctionalLayer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def forward(self, x, y, name=None):
        if False:
            print('Hello World!')
        return math.divide(x, y, name)

class reshape(FloatFunctionalLayer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()

    def forward(self, x, shape, name=None):
        if False:
            for i in range(10):
                print('nop')
        return manipulation.reshape(x, shape, name)

class transpose(FloatFunctionalLayer):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    def forward(self, x, perm, name=None):
        if False:
            while True:
                i = 10
        return manipulation.transpose(x, perm, name)

class concat(FloatFunctionalLayer):

    def __init__(self):
        if False:
            return 10
        super().__init__()

    def forward(self, x, axis=0, name=None):
        if False:
            print('Hello World!')
        return manipulation.concat(x, axis, name)

class flatten(FloatFunctionalLayer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def forward(self, x, start_axis=0, stop_axis=-1, name=None):
        if False:
            return 10
        return manipulation.flatten(x, start_axis, stop_axis, name)

class matmul(FloatFunctionalLayer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()

    def forward(self, x, y, transpose_x=False, transpose_y=False, name=None):
        if False:
            for i in range(10):
                print('nop')
        return linalg.matmul(x, y, transpose_x, transpose_y, name)