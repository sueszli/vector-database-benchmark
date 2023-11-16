from prml.nn.function import Function

class Divide(Function):
    enable_auto_broadcast = True

    @staticmethod
    def _forward(x, y):
        if False:
            while True:
                i = 10
        return x / y

    @staticmethod
    def _backward(delta, x, y):
        if False:
            print('Hello World!')
        dx = delta / y
        dy = -delta * x / y ** 2
        return (dx, dy)

def divide(x, y):
    if False:
        i = 10
        return i + 15
    return Divide().forward(x, y)

def rdivide(x, y):
    if False:
        while True:
            i = 10
    return Divide().forward(y, x)