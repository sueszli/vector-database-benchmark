from prml.nn.function import Function

class Negative(Function):

    @staticmethod
    def _forward(x):
        if False:
            for i in range(10):
                print('nop')
        return -x

    @staticmethod
    def _backward(delta, x):
        if False:
            i = 10
            return i + 15
        return -delta

def negative(x):
    if False:
        print('Hello World!')
    return Negative().forward(x)