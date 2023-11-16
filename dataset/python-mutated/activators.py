import numpy as np

class ReluActivator(object):

    def forward(self, weighted_input):
        if False:
            return 10
        return max(0, weighted_input)

    def backward(self, output):
        if False:
            for i in range(10):
                print('nop')
        return 1 if output > 0 else 0

class IdentityActivator(object):

    def forward(self, weighted_input):
        if False:
            for i in range(10):
                print('nop')
        return weighted_input

    def backward(self, output):
        if False:
            for i in range(10):
                print('nop')
        return 1

class SigmoidActivator(object):

    def forward(self, weighted_input):
        if False:
            for i in range(10):
                print('nop')
        return np.longfloat(1.0 / (1.0 + np.exp(-weighted_input)))

    def backward(self, output):
        if False:
            i = 10
            return i + 15
        return output * (1 - output)

class TanhActivator(object):

    def forward(self, weighted_input):
        if False:
            print('Hello World!')
        return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0

    def backward(self, output):
        if False:
            while True:
                i = 10
        return 1 - output * output