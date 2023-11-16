import math
import random as random
import util
random.seed(0)

def rand(a, b, random=random.random):
    if False:
        while True:
            i = 10
    return (b - a) * random() + a

def makeMatrix(I, J, fill=0.0):
    if False:
        while True:
            i = 10
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m

class NN(object):

    def __init__(self, ni, nh, no):
        if False:
            while True:
                i = 10
        self.ni = ni + 1
        self.nh = nh
        self.no = no
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-2.0, 2.0)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if False:
            i = 10
            return i + 15
        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = 1.0 / (1.0 + math.exp(-sum))
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = 1.0 / (1.0 + math.exp(-sum))
        return self.ao[:]

    def backPropagate(self, targets, N, M):
        if False:
            while True:
                i = 10
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            ao = self.ao[k]
            output_deltas[k] = ao * (1 - ao) * (targets[k] - ao)
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            sum = 0.0
            for k in range(self.no):
                sum = sum + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = self.ah[j] * (1 - self.ah[j]) * sum
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N * change + M * self.co[j][k]
                self.co[j][k] = change
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N * change + M * self.ci[i][j]
                self.ci[i][j] = change
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        if False:
            i = 10
            return i + 15
        for p in patterns:
            print('%s -> %s' % (p[0], self.update(p[0])))

    def weights(self):
        if False:
            for i in range(10):
                print('nop')
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print('')
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=2000, N=0.5, M=0.1):
        if False:
            for i in range(10):
                print('nop')
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)

def demo():
    if False:
        while True:
            i = 10
    pat = [[[0, 0], [0]], [[0, 1], [1]], [[1, 0], [1]], [[1, 1], [0]]]
    n = NN(2, 3, 1)
    n.train(pat, 5000)

def time(fn, *args):
    if False:
        return 10
    import time, traceback
    begin = time.time()
    result = fn(*args)
    end = time.time()
    return (result, end - begin)

def test_bpnn(iterations):
    if False:
        for i in range(10):
            print('nop')
    times = []
    for _ in range(iterations):
        (result, t) = time(demo)
        times.append(t)
    return times
main = test_bpnn
if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser(usage='%prog [options]', description='Test the performance of a neural network.')
    util.add_standard_options_to(parser)
    (options, args) = parser.parse_args()
    util.run_benchmark(options, options.num_runs, test_bpnn)