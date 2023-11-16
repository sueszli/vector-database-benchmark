import unittest
import os, sys
import jittor as jt
import numpy as np
from jittor import nn
from jittor import dataset
mpi = jt.compile_extern.mpi

class Model(nn.Module):

    def __init__(self, input_size):
        if False:
            return 10
        self.linear1 = nn.Linear(input_size, 10)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(10, 10)

    def execute(self, x):
        if False:
            i = 10
            return i + 15
        x = self.linear1(x)
        x = self.relu1(x)
        return self.linear2(x)

def fork_with_mpi(num_procs=4):
    if False:
        while True:
            i = 10
    import sys
    if jt.in_mpi:
        if jt.rank != 0:
            sys.stdout = open('/dev/null', 'w')
        return
    else:
        print(sys.argv)
        cmd = ' '.join(['mpirun', '-np', str(num_procs), sys.executable] + sys.argv)
        print('[RUN CMD]:', cmd)
        os.system(cmd)
        exit(0)

def main():
    if False:
        for i in range(10):
            print('nop')
    mnist = dataset.MNIST()
    model = Model(mnist[0][0].size)
    sgd = jt.optim.SGD(model.parameters(), 0.001)
    fork_with_mpi()
    for (data, label) in mnist:
        pred = model(data.reshape(data.shape[0], -1))
        loss = nn.cross_entropy_loss(pred, label)
        sgd.step(loss)
        print(jt.rank, mnist.epoch_id, mnist.batch_id, loss)
if __name__ == '__main__':
    main()