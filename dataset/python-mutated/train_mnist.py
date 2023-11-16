"""MNIST example with static subgraph optimizations.

This is a version of the Chainer MNIST example that has been modified
to support the static subgraph optimizations feature. Note that
the code is mostly unchanged except for the addition of the
`@static_graph` decorator to the model chain's `__call__()` method.

Note for contributors:
This example code is referred to from the documentation.
If this file is to be modified, please also update the line numbers in
`docs/source/reference/static_graph.rst` accordingly.
"""
from __future__ import print_function
import argparse
import warnings
import numpy
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import static_code
from chainer import static_graph
from chainer import training
from chainer.training import extensions
import chainerx
import matplotlib
matplotlib.use('Agg')

class MLP(chainer.Chain):
    """A fully-connected neural network for digit classification.

    """

    def __init__(self, n_units, n_out):
        if False:
            while True:
                i = 10
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_units)
            self.l3 = L.Linear(None, n_out)

    @static_graph
    def __call__(self, x):
        if False:
            return 10
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

class MLPSideEffect(chainer.Chain):
    """An example of a model with side-effects.

    This uses the same network as ``MLP`` except that it includes an
    example of side-effect code.
    """

    def __init__(self, n_units, n_out):
        if False:
            i = 10
            return i + 15
        super(MLPSideEffect, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_units)
            self.l3 = L.Linear(None, n_out)
        self.side_effect_counter = 0

    @static_code
    def example_side_effect(self):
        if False:
            return 10
        self.side_effect_counter += 1
        if self.side_effect_counter % 1000 == 0:
            print('Side effect counter: ', self.side_effect_counter)

    @static_graph
    def __call__(self, x):
        if False:
            print('Hello World!')
        self.example_side_effect()
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1, help='Frequency of taking a snapshot')
    parser.add_argument('--device', '-d', type=str, default='-1', help='Device specifier. Either ChainerX device specifier or an integer. If non-negative integer, CuPy arrays with specified device id are used. If negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    parser.add_argument('--model', '-m', default='MLP', help='Choose the model: MLP or MLPSideEffect')
    parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000, help='Number of units')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device', type=int, nargs='?', const=0, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    if chainer.get_dtype() == numpy.float16:
        warnings.warn('This example may cause NaN in FP16 mode.', RuntimeWarning)
    device = chainer.get_device(args.device)
    print('Device: {}'.format(device))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')
    device.use()
    if args.model == 'MLP':
        model = L.Classifier(MLP(args.unit, 10))
    elif args.model == 'MLPSideEffect':
        model = L.Classifier(MLPSideEffect(args.unit, 10))
    model.to_device(device)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    (train, test) = chainer.datasets.get_mnist()
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)
    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(test_iter, model, device=device))
    if device.xp is not chainerx:
        trainer.extend(extensions.DumpGraph('main/loss'))
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    if device.xp is not chainerx:
        trainer.run()
    else:
        warnings.warn('Static subgraph optimization does not support ChainerX and will be disabled.', UserWarning)
        with chainer.using_config('use_static_graph', False):
            trainer.run()
if __name__ == '__main__':
    main()