import argparse
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, Variable
from chainer import datasets, iterators, optimizers
from chainer import Chain
from chainer.training import extensions

class ModernMLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        if False:
            print('Hello World!')
        super(ModernMLP, self).__init__(l1=L.Linear(None, n_units), l2=L.Linear(None, n_units), l3=L.Linear(None, n_out))

    def __call__(self, x):
        if False:
            while True:
                i = 10
        h = F.dropout(F.relu(self.l1(x)), ratio=0.3, train=True)
        h = F.dropout(F.relu(self.l2(h)), ratio=0.3, train=True)
        return self.l3(h)

def main():
    if False:
        return 10
    parser = argparse.ArgumentParser(description='Chainer-Tutorial: MLP')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Number of samples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of times to train on data set')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID: -1 indicates CPU')
    parser.add_argument('--frequency', '-f', type=int, default=-1, help='Frequency of taking a snapshot')
    parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
    args = parser.parse_args()
    (train, test) = chainer.datasets.get_mnist()
    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size, repeat=False, shuffle=False)
    model = L.Classifier(ModernMLP(625, 10))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
    optimizer = chainer.optimizers.RMSprop()
    optimizer.setup(model)
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    report_params = ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(report_params))
    trainer.extend(extensions.ProgressBar())
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.LogReport())
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    trainer.run()
if __name__ == '__main__':
    main()
'\nExpected output with 1 gpu.\n\nepoch       main/loss   validation/main/loss  main/accuracy validation/main/accuracy  elapsed_time\n...\n90          0.217452    0.965264              0.958189       0.941456                 294.61\n91          0.196134    1.14531               0.959089       0.944917                 297.859\n92          0.203648    0.956059              0.957148       0.943928                 301.109\n93          0.20284     1.02199               0.960021       0.948378                 304.362\n94          0.195888    1.18072               0.958905       0.945609                 307.619\n95          0.199831    1.2245                0.958356       0.94195                  310.879\n96          0.200486    1.10434               0.960186       0.943038                 314.151\n97          0.202059    1.43919               0.960421       0.943335                 317.447\n98          0.221666    0.947955              0.959305       0.946994                 320.745\n99          0.200717    1.35896               0.961504       0.943137                 324.038\n100         0.182234    0.935365              0.962039       0.946301                 327.323\n'