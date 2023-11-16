import argparse
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, Variable
from chainer import datasets, iterators, optimizers
from chainer import Chain
from chainer.training import extensions

class MLP(chainer.Chain):

    def __init__(self, n_unit, n_out):
        if False:
            i = 10
            return i + 15
        super(MLP, self).__init__(l1=L.Linear(None, n_unit), l2=L.Linear(None, n_out))

    def __call__(self, x):
        if False:
            print('Hello World!')
        h = F.sigmoid(self.l1(x))
        return self.l2(h)

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Chainer-Tutorial: MLP')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Number of samples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of times to train on data set')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID: -1 indicates CPU')
    args = parser.parse_args()
    (train, test) = chainer.datasets.get_mnist()
    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size, repeat=False, shuffle=False)
    model = L.Classifier(MLP(625, 10))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
    optimizer = chainer.optimizers.SGD()
    optimizer.setup(model)
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    report_params = ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(report_params))
    trainer.extend(extensions.ProgressBar())
    trainer.run()
if __name__ == '__main__':
    main()
'\n# Expected output with 1 gpu.\n\nepoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time\n...\n90          0.285929    0.277819              0.918177       0.921084                  198.132\n91          0.285159    0.277838              0.918077       0.920194                  200.3\n92          0.28487     0.277246              0.918453       0.9196                    202.472\n93          0.284443    0.276658              0.918643       0.920589                  204.645\n94          0.283882    0.276925              0.918877       0.920985                  206.818\n95          0.283553    0.276153              0.91906        0.920688                  209.031\n96          0.283272    0.275503              0.919071       0.921282                  211.219\n97          0.282494    0.274468              0.91941        0.921084                  213.428\n98          0.282246    0.274534              0.91936        0.921381                  215.617\n99          0.2818      0.274671              0.919543       0.921875                  217.821\n100         0.281342    0.27406               0.919772       0.922567                  220.023\n'