"""Sample script of recurrent neural network language model.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

Note for contributors:
This example code is referred to from the "RNN Language Models" tutorial.
If this file is to be modified, please also update the line numbers in
`docs/source/examples/ptb.rst` accordingly.

"""
from __future__ import division
import argparse
import sys
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainerx

class RNNForLM(chainer.Chain):

    def __init__(self, n_vocab, n_units):
        if False:
            i = 10
            return i + 15
        super(RNNForLM, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.l1 = L.LSTM(n_units, n_units)
            self.l2 = L.LSTM(n_units, n_units)
            self.l3 = L.Linear(n_units, n_vocab)
        for param in self.params():
            param.array[...] = np.random.uniform(-0.1, 0.1, param.shape)

    def reset_state(self):
        if False:
            for i in range(10):
                print('nop')
        self.l1.reset_state()
        self.l2.reset_state()

    def forward(self, x):
        if False:
            print('Hello World!')
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0))
        h2 = self.l2(F.dropout(h1))
        y = self.l3(F.dropout(h2))
        return y

class ParallelSequentialIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, repeat=True):
        if False:
            for i in range(10):
                print('nop')
        super(ParallelSequentialIterator, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.repeat = repeat
        length = len(dataset)
        self.offsets = [i * length // batch_size for i in range(batch_size)]
        self.reset()

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.epoch = 0
        self.is_new_epoch = False
        self.iteration = 0
        self._previous_epoch_detail = -1.0

    def __next__(self):
        if False:
            return 10
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            raise StopIteration
        cur_words = self.get_words()
        self._previous_epoch_detail = self.epoch_detail
        self.iteration += 1
        next_words = self.get_words()
        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch
        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        if False:
            i = 10
            return i + 15
        return self.iteration * self.batch_size / len(self.dataset)

    @property
    def previous_epoch_detail(self):
        if False:
            while True:
                i = 10
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def get_words(self):
        if False:
            i = 10
            return i + 15
        return [self.dataset[(offset + self.iteration) % len(self.dataset)] for offset in self.offsets]

    def serialize(self, serializer):
        if False:
            print('Hello World!')
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)
        try:
            self._previous_epoch_detail = serializer('previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            self._previous_epoch_detail = self.epoch + (self.current_position - self.batch_size) / len(self.dataset)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(self._previous_epoch_detail, 0.0)
            else:
                self._previous_epoch_detail = -1.0

class BPTTUpdater(training.updaters.StandardUpdater):

    def __init__(self, train_iter, optimizer, bprop_len, device):
        if False:
            print('Hello World!')
        super(BPTTUpdater, self).__init__(train_iter, optimizer, device=device)
        self.bprop_len = bprop_len

    def update_core(self):
        if False:
            return 10
        loss = 0
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        for i in range(self.bprop_len):
            batch = train_iter.__next__()
            (x, t) = self.converter(batch, self.device)
            loss += optimizer.target(x, t)
        optimizer.target.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

def compute_perplexity(result):
    if False:
        i = 10
        return i + 15
    result['perplexity'] = np.exp(result['main/loss'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=20, help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=35, help='Number of words in each mini-batch (= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=39, help='Number of sweeps over the dataset to train')
    parser.add_argument('--device', '-d', type=str, default='-1', help='Device specifier. Either ChainerX device specifier or an integer. If non-negative integer, CuPy arrays with specified device id are used. If negative integer, NumPy arrays are used')
    parser.add_argument('--gradclip', '-c', type=float, default=5, help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    parser.add_argument('--resume', '-r', type=str, help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true', help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=650, help='Number of LSTM units in each layer')
    parser.add_argument('--model', '-m', default='model.npz', help='Model file name to serialize')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device', type=int, nargs='?', const=0, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    device = chainer.get_device(args.device)
    if device.xp is chainerx:
        sys.stderr.write('This example does not support ChainerX devices.\n')
        sys.exit(1)
    device.use()
    (train, val, test) = chainer.datasets.get_ptb_words()
    n_vocab = max(train) + 1
    print('#vocab = {}'.format(n_vocab))
    if args.test:
        train = train[:100]
        val = val[:100]
        test = test[:100]
    train_iter = ParallelSequentialIterator(train, args.batchsize)
    val_iter = ParallelSequentialIterator(val, 1, repeat=False)
    test_iter = ParallelSequentialIterator(test, 1, repeat=False)
    rnn = RNNForLM(n_vocab, args.unit)
    model = L.Classifier(rnn)
    model.compute_accuracy = False
    model.to_device(device)
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(args.gradclip))
    updater = BPTTUpdater(train_iter, optimizer, args.bproplen, device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    eval_model = model.copy()
    eval_rnn = eval_model.predictor
    trainer.extend(extensions.Evaluator(val_iter, eval_model, device=device, eval_hook=lambda _: eval_rnn.reset_state()))
    interval = 10 if args.test else 500
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity, trigger=(interval, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'perplexity', 'val_perplexity']), trigger=(interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=1 if args.test else 10))
    trainer.extend(extensions.snapshot())
    trainer.extend(extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'))
    if args.resume is not None:
        chainer.serializers.load_npz(args.resume, trainer)
    trainer.run()
    print('test')
    eval_rnn.reset_state()
    evaluator = extensions.Evaluator(test_iter, eval_model, device=device)
    result = evaluator()
    print('test perplexity: {}'.format(np.exp(float(result['main/loss']))))
    chainer.serializers.save_npz(args.model, model)
if __name__ == '__main__':
    main()