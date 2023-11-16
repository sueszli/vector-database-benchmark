"""Recurrent neural network language model with static graph optimizations.

This is a modified version of the standard Chainer Penn Tree Bank (ptb)
example that
includes static subgraph optimizations. It is mostly unchanged
from the original model except that that the RNN is unrolled for `bproplen`
slices inside of a static chain.

This was required because the `LSTM` link used by the ptb example
is not fully compatible with the static subgraph
optimizations feature. Specifically, it does not support
multiple calls in the same iteration unless it is called from
inside a single static chain.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

This code is a custom loop version of train_ptb.py. That is, we train
models without using the Trainer class in chainer and instead write a
training loop that manually computes the loss of minibatches and
applies an optimizer to update the model.
"""
from __future__ import print_function
import argparse
import numpy as np
import random
import sys
import chainer
from chainer import configuration
from chainer.dataset import convert
import chainer.functions as F
from chainer.functions.loss import softmax_cross_entropy
import chainer.links as L
from chainer import serializers
from chainer import static_graph
import chainerx

class RNNForLMSlice(chainer.Chain):

    def __init__(self, n_vocab, n_units):
        if False:
            while True:
                i = 10
        super(RNNForLMSlice, self).__init__()
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

    def __call__(self, x):
        if False:
            for i in range(10):
                print('nop')
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0))
        h2 = self.l2(F.dropout(h1))
        y = self.l3(F.dropout(h2))
        return y

class RNNForLMUnrolled(chainer.Chain):

    def __init__(self, n_vocab, n_units):
        if False:
            while True:
                i = 10
        super(RNNForLMUnrolled, self).__init__()
        with self.init_scope():
            self.rnn = RNNForLMSlice(n_vocab, n_units)

    @static_graph(verbosity_level=1)
    def __call__(self, words):
        if False:
            return 10
        'Perform a forward pass on the supplied list of words.\n\n        The RNN is unrolled for a number of time slices equal to the\n        length of the supplied word sequence.\n\n        Args:\n            words_labels (list of Variable): The list of input words to the\n                unrolled neural network.\n\n        Returns the corresponding lest of output variables of the same\n        length as the input sequence.\n        '
        outputs = []
        for ind in range(len(words)):
            word = words[ind]
            y = self.rnn(word)
            outputs.append(y)
        return outputs

class ParallelSequentialIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, repeat=True):
        if False:
            i = 10
            return i + 15
        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = 0
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        self.offsets = [i * length // batch_size for i in range(batch_size)]
        self.iteration = 0
        self._previous_epoch_detail = -1.0

    def __next__(self):
        if False:
            i = 10
            return i + 15
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
            while True:
                i = 10
        return self.iteration * self.batch_size / len(self.dataset)

    @property
    def previous_epoch_detail(self):
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
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

def main():
    if False:
        print('Hello World!')
    np.random.seed(0)
    random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=20, help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=25, help='Number of words in each mini-batch (= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=39, help='Number of sweeps over the dataset to train')
    parser.add_argument('--device', '-d', type=str, default='0', help='Device specifier. Either ChainerX device specifier or an integer. If non-negative integer, CuPy arrays with specified device id are used. If negative integer, NumPy arrays are used')
    parser.add_argument('--gradclip', '-c', type=float, default=5, help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true', help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=650, help='Number of LSTM units in each layer')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device', type=int, nargs='?', const=0, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    device = chainer.get_device(args.device)
    if device.xp is chainerx:
        sys.stderr.write('This example does not support ChainerX devices.\n')
        sys.exit(1)
    device.use()

    def evaluate(model, iter):
        if False:
            while True:
                i = 10
        evaluator = model.copy()
        evaluator.rnn.reset_state()
        sum_perp = 0
        data_count = 0
        words = []
        labels = []
        lossfun = softmax_cross_entropy.softmax_cross_entropy
        with configuration.using_config('train', False):
            iter.reset()
            for batch in iter:
                (word, label) = convert.concat_examples(batch, device)
                words.append(word)
                labels.append(label)
                data_count += 1
            outputs = evaluator(words)
            for ind in range(len(outputs)):
                y = outputs[ind]
                label = labels[ind]
                loss = lossfun(y, label)
                sum_perp += loss.array
        return np.exp(float(sum_perp) / data_count)
    (train, val, test) = chainer.datasets.get_ptb_words()
    n_vocab = max(train) + 1
    print('#vocab =', n_vocab)
    if args.test:
        train = train[:100]
        val = val[:100]
        test = test[:100]
    train_iter = ParallelSequentialIterator(train, args.batchsize)
    val_iter = ParallelSequentialIterator(val, 1, repeat=False)
    test_iter = ParallelSequentialIterator(test, 1, repeat=False)
    model = RNNForLMUnrolled(n_vocab, args.unit)
    lossfun = softmax_cross_entropy.softmax_cross_entropy
    model.to_device(device)
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    sum_perp = 0
    count = 0
    iteration = 0
    while train_iter.epoch < args.epoch:
        iteration += 1
        words = []
        labels = []
        for i in range(args.bproplen):
            batch = train_iter.__next__()
            (word, label) = convert.concat_examples(batch, device)
            words.append(word)
            labels.append(label)
            count += 1
        outputs = model(words)
        loss = 0
        for ind in range(len(outputs)):
            y = outputs[ind]
            label = labels[ind]
            loss += lossfun(y, label)
        sum_perp += loss.array
        optimizer.target.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
        if iteration % 20 == 0:
            print('iteration: ', iteration)
            print('training perplexity: ', np.exp(float(sum_perp) / count))
            sum_perp = 0
            count = 0
        if train_iter.is_new_epoch:
            print('Evaluating model on validation set...')
            print('epoch: ', train_iter.epoch)
            print('validation perplexity: ', evaluate(model, val_iter))
    print('test')
    test_perp = evaluate(model, test_iter)
    print('test perplexity:', test_perp)
    print('save the model')
    serializers.save_npz('rnnlm.model', model)
    print('save the optimizer')
    serializers.save_npz('rnnlm.state', optimizer)
if __name__ == '__main__':
    main()