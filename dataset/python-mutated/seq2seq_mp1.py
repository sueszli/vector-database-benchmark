import argparse
import math
import os.path
import pickle
import re
import sys
import time
from nltk.translate import bleu_score
import numpy
import six
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import reporter
from chainer import training
from chainer.training import extensions
import chainermn
import chainermn.functions
import chainermn.links
import europal

def cached_call(fname, func, *args):
    if False:
        for i in range(10):
            print('nop')
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)
    else:
        val = func(*args)
        with open(fname, 'wb') as f:
            pickle.dump(val, f)
        return val

def read_source(in_dir, cache=None):
    if False:
        while True:
            i = 10
    en_path = os.path.join(in_dir, 'giga-fren.release2.fixed.en')
    source_vocab = ['<eos>', '<unk>'] + europal.count_words(en_path)
    source_data = europal.make_dataset(en_path, source_vocab)
    return (source_vocab, source_data)

def read_target(in_dir, cahce=None):
    if False:
        print('Hello World!')
    fr_path = os.path.join(in_dir, 'giga-fren.release2.fixed.fr')
    target_vocab = ['<eos>', '<unk>'] + europal.count_words(fr_path)
    target_data = europal.make_dataset(fr_path, target_vocab)
    return (target_vocab, target_data)

def sequence_embed(embed, xs):
    if False:
        return 10
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0, force_tuple=True)
    return exs

class Encoder(chainer.Chain):

    def __init__(self, comm, n_layers, n_source_vocab, n_target_vocab, n_units):
        if False:
            print('Hello World!')
        super(Encoder, self).__init__(embed_x=L.EmbedID(n_source_vocab, n_units), mn_encoder=chainermn.links.create_multi_node_n_step_rnn(L.NStepLSTM(n_layers, n_units, n_units, 0.1), comm, rank_in=None, rank_out=1))
        self.comm = comm
        self.n_layers = n_layers
        self.n_units = n_units

    def __call__(self, *inputs):
        if False:
            return 10
        xs = inputs[:len(inputs) // 2]
        xs = [x[::-1] for x in xs]
        exs = sequence_embed(self.embed_x, xs)
        (_, _, _, delegate_variable) = self.mn_encoder(exs)
        return delegate_variable

    def translate(self, xs, max_length=100):
        if False:
            for i in range(10):
                print('nop')
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                xs = [x[::-1] for x in xs]
                exs = sequence_embed(self.embed_x, xs)
                self.mn_encoder(exs)
        return None

class Decoder(chainer.Chain):

    def __init__(self, comm, n_layers, n_source_vocab, n_target_vocab, n_units):
        if False:
            return 10
        super(Decoder, self).__init__(embed_y=L.EmbedID(n_target_vocab, n_units), mn_decoder=chainermn.links.create_multi_node_n_step_rnn(L.NStepLSTM(n_layers, n_units, n_units, 0.1), comm, rank_in=0, rank_out=None), W=L.Linear(n_units, n_target_vocab))
        self.comm = comm
        self.n_layers = n_layers
        self.n_units = n_units

    def __call__(self, *inputs):
        if False:
            i = 10
            return i + 15
        xs = inputs[:len(inputs) // 2]
        ys = inputs[len(inputs) // 2:]
        xs = [x[::-1] for x in xs]
        batch = len(xs)
        eos = self.xp.zeros(1, self.xp.int32)
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]
        eys = sequence_embed(self.embed_y, ys_in)
        (_, _, os, _) = self.mn_decoder(eys)
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.sum(F.softmax_cross_entropy(self.W(concat_os), concat_ys_out, reduce='no')) / batch
        reporter.report({'loss': loss.data}, self)
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.data * batch / n_words)
        reporter.report({'perp': perp}, self)
        return loss

    def translate(self, xs, max_length=100):
        if False:
            while True:
                i = 10
        batch = len(xs)
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                result = []
                ys = self.xp.zeros(batch, self.xp.int32)
                eys = self.embed_y(ys)
                eys = chainer.functions.split_axis(eys, batch, 0, force_tuple=True)
                (h, c, ys, _) = self.mn_decoder(eys)
                cys = chainer.functions.concat(ys, axis=0)
                wy = self.W(cys)
                ys = self.xp.argmax(wy.data, axis=1).astype(self.xp.int32)
                result.append(ys)
                for i in range(1, max_length):
                    eys = self.embed_y(ys)
                    eys = chainer.functions.split_axis(eys, batch, 0, force_tuple=True)
                    (h, c, ys) = self.mn_decoder.actual_rnn(h, c, eys)
                    cys = chainer.functions.concat(ys, axis=0)
                    wy = self.W(cys)
                    ys = self.xp.argmax(wy.data, axis=1).astype(self.xp.int32)
                    result.append(ys)
        result = cuda.to_cpu(self.xp.stack(result).T)
        outs = []
        for y in result:
            inds = numpy.argwhere(y == 0)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs

def convert(batch, device):
    if False:
        i = 10
        return i + 15

    def to_device_batch(batch):
        if False:
            for i in range(10):
                print('nop')
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x) for x in batch[:-1]], dtype=numpy.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev
    return tuple(to_device_batch([x for (x, _) in batch]) + to_device_batch([y for (_, y) in batch]))

class BleuEvaluator(extensions.Evaluator):

    def __init__(self, model, test_data, device=-1, batch=100, max_length=100, comm=None):
        if False:
            i = 10
            return i + 15
        super(BleuEvaluator, self).__init__({'main': None}, model)
        self.model = model
        self.test_data = test_data
        self.batch = batch
        self.device = device
        self.max_length = max_length
        self.comm = comm

    def evaluate(self):
        if False:
            print('Hello World!')
        bt = time.time()
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            observation = {}
            with reporter.report_scope(observation):
                for i in range(0, len(self.test_data), self.batch):
                    (src, trg) = zip(*self.test_data[i:i + self.batch])
                    references.extend([[t.tolist()] for t in trg])
                    src = [chainer.dataset.to_device(self.device, x) for x in src]
                    if self.comm.rank == 0:
                        self.model.translate(src, self.max_length)
                    elif self.comm.rank == 1:
                        ys = [y.tolist() for y in self.model.translate(src, self.max_length)]
                        hypotheses.extend(ys)
                if self.comm.rank == 1:
                    bleu = bleu_score.corpus_bleu(references, hypotheses, smoothing_function=bleu_score.SmoothingFunction().method1)
                    reporter.report({'bleu': bleu}, self.model)
        et = time.time()
        if self.comm.rank == 1:
            print('BleuEvaluator(single)::evaluate(): took {:.3f} [s]'.format(et - bt))
            sys.stdout.flush()
        return observation

def create_optimizer(opt_arg):
    if False:
        i = 10
        return i + 15
    'Parse a string and get an optimizer.\n\n    The syntax is:\n\n        opt(params...)\n\n    where\n        opt := sgd | adam\n        param := [float | key=val]...\n    '
    m = re.match('(adam|sgd)\\(([^)]*)\\)', opt_arg, re.I)
    name = m.group(1).lower()
    args = m.group(2)
    names_dict = {'adadelta': chainer.optimizers.AdaDelta, 'adagrad': chainer.optimizers.AdaGrad, 'adam': chainer.optimizers.Adam, 'momentumsgd': chainer.optimizers.MomentumSGD, 'nesterovag': chainer.optimizers.NesterovAG, 'rmsprop': chainer.optimizers.RMSprop, 'rmspropgraves': chainer.optimizers.RMSpropGraves, 'sgd': chainer.optimizers.SGD, 'smorms3': chainer.optimizers.SMORMS3}
    try:
        opt = names_dict[name]
    except KeyError:
        raise RuntimeError("Unknown optimizer: '{}' in '{}'".format(name, opt_arg))
    pos = []
    kw = {}
    args = args.strip()
    if args:
        for a in re.split(',\\s*', args):
            if a.find('=') >= 0:
                (key, val) = a.split('=')
                kw[key] = float(val)
            else:
                pos.append(float(a))
    return opt(*pos, **kw)

def _get_num_split(excp):
    if False:
        i = 10
        return i + 15
    'Get the preferrable number of split from a DataSizeError error'
    ps = excp.pickled_size
    mx = excp.max_size
    return (ps + mx - 1) // mx

def _slices(excp):
    if False:
        while True:
            i = 10
    'Get a list of slices that are expected to fit in a single send/recv.'
    ds = excp.dataset_size
    nsplit = _get_num_split(excp)
    size = math.ceil(ds / nsplit)
    return [(b, min(e, ds)) for (b, e) in ((i * size, (i + 1) * size) for i in range(0, nsplit))]

def main():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('--batchsize', '-b', type=int, default=64, help='Number of images in each mini-batch')
    parser.add_argument('--bleu', action='store_true', default=False, help='Report BLEU score')
    parser.add_argument('--gpu', '-g', action='store_true', help='Use GPU')
    parser.add_argument('--cache', '-c', default=None, help='Directory to cache pre-processed dataset')
    parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1024, help='Number of units')
    parser.add_argument('--communicator', default='pure_nccl', help='Type of communicator')
    parser.add_argument('--stop', '-s', type=str, default='15e', help='Stop trigger (ex. "500i", "15e")')
    parser.add_argument('--input', '-i', type=str, default='wmt', help='Input directory')
    parser.add_argument('--optimizer', type=str, default='adam()', help='Optimizer and its argument')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    args = parser.parse_args()
    if args.gpu:
        comm = chainermn.create_communicator(args.communicator)
        dev = comm.intra_rank
    else:
        comm = chainermn.create_communicator('naive')
        dev = -1
    if comm.size != 2:
        raise ValueError('This example can only be executed on exactly 2 processes.')
    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        if args.gpu:
            print('Using GPUs')
        print('Using {} communicator'.format(args.communicator))
        print('Num unit: {}'.format(args.unit))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('==========================================')
    if comm.rank == 0 or comm.rank == 1:
        if args.cache and (not os.path.exists(args.cache)):
            os.mkdir(args.cache)
        bt = time.time()
        if args.cache:
            cache_file = os.path.join(args.cache, 'source.pickle')
            (source_vocab, source_data) = cached_call(cache_file, read_source, args.input, args.cache)
        else:
            (source_vocab, source_data) = read_source(args.input, args.cache)
        et = time.time()
        print('RD source done. {:.3f} [s]'.format(et - bt))
        sys.stdout.flush()
        bt = time.time()
        if args.cache:
            cache_file = os.path.join(args.cache, 'target.pickle')
            (target_vocab, target_data) = cached_call(cache_file, read_target, args.input, args.cache)
        else:
            (target_vocab, target_data) = read_target(args.input, args.cache)
        et = time.time()
        print('RD target done. {:.3f} [s]'.format(et - bt))
        sys.stdout.flush()
        print('Original training data size: %d' % len(source_data))
        train_data = [(s, t) for (s, t) in six.moves.zip(source_data, target_data) if 0 < len(s) < 50 and 0 < len(t) < 50]
        print('Filtered training data size: %d' % len(train_data))
        en_path = os.path.join(args.input, 'dev', 'newstest2013.en')
        source_data = europal.make_dataset(en_path, source_vocab)
        fr_path = os.path.join(args.input, 'dev', 'newstest2013.fr')
        target_data = europal.make_dataset(fr_path, target_vocab)
        assert len(source_data) == len(target_data)
        test_data = [(s, t) for (s, t) in six.moves.zip(source_data, target_data) if 0 < len(s) and 0 < len(t)]
        source_ids = {word: index for (index, word) in enumerate(source_vocab)}
        target_ids = {word: index for (index, word) in enumerate(target_vocab)}
    else:
        (train_data, test_data) = (None, None)
        (target_ids, source_ids) = (None, None)
    for i in range(0, comm.size):
        if comm.rank == i:
            print('Rank {} GPU: {}'.format(comm.rank, dev))
        sys.stdout.flush()
        comm.mpi_comm.Barrier()
    source_ids = comm.bcast_obj(source_ids, root=0)
    target_ids = comm.bcast_obj(target_ids, root=0)
    target_words = {i: w for (w, i) in target_ids.items()}
    source_words = {i: w for (w, i) in source_ids.items()}
    if comm.rank == 0:
        print('target_words : {}'.format(len(target_words)))
        print('source_words : {}'.format(len(source_words)))
    n_lstm_layers = 3
    if comm.rank == 0:
        model = Encoder(comm, n_lstm_layers, len(source_ids), len(target_ids), args.unit)
    elif comm.rank == 1:
        model = Decoder(comm, n_lstm_layers, len(source_ids), len(target_ids), args.unit)
    if dev >= 0:
        chainer.cuda.get_device_from_id(dev).use()
        model.to_gpu(dev)
    m = re.match('^(\\d+)e$', args.stop)
    if m:
        trigger = (int(m.group(1)), 'epoch')
    else:
        m = re.match('^(\\d+)i$', args.stop)
        if m:
            trigger = (int(m.group(1)), 'iteration')
        else:
            if comm.rank == 0:
                sys.stderr.write('Error: unknown stop trigger: {}'.format(args.stop))
            exit(-1)
    if comm.rank == 0:
        print('Trigger: {}'.format(trigger))
    optimizer = create_optimizer(args.optimizer)
    optimizer.setup(model)
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize, shuffle=False)
    updater = training.StandardUpdater(train_iter, optimizer, converter=convert, device=dev)
    trainer = training.Trainer(updater, trigger, out=args.out)
    trainer.extend(BleuEvaluator(model, test_data, device=dev, comm=comm))

    def translate_one(source, target):
        if False:
            while True:
                i = 10
        words = europal.split_sentence(source)
        print('# source : ' + ' '.join(words))
        x = model.xp.array([source_ids.get(w, 1) for w in words], model.xp.int32)
        ys = model.translate([x])[0]
        words = [target_words[y] for y in ys]
        print('#  result : ' + ' '.join(words))
        print('#  expect : ' + target)

    def translate(trainer):
        if False:
            return 10
        translate_one('Who are we ?', 'Qui sommes-nous?')
        translate_one('And it often costs over a hundred dollars ' + 'to obtain the required identity card .', 'Or, il en coûte souvent plus de cent dollars ' + "pour obtenir la carte d'identité requise.")
        (source, target) = test_data[numpy.random.choice(len(test_data))]
        source = ' '.join([source_words.get(i, '') for i in source])
        target = ' '.join([target_words.get(i, '') for i in target])
        translate_one(source, target)
    if comm.rank == 1:
        trigger = (1, 'epoch')
        trainer.extend(extensions.LogReport(trigger=trigger), trigger=trigger)
        report = extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'main/perp', 'validation/main/bleu', 'elapsed_time'])
        trainer.extend(report, trigger=trigger)
        trainer.extend(extensions.ProgressBar(update_interval=1))
    comm.mpi_comm.Barrier()
    if comm.rank == 0:
        print('start training')
        sys.stdout.flush()
    trainer.run()
if __name__ == '__main__':
    main()