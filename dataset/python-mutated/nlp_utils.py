import collections
import io
import numpy
import chainer

def split_text(text, char_based=False):
    if False:
        return 10
    if char_based:
        return list(text)
    else:
        return text.split()

def normalize_text(text):
    if False:
        i = 10
        return i + 15
    return text.strip().lower()

def make_vocab(dataset, max_vocab_size=20000, min_freq=2):
    if False:
        i = 10
        return i + 15
    counts = collections.defaultdict(int)
    for (tokens, _) in dataset:
        for token in tokens:
            counts[token] += 1
    vocab = {'<eos>': 0, '<unk>': 1}
    for (w, c) in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if len(vocab) >= max_vocab_size or c < min_freq:
            break
        vocab[w] = len(vocab)
    return vocab

def read_vocab_list(path, max_vocab_size=20000):
    if False:
        print('Hello World!')
    vocab = {'<eos>': 0, '<unk>': 1}
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        for l in f:
            w = l.strip()
            if w not in vocab and w:
                vocab[w] = len(vocab)
            if len(vocab) >= max_vocab_size:
                break
    return vocab

def make_array(tokens, vocab, add_eos=True):
    if False:
        print('Hello World!')
    unk_id = vocab['<unk>']
    eos_id = vocab['<eos>']
    ids = [vocab.get(token, unk_id) for token in tokens]
    if add_eos:
        ids.append(eos_id)
    return numpy.array(ids, numpy.int32)

def transform_to_array(dataset, vocab, with_label=True):
    if False:
        return 10
    if with_label:
        return [(make_array(tokens, vocab), numpy.array([cls], numpy.int32)) for (tokens, cls) in dataset]
    else:
        return [make_array(tokens, vocab) for tokens in dataset]

@chainer.dataset.converter()
def convert_seq(batch, device=None, with_label=True):
    if False:
        print('Hello World!')

    def to_device_batch(batch):
        if False:
            for i in range(10):
                print('nop')
        if device is None:
            return batch
        src_xp = chainer.backend.get_array_module(*batch)
        xp = device.xp
        concat = src_xp.concatenate(batch, axis=0)
        sections = numpy.cumsum([len(x) for x in batch[:-1]], dtype=numpy.int32)
        concat_dev = chainer.dataset.to_device(device, concat)
        batch_dev = xp.split(concat_dev, sections)
        return batch_dev
    if with_label:
        return {'xs': to_device_batch([x for (x, _) in batch]), 'ys': to_device_batch([y for (_, y) in batch])}
    else:
        return to_device_batch([x for x in batch])