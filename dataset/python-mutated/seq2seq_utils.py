"""
Utils for seq2seq models.
"""
from collections import Counter
import random
import json
import torch
import stanza.models.common.seq2seq_constant as constant

def get_optimizer(name, parameters, lr):
    if False:
        for i in range(10):
            print('nop')
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr)
    elif name == 'adam':
        return torch.optim.Adam(parameters)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters)
    else:
        raise Exception('Unsupported optimizer: {}'.format(name))

def change_lr(optimizer, new_lr):
    if False:
        return 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def flatten_indices(seq_lens, width):
    if False:
        while True:
            i = 10
    flat = []
    for (i, l) in enumerate(seq_lens):
        for j in range(l):
            flat.append(i * width + j)
    return flat

def keep_partial_grad(grad, topk):
    if False:
        i = 10
        return i + 15
    '\n    Keep only the topk rows of grads.\n    '
    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad

def save_config(config, path, verbose=True):
    if False:
        print('Hello World!')
    with open(path, 'w') as outfile:
        json.dump(config, outfile, indent=2)
    if verbose:
        print('Config saved to file {}'.format(path))
    return config

def load_config(path, verbose=True):
    if False:
        i = 10
        return i + 15
    with open(path) as f:
        config = json.load(f)
    if verbose:
        print('Config loaded from file {}'.format(path))
    return config

def unmap_with_copy(indices, src_tokens, vocab):
    if False:
        print('Hello World!')
    '\n    Unmap a list of list of indices, by optionally copying from src_tokens.\n    '
    result = []
    for (ind, tokens) in zip(indices, src_tokens):
        words = []
        for idx in ind:
            if idx >= 0:
                words.append(vocab.id2word[idx])
            else:
                idx = -idx - 1
                words.append(tokens[idx])
        result += [words]
    return result

def prune_decoded_seqs(seqs):
    if False:
        while True:
            i = 10
    '\n    Prune decoded sequences after EOS token.\n    '
    out = []
    for s in seqs:
        if constant.EOS in s:
            idx = s.index(constant.EOS_TOKEN)
            out += [s[:idx]]
        else:
            out += [s]
    return out

def prune_hyp(hyp):
    if False:
        return 10
    '\n    Prune a decoded hypothesis\n    '
    if constant.EOS_ID in hyp:
        idx = hyp.index(constant.EOS_ID)
        return hyp[:idx]
    else:
        return hyp

def prune(data_list, lens):
    if False:
        return 10
    assert len(data_list) == len(lens)
    nl = []
    for (d, l) in zip(data_list, lens):
        nl.append(d[:l])
    return nl

def sort(packed, ref, reverse=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sort a series of packed list, according to a ref list.\n    Also return the original index before the sort.\n    '
    assert (isinstance(packed, tuple) or isinstance(packed, list)) and isinstance(ref, list)
    packed = [ref] + [range(len(ref))] + list(packed)
    sorted_packed = [list(t) for t in zip(*sorted(zip(*packed), reverse=reverse))]
    return tuple(sorted_packed[1:])

def unsort(sorted_list, oidx):
    if False:
        for i in range(10):
            print('nop')
    '\n    Unsort a sorted list, based on the original idx.\n    '
    assert len(sorted_list) == len(oidx), 'Number of list elements must match with original indices.'
    (_, unsorted) = [list(t) for t in zip(*sorted(zip(oidx, sorted_list)))]
    return unsorted