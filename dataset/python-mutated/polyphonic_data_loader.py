"""
Data loader logic with two main responsibilities:
(i)  download raw data and process; this logic is initiated upon import
(ii) helper functions for dealing with mini-batches, sequence packing, etc.

Data are taken from

Boulanger-Lewandowski, N., Bengio, Y. and Vincent, P.,
"Modeling Temporal Dependencies in High-Dimensional Sequences: Application to
Polyphonic Music Generation and Transcription"

however, the original source of the data seems to be the Institut fuer Algorithmen
und Kognitive Systeme at Universitaet Karlsruhe.
"""
import os
import pickle
from collections import namedtuple
from urllib.request import urlopen
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from pyro.contrib.examples.util import get_data_directory
dset = namedtuple('dset', ['name', 'url', 'filename'])
JSB_CHORALES = dset('jsb_chorales', 'https://d2hg8soec8ck9v.cloudfront.net/datasets/polyphonic/jsb_chorales.pickle', 'jsb_chorales.pkl')
PIANO_MIDI = dset('piano_midi', 'https://d2hg8soec8ck9v.cloudfront.net/datasets/polyphonic/piano_midi.pickle', 'piano_midi.pkl')
MUSE_DATA = dset('muse_data', 'https://d2hg8soec8ck9v.cloudfront.net/datasets/polyphonic/muse_data.pickle', 'muse_data.pkl')
NOTTINGHAM = dset('nottingham', 'https://d2hg8soec8ck9v.cloudfront.net/datasets/polyphonic/nottingham.pickle', 'nottingham.pkl')

def process_data(base_path, dataset, min_note=21, note_range=88):
    if False:
        return 10
    output = os.path.join(base_path, dataset.filename)
    if os.path.exists(output):
        try:
            with open(output, 'rb') as f:
                return pickle.load(f)
        except (ValueError, UnicodeDecodeError):
            os.remove(output)
    print('processing raw data - {} ...'.format(dataset.name))
    data = pickle.load(urlopen(dataset.url))
    processed_dataset = {}
    for (split, data_split) in data.items():
        processed_dataset[split] = {}
        n_seqs = len(data_split)
        processed_dataset[split]['sequence_lengths'] = torch.zeros(n_seqs, dtype=torch.long)
        processed_dataset[split]['sequences'] = []
        for seq in range(n_seqs):
            seq_length = len(data_split[seq])
            processed_dataset[split]['sequence_lengths'][seq] = seq_length
            processed_sequence = torch.zeros((seq_length, note_range))
            for t in range(seq_length):
                note_slice = torch.tensor(list(data_split[seq][t])) - min_note
                slice_length = len(note_slice)
                if slice_length > 0:
                    processed_sequence[t, note_slice] = torch.ones(slice_length)
            processed_dataset[split]['sequences'].append(processed_sequence)
    pickle.dump(processed_dataset, open(output, 'wb'), pickle.HIGHEST_PROTOCOL)
    print('dumped processed data to %s' % output)
base_path = get_data_directory(__file__)
if not os.path.exists(base_path):
    os.mkdir(base_path)

def load_data(dataset):
    if False:
        i = 10
        return i + 15
    process_data(base_path, dataset)
    file_loc = os.path.join(base_path, dataset.filename)
    with open(file_loc, 'rb') as f:
        dset = pickle.load(f)
        for (k, v) in dset.items():
            sequences = v['sequences']
            dset[k]['sequences'] = pad_sequence(sequences, batch_first=True).type(torch.Tensor)
            dset[k]['sequence_lengths'] = v['sequence_lengths'].to(device=torch.Tensor().device)
    return dset

def reverse_sequences(mini_batch, seq_lengths):
    if False:
        for i in range(10):
            print('nop')
    reversed_mini_batch = torch.zeros_like(mini_batch)
    for b in range(mini_batch.size(0)):
        T = seq_lengths[b]
        time_slice = torch.arange(T - 1, -1, -1, device=mini_batch.device)
        reversed_sequence = torch.index_select(mini_batch[b, :, :], 0, time_slice)
        reversed_mini_batch[b, 0:T, :] = reversed_sequence
    return reversed_mini_batch

def pad_and_reverse(rnn_output, seq_lengths):
    if False:
        print('Hello World!')
    (rnn_output, _) = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    reversed_output = reverse_sequences(rnn_output, seq_lengths)
    return reversed_output

def get_mini_batch_mask(mini_batch, seq_lengths):
    if False:
        i = 10
        return i + 15
    mask = torch.zeros(mini_batch.shape[0:2])
    for b in range(mini_batch.shape[0]):
        mask[b, 0:seq_lengths[b]] = torch.ones(seq_lengths[b])
    return mask

def get_mini_batch(mini_batch_indices, sequences, seq_lengths, cuda=False):
    if False:
        for i in range(10):
            print('nop')
    seq_lengths = seq_lengths[mini_batch_indices]
    (_, sorted_seq_length_indices) = torch.sort(seq_lengths)
    sorted_seq_length_indices = sorted_seq_length_indices.flip(0)
    sorted_seq_lengths = seq_lengths[sorted_seq_length_indices]
    sorted_mini_batch_indices = mini_batch_indices[sorted_seq_length_indices]
    T_max = torch.max(seq_lengths)
    mini_batch = sequences[sorted_mini_batch_indices, 0:T_max, :]
    mini_batch_reversed = reverse_sequences(mini_batch, sorted_seq_lengths)
    mini_batch_mask = get_mini_batch_mask(mini_batch, sorted_seq_lengths)
    if cuda:
        mini_batch = mini_batch.cuda()
        mini_batch_mask = mini_batch_mask.cuda()
        mini_batch_reversed = mini_batch_reversed.cuda()
    mini_batch_reversed = nn.utils.rnn.pack_padded_sequence(mini_batch_reversed, sorted_seq_lengths, batch_first=True)
    return (mini_batch, mini_batch_reversed, mini_batch_mask, sorted_seq_lengths)