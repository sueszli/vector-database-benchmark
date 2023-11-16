import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from name_dataset import NameDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
HIDDEN_SIZE = 100
N_CHARS = 128
N_CLASSES = 18

class RNNClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        if False:
            print('Hello World!')
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        if False:
            while True:
                i = 10
        batch_size = input.size(0)
        input = input.t()
        print('  input', input.size())
        embedded = self.embedding(input)
        print('  embedding', embedded.size())
        hidden = self._init_hidden(batch_size)
        (output, hidden) = self.gru(embedded, hidden)
        print('  gru hidden output', hidden.size())
        fc_output = self.fc(hidden)
        print('  fc output', fc_output.size())
        return fc_output

    def _init_hidden(self, batch_size):
        if False:
            i = 10
            return i + 15
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return Variable(hidden)

def str2ascii_arr(msg):
    if False:
        return 10
    arr = [ord(c) for c in msg]
    return (arr, len(arr))

def pad_sequences(vectorized_seqs, seq_lengths):
    if False:
        return 10
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for (idx, (seq, seq_len)) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    return seq_tensor

def make_variables(names):
    if False:
        return 10
    sequence_and_length = [str2ascii_arr(name) for name in names]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    return pad_sequences(vectorized_seqs, seq_lengths)
if __name__ == '__main__':
    names = ['adylov', 'solan', 'hard', 'san']
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_CLASSES)
    for name in names:
        (arr, _) = str2ascii_arr(name)
        inp = Variable(torch.LongTensor([arr]))
        out = classifier(inp)
        print('in', inp.size(), 'out', out.size())
    inputs = make_variables(names)
    out = classifier(inputs)
    print('batch in', inputs.size(), 'batch out', out.size())