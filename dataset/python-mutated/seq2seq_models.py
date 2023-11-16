import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
MAX_LENGTH = 100
SOS_token = chr(0)
EOS_token = 1

def cuda_variable(tensor):
    if False:
        while True:
            i = 10
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

def str2tensor(msg, eos=False):
    if False:
        for i in range(10):
            print('nop')
    tensor = [ord(c) for c in msg]
    if eos:
        tensor.append(EOS_token)
    return cuda_variable(torch.LongTensor(tensor))

class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers=1):
        if False:
            return 10
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, word_inputs, hidden):
        if False:
            for i in range(10):
                print('nop')
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        (output, hidden) = self.gru(embedded, hidden)
        return (output, hidden)

    def init_hidden(self):
        if False:
            print('Hello World!')
        return cuda_variable(torch.zeros(self.n_layers, 1, self.hidden_size))

class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, n_layers=1):
        if False:
            return 10
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        if False:
            while True:
                i = 10
        output = self.embedding(input).view(1, 1, -1)
        (output, hidden) = self.gru(output, hidden)
        output = self.out(output[0])
        return (output, hidden)

    def init_hidden(self):
        if False:
            i = 10
            return i + 15
        return cuda_variable(torch.zeros(self.n_layers, 1, self.hidden_size))

class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        if False:
            return 10
        super(AttnDecoderRNN, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, word_input, last_hidden, encoder_hiddens):
        if False:
            i = 10
            return i + 15
        rnn_input = self.embedding(word_input).view(1, 1, -1)
        (rnn_output, hidden) = self.gru(rnn_input, last_hidden)
        attn_weights = self.get_att_weight(rnn_output.squeeze(0), encoder_hiddens)
        context = attn_weights.bmm(encoder_hiddens.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        output = self.out(torch.cat((rnn_output, context), 1))
        return (output, hidden, attn_weights)

    def get_att_weight(self, hidden, encoder_hiddens):
        if False:
            while True:
                i = 10
        seq_len = len(encoder_hiddens)
        attn_scores = cuda_variable(torch.zeros(seq_len))
        for i in range(seq_len):
            attn_scores[i] = self.get_att_score(hidden, encoder_hiddens[i])
        return F.softmax(attn_scores).view(1, 1, -1)

    def get_att_score(self, hidden, encoder_hidden):
        if False:
            print('Hello World!')
        score = self.attn(encoder_hidden)
        return torch.dot(hidden.view(-1), score.view(-1))