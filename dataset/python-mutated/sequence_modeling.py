import torch.nn as nn

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        if False:
            for i in range(10):
                print('nop')
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        if False:
            while True:
                i = 10
        '\n        input : visual feature [batch_size x T x input_size]\n        output : contextual feature [batch_size x T x output_size]\n        '
        try:
            self.rnn.flatten_parameters()
        except:
            pass
        (recurrent, _) = self.rnn(input)
        output = self.linear(recurrent)
        return output