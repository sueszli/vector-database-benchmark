import torch.nn as nn

class LSTMCellWithZoneOut(nn.Module):
    """
    Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations
    https://arxiv.org/abs/1606.01305
    """

    def __init__(self, prob: float, input_size: int, hidden_size: int, bias: bool=True):
        if False:
            i = 10
            return i + 15
        super(LSTMCellWithZoneOut, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size, bias=bias)
        self.prob = prob
        if prob > 1.0 or prob < 0.0:
            raise ValueError('zoneout probability must be in the range from 0.0 to 1.0.')

    def zoneout(self, h, next_h, prob):
        if False:
            print('Hello World!')
        if isinstance(h, tuple):
            return tuple([self.zoneout(h[i], next_h[i], prob) for i in range(len(h))])
        if self.training:
            mask = h.new_zeros(*h.size()).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        return prob * h + (1 - prob) * next_h

    def forward(self, x, h):
        if False:
            i = 10
            return i + 15
        return self.zoneout(h, self.lstm_cell(x, h), self.prob)