import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from text_loader import TextDataset
hidden_size = 100
n_layers = 3
batch_size = 1
n_epochs = 100
n_characters = 128

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        if False:
            print('Hello World!')
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        if False:
            while True:
                i = 10
        embed = self.embedding(input.view(1, -1))
        embed = embed.view(1, 1, -1)
        (output, hidden) = self.gru(embed, hidden)
        output = self.linear(output.view(1, -1))
        return (output, hidden)

    def init_hidden(self):
        if False:
            for i in range(10):
                print('nop')
        if torch.cuda.is_available():
            hidden = torch.zeros(self.n_layers, 1, self.hidden_size).cuda()
        else:
            hidden = torch.zeros(self.n_layers, 1, self.hidden_size)
        return Variable(hidden)

def str2tensor(string):
    if False:
        while True:
            i = 10
    tensor = [ord(c) for c in string]
    tensor = torch.LongTensor(tensor)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)

def generate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    if False:
        print('Hello World!')
    hidden = decoder.init_hidden()
    prime_input = str2tensor(prime_str)
    predicted = prime_str
    for p in range(len(prime_str) - 1):
        (_, hidden) = decoder(prime_input[p], hidden)
    inp = prime_input[-1]
    for p in range(predict_len):
        (output, hidden) = decoder(inp, hidden)
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        predicted_char = chr(top_i)
        predicted += predicted_char
        inp = str2tensor(predicted_char)
    return predicted

def train_teacher_forching(line):
    if False:
        print('Hello World!')
    input = str2tensor(line[:-1])
    target = str2tensor(line[1:])
    hidden = decoder.init_hidden()
    loss = 0
    for c in range(len(input)):
        (output, hidden) = decoder(input[c], hidden)
        loss += criterion(output, target[c])
    decoder.zero_grad()
    loss.backward()
    decoder_optimizer.step()
    return loss.data[0] / len(input)

def train(line):
    if False:
        print('Hello World!')
    input = str2tensor(line[:-1])
    target = str2tensor(line[1:])
    hidden = decoder.init_hidden()
    decoder_in = input[0]
    loss = 0
    for c in range(len(input)):
        (output, hidden) = decoder(decoder_in, hidden)
        loss += criterion(output, target[c])
        decoder_in = output.max(1)[1]
    decoder.zero_grad()
    loss.backward()
    decoder_optimizer.step()
    return loss.data[0] / len(input)
if __name__ == '__main__':
    decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
    if torch.cuda.is_available():
        decoder.cuda()
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset=TextDataset(), batch_size=batch_size, shuffle=True)
    print('Training for %d epochs...' % n_epochs)
    for epoch in range(1, n_epochs + 1):
        for (i, (lines, _)) in enumerate(train_loader):
            loss = train(lines[0])
            if i % 100 == 0:
                print('[(%d %d%%) loss: %.4f]' % (epoch, epoch / n_epochs * 100, loss))
                print(generate(decoder, 'Wh', 100), '\n')