import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
torch.manual_seed(777)
idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [0, 1, 0, 2, 3, 3]
one_hot_lookup = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
y_data = [1, 0, 2, 3, 3, 4]
x_one_hot = [one_hot_lookup[x] for x in x_data]
inputs = Variable(torch.Tensor(x_one_hot))
labels = Variable(torch.LongTensor(y_data))
num_classes = 5
input_size = 5
hidden_size = 5
batch_size = 1
sequence_length = 1
num_layers = 1

class Model(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, hidden, x):
        if False:
            return 10
        x = x.view(batch_size, sequence_length, input_size)
        (out, hidden) = self.rnn(x, hidden)
        return (hidden, out.view(-1, num_classes))

    def init_hidden(self):
        if False:
            for i in range(10):
                print('nop')
        return Variable(torch.zeros(num_layers, batch_size, hidden_size))
model = Model()
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
for epoch in range(100):
    optimizer.zero_grad()
    loss = 0
    hidden = model.init_hidden()
    sys.stdout.write('predicted string: ')
    for (input, label) in zip(inputs, labels):
        (hidden, output) = model(hidden, input)
        (val, idx) = output.max(1)
        sys.stdout.write(idx2char[idx.data[0]])
        loss += criterion(output, torch.LongTensor([label]))
    print(', epoch: %d, loss: %1.3f' % (epoch + 1, loss))
    loss.backward()
    optimizer.step()
print('Learning finished!')