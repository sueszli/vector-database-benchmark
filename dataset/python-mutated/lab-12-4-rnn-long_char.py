import torch
import torch.nn as nn
from torch.autograd import Variable
torch.manual_seed(777)
sentence = "if you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea."
char_set = list(set(sentence))
char_dic = {w: i for (i, w) in enumerate(char_set)}
learning_rate = 0.1
num_epochs = 500
input_size = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
sequence_length = 10
num_layers = 2
dataX = []
dataY = []
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1:i + sequence_length + 1]
    print(i, x_str, '->', y_str)
    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]
    dataX.append(x)
    dataY.append(y)
batch_size = len(dataX)
x_data = torch.Tensor(dataX)
y_data = torch.LongTensor(dataY)

def one_hot(x, num_classes):
    if False:
        while True:
            i = 10
    idx = x.long()
    idx = idx.view(-1, 1)
    x_one_hot = torch.zeros(x.size()[0] * x.size()[1], num_classes)
    x_one_hot.scatter_(1, idx, 1)
    x_one_hot = x_one_hot.view(x.size()[0], x.size()[1], num_classes)
    return x_one_hot
x_one_hot = one_hot(x_data, num_classes)
inputs = Variable(x_one_hot)
labels = Variable(y_data)

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        if False:
            i = 10
            return i + 15
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if False:
            while True:
                i = 10
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        (out, _) = self.lstm(x, (h_0, c_0))
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        return out
lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    outputs = lstm(inputs)
    optimizer.zero_grad()
    loss = criterion(outputs, labels.view(-1))
    loss.backward()
    optimizer.step()
    (_, idx) = outputs.max(1)
    idx = idx.data.numpy()
    idx = idx.reshape(-1, sequence_length)
    result_str = [char_set[c] for c in idx[-1]]
    print('epoch: %d, loss: %1.3f' % (epoch + 1, loss.data[0]))
    print('Predicted string: ', ''.join(result_str))
print('Learning finished!')