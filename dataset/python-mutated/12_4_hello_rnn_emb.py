import torch
import torch.nn as nn
from torch.autograd import Variable
torch.manual_seed(777)
idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [[0, 1, 0, 2, 3, 3]]
y_data = [1, 0, 2, 3, 3, 4]
inputs = Variable(torch.LongTensor(x_data))
labels = Variable(torch.LongTensor(y_data))
num_classes = 5
input_size = 5
embedding_size = 10
hidden_size = 5
batch_size = 1
sequence_length = 6
num_layers = 1

class Model(nn.Module):

    def __init__(self, num_layers, hidden_size):
        if False:
            i = 10
            return i + 15
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=5, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if False:
            return 10
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        emb = self.embedding(x)
        emb = emb.view(batch_size, sequence_length, -1)
        (out, _) = self.rnn(emb, h_0)
        return self.fc(out.view(-1, num_classes))
model = Model(num_layers, hidden_size)
print(model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
for epoch in range(100):
    outputs = model(inputs)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    (_, idx) = outputs.max(1)
    idx = idx.data.numpy()
    result_str = [idx2char[c] for c in idx.squeeze()]
    print('epoch: %d, loss: %1.3f' % (epoch + 1, loss.item()))
    print('Predicted string: ', ''.join(result_str))
print('Learning finished!')