import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from name_dataset import NameDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
HIDDEN_SIZE = 100
N_LAYERS = 2
BATCH_SIZE = 256
N_EPOCHS = 100
test_dataset = NameDataset(is_train_set=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_dataset = NameDataset(is_train_set=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
N_COUNTRIES = len(train_dataset.get_countries())
print(N_COUNTRIES, 'countries')
N_CHARS = 128

def time_since(since):
    if False:
        return 10
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def create_variable(tensor):
    if False:
        i = 10
        return i + 15
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

def pad_sequences(vectorized_seqs, seq_lengths, countries):
    if False:
        i = 10
        return i + 15
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for (idx, (seq, seq_len)) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    (seq_lengths, perm_idx) = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    target = countries2tensor(countries)
    if len(countries):
        target = target[perm_idx]
    return (create_variable(seq_tensor), create_variable(seq_lengths), create_variable(target))

def make_variables(names, countries):
    if False:
        print('Hello World!')
    sequence_and_length = [str2ascii_arr(name) for name in names]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    return pad_sequences(vectorized_seqs, seq_lengths, countries)

def str2ascii_arr(msg):
    if False:
        print('Hello World!')
    arr = [ord(c) for c in msg]
    return (arr, len(arr))

def countries2tensor(countries):
    if False:
        for i in range(10):
            print('nop')
    country_ids = [train_dataset.get_country_id(country) for country in countries]
    return torch.LongTensor(country_ids)

class RNNClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        if False:
            while True:
                i = 10
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, seq_lengths):
        if False:
            return 10
        input = input.t()
        batch_size = input.size(1)
        hidden = self._init_hidden(batch_size)
        embedded = self.embedding(input)
        gru_input = pack_padded_sequence(embedded, seq_lengths.data.cpu().numpy())
        self.gru.flatten_parameters()
        (output, hidden) = self.gru(gru_input, hidden)
        fc_output = self.fc(hidden[-1])
        return fc_output

    def _init_hidden(self, batch_size):
        if False:
            i = 10
            return i + 15
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return create_variable(hidden)

def train():
    if False:
        return 10
    total_loss = 0
    for (i, (names, countries)) in enumerate(train_loader, 1):
        (input, seq_lengths, target) = make_variables(names, countries)
        output = classifier(input, seq_lengths)
        loss = criterion(output, target)
        total_loss += loss.data[0]
        classifier.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(time_since(start), epoch, i * len(names), len(train_loader.dataset), 100.0 * i * len(names) / len(train_loader.dataset), total_loss / i * len(names)))
    return total_loss

def test(name=None):
    if False:
        for i in range(10):
            print('nop')
    if name:
        (input, seq_lengths, target) = make_variables([name], [])
        output = classifier(input, seq_lengths)
        pred = output.data.max(1, keepdim=True)[1]
        country_id = pred.cpu().numpy()[0][0]
        print(name, 'is', train_dataset.get_country(country_id))
        return
    print('evaluating trained model ...')
    correct = 0
    train_data_size = len(test_loader.dataset)
    for (names, countries) in test_loader:
        (input, seq_lengths, target) = make_variables(names, countries)
        output = classifier(input, seq_lengths)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, train_data_size, 100.0 * correct / train_data_size))
if __name__ == '__main__':
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRIES, N_LAYERS)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), 'GPUs!')
        classifier = nn.DataParallel(classifier)
    if torch.cuda.is_available():
        classifier.cuda()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    start = time.time()
    print('Training for %d epochs...' % N_EPOCHS)
    for epoch in range(1, N_EPOCHS + 1):
        train()
        test()
        test('Sung')
        test('Jungwoo')
        test('Soojin')
        test('Nako')