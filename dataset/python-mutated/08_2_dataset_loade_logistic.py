from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim
import numpy as np

class DiabetesDataset(Dataset):
    """ Diabetes dataset."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, 0:-1])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        return (self.x_data[index], self.y_data[index])

    def __len__(self):
        if False:
            return 10
        return self.len
dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

class Model(nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        '\n        In the constructor we instantiate two nn.Linear module\n        '
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if False:
            while True:
                i = 10
        '\n        In the forward function we accept a Variable of input data and we must return\n        a Variable of output data. We can use Modules defined in the constructor as\n        well as arbitrary operators on Variables.\n        '
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred
model = Model()
criterion = nn.BCELoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=0.1)
for epoch in range(2):
    for (i, data) in enumerate(train_loader, 0):
        (inputs, labels) = data
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(f'Epoch {epoch + 1} | Batch: {i + 1} | Loss: {loss.item():.4f}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()