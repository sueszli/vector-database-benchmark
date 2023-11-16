from torch import nn, optim, from_numpy
import numpy as np
xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = from_numpy(xy[:, 0:-1])
y_data = from_numpy(xy[:, [-1]])
print(f"X's shape: {x_data.shape} | Y's shape: {y_data.shape}")

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
            return 10
        '\n        In the forward function we accept a Variable of input data and we must return\n        a Variable of output data. We can use Modules defined in the constructor as\n        well as arbitrary operators on Variables.\n        '
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred
model = Model()
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.1)
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()