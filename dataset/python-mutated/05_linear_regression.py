from torch import nn
import torch
from torch import tensor
x_data = tensor([[1.0], [2.0], [3.0]])
y_data = tensor([[2.0], [4.0], [6.0]])

class Model(nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        '\n        In the constructor we instantiate two nn.Linear module\n        '
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        '\n        In the forward function we accept a Variable of input data and we must return\n        a Variable of output data. We can use Modules defined in the constructor as\n        well as arbitrary operators on Variables.\n        '
        y_pred = self.linear(x)
        return y_pred
model = Model()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(500):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
hour_var = tensor([[4.0]])
y_pred = model(hour_var)
print('Prediction (after training)', 4, model(hour_var).data[0][0].item())