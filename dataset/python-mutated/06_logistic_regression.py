from torch import tensor
from torch import nn
from torch import sigmoid
import torch.nn.functional as F
import torch.optim as optim
x_data = tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = tensor([[0.0], [0.0], [1.0], [1.0]])

class Model(nn.Module):

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        In the constructor we instantiate nn.Linear module\n        '
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        if False:
            print('Hello World!')
        '\n        In the forward function we accept a Variable of input data and we must return\n        a Variable of output data.\n        '
        y_pred = sigmoid(self.linear(x))
        return y_pred
model = Model()
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.01)
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f"\nLet's predict the hours need to score above 50%\n{'=' * 50}")
hour_var = model(tensor([[1.0]]))
print(f'Prediction after 1 hour of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}')
hour_var = model(tensor([[7.0]]))
print(f'Prediction after 7 hours of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}')