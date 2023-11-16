import random
import torch
import math

class DynamicNet(torch.nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        '\n        In the constructor we instantiate five parameters and assign them as members.\n        '
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        if False:
            return 10
        '\n        For the forward pass of the model, we randomly choose either 4, 5\n        and reuse the e parameter to compute the contribution of these orders.\n\n        Since each forward pass builds a dynamic computation graph, we can use normal\n        Python control-flow operators like loops or conditional statements when\n        defining the forward pass of the model.\n\n        Here we also see that it is perfectly safe to reuse the same parameter many\n        times when defining a computational graph.\n        '
        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x ** exp
        return y

    def string(self):
        if False:
            return 10
        '\n        Just like any class in Python, you can also define custom method on PyTorch modules\n        '
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4 ? + {self.e.item()} x^5 ?'
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)
model = DynamicNet()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-08, momentum=0.9)
for t in range(30000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    if t % 2000 == 1999:
        print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f'Result: {model.string()}')