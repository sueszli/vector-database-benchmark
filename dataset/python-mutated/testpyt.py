import random
import torch

class DynamicNet(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        if False:
            while True:
                i = 10
        '\n        In the constructor we construct three nn.Linear instances that we will use\n        in the forward pass.\n        '
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        '\n        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3\n        and reuse the middle_linear Module that many times to compute hidden layer\n        representations.\n\n        Since each forward pass builds a dynamic computation graph, we can use normal\n        Python control-flow operators like loops or conditional statements when\n        defining the forward pass of the model.\n\n        Here we also see that it is perfectly safe to reuse the same Module many\n        times when defining a computational graph. This is a big improvement from Lua\n        Torch, where each Module could be used only once.\n        '
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred
(N, D_in, H, D_out) = (64, 1000, 100, 10)
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
model = DynamicNet(D_in, H, D_out)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
for t in range(500):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()