"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.1.11
"""
import torch
import torch.nn.functional as F

class Net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        if False:
            print('Hello World!')
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        if False:
            print('Hello World!')
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
net1 = Net(1, 10, 1)
net2 = torch.nn.Sequential(torch.nn.Linear(1, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1))
print(net1)
'\nNet (\n  (hidden): Linear (1 -> 10)\n  (predict): Linear (10 -> 1)\n)\n'
print(net2)
'\nSequential (\n  (0): Linear (1 -> 10)\n  (1): ReLU ()\n  (2): Linear (10 -> 1)\n)\n'