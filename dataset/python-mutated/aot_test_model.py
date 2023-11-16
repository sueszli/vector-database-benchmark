import torch
from torch import nn

class NeuralNetwork(nn.Module):

    def forward(self, x):
        if False:
            return 10
        return torch.add(x, 10)
model = NeuralNetwork()
script = torch.jit.script(model)
torch.jit.save(script, 'aot_test_model.pt')