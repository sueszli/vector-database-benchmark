"""
==========================
Model ensembling
==========================
This example illustrates how to vectorize model ensembling using vmap.

What is model ensembling?
--------------------------------------------------------------------
Model ensembling combines the predictions from multiple models together.
Traditionally this is done by running each model on some inputs separately
and then combining the predictions. However, if you're running models with
the same architecture, then it may be possible to combine them together
using ``vmap``. ``vmap`` is a function transform that maps functions across
dimensions of the input tensors. One of its use cases is eliminating
for-loops and speeding them up through vectorization.

Let's demonstrate how to do this using an ensemble of simple CNNs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

class SimpleCNN(nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        output = x
        return output
device = 'cuda'
num_models = 10
data = torch.randn(100, 64, 1, 28, 28, device=device)
targets = torch.randint(10, (6400,), device=device)
models = [SimpleCNN().to(device) for _ in range(num_models)]
minibatches = data[:num_models]
predictions1 = [model(minibatch) for (model, minibatch) in zip(models, minibatches)]
minibatch = data[0]
predictions2 = [model(minibatch) for model in models]
from functorch import combine_state_for_ensemble
(fmodel, params, buffers) = combine_state_for_ensemble(models)
[p.requires_grad_() for p in params]
print([p.size(0) for p in params])
assert minibatches.shape == (num_models, 64, 1, 28, 28)
from functorch import vmap
predictions1_vmap = vmap(fmodel)(params, buffers, minibatches)
assert torch.allclose(predictions1_vmap, torch.stack(predictions1), atol=1e-06, rtol=1e-06)
predictions2_vmap = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, minibatch)
assert torch.allclose(predictions2_vmap, torch.stack(predictions2), atol=1e-06, rtol=1e-06)