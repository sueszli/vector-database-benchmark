"""
==========================
Per-sample-gradients
==========================

What is it?
--------------------------------------------------------------------
Per-sample-gradient computation is computing the gradient for each and every
sample in a batch of data. It is a useful quantity in differential privacy
and optimization research.
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
            return 10
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

def loss_fn(predictions, targets):
    if False:
        print('Hello World!')
    return F.nll_loss(predictions, targets)
device = 'cuda'
num_models = 10
batch_size = 64
data = torch.randn(batch_size, 1, 28, 28, device=device)
targets = torch.randint(10, (64,), device=device)
model = SimpleCNN().to(device=device)
predictions = model(data)
loss = loss_fn(predictions, targets)
loss.backward()

def compute_grad(sample, target):
    if False:
        print('Hello World!')
    sample = sample.unsqueeze(0)
    target = target.unsqueeze(0)
    prediction = model(sample)
    loss = loss_fn(prediction, target)
    return torch.autograd.grad(loss, list(model.parameters()))

def compute_sample_grads(data, targets):
    if False:
        i = 10
        return i + 15
    sample_grads = [compute_grad(data[i], targets[i]) for i in range(batch_size)]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads
per_sample_grads = compute_sample_grads(data, targets)
print(per_sample_grads[0].shape)
from functorch import grad, make_functional_with_buffers, vmap
(fmodel, params, buffers) = make_functional_with_buffers(model)

def compute_loss(params, buffers, sample, target):
    if False:
        i = 10
        return i + 15
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)
    predictions = fmodel(params, buffers, batch)
    loss = loss_fn(predictions, targets)
    return loss
ft_compute_grad = grad(compute_loss)
ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, targets)
for (per_sample_grad, ft_per_sample_grad) in zip(per_sample_grads, ft_per_sample_grads):
    assert torch.allclose(per_sample_grad, ft_per_sample_grad, atol=1e-06, rtol=1e-06)