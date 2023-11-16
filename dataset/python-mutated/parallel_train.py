import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad_and_value, stack_module_state, vmap
parser = argparse.ArgumentParser(description='Functorch Ensembled Models')
parser.add_argument('--device', type=str, default='cpu', help="CPU or GPU ID for this process (default: 'cpu')")
args = parser.parse_args()
DEVICE = args.device

def make_spirals(n_samples, noise_std=0.0, rotations=1.0):
    if False:
        while True:
            i = 10
    ts = torch.linspace(0, 1, n_samples, device=DEVICE)
    rs = ts ** 0.5
    thetas = rs * rotations * 2 * math.pi
    signs = torch.randint(0, 2, (n_samples,), device=DEVICE) * 2 - 1
    labels = (signs > 0).to(torch.long).to(DEVICE)
    xs = rs * signs * torch.cos(thetas) + torch.randn(n_samples, device=DEVICE) * noise_std
    ys = rs * signs * torch.sin(thetas) + torch.randn(n_samples, device=DEVICE) * noise_std
    points = torch.stack([xs, ys], dim=1)
    return (points, labels)
(points, labels) = make_spirals(100, noise_std=0.05)

class MLPClassifier(nn.Module):

    def __init__(self, hidden_dim=32, n_classes=2):
        if False:
            while True:
                i = 10
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.fc1 = nn.Linear(2, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

    def forward(self, x):
        if False:
            while True:
                i = 10
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, -1)
        return x
loss_fn = nn.NLLLoss()
model = MLPClassifier().to(DEVICE)

def train_step_fn(weights, batch, targets, lr=0.2):
    if False:
        while True:
            i = 10

    def compute_loss(weights, batch, targets):
        if False:
            while True:
                i = 10
        output = functional_call(model, weights, batch)
        loss = loss_fn(output, targets)
        return loss
    (grad_weights, loss) = grad_and_value(compute_loss)(weights, batch, targets)
    new_weights = {}
    with torch.no_grad():
        for key in grad_weights:
            new_weights[key] = weights[key] - grad_weights[key] * lr
    return (loss, new_weights)

def step4():
    if False:
        while True:
            i = 10
    global weights
    for i in range(2000):
        (loss, weights) = train_step_fn(dict(model.named_parameters()), points, labels)
        if i % 100 == 0:
            print(loss)
step4()

def init_fn(num_models):
    if False:
        i = 10
        return i + 15
    models = [MLPClassifier().to(DEVICE) for _ in range(num_models)]
    (params, _) = stack_module_state(models)
    return params

def step6():
    if False:
        print('Hello World!')
    parallel_train_step_fn = vmap(train_step_fn, in_dims=(0, None, None))
    batched_weights = init_fn(num_models=2)
    for i in range(2000):
        (loss, batched_weights) = parallel_train_step_fn(batched_weights, points, labels)
        if i % 200 == 0:
            print(loss)
step6()