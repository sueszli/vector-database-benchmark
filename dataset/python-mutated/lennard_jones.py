import torch
from torch import nn
from torch.func import jacrev, vmap
from torch.nn.functional import mse_loss
sigma = 0.5
epsilon = 4.0

def lennard_jones(r):
    if False:
        i = 10
        return i + 15
    return epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

def lennard_jones_force(r):
    if False:
        return 10
    'Get magnitude of LJ force'
    return -epsilon * (-12 * sigma ** 12 / r ** 13 + 6 * sigma ** 6 / r ** 7)
training_size = 1000
r = torch.linspace(0.5, 2 * sigma, steps=training_size, requires_grad=True)
drs = torch.outer(r, torch.tensor([1.0, 0, 0]))
norms = torch.norm(drs, dim=1).reshape(-1, 1)
training_energies = torch.stack(list(map(lennard_jones, norms))).reshape(-1, 1)
training_forces = torch.stack([force * dr for (force, dr) in zip(map(lennard_jones_force, norms), drs)])
model = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1))

def make_prediction(model, drs):
    if False:
        for i in range(10):
            print('nop')
    norms = torch.norm(drs, dim=1).reshape(-1, 1)
    energies = model(norms)
    network_derivs = vmap(jacrev(model))(norms).squeeze(-1)
    forces = -network_derivs * drs / norms
    return (energies, forces)

def loss_fn(energies, forces, predicted_energies, predicted_forces):
    if False:
        for i in range(10):
            print('nop')
    return mse_loss(energies, predicted_energies) + 0.01 * mse_loss(forces, predicted_forces) / 3
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(400):
    optimiser.zero_grad()
    (energies, forces) = make_prediction(model, drs)
    loss = loss_fn(training_energies, training_forces, energies, forces)
    loss.backward(retain_graph=True)
    optimiser.step()
    if epoch % 20 == 0:
        print(loss.cpu().item())