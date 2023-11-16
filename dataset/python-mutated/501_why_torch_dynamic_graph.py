"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.4
matplotlib
numpy
"""
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
INPUT_SIZE = 1
LR = 0.02

class RNN(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        if False:
            i = 10
            return i + 15
        (r_out, h_state) = self.rnn(x, h_state)
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return (torch.stack(outs, dim=1), h_state)
rnn = RNN()
print(rnn)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()
h_state = None
plt.figure(1, figsize=(12, 5))
plt.ion()
step = 0
for i in range(60):
    dynamic_steps = np.random.randint(1, 4)
    (start, end) = (step * np.pi, (step + dynamic_steps) * np.pi)
    step += dynamic_steps
    steps = np.linspace(start, end, 10 * dynamic_steps, dtype=np.float32)
    print(len(steps))
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
    (prediction, h_state) = rnn(x, h_state)
    h_state = h_state.data
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.05)
plt.ioff()
plt.show()