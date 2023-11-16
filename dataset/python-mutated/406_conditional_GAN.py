"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.4
numpy
matplotlib
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
BATCH_SIZE = 64
LR_G = 0.0001
LR_D = 0.0001
N_IDEAS = 5
ART_COMPONENTS = 15
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

def artist_works_with_labels():
    if False:
        return 10
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
    labels = a - 1 > 0.5
    paintings = torch.from_numpy(paintings).float()
    labels = torch.from_numpy(labels.astype(np.float32))
    return (paintings, labels)
G = nn.Sequential(nn.Linear(N_IDEAS + 1, 128), nn.ReLU(), nn.Linear(128, ART_COMPONENTS))
D = nn.Sequential(nn.Linear(ART_COMPONENTS + 1, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
plt.ion()
for step in range(10000):
    (artist_paintings, labels) = artist_works_with_labels()
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)
    G_inputs = torch.cat((G_ideas, labels), 1)
    G_paintings = G(G_inputs)
    D_inputs0 = torch.cat((artist_paintings, labels), 1)
    D_inputs1 = torch.cat((G_paintings, labels), 1)
    prob_artist0 = D(D_inputs0)
    prob_artist1 = D(D_inputs1)
    D_score0 = torch.log(prob_artist0)
    D_score1 = torch.log(1.0 - prob_artist1)
    D_loss = -torch.mean(D_score0 + D_score1)
    G_loss = torch.mean(D_score1)
    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)
    opt_D.step()
    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()
    if step % 200 == 0:
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting')
        bound = [0, 0.5] if labels.data[0, 0] == 0 else [0.5, 1]
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + bound[1], c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + bound[0], c='#FF9359', lw=3, label='lower bound')
        plt.text(-0.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
        plt.text(-0.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.text(-0.5, 1.7, 'Class = %i' % int(labels.data[0, 0]), fontdict={'size': 13})
        plt.ylim((0, 3))
        plt.legend(loc='upper right', fontsize=10)
        plt.draw()
        plt.pause(0.1)
plt.ioff()
plt.show()
z = torch.randn(1, N_IDEAS)
label = torch.FloatTensor([[1.0]])
G_inputs = torch.cat((z, label), 1)
G_paintings = G(G_inputs)
plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='G painting for upper class')
plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + bound[1], c='#74BCFF', lw=3, label='upper bound (class 1)')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + bound[0], c='#FF9359', lw=3, label='lower bound (class 1)')
plt.ylim((0, 3))
plt.legend(loc='upper right', fontsize=10)
plt.show()