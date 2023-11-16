"""
=======================
Animated 3D random walk
=======================

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
np.random.seed(19680801)

def random_walk(num_steps, max_step=0.05):
    if False:
        print('Hello World!')
    'Return a 3D random walk as (num_steps, 3) array.'
    start_pos = np.random.random(3)
    steps = np.random.uniform(-max_step, max_step, size=(num_steps, 3))
    walk = start_pos + np.cumsum(steps, axis=0)
    return walk

def update_lines(num, walks, lines):
    if False:
        while True:
            i = 10
    for (line, walk) in zip(lines, walks):
        line.set_data(walk[:num, :2].T)
        line.set_3d_properties(walk[:num, 2])
    return lines
num_steps = 30
walks = [random_walk(num_steps) for index in range(40)]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
lines = [ax.plot([], [], [])[0] for _ in walks]
ax.set(xlim3d=(0, 1), xlabel='X')
ax.set(ylim3d=(0, 1), ylabel='Y')
ax.set(zlim3d=(0, 1), zlabel='Z')
ani = animation.FuncAnimation(fig, update_lines, num_steps, fargs=(walks, lines), interval=100)
plt.show()