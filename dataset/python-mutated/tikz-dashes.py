import numpy as np
import matplotlib.pyplot as plt

def style(index, e=0.001):
    if False:
        i = 10
        return i + 15
    patterns = [('densely dotted', (-0.5, (e, 2 - e))), ('dotted', (-0.5, (e, 3 - e))), ('loosely dotted', (-0.5, (e, 5 - e))), ('densely dashed', (-0.5, (2, 3))), ('dashed', (-0.5, (2, 4))), ('loosely dashed', (-0.5, (2, 7))), ('densely dashdotted', (-0.5, (2, 2, e, 2 - e))), ('dashdotted', (-0.5, (2, 3, e, 2 - e))), ('loosely dashdotted', (-0.5, (2, 5, e, 5 - e))), ('densely dashdotdotted', (-0.5, (2, 2, e, 2 - e, e, 2 - e))), ('dashdotdotted', (-0.5, (2, 3, e, 3 - e, e, 3 - e))), ('loosely dashdotdotted', (-0.5, (2, 5, e, 5 - e, e, 5 - e)))]
    return patterns[index]
fig = plt.figure(figsize=(4.25, 1.5))
ax = fig.add_axes([0, 0, 1, 1], xlim=[-1, 23], ylim=[-1, 7], frameon=False, xticks=[], yticks=[], aspect=1)
for i in range(4):
    for j in range(3):
        (name, pattern) = style(i * 3 + j)
        X = np.array([j * 8, j * 8 + 6])
        Y = np.array([i * 2, i * 2])
        plt.plot(X, Y + 0.25, color='0.4', linewidth=2.0, dash_capstyle='projecting', linestyle=pattern)
        plt.plot(X, Y, color='0.3', linewidth=2.0, dash_capstyle='round', linestyle=pattern)
        plt.text(j * 8 + 3, i * 2 - 0.5, name, ha='center', va='top', size='x-small')
        (name, pattern) = style(i * 3 + j, e=0.25)
        plt.plot(X, Y + 0.5, color='0.5', linewidth=2.0, dash_capstyle='butt', linestyle=pattern)
plt.savefig('tikz-dashes.pdf')
plt.show()