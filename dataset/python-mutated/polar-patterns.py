import numpy as np
import matplotlib.pyplot as plt

def shotgun_pattern(T, a, b, c=1, d=1, e=10000):
    if False:
        print('Hello World!')
    '\n    Equations taken from [1] or [2] + some modifications by Matthieu LEROY\n    in order to squish the pattern and make a log-based pattern.\n    \n    [1] https://math.stackexchange.com/questions/1808380/non-symmetrical-lemniscate-curve-parameterization\n    [2] https://www.jneurosci.org/content/22/18/8201\n    \n    Parameters\n    ----------\n    T: array-like of floats\n        Angle between -pi and pi.\n    a: float\n        Amplitude.\n    b: float\n        Asymetric parameter.\n    c: float\n        Squish parameter??\n    d: float\n        Not sure what it does.\n    e: float\n        Not sure what it does.\n        \n    Returns\n    -------\n    Shotgun pattern.\n    '
    x = a * (np.cos(T) + b) * np.cos(T) / (c + np.sin(T) ** 2)
    y = d * x * np.sin(T)
    res = np.sqrt(x ** 2 + y ** 2)
    return np.exp(res) / e

def plot(ax, title, alpha=1, shotgun=False):
    if False:
        i = 10
        return i + 15
    T = np.linspace(-np.pi, np.pi, 5000)
    if shotgun:
        R = shotgun_pattern(T, 4, 0.2, 0.1, 4) + shotgun_pattern(T - np.pi / 2, 3.5, 0, c=0.15, d=1)
    else:
        R = alpha + (1 - alpha) * np.cos(T)
    R = np.log(1 + np.abs(50 * R)) / np.log(10)
    R = 1000 * (R / R.max())
    ax.set_theta_offset(np.pi / 2)
    ax.set_thetalim(0, 2 * np.pi)
    ax.set_rorigin(0)
    ax.set_rlabel_position(np.pi / 2)
    ax.fill(T, R, zorder=20, color='C1', clip_on=True, alpha=0.25)
    ax.plot(T, R, zorder=30, alpha=0.75, color='C1', linewidth=1.0, linestyle=':', clip_on=False)
    ax.plot(T, R, zorder=40, color='C1', linewidth=1.5, clip_on=True)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.xaxis.set_tick_params('major', pad=-2.5)
    ax.set_xticklabels(['0°', '', '180°', ''], family='Roboto', size='small', horizontalalignment='center', verticalalignment='center')
    ax.set_yticks([200, 400, 600, 800, 1010])
    for (y, label) in zip([390, 590, 790], ['-20 dB', '-15 dB', '-10 dB']):
        ax.text(0, y, label, zorder=10, family='Roboto Condensed', size='small', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='white', edgecolor='None', pad=1.0))
    ax.set_yticklabels([])
    ax.set_ylim(200, 1010)
    ax.set_title(title, family='Roboto', weight='bold', size='large', y=-0.2)
fig = plt.figure(figsize=(8, 7))
fig.suptitle('Microphone polar patterns', family='Roboto', weight='bold', size='xx-large')
ax = plt.subplot(2, 3, 1, projection='polar')
plot(ax, 'Omnidirectional', 1.0)
ax = plt.subplot(2, 3, 2, projection='polar')
plot(ax, 'Subcardioid', 0.75)
ax = plt.subplot(2, 3, 3, projection='polar')
plot(ax, 'Cardioid', 0.5)
ax = plt.subplot(2, 3, 4, projection='polar')
plot(ax, 'Supercardioid', 0.25)
ax = plt.subplot(2, 3, 5, projection='polar')
plot(ax, 'Bidirectional', 0.0)
ax = plt.subplot(2, 3, 6, projection='polar')
plot(ax, 'Shotgun', shotgun=True)
plt.tight_layout()
plt.savefig('../../figures/scales-projections/polar-patterns.pdf')
plt.show()