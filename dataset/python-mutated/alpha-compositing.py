import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

def circle(center, radius):
    if False:
        for i in range(10):
            print('nop')
    ' Regular circle path '
    T = np.arange(0, np.pi * 2.0, 0.01)
    T = T.reshape(-1, 1)
    X = center[0] + radius * np.cos(T)
    Y = center[1] + radius * np.sin(T)
    vertices = np.hstack((X, Y))
    codes = np.ones(len(vertices), dtype=mpath.Path.code_type) * mpath.Path.LINETO
    codes[0] = mpath.Path.MOVETO
    return (vertices, codes)

def rectangle(center, size):
    if False:
        print('Hello World!')
    ' Regular rectangle path '
    ((x, y), (w, h)) = (center, size)
    vertices = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])
    codes = np.array([mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.LINETO])
    return (vertices, codes)

def patch(ax, path, facecolor='0.9', edgecolor='black', linewidth=0, linestyle='-', antialiased=True, clip=None):
    if False:
        i = 10
        return i + 15
    ' Build a patch with potential clipping path '
    patch = mpatches.PathPatch(path, antialiased=antialiased, linewidth=linewidth, linestyle=linestyle, facecolor=facecolor, edgecolor=edgecolor)
    if clip:
        clip_patch = mpatches.PathPatch(clip, linewidth=0, facecolor='None', edgecolor='None')
        ax.add_patch(clip_patch)
        patch.set_clip_path(clip_patch)
    ax.add_patch(patch)

def subplot(cols, rows, index, title):
    if False:
        return 10
    ' Shortcut to subplot to factorize options'
    ax = plt.subplot(cols, rows, index, aspect=1)
    ax.set_title(title, weight='bold')
    (ax.set_xlim(-1.5, +1.5), ax.set_xticks([]))
    (ax.set_ylim(-1, +1), ax.set_yticks([]))
    return ax

def blend(A, B, f):
    if False:
        while True:
            i = 10
    (xA, xB) = (np.array(A), np.array(B))
    (aA, aB) = (xA[3], xB[3])
    aR = aA + aB * (1 - aA)
    (xaA, xaB) = (aA * xA, aB * xB)
    xR = 1.0 / aR * ((1 - aB) * xaA + (1 - aA) * xaB + aA * aB * f(xA, xB))
    return xR[:3]

def blend_multiply(A, B):
    if False:
        while True:
            i = 10
    return blend(A, B, lambda x, y: x * y)

def blend_screen(A, B):
    if False:
        i = 10
        return i + 15
    return blend(A, B, lambda x, y: x + y - x * y)

def blend_darken(A, B):
    if False:
        print('Hello World!')
    return blend(A, B, lambda x, y: np.minimum(x, y))

def blend_lighten(A, B):
    if False:
        while True:
            i = 10
    return blend(A, B, lambda x, y: np.minimum(x, y))

def blend_color_dodge(A, B):
    if False:
        return 10
    return blend(A, B, lambda x, y: np.where(A < 1, np.minimum(1, B / (1 - A)), 1))

def blend_color_burn(A, B):
    if False:
        return 10
    return blend(A, B, lambda x, y: np.where(A > 0, 1 - np.minimum(1, (1 - B) / A), 0))
(V1, C1) = rectangle((-1.5, -1), size=(3, 2))
(V2, C2) = circle((-0.5, 0), radius=0.75)
A = mpath.Path(V2, C2)
(V2, C2) = circle((+0.5, 0), radius=0.75)
B = mpath.Path(V2, C2)
fig = plt.figure(figsize=(9, 5))
(rows, cols) = (3, 4)
cA = np.array([0.7, 0.0, 0.0, 0.8])
cB = np.array([0.0, 0.0, 0.9, 0.4])
ax = subplot(rows, cols, 1, 'No blend')
patch(ax, B, facecolor=cB)
patch(ax, B, clip=A, facecolor='white', antialiased=False)
patch(ax, A, facecolor=cA)
ax = subplot(rows, cols, 2, 'Default blend')
patch(ax, A, facecolor=cA)
patch(ax, B, facecolor=cB)
ax = subplot(rows, cols, 3, 'Multiply')
patch(ax, A, facecolor=cA)
patch(ax, B, clip=A, facecolor=blend_multiply(cA, cB))
patch(ax, B, facecolor=cB)
ax = subplot(rows, cols, 4, 'Screen')
patch(ax, A, facecolor=cA)
patch(ax, B, clip=A, facecolor=blend_screen(cA, cB))
patch(ax, B, facecolor=cB)
ax = subplot(rows, cols, 5, 'Darken')
patch(ax, A, facecolor=cA)
patch(ax, B, clip=A, facecolor=blend_darken(cA, cB))
patch(ax, B, facecolor=cB)
ax = subplot(rows, cols, 6, 'Lighten')
patch(ax, A, facecolor=cA)
patch(ax, B, clip=A, facecolor=blend_lighten(cA, cB))
patch(ax, B, facecolor=cB)
ax = subplot(rows, cols, 7, 'Color dodge')
patch(ax, A, facecolor=cA)
patch(ax, B, clip=A, facecolor=blend_color_dodge(cA, cB))
patch(ax, B, facecolor=cB)
ax = subplot(rows, cols, 8, 'Color burn')
patch(ax, A, facecolor=cA)
patch(ax, B, clip=A, facecolor=blend_color_burn(cA, cB))
patch(ax, B, facecolor=cB)
ax = subplot(rows, cols, 9, '')
ax = subplot(rows, cols, 10, '')
ax = subplot(rows, cols, 11, '')
ax = subplot(rows, cols, 12, '')
plt.tight_layout()
plt.show()