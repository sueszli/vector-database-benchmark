"""
===============================
Rectangle and ellipse selectors
===============================

Click somewhere, move the mouse, and release the mouse button.
`.RectangleSelector` and `.EllipseSelector` draw a rectangle or an ellipse
from the initial click position to the current mouse position (within the same
axes) until the button is released.  A connected callback receives the click-
and release-events.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import EllipseSelector, RectangleSelector

def select_callback(eclick, erelease):
    if False:
        while True:
            i = 10
    '\n    Callback for line selection.\n\n    *eclick* and *erelease* are the press and release events.\n    '
    (x1, y1) = (eclick.xdata, eclick.ydata)
    (x2, y2) = (erelease.xdata, erelease.ydata)
    print(f'({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})')
    print(f'The buttons you used were: {eclick.button} {erelease.button}')

def toggle_selector(event):
    if False:
        print('Hello World!')
    print('Key pressed.')
    if event.key == 't':
        for selector in selectors:
            name = type(selector).__name__
            if selector.active:
                print(f'{name} deactivated.')
                selector.set_active(False)
            else:
                print(f'{name} activated.')
                selector.set_active(True)
fig = plt.figure(layout='constrained')
axs = fig.subplots(2)
N = 100000
x = np.linspace(0, 10, N)
selectors = []
for (ax, selector_class) in zip(axs, [RectangleSelector, EllipseSelector]):
    ax.plot(x, np.sin(2 * np.pi * x))
    ax.set_title(f'Click and drag to draw a {selector_class.__name__}.')
    selectors.append(selector_class(ax, select_callback, useblit=True, button=[1, 3], minspanx=5, minspany=5, spancoords='pixels', interactive=True))
    fig.canvas.mpl_connect('key_press_event', toggle_selector)
axs[0].set_title("Press 't' to toggle the selectors on and off.\n" + axs[0].get_title())
plt.show()