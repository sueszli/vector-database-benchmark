"""
=================
Custom box styles
=================

This example demonstrates the implementation of a custom `.BoxStyle`.
Custom `.ConnectionStyle`\\s and `.ArrowStyle`\\s can be similarly defined.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import BoxStyle
from matplotlib.path import Path

def custom_box_style(x0, y0, width, height, mutation_size):
    if False:
        i = 10
        return i + 15
    '\n    Given the location and size of the box, return the path of the box around\n    it.\n\n    Rotation is automatically taken care of.\n\n    Parameters\n    ----------\n    x0, y0, width, height : float\n        Box location and size.\n    mutation_size : float\n        Mutation reference scale, typically the text font size.\n    '
    mypad = 0.3
    pad = mutation_size * mypad
    width = width + 2 * pad
    height = height + 2 * pad
    (x0, y0) = (x0 - pad, y0 - pad)
    (x1, y1) = (x0 + width, y0 + height)
    return Path([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0 - pad, (y0 + y1) / 2), (x0, y0), (x0, y0)], closed=True)
(fig, ax) = plt.subplots(figsize=(3, 3))
ax.text(0.5, 0.5, 'Test', size=30, va='center', ha='center', rotation=30, bbox=dict(boxstyle=custom_box_style, alpha=0.2))

class MyStyle:
    """A simple box."""

    def __init__(self, pad=0.3):
        if False:
            print('Hello World!')
        '\n        The arguments must be floats and have default values.\n\n        Parameters\n        ----------\n        pad : float\n            amount of padding\n        '
        self.pad = pad
        super().__init__()

    def __call__(self, x0, y0, width, height, mutation_size):
        if False:
            print('Hello World!')
        '\n        Given the location and size of the box, return the path of the box\n        around it.\n\n        Rotation is automatically taken care of.\n\n        Parameters\n        ----------\n        x0, y0, width, height : float\n            Box location and size.\n        mutation_size : float\n            Reference scale for the mutation, typically the text font size.\n        '
        pad = mutation_size * self.pad
        width = width + 2.0 * pad
        height = height + 2.0 * pad
        (x0, y0) = (x0 - pad, y0 - pad)
        (x1, y1) = (x0 + width, y0 + height)
        return Path([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0 - pad, (y0 + y1) / 2.0), (x0, y0), (x0, y0)], closed=True)
BoxStyle._style_list['angled'] = MyStyle
(fig, ax) = plt.subplots(figsize=(3, 3))
ax.text(0.5, 0.5, 'Test', size=30, va='center', ha='center', rotation=30, bbox=dict(boxstyle='angled,pad=0.5', alpha=0.2))
del BoxStyle._style_list['angled']
plt.show()