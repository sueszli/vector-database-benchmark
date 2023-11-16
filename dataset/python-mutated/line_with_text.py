"""
=======================
Artist within an artist
=======================

Override basic methods so an artist can contain another
artist.  In this case, the line contains a Text instance to label it.
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as lines
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms

class MyLine(lines.Line2D):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.text = mtext.Text(0, 0, '')
        super().__init__(*args, **kwargs)
        self.text.set_text(self.get_label())

    def set_figure(self, figure):
        if False:
            print('Hello World!')
        self.text.set_figure(figure)
        super().set_figure(figure)

    @lines.Line2D.axes.setter
    def axes(self, new_axes):
        if False:
            i = 10
            return i + 15
        self.text.axes = new_axes
        lines.Line2D.axes.fset(self, new_axes)

    def set_transform(self, transform):
        if False:
            i = 10
            return i + 15
        texttrans = transform + mtransforms.Affine2D().translate(2, 2)
        self.text.set_transform(texttrans)
        super().set_transform(transform)

    def set_data(self, x, y):
        if False:
            i = 10
            return i + 15
        if len(x):
            self.text.set_position((x[-1], y[-1]))
        super().set_data(x, y)

    def draw(self, renderer):
        if False:
            i = 10
            return i + 15
        super().draw(renderer)
        self.text.draw(renderer)
np.random.seed(19680801)
(fig, ax) = plt.subplots()
(x, y) = np.random.rand(2, 20)
line = MyLine(x, y, mfc='red', ms=12, label='line label')
line.text.set_color('red')
line.text.set_fontsize(16)
ax.add_line(line)
plt.show()