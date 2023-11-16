"""
.. redirect-from:: /tutorials/advanced/blitting

.. _blitting:

==================================
Faster rendering by using blitting
==================================

*Blitting* is a `standard technique
<https://en.wikipedia.org/wiki/Bit_blit>`__ in raster graphics that,
in the context of Matplotlib, can be used to (drastically) improve
performance of interactive figures. For example, the
:mod:`.animation` and :mod:`.widgets` modules use blitting
internally. Here, we demonstrate how to implement your own blitting, outside
of these classes.

Blitting speeds up repetitive drawing by rendering all non-changing
graphic elements into a background image once. Then, for every draw, only the
changing elements need to be drawn onto this background. For example,
if the limits of an Axes have not changed, we can render the empty Axes
including all ticks and labels once, and only draw the changing data later.

The strategy is

- Prepare the constant background:

  - Draw the figure, but exclude all artists that you want to animate by
    marking them as *animated* (see `.Artist.set_animated`).
  - Save a copy of the RBGA buffer.

- Render the individual images:

  - Restore the copy of the RGBA buffer.
  - Redraw the animated artists using `.Axes.draw_artist` /
    `.Figure.draw_artist`.
  - Show the resulting image on the screen.

One consequence of this procedure is that your animated artists are always
drawn on top of the static artists.

Not all backends support blitting.  You can check if a given canvas does via
the `.FigureCanvasBase.supports_blit` property.

.. warning::

   This code does not work with the OSX backend (but does work with other
   GUI backends on Mac).

Minimal example
---------------

We can use the `.FigureCanvasAgg` methods
`~.FigureCanvasAgg.copy_from_bbox` and
`~.FigureCanvasAgg.restore_region` in conjunction with setting
``animated=True`` on our artist to implement a minimal example that
uses blitting to accelerate rendering

"""
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 2 * np.pi, 100)
(fig, ax) = plt.subplots()
(ln,) = ax.plot(x, np.sin(x), animated=True)
plt.show(block=False)
plt.pause(0.1)
bg = fig.canvas.copy_from_bbox(fig.bbox)
ax.draw_artist(ln)
fig.canvas.blit(fig.bbox)
for j in range(100):
    fig.canvas.restore_region(bg)
    ln.set_ydata(np.sin(x + j / 100 * np.pi))
    ax.draw_artist(ln)
    fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()

class BlitManager:

    def __init__(self, canvas, animated_artists=()):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        canvas : FigureCanvasAgg\n            The canvas to work with, this only works for subclasses of the Agg\n            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and\n            `~FigureCanvasAgg.restore_region` methods.\n\n        animated_artists : Iterable[Artist]\n            List of the artists to manage\n        '
        self.canvas = canvas
        self._bg = None
        self._artists = []
        for a in animated_artists:
            self.add_artist(a)
        self.cid = canvas.mpl_connect('draw_event', self.on_draw)

    def on_draw(self, event):
        if False:
            print('Hello World!')
        "Callback to register with 'draw_event'."
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        if False:
            print('Hello World!')
        "\n        Add an artist to be managed.\n\n        Parameters\n        ----------\n        art : Artist\n\n            The artist to be added.  Will be set to 'animated' (just\n            to be safe).  *art* must be in the figure associated with\n            the canvas this class is managing.\n\n        "
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        if False:
            for i in range(10):
                print('nop')
        'Draw all of the animated artists.'
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        if False:
            for i in range(10):
                print('nop')
        'Update the screen with animated artists.'
        cv = self.canvas
        fig = cv.figure
        if self._bg is None:
            self.on_draw(None)
        else:
            cv.restore_region(self._bg)
            self._draw_animated()
            cv.blit(fig.bbox)
        cv.flush_events()
(fig, ax) = plt.subplots()
(ln,) = ax.plot(x, np.sin(x), animated=True)
fr_number = ax.annotate('0', (0, 1), xycoords='axes fraction', xytext=(10, -10), textcoords='offset points', ha='left', va='top', animated=True)
bm = BlitManager(fig.canvas, [ln, fr_number])
plt.show(block=False)
plt.pause(0.1)
for j in range(100):
    ln.set_ydata(np.sin(x + j / 100 * np.pi))
    fr_number.set_text(f'frame: {j}')
    bm.update()