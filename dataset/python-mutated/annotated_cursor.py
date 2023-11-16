"""
================
Annotated cursor
================

Display a data cursor including a text box, which shows the plot point close
to the mouse pointer.

The new cursor inherits from `~matplotlib.widgets.Cursor` and demonstrates the
creation of new widgets and their event callbacks.

See also the :doc:`cross hair cursor
</gallery/event_handling/cursor_demo>`, which implements a cursor tracking the
plotted data, but without using inheritance and without displaying the
currently tracked coordinates.

.. note::
    The figure related to this example does not show the cursor, because that
    figure is automatically created in a build queue, where the first mouse
    movement, which triggers the cursor creation, is missing.

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Cursor

class AnnotatedCursor(Cursor):
    """
    A crosshair cursor like `~matplotlib.widgets.Cursor` with a text showing     the current coordinates.

    For the cursor to remain responsive you must keep a reference to it.
    The data of the axis specified as *dataaxis* must be in ascending
    order. Otherwise, the `numpy.searchsorted` call might fail and the text
    disappears. You can satisfy the requirement by sorting the data you plot.
    Usually the data is already sorted (if it was created e.g. using
    `numpy.linspace`), but e.g. scatter plots might cause this problem.
    The cursor sticks to the plotted line.

    Parameters
    ----------
    line : `matplotlib.lines.Line2D`
        The plot line from which the data coordinates are displayed.

    numberformat : `python format string <https://docs.python.org/3/    library/string.html#formatstrings>`_, optional, default: "{0:.4g};{1:.4g}"
        The displayed text is created by calling *format()* on this string
        with the two coordinates.

    offset : (float, float) default: (5, 5)
        The offset in display (pixel) coordinates of the text position
        relative to the cross-hair.

    dataaxis : {"x", "y"}, optional, default: "x"
        If "x" is specified, the vertical cursor line sticks to the mouse
        pointer. The horizontal cursor line sticks to *line*
        at that x value. The text shows the data coordinates of *line*
        at the pointed x value. If you specify "y", it works in the opposite
        manner. But: For the "y" value, where the mouse points to, there might
        be multiple matching x values, if the plotted function is not biunique.
        Cursor and text coordinate will always refer to only one x value.
        So if you use the parameter value "y", ensure that your function is
        biunique.

    Other Parameters
    ----------------
    textprops : `matplotlib.text` properties as dictionary
        Specifies the appearance of the rendered text object.

    **cursorargs : `matplotlib.widgets.Cursor` properties
        Arguments passed to the internal `~matplotlib.widgets.Cursor` instance.
        The `matplotlib.axes.Axes` argument is mandatory! The parameter
        *useblit* can be set to *True* in order to achieve faster rendering.

    """

    def __init__(self, line, numberformat='{0:.4g};{1:.4g}', offset=(5, 5), dataaxis='x', textprops=None, **cursorargs):
        if False:
            for i in range(10):
                print('nop')
        if textprops is None:
            textprops = {}
        self.line = line
        self.numberformat = numberformat
        self.offset = np.array(offset)
        self.dataaxis = dataaxis
        super().__init__(**cursorargs)
        self.set_position(self.line.get_xdata()[0], self.line.get_ydata()[0])
        self.text = self.ax.text(self.ax.get_xbound()[0], self.ax.get_ybound()[0], '0, 0', animated=bool(self.useblit), visible=False, **textprops)
        self.lastdrawnplotpoint = None

    def onmove(self, event):
        if False:
            print('Hello World!')
        '\n        Overridden draw callback for cursor. Called when moving the mouse.\n        '
        if self.ignore(event):
            self.lastdrawnplotpoint = None
            return
        if not self.canvas.widgetlock.available(self):
            self.lastdrawnplotpoint = None
            return
        if event.inaxes != self.ax:
            self.lastdrawnplotpoint = None
            self.text.set_visible(False)
            super().onmove(event)
            return
        plotpoint = None
        if event.xdata is not None and event.ydata is not None:
            plotpoint = self.set_position(event.xdata, event.ydata)
            if plotpoint is not None:
                event.xdata = plotpoint[0]
                event.ydata = plotpoint[1]
        if plotpoint is not None and plotpoint == self.lastdrawnplotpoint:
            return
        super().onmove(event)
        if not self.get_active() or not self.visible:
            return
        if plotpoint is not None:
            temp = [event.xdata, event.ydata]
            temp = self.ax.transData.transform(temp)
            temp = temp + self.offset
            temp = self.ax.transData.inverted().transform(temp)
            self.text.set_position(temp)
            self.text.set_text(self.numberformat.format(*plotpoint))
            self.text.set_visible(self.visible)
            self.needclear = True
            self.lastdrawnplotpoint = plotpoint
        else:
            self.text.set_visible(False)
        if self.useblit:
            self.ax.draw_artist(self.text)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def set_position(self, xpos, ypos):
        if False:
            return 10
        "\n        Finds the coordinates, which have to be shown in text.\n\n        The behaviour depends on the *dataaxis* attribute. Function looks\n        up the matching plot coordinate for the given mouse position.\n\n        Parameters\n        ----------\n        xpos : float\n            The current x position of the cursor in data coordinates.\n            Important if *dataaxis* is set to 'x'.\n        ypos : float\n            The current y position of the cursor in data coordinates.\n            Important if *dataaxis* is set to 'y'.\n\n        Returns\n        -------\n        ret : {2D array-like, None}\n            The coordinates which should be displayed.\n            *None* is the fallback value.\n        "
        xdata = self.line.get_xdata()
        ydata = self.line.get_ydata()
        if self.dataaxis == 'x':
            pos = xpos
            data = xdata
            lim = self.ax.get_xlim()
        elif self.dataaxis == 'y':
            pos = ypos
            data = ydata
            lim = self.ax.get_ylim()
        else:
            raise ValueError(f"The data axis specifier {self.dataaxis} should be 'x' or 'y'")
        if pos is not None and lim[0] <= pos <= lim[-1]:
            index = np.searchsorted(data, pos)
            if index < 0 or index >= len(data):
                return None
            return (xdata[index], ydata[index])
        return None

    def clear(self, event):
        if False:
            i = 10
            return i + 15
        '\n        Overridden clear callback for cursor, called before drawing the figure.\n        '
        super().clear(event)
        if self.ignore(event):
            return
        self.text.set_visible(False)

    def _update(self):
        if False:
            i = 10
            return i + 15
        '\n        Overridden method for either blitting or drawing the widget canvas.\n\n        Passes call to base class if blitting is activated, only.\n        In other cases, one draw_idle call is enough, which is placed\n        explicitly in this class (see *onmove()*).\n        In that case, `~matplotlib.widgets.Cursor` is not supposed to draw\n        something using this method.\n        '
        if self.useblit:
            super()._update()
(fig, ax) = plt.subplots(figsize=(8, 6))
ax.set_title('Cursor Tracking x Position')
x = np.linspace(-5, 5, 1000)
y = x ** 2
(line,) = ax.plot(x, y)
ax.set_xlim(-5, 5)
ax.set_ylim(0, 25)
cursor = AnnotatedCursor(line=line, numberformat='{0:.2f}\n{1:.2f}', dataaxis='x', offset=[10, 10], textprops={'color': 'blue', 'fontweight': 'bold'}, ax=ax, useblit=True, color='red', linewidth=2)
t = ax.transData
MouseEvent('motion_notify_event', ax.figure.canvas, *t.transform((-2, 10)))._process()
plt.show()
(fig, ax) = plt.subplots(figsize=(8, 6))
ax.set_title('Cursor Tracking y Position')
(line,) = ax.plot(x, y)
ax.set_xlim(-5, 5)
ax.set_ylim(0, 25)
cursor = AnnotatedCursor(line=line, numberformat='{0:.2f}\n{1:.2f}', dataaxis='y', offset=[10, 10], textprops={'color': 'blue', 'fontweight': 'bold'}, ax=ax, useblit=True, color='red', linewidth=2)
t = ax.transData
MouseEvent('motion_notify_event', ax.figure.canvas, *t.transform((-2, 10)))._process()
plt.show()