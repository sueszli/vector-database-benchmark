import numpy as np
from .widget import Widget
from ...visuals import AxisVisual

class AxisWidget(Widget):
    """Widget containing an axis

    Parameters
    ----------
    orientation : str
        Orientation of the axis, 'left' or 'bottom'.
    **kwargs : dict
        Keyword arguments to pass to AxisVisual.
    """

    def __init__(self, orientation='left', **kwargs):
        if False:
            print('Hello World!')
        if 'tick_direction' not in kwargs:
            tickdir = {'left': (-1, 0), 'right': (1, 0), 'bottom': (0, 1), 'top': (0, -1)}[orientation]
            kwargs['tick_direction'] = tickdir
        self.axis = AxisVisual(**kwargs)
        self.orientation = orientation
        self._linked_view = None
        Widget.__init__(self)
        self.add_subvisual(self.axis)

    def on_resize(self, event):
        if False:
            return 10
        'Resize event handler\n\n        Parameters\n        ----------\n        event : instance of Event\n            The event.\n        '
        self._update_axis()

    def _update_axis(self):
        if False:
            while True:
                i = 10
        self.axis.pos = self._axis_ends()

    def _axis_ends(self):
        if False:
            while True:
                i = 10
        r = self.rect
        if self.orientation == 'left':
            return np.array([[r.right, r.top], [r.right, r.bottom]])
        elif self.orientation == 'bottom':
            return np.array([[r.left, r.bottom], [r.right, r.bottom]])
        elif self.orientation == 'right':
            return np.array([[r.left, r.top], [r.left, r.bottom]])
        elif self.orientation == 'top':
            return np.array([[r.left, r.top], [r.right, r.top]])
        else:
            raise RuntimeError('Orientation %s not supported.' % self.orientation)

    def link_view(self, view):
        if False:
            for i in range(10):
                print('nop')
        "Link this axis to a ViewBox\n\n        This makes it so that the axis's domain always matches the\n        visible range in the ViewBox.\n\n        Parameters\n        ----------\n        view : instance of ViewBox\n            The ViewBox to link.\n        "
        if view is self._linked_view:
            return
        if self._linked_view is not None:
            self._linked_view.scene.transform.changed.disconnect(self._view_changed)
        self._linked_view = view
        view.scene.transform.changed.connect(self._view_changed)
        self._view_changed()

    def _view_changed(self, event=None):
        if False:
            return 10
        'Linked view transform has changed; update ticks.'
        tr = self.node_transform(self._linked_view.scene)
        (p1, p2) = tr.map(self._axis_ends())
        if self.orientation in ('left', 'right'):
            self.axis.domain = (p1[1], p2[1])
        else:
            self.axis.domain = (p1[0], p2[0])