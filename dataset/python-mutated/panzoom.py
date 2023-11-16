from __future__ import division
import numpy as np
from .base_camera import BaseCamera
from ...geometry import Rect
from ...visuals.transforms import STTransform, MatrixTransform
DEFAULT_RECT_TUPLE = (0, 0, 1, 1)

class PanZoomCamera(BaseCamera):
    """Camera implementing 2D pan/zoom mouse interaction.

    For this camera, the ``scale_factor`` indicates the zoom level, and
    the ``center`` indicates the center position of the view.

    By default, this camera inverts the y axis of the scene. This usually
    results in the scene +y axis pointing upward because widgets (including
    ViewBox) have their +y axis pointing downward.

    Parameters
    ----------
    rect : Rect
        A Rect object or 4-element tuple that specifies the rectangular
        area to show.
    aspect : float | None
        The aspect ratio (i.e. scaling) between x and y dimension of
        the scene. E.g. to show a square image as square, the aspect
        should be 1. If None (default) the x and y dimensions are scaled
        independently.
    **kwargs : dict
        Keyword arguments to pass to `BaseCamera`.

    Notes
    -----
    Interaction:

        * LMB: pan the view
        * RMB or scroll: zooms the view

    """
    _state_props = BaseCamera._state_props + ('rect',)

    def __init__(self, rect=DEFAULT_RECT_TUPLE, aspect=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(PanZoomCamera, self).__init__(**kwargs)
        self.transform = STTransform()
        self.tf_mat = MatrixTransform()
        self.aspect = aspect
        self._rect = None
        self.rect = rect

    @property
    def aspect(self):
        if False:
            print('Hello World!')
        'The ratio between the x and y dimension. E.g. to show a\n        square image as square, the aspect should be 1. If None, the\n        dimensions are scaled automatically, dependening on the\n        available space. Otherwise the ratio between the dimensions\n        is fixed.\n        '
        return self._aspect

    @aspect.setter
    def aspect(self, value):
        if False:
            print('Hello World!')
        if value is None:
            self._aspect = None
        else:
            self._aspect = float(value)
        self.view_changed()

    def zoom(self, factor, center=None):
        if False:
            while True:
                i = 10
        'Zoom in (or out) at the given center\n\n        Parameters\n        ----------\n        factor : float or tuple\n            Fraction by which the scene should be zoomed (e.g. a factor of 2\n            causes the scene to appear twice as large).\n        center : tuple of 2-4 elements\n            The center of the view. If not given or None, use the\n            current center.\n        '
        center = center if center is not None else self.center
        assert len(center) in (2, 3, 4)
        if np.isscalar(factor):
            scale = [factor, factor]
        else:
            if len(factor) != 2:
                raise TypeError('factor must be scalar or length-2 sequence.')
            scale = list(factor)
        if self.aspect is not None:
            scale[0] = scale[1]
        rect = Rect(self.rect)
        left_space = center[0] - rect.left
        right_space = rect.right - center[0]
        bottom_space = center[1] - rect.bottom
        top_space = rect.top - center[1]
        rect.left = center[0] - left_space * scale[0]
        rect.right = center[0] + right_space * scale[0]
        rect.bottom = center[1] - bottom_space * scale[1]
        rect.top = center[1] + top_space * scale[1]
        self.rect = rect

    def pan(self, *pan):
        if False:
            print('Hello World!')
        'Pan the view.\n\n        Parameters\n        ----------\n        *pan : length-2 sequence\n            The distance to pan the view, in the coordinate system of the\n            scene.\n        '
        if len(pan) == 1:
            pan = pan[0]
        self.rect = self.rect + pan

    @property
    def rect(self):
        if False:
            print('Hello World!')
        "The rectangular border of the ViewBox visible area.\n        \n        This is expressed in the coordinate system of the scene.\n        See :class:`~vispy.geometry.rect.Rect` for different ways this can\n        be specified.\n\n        Note that the rectangle can have negative width or height, in\n        which case the corresponding dimension is flipped (this flipping\n        is independent from the camera's ``flip`` property).\n        "
        return self._rect

    @rect.setter
    def rect(self, args):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(args, tuple):
            rect = Rect(*args)
        else:
            rect = Rect(args)
        if self._rect != rect:
            self._rect = rect
            self.view_changed()

    @property
    def center(self):
        if False:
            return 10
        rect = self._rect
        return (*rect.center, 0)

    @center.setter
    def center(self, center):
        if False:
            while True:
                i = 10
        if not (isinstance(center, (tuple, list)) and len(center) in (2, 3)):
            raise ValueError('center must be a 2 or 3 element tuple')
        rect = Rect(self.rect) or Rect(*DEFAULT_RECT_TUPLE)
        rect.center = center[:2]
        self.rect = rect

    def _set_range(self, init):
        if False:
            return 10
        if init and self._rect is not None:
            return
        w = self._xlim[1] - self._xlim[0]
        h = self._ylim[1] - self._ylim[0]
        self.rect = (self._xlim[0], self._ylim[0], w, h)

    def viewbox_resize_event(self, event):
        if False:
            print('Hello World!')
        'Modify the data aspect and scale factor, to adjust to\n        the new window size.\n\n        Parameters\n        ----------\n        event : instance of Event\n            The event.\n        '
        self.view_changed()

    def viewbox_mouse_event(self, event):
        if False:
            while True:
                i = 10
        '\n        The SubScene received a mouse event; update transform\n        accordingly.\n\n        Parameters\n        ----------\n        event : instance of Event\n            The event.\n        '
        if event.handled or not self.interactive:
            return
        BaseCamera.viewbox_mouse_event(self, event)
        if event.type == 'mouse_wheel':
            center = self._scene_transform.imap(event.pos)
            self.zoom((1 + self.zoom_factor) ** (-event.delta[1] * 30), center)
            event.handled = True
        elif event.type == 'gesture_zoom':
            center = self._scene_transform.imap(event.pos)
            self.zoom(1 - event.scale, center)
            event.handled = True
        elif event.type == 'mouse_move':
            if event.press_event is None:
                return
            modifiers = event.mouse_event.modifiers
            p1 = event.mouse_event.press_event.pos
            p2 = event.mouse_event.pos
            if 1 in event.buttons and (not modifiers):
                p1 = np.array(event.last_event.pos)[:2]
                p2 = np.array(event.pos)[:2]
                p1s = self._transform.imap(p1)
                p2s = self._transform.imap(p2)
                self.pan(p1s - p2s)
                event.handled = True
            elif 2 in event.buttons and (not modifiers):
                p1c = np.array(event.last_event.pos)[:2]
                p2c = np.array(event.pos)[:2]
                scale = (1 + self.zoom_factor) ** ((p1c - p2c) * np.array([1, -1]))
                center = self._transform.imap(event.press_event.pos[:2])
                self.zoom(scale, center)
                event.handled = True
            else:
                event.handled = False
        elif event.type == 'mouse_press':
            event.handled = event.button in [1, 2]
        else:
            event.handled = False

    def _update_transform(self):
        if False:
            i = 10
            return i + 15
        rect = self.rect
        self._real_rect = Rect(rect)
        vbr = self._viewbox.rect.flipped(x=self.flip[0], y=not self.flip[1])
        d = self.depth_value
        if self._aspect is not None:
            requested_aspect = rect.width / rect.height if rect.height != 0 else 1
            view_aspect = vbr.width / vbr.height if vbr.height != 0 else 1
            constrained_aspect = abs(view_aspect / self._aspect)
            if requested_aspect > constrained_aspect:
                dy = 0.5 * (rect.width / constrained_aspect - rect.height)
                self._real_rect.top += dy
                self._real_rect.bottom -= dy
            else:
                dx = 0.5 * (rect.height * constrained_aspect - rect.width)
                self._real_rect.left -= dx
                self._real_rect.right += dx
        self.transform.set_mapping(self._real_rect, vbr, update=False)
        self.transform.zoom((1, 1, 1 / d))
        if self.up == '+z':
            self.tf_mat.matrix = self.transform.as_matrix().matrix
        else:
            rr = self._real_rect
            d = d if self.up[0] == '+' else -d
            pp1 = [(vbr.left, vbr.bottom, 0), (vbr.left, vbr.top, 0), (vbr.right, vbr.bottom, 0), (vbr.left, vbr.bottom, 1)]
            if self.up[1] == 'z':
                pp2 = [(rr.left, rr.bottom, 0), (rr.left, rr.top, 0), (rr.right, rr.bottom, 0), (rr.left, rr.bottom, d)]
            elif self.up[1] == 'y':
                pp2 = [(rr.left, 0, rr.bottom), (rr.left, 0, rr.top), (rr.right, 0, rr.bottom), (rr.left, d, rr.bottom)]
            elif self.up[1] == 'x':
                pp2 = [(0, rr.left, rr.bottom), (0, rr.left, rr.top), (0, rr.right, rr.bottom), (d, rr.left, rr.bottom)]
            self.tf_mat.set_mapping(np.array(pp2), np.array(pp1))
        self._set_scene_transform(self.tf_mat)