from __future__ import division
from ...util import keys
from ..node import Node
from ...visuals.transforms import STTransform, MatrixTransform, NullTransform, TransformCache

def nested_getattr(obj, names):
    if False:
        return 10
    for name in names:
        obj = getattr(obj, name)
    return obj

def nested_setattr(obj, names, val):
    if False:
        print('Hello World!')
    target = nested_getattr(obj, names[:-1])
    setattr(target, names[-1], val)

class BaseCamera(Node):
    """Base camera class.

    The Camera describes the perspective from which a ViewBox views its
    subscene, and the way that user interaction affects that perspective.

    Most functionality is implemented in subclasses. This base class has
    no user interaction and causes the subscene to use the same coordinate
    system as the ViewBox.

    Parameters
    ----------
    interactive : bool
        Whether the camera processes mouse and keyboard events.
    flip : tuple of bools
        For each dimension, specify whether it is flipped.
    up : {'+z', '-z', '+y', '-y', '+x', '-x'}
        The direction that is considered up. Default '+z'. Not all
        camera's may support this (yet).
    parent : Node
        The parent of the camera.
    name : str
        Name used to identify the camera in the scene.
    """
    _state_props = ()
    zoom_factor = 0.007

    def __init__(self, interactive=True, flip=None, up='+z', parent=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        super(BaseCamera, self).__init__(parent, name)
        self._viewbox = None
        self._linked_cameras = {}
        self._linked_cameras_no_update = None
        self.transform = NullTransform()
        self._pre_transform = None
        self._viewbox_tr = STTransform()
        self._projection = MatrixTransform()
        self._transform_cache = TransformCache()
        self._event_value = None
        self._resetting = False
        self._key_events_bound = False
        self._set_range_args = None
        self._xlim = None
        self._ylim = None
        self._zlim = None
        self._default_state = None
        self._fov = 0.0
        self._center = None
        self._depth_value = 1000000.0
        self.interactive = bool(interactive)
        self.flip = flip if flip is not None else (False, False, False)
        self.up = up

    @property
    def depth_value(self):
        if False:
            i = 10
            return i + 15
        'The depth value to use  in orthographic and perspective projection\n\n        For orthographic projections, ``depth_value`` is the distance between\n        the near and far clipping planes. For perspective projections, it is\n        the ratio between the near and far clipping plane distances.\n\n        GL has a fixed amount of precision in the depth buffer, and a fixed\n        constant will not work for both a very large range and very high\n        precision. This property provides the user a way to override\n        the default value if necessary.\n        '
        return self._depth_value

    @depth_value.setter
    def depth_value(self, value):
        if False:
            i = 10
            return i + 15
        value = float(value)
        if value <= 0:
            raise ValueError('depth value must be positive')
        self._depth_value = value
        self.view_changed()

    def _depth_to_z(self, depth):
        if False:
            while True:
                i = 10
        'Get the z-coord, given the depth value.'
        val = self.depth_value
        return val - depth * 2 * val

    def _viewbox_set(self, viewbox):
        if False:
            while True:
                i = 10
        'Friend method of viewbox to register itself.'
        self._viewbox = viewbox
        viewbox.events.mouse_press.connect(self.viewbox_mouse_event)
        viewbox.events.mouse_release.connect(self.viewbox_mouse_event)
        viewbox.events.mouse_move.connect(self.viewbox_mouse_event)
        viewbox.events.mouse_wheel.connect(self.viewbox_mouse_event)
        viewbox.events.gesture_zoom.connect(self.viewbox_mouse_event)
        viewbox.events.gesture_rotate.connect(self.viewbox_mouse_event)
        viewbox.events.resize.connect(self.viewbox_resize_event)

    def _viewbox_unset(self, viewbox):
        if False:
            return 10
        'Friend method of viewbox to unregister itself.'
        self._viewbox = None
        viewbox.events.mouse_press.disconnect(self.viewbox_mouse_event)
        viewbox.events.mouse_release.disconnect(self.viewbox_mouse_event)
        viewbox.events.mouse_move.disconnect(self.viewbox_mouse_event)
        viewbox.events.mouse_wheel.disconnect(self.viewbox_mouse_event)
        viewbox.events.gesture_zoom.disconnect(self.viewbox_mouse_event)
        viewbox.events.gesture_rotate.disconnect(self.viewbox_mouse_event)
        viewbox.events.resize.disconnect(self.viewbox_resize_event)

    @property
    def viewbox(self):
        if False:
            while True:
                i = 10
        'The viewbox that this camera applies to.'
        return self._viewbox

    @property
    def interactive(self):
        if False:
            return 10
        'Boolean describing whether the camera should enable or disable\n        user interaction.\n        '
        return self._interactive

    @interactive.setter
    def interactive(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._interactive = bool(value)

    @property
    def flip(self):
        if False:
            i = 10
            return i + 15
        return self._flip

    @flip.setter
    def flip(self, value):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(value, (list, tuple)):
            raise ValueError('Flip must be a tuple or list.')
        if len(value) == 2:
            self._flip = (bool(value[0]), bool(value[1]), False)
        elif len(value) == 3:
            self._flip = (bool(value[0]), bool(value[1]), bool(value[2]))
        else:
            raise ValueError('Flip must have 2 or 3 elements.')
        self._flip_factors = tuple([1 - x * 2 for x in self._flip])
        self.view_changed()

    @property
    def up(self):
        if False:
            return 10
        'The dimension that is considered up.'
        return self._up

    @up.setter
    def up(self, value):
        if False:
            while True:
                i = 10
        value = value.lower()
        value = '+' + value if value in 'zyx' else value
        if value not in ('+z', '-z', '+y', '-y', '+x', '-x'):
            raise ValueError('Invalid value for up.')
        self._up = value
        self.view_changed()

    @property
    def center(self):
        if False:
            for i in range(10):
                print('nop')
        'The center location for this camera\n\n        The exact meaning of this value differs per type of camera, but\n        generally means the point of interest or the rotation point.\n        '
        return self._center or (0, 0, 0)

    @center.setter
    def center(self, val):
        if False:
            i = 10
            return i + 15
        if len(val) == 2:
            self._center = (float(val[0]), float(val[1]), 0.0)
        elif len(val) == 3:
            self._center = (float(val[0]), float(val[1]), float(val[2]))
        else:
            raise ValueError('Center must be a 2 or 3 element tuple')
        self.view_changed()

    @property
    def fov(self):
        if False:
            i = 10
            return i + 15
        'Field-of-view angle of the camera. If 0, the camera is in\n        orthographic mode.\n        '
        return self._fov

    @fov.setter
    def fov(self, fov):
        if False:
            print('Hello World!')
        fov = float(fov)
        if fov < 0 or fov > 180:
            raise ValueError('fov must be 0 <= fov <= 180.')
        self._fov = fov
        self.view_changed()

    def set_range(self, x=None, y=None, z=None, margin=0.05):
        if False:
            i = 10
            return i + 15
        'Set the range of the view region for the camera\n\n        Parameters\n        ----------\n        x : tuple | None\n            X range.\n        y : tuple | None\n            Y range.\n        z : tuple | None\n            Z range.\n        margin : float\n            Margin to use.\n\n        Notes\n        -----\n        The view is set to the given range or to the scene boundaries\n        if ranges are not specified. The ranges should be 2-element\n        tuples specifying the min and max for each dimension.\n\n        For the PanZoomCamera the view is fully defined by the range.\n        For e.g. the TurntableCamera the elevation and azimuth are not\n        set. One should use reset() for that.\n        '
        init = self._xlim is None
        bounds = [None, None, None]
        if x is not None:
            bounds[0] = (float(x[0]), float(x[1]))
        if y is not None:
            bounds[1] = (float(y[0]), float(y[1]))
        if z is not None:
            bounds[2] = (float(z[0]), float(z[1]))
        if self._viewbox is None:
            self._set_range_args = (bounds[0], bounds[1], bounds[2], margin)
            return
        self._resetting = True
        if all([b is None for b in bounds]):
            bounds = self._viewbox.get_scene_bounds()
        else:
            for i in range(3):
                if bounds[i] is None:
                    bounds[i] = self._viewbox.get_scene_bounds(i)
        ranges = [b[1] - b[0] for b in bounds]
        margins = [r * margin or 0.1 for r in ranges]
        bounds_margins = [(b[0] - m, b[1] + m) for (b, m) in zip(bounds, margins)]
        (self._xlim, self._ylim, self._zlim) = bounds_margins
        if not init or self._center is None:
            self._center = [b[0] + r / 2 for (b, r) in zip(bounds, ranges)]
        self._set_range(init)
        self._resetting = False
        self.view_changed()

    def _set_range(self, init):
        if False:
            i = 10
            return i + 15
        pass

    def reset(self):
        if False:
            print('Hello World!')
        'Reset the view to the default state.'
        self.set_state(self._default_state)

    def set_default_state(self):
        if False:
            return 10
        'Set the current state to be the default state to be applied\n        when calling reset().\n        '
        self._default_state = self.get_state()

    def get_state(self, props=None):
        if False:
            print('Hello World!')
        'Get the current view state of the camera\n\n        Returns a dict of key-value pairs. The exact keys depend on the\n        camera. Can be passed to set_state() (of this or another camera\n        of the same type) to reproduce the state.\n\n        Parameters\n        ----------\n        props : list of strings | None\n            List of properties to include in the returned dict. If None,\n            all of camera state is returned.\n        '
        if props is None:
            props = self._state_props
        state = {}
        for key in props:
            if isinstance(key, tuple):
                state[key] = nested_getattr(self, key)
            else:
                state[key] = getattr(self, key)
        return state

    def set_state(self, state=None, **kwargs):
        if False:
            while True:
                i = 10
        'Set the view state of the camera\n\n        Should be a dict (or kwargs) as returned by get_state. It can\n        be an incomlete dict, in which case only the specified\n        properties are set.\n\n        Parameters\n        ----------\n        state : dict\n            The camera state.\n        **kwargs : dict\n            Unused keyword arguments.\n        '
        state = state or {}
        state.update(kwargs)
        for key in list(state.keys()):
            if isinstance(key, tuple):
                key1 = key[0]
                if key1 not in state:
                    root_prop = getattr(self, key1)
                    state[key1] = root_prop.__class__(root_prop)
                nested_setattr(state[key1], key[1:], state[key])
        for (key, val) in state.items():
            if isinstance(key, tuple):
                continue
            if key not in self._state_props:
                raise KeyError('Not a valid camera state property %r' % key)
            setattr(self, key, val)

    def link(self, camera, props=None, axis=None):
        if False:
            i = 10
            return i + 15
        'Link this camera with another camera of the same type\n\n        Linked camera\'s keep each-others\' state in sync.\n\n        Parameters\n        ----------\n        camera : instance of Camera\n            The other camera to link.\n        props : list of strings | tuple of strings | None\n            List of camera state properties to keep in sync between\n            the two cameras. If None, all of camera state is kept in sync.\n        axis : "x" | "y" | None\n            An axis to link between two PanZoomCamera instances. If not None,\n            view limits in the selected axis only will be kept in sync between\n            the cameras.\n        '
        if axis is not None:
            props = props or []
            if axis == 'x':
                props += [('rect', 'left'), ('rect', 'right')]
            elif axis == 'y':
                props += [('rect', 'bottom'), ('rect', 'top')]
            else:
                raise ValueError("Axis can be 'x' or 'y', not %r" % axis)
        if props is None:
            props = self._state_props
        (cam1, cam2) = (self, camera)
        while cam1 in cam2._linked_cameras:
            del cam2._linked_cameras[cam1]
        while cam2 in cam1._linked_cameras:
            del cam1._linked_cameras[cam2]
        cam1._linked_cameras[cam2] = props
        cam2._linked_cameras[cam1] = props

    def view_changed(self):
        if False:
            return 10
        'Called when this camera is changes its view. Also called\n        when its associated with a viewbox.\n        '
        if self._resetting:
            return
        if self._viewbox:
            if self._xlim is None:
                args = self._set_range_args or ()
                self.set_range(*args)
            if self._default_state is None:
                self.set_default_state()
            self._update_transform()

    @property
    def pre_transform(self):
        if False:
            print('Hello World!')
        'A transform to apply to the beginning of the scene transform, in\n        addition to anything else provided by this Camera.\n        '
        return self._pre_transform

    @pre_transform.setter
    def pre_transform(self, tr):
        if False:
            print('Hello World!')
        self._pre_transform = tr
        self.view_changed()

    def viewbox_mouse_event(self, event):
        if False:
            return 10
        'Viewbox mouse event handler\n\n        Parameters\n        ----------\n        event : instance of Event\n            The event.\n        '
        pass

    def on_canvas_change(self, event):
        if False:
            while True:
                i = 10
        'Canvas change event handler\n\n        Parameters\n        ----------\n        event : instance of Event\n            The event.\n        '
        if event.old is not None:
            event.old.events.key_press.disconnect(self.viewbox_key_event)
            event.old.events.key_release.disconnect(self.viewbox_key_event)
        if event.new is not None:
            event.new.events.key_press.connect(self.viewbox_key_event)
            event.new.events.key_release.connect(self.viewbox_key_event)

    def viewbox_key_event(self, event):
        if False:
            return 10
        'The ViewBox key event handler\n\n        Parameters\n        ----------\n        event : instance of Event\n            The event.\n        '
        if event.key == keys.BACKSPACE:
            self.reset()

    def viewbox_resize_event(self, event):
        if False:
            i = 10
            return i + 15
        'The ViewBox resize handler to update the transform\n\n        Parameters\n        ----------\n        event : instance of Event\n            The event.\n        '
        pass

    def _update_transform(self):
        if False:
            return 10
        'Subclasses should reimplement this method to update the scene\n        transform by calling self._set_scene_transform.\n        '
        self._set_scene_transform(self.transform)

    def _set_scene_transform(self, tr):
        if False:
            return 10
        'Called by subclasses to configure the viewbox scene transform.'
        pre_tr = self.pre_transform
        if pre_tr is None:
            self._scene_transform = tr
        else:
            self._transform_cache.roll()
            self._scene_transform = self._transform_cache.get([pre_tr, tr])
        self._scene_transform.dynamic = True
        self._viewbox.scene.transform = self._scene_transform
        self._viewbox.update()
        for cam in self._linked_cameras:
            if cam is self._linked_cameras_no_update:
                continue
            try:
                cam._linked_cameras_no_update = self
                linked_props = self._linked_cameras[cam]
                cam.set_state(self.get_state(linked_props))
            finally:
                cam._linked_cameras_no_update = None