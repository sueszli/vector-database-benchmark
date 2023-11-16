"""
.. _motionevent:

Motion Event
============

The :class:`MotionEvent` is the base class used for events provided by
pointing devices (touch and non-touch). This class defines all the properties
and methods needed to handle 2D and 3D movements but has many more
capabilities.

Usually you would never need to create the :class:`MotionEvent` yourself as
this is the role of the :mod:`~kivy.input.providers`.

Flow of the motion events
-------------------------

1. The :class:`MotionEvent` 's are gathered from input providers by
   :class:`~kivy.base.EventLoopBase`.
2. Post processing is performed by registered processors
   :mod:`~kivy.input.postproc`.
3. :class:`~kivy.base.EventLoopBase` dispatches all motion events using
   `on_motion` event to all registered listeners including the
   :class:`~kivy.core.window.WindowBase`.
4. Once received in :meth:`~kivy.core.window.WindowBase.on_motion` events
   (touch or non-touch) are all registered managers. If a touch event is not
   handled by at least one manager, then it is dispatched through
   :meth:`~kivy.core.window.WindowBase.on_touch_down`,
   :meth:`~kivy.core.window.WindowBase.on_touch_move` and
   :meth:`~kivy.core.window.WindowBase.on_touch_up`.
5. Widgets receive events in :meth:`~kivy.uix.widget.Widget.on_motion` method
   (if passed by a manager) or on `on_touch_xxx` methods.

Motion events and event managers
--------------------------------

A motion event is a touch event if its :attr:`MotionEvent.is_touch` is set to
`True`. Beside `is_touch` attribute, :attr:`MotionEvent.type_id` can be used to
check for event's general type. Currently two types are dispatched by
input providers: "touch" and "hover".

Event managers can be used to dispatch any motion event throughout the widget
tree and a manager uses `type_id` to specify which event types it want to
receive. See :mod:`~kivy.eventmanager` to learn how to define and register
an event manager.

A manager can also assign a new `type_id` to
:attr:`MotionEvent.type_id` before dispatching it to the widgets. This useful
when dispatching a specific event::

    class MouseTouchManager(EventManagerBase):

        type_ids = ('touch',)

        def dispatch(self, etype, me):
            accepted = False
            if me.device == 'mouse':
                me.push() # Save current type_id and other values
                me.type_id = 'mouse_touch'
                self.window.transform_motion_event_2d(me)
                # Dispatch mouse touch event to widgets which registered
                # to receive 'mouse_touch'
                for widget in self.window.children[:]:
                    if widget.dispatch('on_motion', etype, me):
                        accepted = True
                        break
                me.pop() # Restore
            return accepted

Listening to a motion event
---------------------------

If you want to receive all motion events, touch or not, you can bind the
MotionEvent from the :class:`~kivy.core.window.Window` to your own callback::

    def on_motion(self, etype, me):
        # will receive all motion events.
        pass

    Window.bind(on_motion=on_motion)

You can also listen to changes of the mouse position by watching
:attr:`~kivy.core.window.WindowBase.mouse_pos`.

Profiles
--------

The :class:`MotionEvent` stores device specific information in various
properties listed in the :attr:`~MotionEvent.profile`.
For example, you can receive a MotionEvent that has an angle, a fiducial
ID, or even a shape. You can check the :attr:`~MotionEvent.profile`
attribute to see what is currently supported by the MotionEvent provider.

This is a short list of the profile values supported by default. Please check
the :attr:`MotionEvent.profile` property to see what profile values are
available.

============== ================================================================
Profile value   Description
-------------- ----------------------------------------------------------------
angle          2D angle. Accessed via the `a` property.
button         Mouse button ('left', 'right', 'middle', 'scrollup' or
               'scrolldown'). Accessed via the `button` property.
markerid       Marker or Fiducial ID. Accessed via the `fid` property.
pos            2D position. Accessed via the `x`, `y` or `pos` properties.
pos3d          3D position. Accessed via the `x`, `y` or `z` properties.
pressure       Pressure of the contact. Accessed via the `pressure` property.
shape          Contact shape. Accessed via the `shape` property .
============== ================================================================

If you want to know whether the current :class:`MotionEvent` has an angle::

    def on_touch_move(self, touch):
        if 'angle' in touch.profile:
            print('The touch angle is', touch.a)

If you want to select only the fiducials::

    def on_touch_move(self, touch):
        if 'markerid' not in touch.profile:
            return

"""
__all__ = ('MotionEvent',)
import weakref
from inspect import isroutine
from copy import copy
from time import time
from kivy.eventmanager import MODE_DEFAULT_DISPATCH
from kivy.vector import Vector

class EnhancedDictionary(dict):

    def __getattr__(self, attr):
        if False:
            print('Hello World!')
        try:
            return self.__getitem__(attr)
        except KeyError:
            return super(EnhancedDictionary, self).__getattr__(attr)

    def __setattr__(self, attr, value):
        if False:
            while True:
                i = 10
        self.__setitem__(attr, value)

class MotionEventMetaclass(type):

    def __new__(mcs, name, bases, attrs):
        if False:
            print('Hello World!')
        __attrs__ = []
        for base in bases:
            if hasattr(base, '__attrs__'):
                __attrs__.extend(base.__attrs__)
        if '__attrs__' in attrs:
            __attrs__.extend(attrs['__attrs__'])
        attrs['__attrs__'] = tuple(__attrs__)
        return super(MotionEventMetaclass, mcs).__new__(mcs, name, bases, attrs)
MotionEventBase = MotionEventMetaclass('MotionEvent', (object,), {})

class MotionEvent(MotionEventBase):
    """Abstract class that represents an input event.

    :Parameters:
        `id`: str
            unique ID of the MotionEvent
        `args`: list
            list of parameters, passed to the depack() function
    """
    __uniq_id = 0
    __attrs__ = ('device', 'push_attrs', 'push_attrs_stack', 'is_touch', 'type_id', 'id', 'dispatch_mode', 'shape', 'profile', 'sx', 'sy', 'sz', 'osx', 'osy', 'osz', 'psx', 'psy', 'psz', 'dsx', 'dsy', 'dsz', 'x', 'y', 'z', 'ox', 'oy', 'oz', 'px', 'py', 'pz', 'dx', 'dy', 'dz', 'time_start', 'is_double_tap', 'double_tap_time', 'is_triple_tap', 'triple_tap_time', 'ud')

    def __init__(self, device, id, args, is_touch=False, type_id=None):
        if False:
            for i in range(10):
                print('nop')
        if self.__class__ == MotionEvent:
            raise NotImplementedError('class MotionEvent is abstract')
        MotionEvent.__uniq_id += 1
        self.is_touch = is_touch
        self.type_id = type_id
        self.dispatch_mode = MODE_DEFAULT_DISPATCH
        self.push_attrs_stack = []
        self.push_attrs = ('x', 'y', 'z', 'dx', 'dy', 'dz', 'ox', 'oy', 'oz', 'px', 'py', 'pz', 'pos', 'type_id', 'dispatch_mode')
        self.uid = MotionEvent.__uniq_id
        self.device = device
        self.grab_list = []
        self.grab_exclusive_class = None
        self.grab_state = False
        self.grab_current = None
        self.button = None
        self.profile = []
        self.id = id
        self.shape = None
        self.sx = 0.0
        self.sy = 0.0
        self.sz = 0.0
        self.osx = None
        self.osy = None
        self.osz = None
        self.psx = None
        self.psy = None
        self.psz = None
        self.dsx = None
        self.dsy = None
        self.dsz = None
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.ox = None
        self.oy = None
        self.oz = None
        self.px = None
        self.py = None
        self.pz = None
        self.dx = None
        self.dy = None
        self.dz = None
        self.pos = (0.0, 0.0)
        self.time_start = time()
        self.time_update = self.time_start
        self.time_end = -1
        self.is_double_tap = False
        self.is_triple_tap = False
        self.double_tap_time = 0
        self.triple_tap_time = 0
        self.ud = EnhancedDictionary()
        self.sync_with_dispatch = True
        self._keep_prev_pos = True
        self._first_dispatch_done = False
        self.depack(args)

    def depack(self, args):
        if False:
            return 10
        'Depack `args` into attributes of the class'
        if self.osx is None or (self.sync_with_dispatch and (not self._first_dispatch_done)):
            self.osx = self.psx = self.sx
            self.osy = self.psy = self.sy
            self.osz = self.psz = self.sz
        self.dsx = self.sx - self.psx
        self.dsy = self.sy - self.psy
        self.dsz = self.sz - self.psz

    def grab(self, class_instance, exclusive=False):
        if False:
            for i in range(10):
                print('nop')
        "Grab this motion event.\n\n        If this event is a touch you can grab it if you want to receive\n        subsequent :meth:`~kivy.uix.widget.Widget.on_touch_move` and\n        :meth:`~kivy.uix.widget.Widget.on_touch_up` events, even if the touch\n        is not dispatched by the parent:\n\n        .. code-block:: python\n\n            def on_touch_down(self, touch):\n                touch.grab(self)\n\n            def on_touch_move(self, touch):\n                if touch.grab_current is self:\n                    # I received my grabbed touch\n                else:\n                    # it's a normal touch\n\n            def on_touch_up(self, touch):\n                if touch.grab_current is self:\n                    # I receive my grabbed touch, I must ungrab it!\n                    touch.ungrab(self)\n                else:\n                    # it's a normal touch\n                    pass\n\n        .. versionchanged:: 2.1.0\n            Allowed grab for non-touch events.\n        "
        if self.grab_exclusive_class is not None:
            raise Exception('Event is exclusive and cannot be grabbed')
        class_instance = weakref.ref(class_instance.__self__)
        if exclusive:
            self.grab_exclusive_class = class_instance
        self.grab_list.append(class_instance)

    def ungrab(self, class_instance):
        if False:
            while True:
                i = 10
        'Ungrab a previously grabbed motion event.\n        '
        class_instance = weakref.ref(class_instance.__self__)
        if self.grab_exclusive_class == class_instance:
            self.grab_exclusive_class = None
        if class_instance in self.grab_list:
            self.grab_list.remove(class_instance)

    def dispatch_done(self):
        if False:
            print('Hello World!')
        'Notify that dispatch to the listeners is done.\n\n        Called by the :meth:`EventLoopBase.post_dispatch_input`.\n\n        .. versionadded:: 2.1.0\n        '
        self._keep_prev_pos = True
        self._first_dispatch_done = True

    def move(self, args):
        if False:
            print('Hello World!')
        'Move to another position.\n        '
        if self.sync_with_dispatch:
            if self._keep_prev_pos:
                (self.psx, self.psy, self.psz) = (self.sx, self.sy, self.sz)
                self._keep_prev_pos = False
        else:
            (self.psx, self.psy, self.psz) = (self.sx, self.sy, self.sz)
        self.time_update = time()
        self.depack(args)

    def scale_for_screen(self, w, h, p=None, rotation=0, smode='None', kheight=0):
        if False:
            while True:
                i = 10
        'Scale position for the screen.\n\n        .. versionchanged:: 2.1.0\n            Max value for `x`, `y` and `z` is changed respectively to `w` - 1,\n            `h` - 1 and `p` - 1.\n        '
        (x_max, y_max) = (max(0, w - 1), max(0, h - 1))
        absolute = self.to_absolute_pos
        (self.x, self.y) = absolute(self.sx, self.sy, x_max, y_max, rotation)
        (self.ox, self.oy) = absolute(self.osx, self.osy, x_max, y_max, rotation)
        (self.px, self.py) = absolute(self.psx, self.psy, x_max, y_max, rotation)
        z_max = 0 if p is None else max(0, p - 1)
        self.z = self.sz * z_max
        self.oz = self.osz * z_max
        self.pz = self.psz * z_max
        if smode:
            if smode == 'pan' or smode == 'below_target':
                self.y -= kheight
                self.oy -= kheight
                self.py -= kheight
            elif smode == 'scale':
                offset = kheight * (self.y - h) / (h - kheight)
                self.y += offset
                self.oy += offset
                self.py += offset
        self.dx = self.x - self.px
        self.dy = self.y - self.py
        self.dz = self.z - self.pz
        self.pos = (self.x, self.y)

    def to_absolute_pos(self, nx, ny, x_max, y_max, rotation):
        if False:
            i = 10
            return i + 15
        'Transforms normalized (0-1) coordinates `nx` and `ny` to absolute\n        coordinates using `x_max`, `y_max` and `rotation`.\n\n        :raises:\n            `ValueError`: If `rotation` is not one of: 0, 90, 180 or 270\n\n        .. versionadded:: 2.1.0\n        '
        if rotation == 0:
            return (nx * x_max, ny * y_max)
        elif rotation == 90:
            return (ny * y_max, (1 - nx) * x_max)
        elif rotation == 180:
            return ((1 - nx) * x_max, (1 - ny) * y_max)
        elif rotation == 270:
            return ((1 - ny) * y_max, nx * x_max)
        raise ValueError('Invalid rotation %s, valid values are 0, 90, 180 or 270' % rotation)

    def push(self, attrs=None):
        if False:
            i = 10
            return i + 15
        'Push attribute values in `attrs` onto the stack.\n        '
        if attrs is None:
            attrs = self.push_attrs
        values = [getattr(self, x) for x in attrs]
        self.push_attrs_stack.append((attrs, values))

    def pop(self):
        if False:
            i = 10
            return i + 15
        'Pop attributes values from the stack.\n        '
        (attrs, values) = self.push_attrs_stack.pop()
        for i in range(len(attrs)):
            setattr(self, attrs[i], values[i])

    def apply_transform_2d(self, transform):
        if False:
            i = 10
            return i + 15
        'Apply a transformation on x, y, z, px, py, pz,\n        ox, oy, oz, dx, dy, dz.\n        '
        (self.x, self.y) = self.pos = transform(self.x, self.y)
        (self.px, self.py) = transform(self.px, self.py)
        (self.ox, self.oy) = transform(self.ox, self.oy)
        self.dx = self.x - self.px
        self.dy = self.y - self.py

    def copy_to(self, to):
        if False:
            while True:
                i = 10
        'Copy some attribute to another motion event object.'
        for attr in self.__attrs__:
            to.__setattr__(attr, copy(self.__getattribute__(attr)))

    def distance(self, other_touch):
        if False:
            print('Hello World!')
        'Return the distance between the two events.\n        '
        return Vector(self.pos).distance(other_touch.pos)

    def update_time_end(self):
        if False:
            i = 10
            return i + 15
        self.time_end = time()

    @property
    def dpos(self):
        if False:
            while True:
                i = 10
        'Return delta between last position and current position, in the\n        screen coordinate system (self.dx, self.dy).'
        return (self.dx, self.dy)

    @property
    def opos(self):
        if False:
            i = 10
            return i + 15
        'Return the initial position of the motion event in the screen\n        coordinate system (self.ox, self.oy).'
        return (self.ox, self.oy)

    @property
    def ppos(self):
        if False:
            print('Hello World!')
        'Return the previous position of the motion event in the screen\n        coordinate system (self.px, self.py).'
        return (self.px, self.py)

    @property
    def spos(self):
        if False:
            return 10
        'Return the position in the 0-1 coordinate system (self.sx, self.sy).\n        '
        return (self.sx, self.sy)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        basename = str(self.__class__)
        classname = basename.split('.')[-1].replace('>', '').replace("'", '')
        return '<%s spos=%s pos=%s>' % (classname, self.spos, self.pos)

    def __repr__(self):
        if False:
            while True:
                i = 10
        out = []
        for x in dir(self):
            v = getattr(self, x)
            if x[0] == '_':
                continue
            if isroutine(v):
                continue
            out.append('%s="%s"' % (x, v))
        return '<%s %s>' % (self.__class__.__name__, ' '.join(out))

    @property
    def is_mouse_scrolling(self, *args):
        if False:
            while True:
                i = 10
        'Returns True if the touch event is a mousewheel scrolling\n\n        .. versionadded:: 1.6.0\n        '
        return 'button' in self.profile and 'scroll' in self.button