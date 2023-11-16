"""
ModalView
=========

.. versionadded:: 1.4.0

The :class:`ModalView` widget is used to create modal views. By default, the
view will cover the whole "main" window.

Remember that the default size of a Widget is size_hint=(1, 1). If you don't
want your view to be fullscreen, either use size hints with values lower than
1 (for instance size_hint=(.8, .8)) or deactivate the size_hint and use fixed
size attributes.

Examples
--------

Example of a simple 400x400 Hello world view::

    view = ModalView(size_hint=(None, None), size=(400, 400))
    view.add_widget(Label(text='Hello world'))

By default, any click outside the view will dismiss it. If you don't
want that, you can set :attr:`ModalView.auto_dismiss` to False::

    view = ModalView(auto_dismiss=False)
    view.add_widget(Label(text='Hello world'))
    view.open()

To manually dismiss/close the view, use the :meth:`ModalView.dismiss` method of
the ModalView instance::

    view.dismiss()

Both :meth:`ModalView.open` and :meth:`ModalView.dismiss` are bind-able. That
means you can directly bind the function to an action, e.g. to a button's
on_press ::

    # create content and add it to the view
    content = Button(text='Close me!')
    view = ModalView(auto_dismiss=False)
    view.add_widget(content)

    # bind the on_press event of the button to the dismiss function
    content.bind(on_press=view.dismiss)

    # open the view
    view.open()


ModalView Events
----------------

There are four events available: `on_pre_open` and `on_open` which are raised
when the view is opening; `on_pre_dismiss` and `on_dismiss` which are raised
when the view is closed.

For `on_dismiss`, you can prevent the view from closing by explicitly
returning `True` from your callback::

    def my_callback(instance):
        print('ModalView', instance, 'is being dismissed, but is prevented!')
        return True
    view = ModalView()
    view.add_widget(Label(text='Hello world'))
    view.bind(on_dismiss=my_callback)
    view.open()


.. versionchanged:: 1.5.0
    The ModalView can be closed by hitting the escape key on the
    keyboard if the :attr:`ModalView.auto_dismiss` property is True (the
    default).

"""
__all__ = ('ModalView',)
from kivy.animation import Animation
from kivy.properties import StringProperty, BooleanProperty, ObjectProperty, NumericProperty, ListProperty, ColorProperty
from kivy.uix.anchorlayout import AnchorLayout

class ModalView(AnchorLayout):
    """ModalView class. See module documentation for more information.

    :Events:
        `on_pre_open`:
            Fired before the ModalView is opened. When this event is fired
            ModalView is not yet added to window.
        `on_open`:
            Fired when the ModalView is opened.
        `on_pre_dismiss`:
            Fired before the ModalView is closed.
        `on_dismiss`:
            Fired when the ModalView is closed. If the callback returns True,
            the dismiss will be canceled.

    .. versionchanged:: 1.11.0
        Added events `on_pre_open` and `on_pre_dismiss`.

    .. versionchanged:: 2.0.0
        Added property 'overlay_color'.

    .. versionchanged:: 2.1.0
        Marked `attach_to` property as deprecated.

    """
    auto_dismiss = BooleanProperty(True)
    'This property determines if the view is automatically\n    dismissed when the user clicks outside it.\n\n    :attr:`auto_dismiss` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to True.\n    '
    attach_to = ObjectProperty(None, deprecated=True)
    'If a widget is set on attach_to, the view will attach to the nearest\n    parent window of the widget. If none is found, it will attach to the\n    main/global Window.\n\n    :attr:`attach_to` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to None.\n    '
    background_color = ColorProperty([1, 1, 1, 1])
    "Background color, in the format (r, g, b, a).\n\n    This acts as a *multiplier* to the texture color. The default\n    texture is grey, so just setting the background color will give\n    a darker result. To set a plain color, set the\n    :attr:`background_normal` to ``''``.\n\n    The :attr:`background_color` is a\n    :class:`~kivy.properties.ColorProperty` and defaults to [1, 1, 1, 1].\n\n    .. versionchanged:: 2.0.0\n        Changed behavior to affect the background of the widget itself, not\n        the overlay dimming.\n        Changed from :class:`~kivy.properties.ListProperty` to\n        :class:`~kivy.properties.ColorProperty`.\n    "
    background = StringProperty('atlas://data/images/defaulttheme/modalview-background')
    "Background image of the view used for the view background.\n\n    :attr:`background` is a :class:`~kivy.properties.StringProperty` and\n    defaults to 'atlas://data/images/defaulttheme/modalview-background'.\n    "
    border = ListProperty([16, 16, 16, 16])
    'Border used for :class:`~kivy.graphics.vertex_instructions.BorderImage`\n    graphics instruction. Used for the :attr:`background_normal` and the\n    :attr:`background_down` properties. Can be used when using custom\n    backgrounds.\n\n    It must be a list of four values: (bottom, right, top, left). Read the\n    BorderImage instructions for more information about how to use it.\n\n    :attr:`border` is a :class:`~kivy.properties.ListProperty` and defaults to\n    (16, 16, 16, 16).\n    '
    overlay_color = ColorProperty([0, 0, 0, 0.7])
    'Overlay color in the format (r, g, b, a).\n    Used for dimming the window behind the modal view.\n\n    :attr:`overlay_color` is a :class:`~kivy.properties.ColorProperty` and\n    defaults to [0, 0, 0, .7].\n\n    .. versionadded:: 2.0.0\n    '
    _anim_alpha = NumericProperty(0)
    _anim_duration = NumericProperty(0.1)
    _window = ObjectProperty(allownone=True, rebind=True)
    _is_open = BooleanProperty(False)
    _touch_started_inside = None
    __events__ = ('on_pre_open', 'on_open', 'on_pre_dismiss', 'on_dismiss')

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._parent = None
        super(ModalView, self).__init__(**kwargs)

    def open(self, *_args, **kwargs):
        if False:
            return 10
        "Display the modal in the Window.\n\n        When the view is opened, it will be faded in with an animation. If you\n        don't want the animation, use::\n\n            view.open(animation=False)\n\n        "
        from kivy.core.window import Window
        if self._is_open:
            return
        self._window = Window
        self._is_open = True
        self.dispatch('on_pre_open')
        Window.add_widget(self)
        Window.bind(on_resize=self._align_center, on_keyboard=self._handle_keyboard)
        self.center = Window.center
        self.fbind('center', self._align_center)
        self.fbind('size', self._align_center)
        if kwargs.get('animation', True):
            ani = Animation(_anim_alpha=1.0, d=self._anim_duration)
            ani.bind(on_complete=lambda *_args: self.dispatch('on_open'))
            ani.start(self)
        else:
            self._anim_alpha = 1.0
            self.dispatch('on_open')

    def dismiss(self, *_args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        " Close the view if it is open.\n\n        If you really want to close the view, whatever the on_dismiss\n        event returns, you can use the *force* keyword argument::\n\n            view = ModalView()\n            view.dismiss(force=True)\n\n        When the view is dismissed, it will be faded out before being\n        removed from the parent. If you don't want this animation, use::\n\n            view.dismiss(animation=False)\n\n        "
        if not self._is_open:
            return
        self.dispatch('on_pre_dismiss')
        if self.dispatch('on_dismiss') is True:
            if kwargs.get('force', False) is not True:
                return
        if kwargs.get('animation', True):
            Animation(_anim_alpha=0.0, d=self._anim_duration).start(self)
        else:
            self._anim_alpha = 0
            self._real_remove_widget()

    def _align_center(self, *_args):
        if False:
            for i in range(10):
                print('nop')
        if self._is_open:
            self.center = self._window.center

    def on_motion(self, etype, me):
        if False:
            return 10
        super().on_motion(etype, me)
        return True

    def on_touch_down(self, touch):
        if False:
            while True:
                i = 10
        ' touch down event handler. '
        self._touch_started_inside = self.collide_point(*touch.pos)
        if not self.auto_dismiss or self._touch_started_inside:
            super().on_touch_down(touch)
        return True

    def on_touch_move(self, touch):
        if False:
            while True:
                i = 10
        ' touch moved event handler. '
        if not self.auto_dismiss or self._touch_started_inside:
            super().on_touch_move(touch)
        return True

    def on_touch_up(self, touch):
        if False:
            print('Hello World!')
        ' touch up event handler. '
        if self.auto_dismiss and self._touch_started_inside is False:
            self.dismiss()
        else:
            super().on_touch_up(touch)
        self._touch_started_inside = None
        return True

    def on__anim_alpha(self, _instance, value):
        if False:
            for i in range(10):
                print('nop')
        ' animation progress callback. '
        if value == 0 and self._is_open:
            self._real_remove_widget()

    def _real_remove_widget(self):
        if False:
            while True:
                i = 10
        if not self._is_open:
            return
        self._window.remove_widget(self)
        self._window.unbind(on_resize=self._align_center, on_keyboard=self._handle_keyboard)
        self._is_open = False
        self._window = None

    def on_pre_open(self):
        if False:
            while True:
                i = 10
        ' default pre-open event handler. '

    def on_open(self):
        if False:
            while True:
                i = 10
        ' default open event handler. '

    def on_pre_dismiss(self):
        if False:
            print('Hello World!')
        ' default pre-dismiss event handler. '

    def on_dismiss(self):
        if False:
            i = 10
            return i + 15
        ' default dismiss event handler. '

    def _handle_keyboard(self, _window, key, *_args):
        if False:
            while True:
                i = 10
        if key == 27 and self.auto_dismiss:
            self.dismiss()
            return True
if __name__ == '__main__':
    from kivy.base import runTouchApp
    from kivy.uix.button import Button
    from kivy.core.window import Window
    from kivy.uix.label import Label
    from kivy.uix.gridlayout import GridLayout
    content = GridLayout(cols=1)
    content.add_widget(Label(text='This is a hello world'))
    view = ModalView(size_hint=(None, None), size=(256, 256))
    view.add_widget(content)
    layout = GridLayout(cols=3)
    for x in range(9):
        btn = Button(text=f'click me {x}')
        btn.bind(on_release=view.open)
        layout.add_widget(btn)
    Window.add_widget(layout)
    view.open()
    runTouchApp()