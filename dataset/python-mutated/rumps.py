import Foundation
import AppKit
from Foundation import NSDate, NSTimer, NSRunLoop, NSDefaultRunLoopMode, NSSearchPathForDirectoriesInDomains, NSMakeRect, NSLog, NSObject, NSMutableDictionary, NSString, NSUserDefaults
from AppKit import NSApplication, NSStatusBar, NSMenu, NSMenuItem, NSAlert, NSTextField, NSSecureTextField, NSImage, NSSlider, NSSize, NSWorkspace, NSWorkspaceWillSleepNotification, NSWorkspaceDidWakeNotification
from PyObjCTools import AppHelper
import os
import pickle
import traceback
import weakref
from .compat import text_type, string_types, iteritems, collections_abc
from .text_field import Editing, SecureEditing
from .utils import ListDict
from . import _internal
from . import events
from . import notifications
_TIMERS = weakref.WeakKeyDictionary()
separator = object()

def debug_mode(choice):
    if False:
        print('Hello World!')
    'Enable/disable printing helpful information for debugging the program. Default is off.'
    global _log
    if choice:

        def _log(*args):
            if False:
                for i in range(10):
                    print('nop')
            NSLog(' '.join(map(str, args)))
    else:

        def _log(*_):
            if False:
                for i in range(10):
                    print('nop')
            pass
debug_mode(False)

def alert(title=None, message='', ok=None, cancel=None, other=None, icon_path=None):
    if False:
        for i in range(10):
            print('nop')
    'Generate a simple alert window.\n\n    .. versionchanged:: 0.2.0\n        Providing a `cancel` string will set the button text rather than only using text "Cancel". `title` is no longer\n        a required parameter.\n\n    .. versionchanged:: 0.3.0\n        Add `other` button functionality as well as `icon_path` to change the alert icon.\n\n    :param title: the text positioned at the top of the window in larger font. If ``None``, a default localized title\n                  is used. If not ``None`` or a string, will use the string representation of the object.\n    :param message: the text positioned below the `title` in smaller font. If not a string, will use the string\n                    representation of the object.\n    :param ok: the text for the "ok" button. Must be either a string or ``None``. If ``None``, a default\n               localized button title will be used.\n    :param cancel: the text for the "cancel" button. If a string, the button will have that text. If `cancel`\n                   evaluates to ``True``, will create a button with text "Cancel". Otherwise, this button will not be\n                   created.\n    :param other: the text for the "other" button. If a string, the button will have that text. Otherwise, this button will not be\n                   created.\n    :param icon_path: a path to an image. If ``None``, the applications icon is used.\n    :return: a number representing the button pressed. The "ok" button is ``1`` and "cancel" is ``0``.\n    '
    message = text_type(message)
    message = message.replace('%', '%%')
    if title is not None:
        title = text_type(title)
    _internal.require_string_or_none(ok)
    if not isinstance(cancel, string_types):
        cancel = 'Cancel' if cancel else None
    alert = NSAlert.alertWithMessageText_defaultButton_alternateButton_otherButton_informativeTextWithFormat_(title, ok, cancel, other, message)
    if NSUserDefaults.standardUserDefaults().stringForKey_('AppleInterfaceStyle') == 'Dark':
        alert.window().setAppearance_(AppKit.NSAppearance.appearanceNamed_('NSAppearanceNameVibrantDark'))
    alert.setAlertStyle_(0)
    if icon_path is not None:
        icon = _nsimage_from_file(icon_path)
        alert.setIcon_(icon)
    _log('alert opened with message: {0}, title: {1}'.format(repr(message), repr(title)))
    return alert.runModal()

def application_support(name):
    if False:
        i = 10
        return i + 15
    "Return the application support folder path for the given `name`, creating it if it doesn't exist."
    app_support_path = os.path.join(NSSearchPathForDirectoriesInDomains(14, 1, 1).objectAtIndex_(0), name)
    if not os.path.isdir(app_support_path):
        os.mkdir(app_support_path)
    return app_support_path

def timers():
    if False:
        for i in range(10):
            print('nop')
    'Return a list of all :class:`rumps.Timer` objects. These can be active or inactive.'
    return list(_TIMERS)

def quit_application(sender=None):
    if False:
        i = 10
        return i + 15
    'Quit the application. Some menu item should call this function so that the application can exit gracefully.'
    nsapplication = NSApplication.sharedApplication()
    _log('closing application')
    nsapplication.terminate_(sender)

def _nsimage_from_file(filename, dimensions=None, template=None):
    if False:
        return 10
    'Take a path to an image file and return an NSImage object.'
    try:
        _log('attempting to open image at {0}'.format(filename))
        with open(filename):
            pass
    except IOError:
        try:
            from __main__ import __file__ as main_script_path
            main_script_path = os.path.dirname(main_script_path)
            filename = os.path.join(main_script_path, filename)
        except ImportError:
            pass
        _log('attempting (again) to open image at {0}'.format(filename))
        with open(filename):
            pass
    image = NSImage.alloc().initByReferencingFile_(filename)
    image.setScalesWhenResized_(True)
    image.setSize_((20, 20) if dimensions is None else dimensions)
    if not template is None:
        image.setTemplate_(template)
    return image

def timer(interval):
    if False:
        while True:
            i = 10
    "Decorator for registering a function as a callback in a new thread. The function will be repeatedly called every\n    `interval` seconds. This decorator accomplishes the same thing as creating a :class:`rumps.Timer` object by using\n    the decorated function and `interval` as parameters and starting it on application launch.\n\n    .. code-block:: python\n\n        @rumps.timer(2)\n        def repeating_function(sender):\n            print 'hi'\n\n    :param interval: a number representing the time in seconds before the decorated function should be called.\n    "

    def decorator(f):
        if False:
            print('Hello World!')
        timers = timer.__dict__.setdefault('*timers', [])
        timers.append(Timer(f, interval))
        return f
    return decorator

def clicked(*args, **options):
    if False:
        return 10
    'Decorator for registering a function as a callback for a click action on a :class:`rumps.MenuItem` within the\n    application. The passed `args` must specify an existing path in the main menu. The :class:`rumps.MenuItem`\n    instance at the end of that path will have its :meth:`rumps.MenuItem.set_callback` method called, passing in the\n    decorated function.\n\n    .. versionchanged:: 0.2.1\n        Accepts `key` keyword argument.\n\n    .. code-block:: python\n\n        @rumps.clicked(\'Animal\', \'Dog\', \'Corgi\')\n        def corgi_button(sender):\n            import subprocess\n            subprocess.call([\'say\', \'"corgis are the cutest"\'])\n\n    :param args: a series of strings representing the path to a :class:`rumps.MenuItem` in the main menu of the\n                 application.\n    :param key: a string representing the key shortcut as an alternative means of clicking the menu item.\n    '

    def decorator(f):
        if False:
            i = 10
            return i + 15

        def register_click(self):
            if False:
                i = 10
                return i + 15
            menuitem = self._menu
            if menuitem is None:
                raise ValueError('no menu created')
            for arg in args:
                try:
                    menuitem = menuitem[arg]
                except KeyError:
                    menuitem.add(arg)
                    menuitem = menuitem[arg]
            menuitem.set_callback(f, options.get('key'))
        buttons = clicked.__dict__.setdefault('*buttons', [])
        buttons.append(register_click)
        return f
    return decorator

def slider(*args, **options):
    if False:
        i = 10
        return i + 15
    'Decorator for registering a function as a callback for a slide action on a :class:`rumps.SliderMenuItem` within\n    the application. All elements of the provided path will be created as :class:`rumps.MenuItem` objects. The\n    :class:`rumps.SliderMenuItem` will be created as a child of the last menu item.\n\n    Accepts the same keyword arguments as :class:`rumps.SliderMenuItem`.\n\n    .. versionadded:: 0.3.0\n\n    :param args: a series of strings representing the path to a :class:`rumps.SliderMenuItem` in the main menu of the\n                 application.\n    '

    def decorator(f):
        if False:
            i = 10
            return i + 15

        def register_click(self):
            if False:
                while True:
                    i = 10
            menuitem = self._menu
            if menuitem is None:
                raise ValueError('no menu created')
            slider_menu_item = SliderMenuItem(**options)
            slider_menu_item.set_callback(f)
            for arg in args:
                try:
                    menuitem = menuitem[arg]
                except KeyError:
                    menuitem.add(arg)
                    menuitem = menuitem[arg]
            menuitem.add(slider_menu_item)
        buttons = clicked.__dict__.setdefault('*buttons', [])
        buttons.append(register_click)
        return f
    return decorator

class Menu(ListDict):
    """Wrapper for Objective-C's NSMenu class.

    Implements core functionality of menus in rumps. :class:`rumps.MenuItem` subclasses `Menu`.
    """
    _choose_key = object()

    def __init__(self):
        if False:
            while True:
                i = 10
        self._counts = {}
        if not hasattr(self, '_menu'):
            self._menu = NSMenu.alloc().init()
        super(Menu, self).__init__()

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        if key not in self:
            (key, value) = self._process_new_menuitem(key, value)
            self._menu.addItem_(value._menuitem)
            super(Menu, self).__setitem__(key, value)

    def __delitem__(self, key):
        if False:
            while True:
                i = 10
        value = self[key]
        self._menu.removeItem_(value._menuitem)
        super(Menu, self).__delitem__(key)

    def add(self, menuitem):
        if False:
            i = 10
            return i + 15
        'Adds the object to the menu as a :class:`rumps.MenuItem` using the :attr:`rumps.MenuItem.title` as the\n        key. `menuitem` will be converted to a `MenuItem` object if not one already.\n        '
        self.__setitem__(self._choose_key, menuitem)

    def clear(self):
        if False:
            i = 10
            return i + 15
        'Remove all `MenuItem` objects from within the menu of this `MenuItem`.'
        self._menu.removeAllItems()
        super(Menu, self).clear()

    def copy(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    @classmethod
    def fromkeys(cls, *args, **kwargs):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def update(self, iterable, **kwargs):
        if False:
            i = 10
            return i + 15
        'Update with objects from `iterable` after each is converted to a :class:`rumps.MenuItem`, ignoring\n        existing keys. This update is a bit different from the usual ``dict.update`` method. It works recursively and\n        will parse a variety of Python containers and objects, creating `MenuItem` object and submenus as necessary.\n\n        If the `iterable` is an instance of :class:`rumps.MenuItem`, then add to the menu.\n\n        Otherwise, for each element in the `iterable`,\n\n            - if the element is a string or is not an iterable itself, it will be converted to a\n              :class:`rumps.MenuItem` and the key will be its string representation.\n            - if the element is a :class:`rumps.MenuItem` already, it will remain the same and the key will be its\n              :attr:`rumps.MenuItem.title` attribute.\n            - if the element is an iterable having a length of 2, the first value will be converted to a\n              :class:`rumps.MenuItem` and the second will act as the submenu for that `MenuItem`\n            - if the element is an iterable having a length of anything other than 2, a ``ValueError`` will be raised\n            - if the element is a mapping, each key-value pair will act as an iterable having a length of 2\n\n        '

        def parse_menu(iterable, menu, depth):
            if False:
                while True:
                    i = 10
            if isinstance(iterable, MenuItem):
                menu.add(iterable)
                return
            for (n, ele) in enumerate(iteritems(iterable) if isinstance(iterable, collections_abc.Mapping) else iterable):
                if not isinstance(ele, MenuItem) and isinstance(ele, collections_abc.Mapping):
                    parse_menu(ele, menu, depth)
                elif not isinstance(ele, (string_types, MenuItem)) and isinstance(ele, collections_abc.Iterable):
                    try:
                        (menuitem, submenu) = ele
                    except TypeError:
                        raise ValueError('menu iterable element #{0} at depth {1} has length {2}; must be a single menu item or a pair consisting of a menu item and its submenu'.format(n, depth, len(tuple(ele))))
                    menuitem = MenuItem(menuitem)
                    menu.add(menuitem)
                    parse_menu(submenu, menuitem, depth + 1)
                else:
                    menu.add(ele)
        parse_menu(iterable, self, 0)
        parse_menu(kwargs, self, 0)

    def insert_after(self, existing_key, menuitem):
        if False:
            print('Hello World!')
        'Insert a :class:`rumps.MenuItem` in the menu after the `existing_key`.\n\n        :param existing_key: a string key for an existing `MenuItem` value.\n        :param menuitem: an object to be added. It will be converted to a `MenuItem` if not one already.\n        '
        (key, menuitem) = self._process_new_menuitem(self._choose_key, menuitem)
        self._insert_helper(existing_key, key, menuitem, 1)
        super(Menu, self).insert_after(existing_key, (key, menuitem))

    def insert_before(self, existing_key, menuitem):
        if False:
            i = 10
            return i + 15
        'Insert a :class:`rumps.MenuItem` in the menu before the `existing_key`.\n\n        :param existing_key: a string key for an existing `MenuItem` value.\n        :param menuitem: an object to be added. It will be converted to a `MenuItem` if not one already.\n        '
        (key, menuitem) = self._process_new_menuitem(self._choose_key, menuitem)
        self._insert_helper(existing_key, key, menuitem, 0)
        super(Menu, self).insert_before(existing_key, (key, menuitem))

    def _insert_helper(self, existing_key, key, menuitem, pos):
        if False:
            for i in range(10):
                print('nop')
        if existing_key == key:
            raise ValueError('same key provided for location and insertion')
        existing_menuitem = self[existing_key]
        index = self._menu.indexOfItem_(existing_menuitem._menuitem)
        self._menu.insertItem_atIndex_(menuitem._menuitem, index + pos)

    def _process_new_menuitem(self, key, value):
        if False:
            while True:
                i = 10
        if value is None or value is separator:
            value = SeparatorMenuItem()
        if not hasattr(value, '_menuitem'):
            value = MenuItem(value)
        if key is self._choose_key:
            if hasattr(value, 'title'):
                key = value.title
            else:
                cls = type(value)
                count = self._counts[cls] = self._counts.get(cls, 0) + 1
                key = '%s_%d' % (cls.__name__, count)
        if hasattr(value, 'title') and key != value.title:
            _log('WARNING: key {0} is not the same as the title of the corresponding MenuItem {1}; while this would occur if the title is dynamically altered, having different names at the time of menu creation may not be desired '.format(repr(key), repr(value.title)))
        return (key, value)

class MenuItem(Menu):
    """Represents an item within the application's menu.

    A :class:`rumps.MenuItem` is a button inside a menu but it can also serve as a menu itself whose elements are
    other `MenuItem` instances.

    Encapsulates and abstracts Objective-C NSMenuItem (and possibly a corresponding NSMenu as a submenu).

    A couple of important notes:

        - A new `MenuItem` instance can be created from any object with a string representation.
        - Attempting to create a `MenuItem` by passing an existing `MenuItem` instance as the first parameter will not
          result in a new instance but will instead return the existing instance.

    Remembers the order of items added to menu and has constant time lookup. Can insert new `MenuItem` object before or
    after other specified ones.

    .. note::
       When adding a `MenuItem` instance to a menu, the value of :attr:`title` at that time will serve as its key for
       lookup performed on menus even if the `title` changes during program execution.

    :param title: the name of this menu item. If not a string, will use the string representation of the object.
    :param callback: the function serving as callback for when a click event occurs on this menu item.
    :param key: the key shortcut to click this menu item. Must be a string or ``None``.
    :param icon: a path to an image. If set to ``None``, the current image (if any) is removed.
    :param dimensions: a sequence of numbers whose length is two, specifying the dimensions of the icon.
    :param template: a boolean, specifying template mode for a given icon (proper b/w display in dark menu bar)
    """

    def __new__(cls, *args, **kwargs):
        if False:
            print('Hello World!')
        if args and isinstance(args[0], MenuItem):
            return args[0]
        return super(MenuItem, cls).__new__(cls, *args, **kwargs)

    def __init__(self, title, callback=None, key=None, icon=None, dimensions=None, template=None):
        if False:
            i = 10
            return i + 15
        if isinstance(title, MenuItem):
            return
        self._menuitem = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(text_type(title), None, '')
        self._menuitem.setTarget_(NSApp)
        self._menu = self._icon = None
        self.set_callback(callback, key)
        self._template = template
        self.set_icon(icon, dimensions, template)
        super(MenuItem, self).__init__()

    def __setitem__(self, key, value):
        if False:
            while True:
                i = 10
        if self._menu is None:
            self._menu = NSMenu.alloc().init()
            self._menuitem.setSubmenu_(self._menu)
        super(MenuItem, self).__setitem__(key, value)

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<{0}: [{1} -> {2}; callback: {3}]>'.format(type(self).__name__, repr(self.title), list(map(str, self)), repr(self.callback))

    @property
    def title(self):
        if False:
            print('Hello World!')
        'The text displayed in a menu for this menu item. If not a string, will use the string representation of the\n        object.\n        '
        return self._menuitem.title()

    @title.setter
    def title(self, new_title):
        if False:
            return 10
        new_title = text_type(new_title)
        self._menuitem.setTitle_(new_title)

    @property
    def icon(self):
        if False:
            print('Hello World!')
        'The path to an image displayed next to the text for this menu item. If set to ``None``, the current image\n        (if any) is removed.\n\n        .. versionchanged:: 0.2.0\n           Setting icon to ``None`` after setting it to an image will correctly remove the icon. Returns the path to an\n           image rather than exposing a `PyObjC` class.\n\n        '
        return self._icon

    @icon.setter
    def icon(self, icon_path):
        if False:
            while True:
                i = 10
        self.set_icon(icon_path, template=self._template)

    @property
    def template(self):
        if False:
            i = 10
            return i + 15
        'Template mode for an icon. If set to ``None``, the current icon (if any) is displayed as a color icon.\n        If set to ``True``, template mode is enabled and the icon will be displayed correctly in dark menu bar mode.\n        '
        return self._template

    @template.setter
    def template(self, template_mode):
        if False:
            i = 10
            return i + 15
        self._template = template_mode
        self.set_icon(self.icon, template=template_mode)

    def set_icon(self, icon_path, dimensions=None, template=None):
        if False:
            i = 10
            return i + 15
        'Sets the icon displayed next to the text for this menu item. If set to ``None``, the current image (if any)\n        is removed. Can optionally supply `dimensions`.\n\n        .. versionchanged:: 0.2.0\n           Setting `icon` to ``None`` after setting it to an image will correctly remove the icon. Passing `dimensions`\n           a sequence whose length is not two will no longer silently error.\n\n        :param icon_path: a file path to an image.\n        :param dimensions: a sequence of numbers whose length is two.\n        :param template: a boolean who defines the template mode for the icon.\n        '
        new_icon = _nsimage_from_file(icon_path, dimensions, template) if icon_path is not None else None
        self._icon = icon_path
        self._menuitem.setImage_(new_icon)

    @property
    def state(self):
        if False:
            for i in range(10):
                print('nop')
        'The state of the menu item. The "on" state is symbolized by a check mark. The "mixed" state is symbolized\n        by a dash.\n\n        .. table:: Setting states\n\n           =====  ======\n           State  Number\n           =====  ======\n            ON      1\n            OFF     0\n           MIXED   -1\n           =====  ======\n\n        '
        return self._menuitem.state()

    @state.setter
    def state(self, new_state):
        if False:
            print('Hello World!')
        self._menuitem.setState_(new_state)

    @property
    def hidden(self):
        if False:
            return 10
        'Indicates whether the menu item is hidden.\n\n        .. versionadded:: 0.4.0\n\n        '
        return self._menuitem.isHidden()

    @hidden.setter
    def hidden(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._menuitem.setHidden_(value)

    def hide(self):
        if False:
            i = 10
            return i + 15
        'Hide the menu item.\n\n        .. versionadded:: 0.4.0\n\n        '
        self.hidden = True

    def show(self):
        if False:
            return 10
        'Show the menu item.\n\n        .. versionadded:: 0.4.0\n\n        '
        self.hidden = False

    def set_callback(self, callback, key=None):
        if False:
            i = 10
            return i + 15
        'Set the function serving as callback for when a click event occurs on this menu item. When `callback` is\n        ``None``, it will disable the callback function and grey out the menu item. If `key` is a string, set as the\n        key shortcut. If it is ``None``, no adjustment will be made to the current key shortcut.\n\n        .. versionchanged:: 0.2.0\n           Allowed passing ``None`` as both `callback` and `key`. Additionally, passing a `key` that is neither a\n           string nor ``None`` will result in a standard ``TypeError`` rather than various, uninformative `PyObjC`\n           internal errors depending on the object.\n\n        :param callback: the function to be called when the user clicks on this menu item.\n        :param key: the key shortcut to click this menu item.\n        '
        _internal.require_string_or_none(key)
        if key is not None:
            self._menuitem.setKeyEquivalent_(key)
        NSApp._ns_to_py_and_callback[self._menuitem] = (self, callback)
        self._menuitem.setAction_('callback:' if callback is not None else None)

    @property
    def callback(self):
        if False:
            print('Hello World!')
        'Return the current callback function.\n\n        .. versionadded:: 0.2.0\n\n        '
        return NSApp._ns_to_py_and_callback[self._menuitem][1]

    @property
    def key(self):
        if False:
            return 10
        'The key shortcut to click this menu item.\n\n        .. versionadded:: 0.2.0\n\n        '
        return self._menuitem.keyEquivalent()

class SliderMenuItem(object):
    """Represents a slider menu item within the application's menu.

    .. versionadded:: 0.3.0

    :param value: a number for the current position of the slider.
    :param min_value: a number for the minimum position to which a slider can be moved.
    :param max_value: a number for the maximum position to which a slider can be moved.
    :param callback: the function serving as callback for when a slide event occurs on this menu item.
    :param dimensions: a sequence of numbers whose length is two, specifying the dimensions of the slider.
    """

    def __init__(self, value=50, min_value=0, max_value=100, callback=None, dimensions=(180, 15)):
        if False:
            while True:
                i = 10
        self._slider = NSSlider.alloc().init()
        self._slider.setMinValue_(min_value)
        self._slider.setMaxValue_(max_value)
        self._slider.setDoubleValue_(value)
        self._slider.setFrameSize_(NSSize(*dimensions))
        self._slider.setTarget_(NSApp)
        self._menuitem = NSMenuItem.alloc().init()
        self._menuitem.setTarget_(NSApp)
        self._menuitem.setView_(self._slider)
        self.set_callback(callback)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<{0}: [value: {1}; callback: {2}]>'.format(type(self).__name__, self.value, repr(self.callback))

    def set_callback(self, callback):
        if False:
            i = 10
            return i + 15
        'Set the function serving as callback for when a slide event occurs on this menu item.\n\n        :param callback: the function to be called when the user drags the marker on the slider.\n        '
        NSApp._ns_to_py_and_callback[self._slider] = (self, callback)
        self._slider.setAction_('callback:' if callback is not None else None)

    @property
    def callback(self):
        if False:
            print('Hello World!')
        return NSApp._ns_to_py_and_callback[self._slider][1]

    @property
    def value(self):
        if False:
            return 10
        'The current position of the slider.'
        return self._slider.doubleValue()

    @value.setter
    def value(self, new_value):
        if False:
            for i in range(10):
                print('nop')
        self._slider.setDoubleValue_(new_value)

class SeparatorMenuItem(object):
    """Visual separator between :class:`rumps.MenuItem` objects in the application menu."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._menuitem = NSMenuItem.separatorItem()

class Timer(object):
    """
    Python abstraction of an Objective-C event timer in a new thread for application. Controls the callback function,
    interval, and starting/stopping the run loop.

    .. versionchanged:: 0.2.0
       Method `__call__` removed.

    :param callback: Function that should be called every `interval` seconds. It will be passed this
                     :class:`rumps.Timer` object as its only parameter.
    :param interval: The time in seconds to wait before calling the `callback` function.
    """

    def __init__(self, callback, interval):
        if False:
            return 10
        self.set_callback(callback)
        self._interval = interval
        self._status = False

    def __repr__(self):
        if False:
            return 10
        return '<{0}: [callback: {1}; interval: {2}; status: {3}]>'.format(type(self).__name__, repr(getattr(self, '*callback').__name__), self._interval, 'ON' if self._status else 'OFF')

    @property
    def interval(self):
        if False:
            print('Hello World!')
        'The time in seconds to wait before calling the :attr:`callback` function.'
        return self._interval

    @interval.setter
    def interval(self, new_interval):
        if False:
            return 10
        if self._status:
            if abs(self._nsdate.timeIntervalSinceNow()) >= self._nstimer.timeInterval():
                self.stop()
                self._interval = new_interval
                self.start()
        else:
            self._interval = new_interval

    @property
    def callback(self):
        if False:
            i = 10
            return i + 15
        'The current function specified as the callback.'
        return getattr(self, '*callback')

    def is_alive(self):
        if False:
            return 10
        'Whether the timer thread loop is currently running.'
        return self._status

    def start(self):
        if False:
            return 10
        'Start the timer thread loop.'
        if not self._status:
            self._nsdate = NSDate.date()
            self._nstimer = NSTimer.alloc().initWithFireDate_interval_target_selector_userInfo_repeats_(self._nsdate, self._interval, self, 'callback:', None, True)
            NSRunLoop.currentRunLoop().addTimer_forMode_(self._nstimer, NSDefaultRunLoopMode)
            _TIMERS[self] = None
            self._status = True

    def stop(self):
        if False:
            while True:
                i = 10
        'Stop the timer thread loop.'
        if self._status:
            self._nstimer.invalidate()
            del self._nstimer
            del self._nsdate
            self._status = False

    def set_callback(self, callback):
        if False:
            return 10
        'Set the function that should be called every :attr:`interval` seconds. It will be passed this\n        :class:`rumps.Timer` object as its only parameter.\n        '
        setattr(self, '*callback', callback)

    def callback_(self, _):
        if False:
            return 10
        _log(self)
        try:
            return _internal.call_as_function_or_method(getattr(self, '*callback'), self)
        except Exception:
            traceback.print_exc()

class Window(object):
    """Generate a window to consume user input in the form of both text and button clicked.

    .. versionchanged:: 0.2.0
        Providing a `cancel` string will set the button text rather than only using text "Cancel". `message` is no
        longer a required parameter.

    .. versionchanged:: 0.3.0
        Add `secure` text input field functionality.

    :param message: the text positioned below the `title` in smaller font. If not a string, will use the string
                    representation of the object.
    :param title: the text positioned at the top of the window in larger font. If not a string, will use the string
                  representation of the object.
    :param default_text: the text within the editable textbox. If not a string, will use the string representation of
                         the object.
    :param ok: the text for the "ok" button. Must be either a string or ``None``. If ``None``, a default
               localized button title will be used.
    :param cancel: the text for the "cancel" button. If a string, the button will have that text. If `cancel`
                   evaluates to ``True``, will create a button with text "Cancel". Otherwise, this button will not be
                   created.
    :param dimensions: the size of the editable textbox. Must be sequence with a length of 2.
    :param secure: should the text field be secured or not. With ``True`` the window can be used for passwords.
    """

    def __init__(self, message='', title='', default_text='', ok=None, cancel=None, dimensions=(320, 160), secure=False):
        if False:
            while True:
                i = 10
        message = text_type(message)
        message = message.replace('%', '%%')
        title = text_type(title)
        self._cancel = bool(cancel)
        self._icon = None
        _internal.require_string_or_none(ok)
        if not isinstance(cancel, string_types):
            cancel = 'Cancel' if cancel else None
        self._alert = NSAlert.alertWithMessageText_defaultButton_alternateButton_otherButton_informativeTextWithFormat_(title, ok, cancel, None, message)
        self._alert.setAlertStyle_(0)
        if secure:
            self._textfield = SecureEditing.alloc().initWithFrame_(NSMakeRect(0, 0, *dimensions))
        else:
            self._textfield = Editing.alloc().initWithFrame_(NSMakeRect(0, 0, *dimensions))
        self._textfield.setSelectable_(True)
        self._alert.setAccessoryView_(self._textfield)
        self.default_text = default_text

    @property
    def title(self):
        if False:
            for i in range(10):
                print('nop')
        'The text positioned at the top of the window in larger font. If not a string, will use the string\n        representation of the object.\n        '
        return self._alert.messageText()

    @title.setter
    def title(self, new_title):
        if False:
            for i in range(10):
                print('nop')
        new_title = text_type(new_title)
        self._alert.setMessageText_(new_title)

    @property
    def message(self):
        if False:
            return 10
        'The text positioned below the :attr:`title` in smaller font. If not a string, will use the string\n        representation of the object.\n        '
        return self._alert.informativeText()

    @message.setter
    def message(self, new_message):
        if False:
            for i in range(10):
                print('nop')
        new_message = text_type(new_message)
        self._alert.setInformativeText_(new_message)

    @property
    def default_text(self):
        if False:
            i = 10
            return i + 15
        'The text within the editable textbox. An example would be\n\n            "Type your message here."\n\n        If not a string, will use the string representation of the object.\n        '
        return self._default_text

    @default_text.setter
    def default_text(self, new_text):
        if False:
            i = 10
            return i + 15
        new_text = text_type(new_text)
        self._default_text = new_text
        self._textfield.setStringValue_(new_text)

    @property
    def icon(self):
        if False:
            return 10
        'The path to an image displayed for this window. If set to ``None``, will default to the icon for the\n        application using :attr:`rumps.App.icon`.\n\n        .. versionchanged:: 0.2.0\n           If the icon is set to an image then changed to ``None``, it will correctly be changed to the application\n           icon.\n\n        '
        return self._icon

    @icon.setter
    def icon(self, icon_path):
        if False:
            return 10
        new_icon = _nsimage_from_file(icon_path) if icon_path is not None else None
        self._icon = icon_path
        self._alert.setIcon_(new_icon)

    def add_button(self, name):
        if False:
            i = 10
            return i + 15
        'Create a new button.\n\n        .. versionchanged:: 0.2.0\n           The `name` parameter is required to be a string.\n\n        :param name: the text for a new button. Must be a string.\n        '
        _internal.require_string(name)
        self._alert.addButtonWithTitle_(name)

    def add_buttons(self, iterable=None, *args):
        if False:
            for i in range(10):
                print('nop')
        'Create multiple new buttons.\n\n        .. versionchanged:: 0.2.0\n           Since each element is passed to :meth:`rumps.Window.add_button`, they must be strings.\n\n        '
        if iterable is None:
            return
        if isinstance(iterable, string_types):
            self.add_button(iterable)
        else:
            for ele in iterable:
                self.add_button(ele)
        for arg in args:
            self.add_button(arg)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        'Launch the window. :class:`rumps.Window` instances can be reused to retrieve user input as many times as\n        needed.\n\n        :return: a :class:`rumps.rumps.Response` object that contains the text and the button clicked as an integer.\n        '
        _log(self)
        if NSUserDefaults.standardUserDefaults().stringForKey_('AppleInterfaceStyle') == 'Dark':
            self._alert.window().setAppearance_(AppKit.NSAppearance.appearanceNamed_('NSAppearanceNameVibrantDark'))
        clicked = self._alert.runModal() % 999
        if clicked > 2 and self._cancel:
            clicked -= 1
        self._textfield.validateEditing()
        text = self._textfield.stringValue()
        self.default_text = self._default_text
        return Response(clicked, text)

class Response(object):
    """Holds information from user interaction with a :class:`rumps.Window` after it has been closed."""

    def __init__(self, clicked, text):
        if False:
            i = 10
            return i + 15
        self._clicked = clicked
        self._text = text

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        shortened_text = self._text if len(self._text) < 21 else self._text[:17] + '...'
        return '<{0}: [clicked: {1}, text: {2}]>'.format(type(self).__name__, self._clicked, repr(shortened_text))

    @property
    def clicked(self):
        if False:
            i = 10
            return i + 15
        'Return a number representing the button pressed by the user.\n\n        The "ok" button will return ``1`` and the "cancel" button will return ``0``. This makes it convenient to write\n        a conditional like,\n\n        .. code-block:: python\n\n            if response.clicked:\n                do_thing_for_ok_pressed()\n            else:\n                do_thing_for_cancel_pressed()\n\n        Where `response` is an instance of :class:`rumps.rumps.Response`.\n\n        Additional buttons added using methods :meth:`rumps.Window.add_button` and :meth:`rumps.Window.add_buttons`\n        will return ``2``, ``3``, ... in the order they were added.\n        '
        return self._clicked

    @property
    def text(self):
        if False:
            print('Hello World!')
        'Return the text collected from the user.'
        return self._text

class NSApp(NSObject):
    """Objective-C delegate class for NSApplication. Don't instantiate - use App instead."""
    _ns_to_py_and_callback = {}

    def userNotificationCenter_didActivateNotification_(self, notification_center, notification):
        if False:
            return 10
        notifications._clicked(notification_center, notification)

    def initializeStatusBar(self):
        if False:
            while True:
                i = 10
        self.nsstatusitem = NSStatusBar.systemStatusBar().statusItemWithLength_(-1)
        self.nsstatusitem.setHighlightMode_(True)
        self.setStatusBarIcon()
        self.setStatusBarTitle()
        mainmenu = self._app['_menu']
        quit_button = self._app['_quit_button']
        if quit_button is not None:
            quit_button.set_callback(quit_application)
            mainmenu.add(quit_button)
        else:
            _log('WARNING: the default quit button is disabled. To exit the application gracefully, another button should have a callback of quit_application or call it indirectly.')
        self.nsstatusitem.setMenu_(mainmenu._menu)

    def setStatusBarTitle(self):
        if False:
            while True:
                i = 10
        self.nsstatusitem.setTitle_(self._app['_title'])
        self.fallbackOnName()

    def setStatusBarIcon(self):
        if False:
            while True:
                i = 10
        self.nsstatusitem.setImage_(self._app['_icon_nsimage'])
        self.fallbackOnName()

    def fallbackOnName(self):
        if False:
            print('Hello World!')
        if not (self.nsstatusitem.title() or self.nsstatusitem.image()):
            self.nsstatusitem.setTitle_(self._app['_name'])

    def applicationDidFinishLaunching_(self, notification):
        if False:
            return 10
        workspace = NSWorkspace.sharedWorkspace()
        notificationCenter = workspace.notificationCenter()
        notificationCenter.addObserver_selector_name_object_(self, self.receiveSleepNotification_, NSWorkspaceWillSleepNotification, None)
        notificationCenter.addObserver_selector_name_object_(self, self.receiveWakeNotification_, NSWorkspaceDidWakeNotification, None)

    def receiveSleepNotification_(self, ns_notification):
        if False:
            return 10
        _log('receiveSleepNotification')
        events.on_sleep.emit()

    def receiveWakeNotification_(self, ns_notification):
        if False:
            print('Hello World!')
        _log('receiveWakeNotification')
        events.on_wake.emit()

    def applicationWillTerminate_(self, ns_notification):
        if False:
            i = 10
            return i + 15
        _log('applicationWillTerminate')
        events.before_quit.emit()

    @classmethod
    def callback_(cls, nsmenuitem):
        if False:
            return 10
        (self, callback) = cls._ns_to_py_and_callback[nsmenuitem]
        _log(self)
        try:
            return _internal.call_as_function_or_method(callback, self)
        except Exception:
            traceback.print_exc()

class App(object):
    """Represents the statusbar application.

    Provides a simple and pythonic interface for all those long and ugly `PyObjC` calls. :class:`rumps.App` may be
    subclassed so that the application logic can be encapsulated within a class. Alternatively, an `App` can be
    instantiated and the various callback functions can exist at module level.

    .. versionchanged:: 0.2.0
       `name` parameter must be a string and `title` must be either a string or ``None``. `quit_button` parameter added.

    :param name: the name of the application.
    :param title: text that will be displayed for the application in the statusbar.
    :param icon: file path to the icon that will be displayed for the application in the statusbar.
    :param menu: an iterable of Python objects or pairs of objects that will be converted into the main menu for the
                 application. Parsing is implemented by calling :meth:`rumps.MenuItem.update`.
    :param quit_button: the quit application menu item within the main menu. If ``None``, the default quit button will
                        not be added.
    """
    serializer = pickle

    def __init__(self, name, title=None, icon=None, template=None, menu=None, quit_button='Quit'):
        if False:
            while True:
                i = 10
        _internal.require_string(name)
        self._name = name
        self._icon = self._icon_nsimage = self._title = None
        self._template = template
        self.icon = icon
        self.title = title
        self.quit_button = quit_button
        self._menu = Menu()
        if menu is not None:
            self.menu = menu
        self._application_support = application_support(self._name)

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        'The name of the application. Determines the application support folder name. Will also serve as the title\n        text of the application if :attr:`title` is not set.\n        '
        return self._name

    @property
    def title(self):
        if False:
            for i in range(10):
                print('nop')
        'The text that will be displayed for the application in the statusbar. Can be ``None`` in which case the icon\n        will be used or, if there is no icon set the application text will fallback on the application :attr:`name`.\n\n        .. versionchanged:: 0.2.0\n           If the title is set then changed to ``None``, it will correctly be removed. Must be either a string or\n           ``None``.\n\n        '
        return self._title

    @title.setter
    def title(self, title):
        if False:
            for i in range(10):
                print('nop')
        _internal.require_string_or_none(title)
        self._title = title
        try:
            self._nsapp.setStatusBarTitle()
        except AttributeError:
            pass

    @property
    def icon(self):
        if False:
            i = 10
            return i + 15
        'A path to an image representing the icon that will be displayed for the application in the statusbar.\n        Can be ``None`` in which case the text from :attr:`title` will be used.\n\n        .. versionchanged:: 0.2.0\n           If the icon is set to an image then changed to ``None``, it will correctly be removed.\n\n        '
        return self._icon

    @icon.setter
    def icon(self, icon_path):
        if False:
            print('Hello World!')
        new_icon = _nsimage_from_file(icon_path, template=self._template) if icon_path is not None else None
        self._icon = icon_path
        self._icon_nsimage = new_icon
        try:
            self._nsapp.setStatusBarIcon()
        except AttributeError:
            pass

    @property
    def template(self):
        if False:
            while True:
                i = 10
        'Template mode for an icon. If set to ``None``, the current icon (if any) is displayed as a color icon.\n        If set to ``True``, template mode is enabled and the icon will be displayed correctly in dark menu bar mode.\n        '
        return self._template

    @template.setter
    def template(self, template_mode):
        if False:
            while True:
                i = 10
        self._template = template_mode
        self.icon = self._icon

    @property
    def menu(self):
        if False:
            return 10
        'Represents the main menu of the statusbar application. Setting `menu` works by calling\n        :meth:`rumps.MenuItem.update`.\n        '
        return self._menu

    @menu.setter
    def menu(self, iterable):
        if False:
            return 10
        self._menu.update(iterable)

    @property
    def quit_button(self):
        if False:
            for i in range(10):
                print('nop')
        'The quit application menu item within the main menu. This is a special :class:`rumps.MenuItem` object that\n        will both replace any function callback with :func:`rumps.quit_application` and add itself to the end of the\n        main menu when :meth:`rumps.App.run` is called. If set to ``None``, the default quit button will not be added.\n\n        .. warning::\n           If set to ``None``, some other menu item should call :func:`rumps.quit_application` so that the\n           application can exit gracefully.\n\n        .. versionadded:: 0.2.0\n\n        '
        return self._quit_button

    @quit_button.setter
    def quit_button(self, quit_text):
        if False:
            return 10
        if quit_text is None:
            self._quit_button = None
        else:
            self._quit_button = MenuItem(quit_text)

    def open(self, *args):
        if False:
            print('Hello World!')
        "Open a file within the application support folder for this application.\n\n        .. code-block:: python\n\n            app = App('Cool App')\n            with app.open('data.json') as f:\n                pass\n\n        Is a shortcut for,\n\n        .. code-block:: python\n\n            app = App('Cool App')\n            filename = os.path.join(application_support(app.name), 'data.json')\n            with open(filename) as f:\n                pass\n\n        "
        return open(os.path.join(self._application_support, args[0]), *args[1:])

    def run(self, **options):
        if False:
            i = 10
            return i + 15
        'Performs various setup tasks including creating the underlying Objective-C application, starting the timers,\n        and registering callback functions for click events. Then starts the application run loop.\n\n        .. versionchanged:: 0.2.1\n            Accepts `debug` keyword argument.\n\n        :param debug: determines if application should log information useful for debugging. Same effect as calling\n                      :func:`rumps.debug_mode`.\n\n        '
        dont_change = object()
        debug = options.get('debug', dont_change)
        if debug is not dont_change:
            debug_mode(debug)
        nsapplication = NSApplication.sharedApplication()
        nsapplication.activateIgnoringOtherApps_(True)
        self._nsapp = NSApp.alloc().init()
        self._nsapp._app = self.__dict__
        nsapplication.setDelegate_(self._nsapp)
        notifications._init_nsapp(self._nsapp)
        setattr(App, '*app_instance', self)
        t = b = None
        for t in getattr(timer, '*timers', []):
            t.start()
        for b in getattr(clicked, '*buttons', []):
            b(self)
        del t, b
        self._nsapp.initializeStatusBar()
        AppHelper.installMachInterrupt()
        events.before_start.emit()
        AppHelper.runEventLoop()