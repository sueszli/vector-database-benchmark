"""Base class for all wrappers in all backends"""
from __future__ import unicode_literals
from __future__ import print_function
import abc
import locale
import re
import sys
import six
try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None
from time import sleep
from .actionlogger import ActionLogger
from .mouse import _get_cursor_pos
from .timings import TimeoutError
from .timings import Timings
from .timings import wait_until

def remove_non_alphanumeric_symbols(s):
    if False:
        i = 10
        return i + 15
    'Make text usable for attribute name'
    return re.sub('\\W', '_', s)

class InvalidElement(RuntimeError):
    """Raises when an invalid element is passed"""
    pass

class ElementNotEnabled(RuntimeError):
    """Raised when an element is not enabled"""
    pass

class ElementNotVisible(RuntimeError):
    """Raised when an element is not visible"""
    pass

class ElementNotActive(RuntimeError):
    """Raised when an element is not active"""
    pass

@six.add_metaclass(abc.ABCMeta)
class BaseMeta(abc.ABCMeta):
    """Abstract metaclass for Wrapper objects"""

    @staticmethod
    def find_wrapper(element):
        if False:
            print('Hello World!')
        'Abstract static method to find an appropriate wrapper'
        raise NotImplementedError()

@six.add_metaclass(BaseMeta)
class BaseWrapper(object):
    """
    Abstract wrapper for elements.

    All other wrappers are derived from this.
    """
    friendlyclassname = None
    windowclasses = []
    can_be_label = False
    has_title = True

    def __new__(cls, element_info, active_backend):
        if False:
            return 10
        return BaseWrapper._create_wrapper(cls, element_info, BaseWrapper)

    @staticmethod
    def _create_wrapper(cls_spec, element_info, myself):
        if False:
            while True:
                i = 10
        'Create a wrapper object according to the specified element info'
        if cls_spec != myself:
            obj = object.__new__(cls_spec)
            obj.__init__(element_info)
            return obj
        new_class = cls_spec.find_wrapper(element_info)
        obj = object.__new__(new_class)
        obj.__init__(element_info)
        return obj

    def __init__(self, element_info, active_backend):
        if False:
            i = 10
            return i + 15
        '\n        Initialize the element\n\n        * **element_info** is instance of int or one of ElementInfo childs\n        '
        self.backend = active_backend
        if element_info:
            self._element_info = element_info
            self.handle = self._element_info.handle
            self._as_parameter_ = self.handle
            self.ref = None
            self.appdata = None
            self._cache = {}
            self.actions = ActionLogger()
        else:
            raise RuntimeError('NULL pointer was used to initialize BaseWrapper')

    def by(self, **criteria):
        if False:
            while True:
                i = 10
        '\n        Create WindowSpecification for search in descendants by criteria\n\n        Current wrapper object is used as a parent while searching in the subtree.\n        '
        from .base_application import WindowSpecification
        if 'top_level_only' not in criteria:
            criteria['top_level_only'] = False
        criteria['backend'] = self.backend.name
        criteria['parent'] = self.element_info
        child_specification = WindowSpecification(criteria)
        return child_specification

    def __repr_texts(self):
        if False:
            return 10
        'Internal common method to be called from __str__ and __repr__'
        module = self.__class__.__module__
        module = module[module.rfind('.') + 1:]
        type_name = module + '.' + self.__class__.__name__
        title = self.window_text()
        class_name = self.friendly_class_name()
        if six.PY2:
            if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding is not None:
                title = title.encode(sys.stdout.encoding, errors='backslashreplace')
            else:
                title = title.encode(locale.getpreferredencoding(), errors='backslashreplace')
        return (type_name, title, class_name)

    def __repr__(self):
        if False:
            print('Hello World!')
        'Representation of the wrapper object\n\n        The method prints the following info:\n        * type name as a module name and a class name of the object\n        * title of the control or empty string\n        * friendly class name of the control\n        * unique ID of the control calculated as a hash value from a backend specific ID.\n\n        Notice that the reported title and class name can be used as hints to prepare\n        a windows specification to access the control, while the unique ID is more for\n        debugging purposes helping to distinguish between the runtime objects.\n        '
        (type_name, title, class_name) = self.__repr_texts()
        if six.PY2:
            return b"<{0} - '{1}', {2}, {3}>".format(type_name, title, class_name, self.__hash__())
        else:
            return "<{0} - '{1}', {2}, {3}>".format(type_name, title, class_name, self.__hash__())

    def __str__(self):
        if False:
            return 10
        'Pretty print representation of the wrapper object\n\n        The method prints the following info:\n        * type name as a module name and class name of the object\n        * title of the wrapped control or empty string\n        * friendly class name of the wrapped control\n\n        Notice that the reported title and class name can be used as hints\n        to prepare a window specification to access the control\n        '
        (type_name, title, class_name) = self.__repr_texts()
        if six.PY2:
            return b"{0} - '{1}', {2}".format(type_name, title, class_name)
        else:
            return "{0} - '{1}', {2}".format(type_name, title, class_name)

    @property
    def writable_props(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Build the list of the default properties to be written.\n\n        Derived classes may override or extend this list depending\n        on how much control they need.\n        '
        props = ['class_name', 'friendly_class_name', 'texts', 'control_id', 'rectangle', 'is_visible', 'is_enabled', 'control_count']
        return props

    @property
    def _needs_image_prop(self):
        if False:
            return 10
        'Specify whether we need to grab an image of ourselves\n\n        when asked for properties.\n        '
        return False

    @property
    def element_info(self):
        if False:
            i = 10
            return i + 15
        'Read-only property to get **ElementInfo** object'
        return self._element_info

    def from_point(self, x, y):
        if False:
            return 10
        'Get wrapper object for element at specified screen coordinates (x, y)'
        element_info = self.backend.element_info_class.from_point(x, y)
        return self.backend.generic_wrapper_class(element_info)

    def top_from_point(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        'Get wrapper object for top level element at specified screen coordinates (x, y)'
        top_element_info = self.backend.element_info_class.top_from_point(x, y)
        return self.backend.generic_wrapper_class(top_element_info)

    def get_active(self):
        if False:
            for i in range(10):
                print('nop')
        'Get wrapper object for active element'
        element_info = self.backend.element_info_class.get_active()
        return self.backend.generic_wrapper_class(element_info)

    def friendly_class_name(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the friendly class name for the control\n\n        This differs from the class of the control in some cases.\n        class_name() is the actual \'Registered\' element class of the control\n        while friendly_class_name() is hopefully something that will make\n        more sense to the user.\n\n        For example Checkboxes are implemented as Buttons - so the class\n        of a CheckBox is "Button" - but the friendly class is "CheckBox"\n        '
        if self.friendlyclassname is None:
            self.friendlyclassname = self.class_name()
        return self.friendlyclassname

    def class_name(self):
        if False:
            while True:
                i = 10
        'Return the class name of the elenemt'
        return self.element_info.class_name

    def window_text(self):
        if False:
            while True:
                i = 10
        '\n        Window text of the element\n\n        Quite a few controls have other text that is visible, for example\n        Edit controls usually have an empty string for window_text but still\n        have text displayed in the edit window.\n        '
        return self.element_info.rich_text

    def control_id(self):
        if False:
            print('Hello World!')
        "\n        Return the ID of the element\n\n        Only controls have a valid ID - dialogs usually have no ID assigned.\n\n        The ID usually identified the control in the window - but there can\n        be duplicate ID's for example lables in a dialog may have duplicate\n        ID's.\n        "
        return self.element_info.control_id

    def is_visible(self):
        if False:
            print('Hello World!')
        '\n        Whether the element is visible or not\n\n        Checks that both the top level parent (probably dialog) that\n        owns this element and the element itself are both visible.\n\n        If you want to wait for an element to become visible (or wait\n        for it to become hidden) use ``BaseWrapper.wait_visible()`` or\n        ``BaseWrapper.wait_not_visible()``.\n\n        If you want to raise an exception immediately if an element is\n        not visible then you can use the ``BaseWrapper.verify_visible()``.\n        ``BaseWrapper.verify_actionable()`` raises if the element is not both\n        visible and enabled.\n        '
        return self.element_info.visible

    def is_enabled(self):
        if False:
            return 10
        '\n        Whether the element is enabled or not\n\n        Checks that both the top level parent (probably dialog) that\n        owns this element and the element itself are both enabled.\n\n        If you want to wait for an element to become enabled (or wait\n        for it to become disabled) use ``BaseWrapper.wait_enabled()`` or\n        ``BaseWrapper.wait_not_enabled()``.\n\n        If you want to raise an exception immediately if an element is\n        not enabled then you can use the ``BaseWrapper.verify_enabled()``.\n        ``BaseWrapper.VerifyReady()`` raises if the window is not both\n        visible and enabled.\n        '
        return self.element_info.enabled

    def is_active(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Whether the element is active or not\n\n        Checks that both the top level parent (probably dialog) that\n        owns this element and the element itself are both active.\n\n        If you want to wait for an element to become active (or wait\n        for it to become not active) use ``BaseWrapper.wait_active()`` or\n        ``BaseWrapper.wait_not_active()``.\n\n        If you want to raise an exception immediately if an element is\n        not active then you can use the ``BaseWrapper.verify_active()``.\n        '
        return self.element_info.active

    def was_maximized(self):
        if False:
            for i in range(10):
                print('nop')
        'Indicate whether the window was maximized before minimizing or not'
        raise NotImplementedError

    def rectangle(self):
        if False:
            while True:
                i = 10
        '\n        Return the rectangle of element\n\n        The rectangle() is the rectangle of the element on the screen.\n        Coordinates are given from the top left of the screen.\n\n        This method returns a RECT structure, Which has attributes - top,\n        left, right, bottom. and has methods width() and height().\n        See win32structures.RECT for more information.\n        '
        return self.element_info.rectangle

    def client_to_screen(self, client_point):
        if False:
            print('Hello World!')
        'Maps point from client to screen coordinates'
        rect = self.element_info.rectangle
        return (client_point[0] + rect.left, client_point[1] + rect.top)

    def process_id(self):
        if False:
            print('Hello World!')
        'Return the ID of process that owns this window'
        return self.element_info.process_id

    def is_dialog(self):
        if False:
            return 10
        'Return True if the control is a top level window'
        if self.parent():
            return self == self.top_level_parent()
        else:
            return False

    def parent(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the parent of this element\n\n        Note that the parent of a control is not necesarily a dialog or\n        other main window. A group box may be the parent of some radio\n        buttons for example.\n\n        To get the main (or top level) window then use\n        BaseWrapper.top_level_parent().\n        '
        parent_elem = self.element_info.parent
        if parent_elem:
            return self.backend.generic_wrapper_class(parent_elem)
        else:
            return None

    def root(self):
        if False:
            return 10
        'Return wrapper for root element (desktop)'
        return self.backend.generic_wrapper_class(self.backend.element_info_class())

    def top_level_parent(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the top level window of this control\n\n        The TopLevel parent is different from the parent in that the parent\n        is the element that owns this element - but it may not be a dialog/main\n        window. For example most Comboboxes have an Edit. The ComboBox is the\n        parent of the Edit control.\n\n        This will always return a valid window element (if the control has\n        no top level parent then the control itself is returned - as it is\n        a top level window already!)\n        '
        if not 'top_level_parent' in self._cache.keys():
            self._cache['top_level_parent'] = self.backend.generic_wrapper_class(self.element_info.top_level_parent)
        return self._cache['top_level_parent']

    def texts(self):
        if False:
            while True:
                i = 10
        '\n        Return the text for each item of this control\n\n        It is a list of strings for the control. It is frequently overridden\n        to extract all strings from a control with multiple items.\n\n        It is always a list with one or more strings:\n\n          * The first element is the window text of the control\n          * Subsequent elements contain the text of any items of the\n            control (e.g. items in a listbox/combobox, tabs in a tabcontrol)\n        '
        texts_list = [self.window_text()]
        return texts_list

    def children(self, **kwargs):
        if False:
            return 10
        '\n        Return the children of this element as a list\n\n        It returns a list of BaseWrapper (or subclass) instances.\n        An empty list is returned if there are no children.\n        '
        child_elements = self.element_info.children(**kwargs)
        return [self.backend.generic_wrapper_class(element_info) for element_info in child_elements]

    def iter_children(self, **kwargs):
        if False:
            print('Hello World!')
        '\n        Iterate over the children of this element\n\n        It returns a generator of BaseWrapper (or subclass) instances.\n        '
        child_elements = self.element_info.iter_children(**kwargs)
        for element_info in child_elements:
            yield self.backend.generic_wrapper_class(element_info)

    def descendants(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the descendants of this element as a list\n\n        It returns a list of BaseWrapper (or subclass) instances.\n        An empty list is returned if there are no descendants.\n        '
        desc_elements = self.element_info.descendants(**kwargs)
        return [self.backend.generic_wrapper_class(element_info) for element_info in desc_elements]

    def iter_descendants(self, **kwargs):
        if False:
            print('Hello World!')
        '\n        Iterate over the descendants of this element\n\n        It returns a generator of BaseWrapper (or subclass) instances.\n        '
        desc_elements = self.element_info.iter_descendants(**kwargs)
        for element_info in desc_elements:
            yield self.backend.generic_wrapper_class(element_info)

    def control_count(self):
        if False:
            return 10
        'Return the number of children of this control'
        return len(self.element_info.children(process=self.process_id()))

    def capture_as_image(self, rect=None):
        if False:
            i = 10
            return i + 15
        '\n        Return a PIL image of the control.\n\n        See PIL documentation to know what you can do with the resulting\n        image.\n        '
        control_rectangle = self.rectangle()
        if not (control_rectangle.width() and control_rectangle.height()):
            return None
        if not ImageGrab:
            print('PIL does not seem to be installed. PIL is required for capture_as_image')
            self.actions.log('PIL does not seem to be installed. PIL is required for capture_as_image')
            return None
        if rect:
            control_rectangle = rect
        left = control_rectangle.left
        right = control_rectangle.right
        top = control_rectangle.top
        bottom = control_rectangle.bottom
        box = (left, top, right, bottom)
        return ImageGrab.grab(box)

    def get_properties(self):
        if False:
            i = 10
            return i + 15
        'Return the properties of the control as a dictionary.'
        props = {}
        for propname in self.writable_props:
            props[propname] = getattr(self, propname)()
        if self._needs_image_prop:
            props['image'] = self.capture_as_image()
        return props

    def draw_outline(self, colour='green', thickness=2, fill=None, rect=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Draw an outline around the window.\n\n        * **colour** can be either an integer or one of 'red', 'green', 'blue'\n          (default 'green')\n        * **thickness** thickness of rectangle (default 2)\n        * **fill** how to fill in the rectangle (default BS_NULL)\n        * **rect** the coordinates of the rectangle to draw (defaults to\n          the rectangle of the control)\n        "
        raise NotImplementedError()

    def is_child(self, parent):
        if False:
            i = 10
            return i + 15
        "\n        Return True if this element is a child of 'parent'.\n\n        An element is a child of another element when it is a direct of the\n        other element. An element is a direct descendant of a given\n        element if the parent element is the the chain of parent elements\n        for the child element.\n        "
        return self in parent.children(class_name=self.class_name())

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        "Return a unique hash value based on the element's handle"
        return self.element_info.__hash__()

    def __eq__(self, other):
        if False:
            return 10
        "Return True if 2 BaseWrapper's describe 1 actual element"
        if hasattr(other, 'element_info'):
            return self.element_info == other.element_info
        else:
            return self.element_info == other

    def __ne__(self, other):
        if False:
            return 10
        "Return False if the elements described by 2 BaseWrapper's are different"
        return not self == other

    def verify_actionable(self):
        if False:
            i = 10
            return i + 15
        '\n        Verify that the element is both visible and enabled\n\n        Raise either ElementNotEnalbed or ElementNotVisible if not\n        enabled or visible respectively.\n        '
        self.wait_for_idle()
        self.verify_visible()
        self.verify_enabled()

    def verify_enabled(self):
        if False:
            i = 10
            return i + 15
        "\n        Verify that the element is enabled\n\n        Check first if the element's parent is enabled (skip if no parent),\n        then check if element itself is enabled.\n        "
        if not self.is_enabled():
            raise ElementNotEnabled()

    def verify_visible(self):
        if False:
            while True:
                i = 10
        "\n        Verify that the element is visible\n\n        Check first if the element's parent is visible. (skip if no parent),\n        then check if element itself is visible.\n        "
        if not self.is_visible():
            raise ElementNotVisible()

    def verify_active(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Verify that the element is active\n\n        Check first if the element's parent is active. (skip if no parent),\n        then check if element itself is active.\n        "
        if not self.is_active():
            raise ElementNotActive()

    def click_input(self, button='left', coords=(None, None), button_down=True, button_up=True, double=False, wheel_dist=0, use_log=True, pressed='', absolute=False, key_down=True, key_up=True, fast_move=False):
        if False:
            for i in range(10):
                print('nop')
        "Click at the specified coordinates\n\n        * **button** The mouse button to click. One of 'left', 'right',\n          'middle' or 'x' (Default: 'left', 'move' is a special case)\n        * **coords** The coordinates to click at.(Default: the center of the control)\n        * **double** Whether to perform a double click or not (Default: False)\n        * **wheel_dist** The distance to move the mouse wheel (default: 0)\n\n        NOTES:\n           This is different from click method in that it requires the control\n           to be visible on the screen but performs a more realistic 'click'\n           simulation.\n\n           This method is also vulnerable if the mouse is moved by the user\n           as that could easily move the mouse off the control before the\n           click_input has finished.\n        "
        raise NotImplementedError()

    def double_click_input(self, button='left', coords=(None, None)):
        if False:
            while True:
                i = 10
        'Double click at the specified coordinates'
        self.click_input(button, coords, double=True)

    def right_click_input(self, coords=(None, None)):
        if False:
            return 10
        'Right click at the specified coords'
        self.click_input(button='right', coords=coords)

    def press_mouse_input(self, button='left', coords=(None, None), pressed='', absolute=True, key_down=True, key_up=True):
        if False:
            for i in range(10):
                print('nop')
        'Press a mouse button using SendInput'
        self.click_input(button=button, coords=coords, button_down=True, button_up=False, pressed=pressed, absolute=absolute, key_down=key_down, key_up=key_up)

    def release_mouse_input(self, button='left', coords=(None, None), pressed='', absolute=True, key_down=True, key_up=True):
        if False:
            print('Hello World!')
        'Release the mouse button'
        self.click_input(button, coords, button_down=False, button_up=True, pressed=pressed, absolute=absolute, key_down=key_down, key_up=key_up)

    def move_mouse_input(self, coords=(0, 0), pressed='', absolute=True, duration=0.0):
        if False:
            print('Hello World!')
        'Move the mouse'
        if not absolute:
            self.actions.log('Moving mouse to relative (client) coordinates ' + str(coords).replace('\n', ', '))
            coords = self.client_to_screen(coords)
        if not isinstance(duration, float):
            raise TypeError('duration must be float (in seconds)')
        minimum_duration = 0.05
        if duration >= minimum_duration:
            (x_start, y_start) = _get_cursor_pos()
            delta_x = coords[0] - x_start
            delta_y = coords[1] - y_start
            max_delta = max(abs(delta_x), abs(delta_y))
            num_steps = max_delta
            sleep_amount = duration / max(num_steps, 1)
            if sleep_amount < minimum_duration:
                num_steps = int(num_steps * sleep_amount / minimum_duration)
                sleep_amount = minimum_duration
            delta_x /= max(num_steps, 1)
            delta_y /= max(num_steps, 1)
            for step in range(num_steps):
                self.click_input(button='move', coords=(x_start + int(delta_x * step), y_start + int(delta_y * step)), absolute=True, pressed=pressed, fast_move=True)
                sleep(sleep_amount)
        self.click_input(button='move', coords=coords, absolute=True, pressed=pressed)
        self.wait_for_idle()
        return self

    def _calc_click_coords(self):
        if False:
            return 10
        'A helper that tries to get click coordinates of the control\n\n        The calculated coordinates are absolute and returned as\n        a tuple with x and y values.\n        '
        coords = self.rectangle().mid_point()
        return (coords.x, coords.y)

    def drag_mouse_input(self, dst=(0, 0), src=None, button='left', pressed='', absolute=True, duration=0.0):
        if False:
            for i in range(10):
                print('nop')
        'Click on **src**, drag it and drop on **dst**\n\n        * **dst** is a destination wrapper object or just coordinates.\n        * **src** is a source wrapper object or coordinates.\n          If **src** is None the self is used as a source object.\n        * **button** is a mouse button to hold during the drag.\n          It can be "left", "right", "middle" or "x"\n        * **pressed** is a key on the keyboard to press during the drag.\n        * **absolute** specifies whether to use absolute coordinates\n          for the mouse pointer locations\n        '
        raise NotImplementedError()

    def wheel_mouse_input(self, coords=(None, None), wheel_dist=1, pressed=''):
        if False:
            while True:
                i = 10
        'Do mouse wheel'
        self.click_input(button='wheel', coords=coords, wheel_dist=wheel_dist, pressed=pressed)
        return self

    def wait_for_idle(self):
        if False:
            while True:
                i = 10
        'Backend specific function to wait for idle state of a thread or a window'
        pass

    def type_keys(self, keys, pause=None, with_spaces=False, with_tabs=False, with_newlines=False, turn_off_numlock=True, set_foreground=True, vk_packet=True):
        if False:
            while True:
                i = 10
        '\n        Type keys to the element using keyboard.send_keys\n\n        This uses the re-written keyboard_ python module where you can\n        find documentation on what to use for the **keys**.\n\n        .. _keyboard: pywinauto.keyboard.html\n        '
        raise NotImplementedError()

    def set_focus(self):
        if False:
            return 10
        'Set the focus to this element'
        pass

    def wait_visible(self, timeout, retry_interval):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wait until control is visible.\n\n        :param timeout: Raise an :func:`pywinauto.timings.TimeoutError` if the window\n            is not visible after this number of seconds.\n            Default: :py:attr:`pywinauto.timings.Timings.window_find_timeout`.\n\n        :param retry_interval: How long to sleep between each retry.\n            Default: :py:attr:`pywinauto.timings.Timings.window_find_retry`.\n        '
        if timeout is None:
            timeout = Timings.window_find_timeout
        if retry_interval is None:
            retry_interval = Timings.window_find_retry
        try:
            wait_until(timeout, retry_interval, self.is_visible)
            return self
        except TimeoutError as e:
            raise e

    def wait_not_visible(self, timeout, retry_interval):
        if False:
            return 10
        '\n        Wait until control is not visible.\n\n        :param timeout: Raise an :func:`pywinauto.timings.TimeoutError` if the window\n            is still visible after this number of seconds.\n            Default: :py:attr:`pywinauto.timings.Timings.window_find_timeout`.\n\n        :param retry_interval: How long to sleep between each retry.\n            Default: :py:attr:`pywinauto.timings.Timings.window_find_retry`.\n        '
        if timeout is None:
            timeout = Timings.window_find_timeout
        if retry_interval is None:
            retry_interval = Timings.window_find_retry
        try:
            wait_until(timeout, retry_interval, self.is_visible, False)
        except TimeoutError as e:
            raise e

    def wait_enabled(self, timeout, retry_interval):
        if False:
            return 10
        '\n        Wait until control is enabled.\n\n        :param timeout: Raise an :func:`pywinauto.timings.TimeoutError` if the window\n            is not enabled after this number of seconds.\n            Default: :py:attr:`pywinauto.timings.Timings.window_find_timeout`.\n\n        :param retry_interval: How long to sleep between each retry.\n            Default: :py:attr:`pywinauto.timings.Timings.window_find_retry`.\n        '
        if timeout is None:
            timeout = Timings.window_find_timeout
        if retry_interval is None:
            retry_interval = Timings.window_find_retry
        try:
            wait_until(timeout, retry_interval, self.is_enabled)
            return self
        except TimeoutError as e:
            raise e

    def wait_not_enabled(self, timeout, retry_interval):
        if False:
            while True:
                i = 10
        '\n        Wait until control is not enabled.\n\n        :param timeout: Raise an :func:`pywinauto.timings.TimeoutError` if the window\n            is still enabled after this number of seconds.\n            Default: :py:attr:`pywinauto.timings.Timings.window_find_timeout`.\n\n        :param retry_interval: How long to sleep between each retry.\n            Default: :py:attr:`pywinauto.timings.Timings.window_find_retry`.\n        '
        if timeout is None:
            timeout = Timings.window_find_timeout
        if retry_interval is None:
            retry_interval = Timings.window_find_retry
        try:
            wait_until(timeout, retry_interval, self.is_enabled, False)
        except TimeoutError as e:
            raise e

    def wait_active(self, timeout, retry_interval):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wait until control is active.\n\n        :param timeout: Raise an :func:`pywinauto.timings.TimeoutError` if the window\n            is not active after this number of seconds.\n            Default: :py:attr:`pywinauto.timings.Timings.window_find_timeout`.\n\n        :param retry_interval: How long to sleep between each retry.\n            Default: :py:attr:`pywinauto.timings.Timings.window_find_retry`.\n        '
        if timeout is None:
            timeout = Timings.window_find_timeout
        if retry_interval is None:
            retry_interval = Timings.window_find_retry
        try:
            wait_until(timeout, retry_interval, self.is_active)
            return self
        except TimeoutError as e:
            raise e

    def wait_not_active(self, timeout, retry_interval):
        if False:
            i = 10
            return i + 15
        '\n        :param timeout: Raise an :func:`pywinauto.timings.TimeoutError` if the window\n            is still active after this number of seconds.\n            Default: :py:attr:`pywinauto.timings.Timings.window_find_timeout`.\n\n        :param retry_interval: How long to sleep between each retry.\n            Default: :py:attr:`pywinauto.timings.Timings.window_find_retry`.\n        '
        if timeout is None:
            timeout = Timings.window_find_timeout
        if retry_interval is None:
            retry_interval = Timings.window_find_retry
        try:
            wait_until(timeout, retry_interval, self.is_active, False)
        except TimeoutError as e:
            raise e