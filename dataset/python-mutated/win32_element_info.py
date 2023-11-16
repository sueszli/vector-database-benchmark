"""Implementation of the class to deal with a native element (window with a handle)"""
import ctypes
import six
import win32gui
from . import win32functions
from . import win32structures
from .. import handleprops
from ..element_info import ElementInfo
from .remote_memory_block import RemoteMemoryBlock

def _register_win_msg(msg_name):
    if False:
        i = 10
        return i + 15
    msg_id = win32functions.RegisterWindowMessage(six.text_type(msg_name))
    if not isinstance(msg_id, six.integer_types):
        return -1
    if msg_id > 0:
        return msg_id
    else:
        raise Exception('Cannot register {}'.format(msg_name))

class HwndElementInfo(ElementInfo):
    """Wrapper for window handler"""
    wm_get_ctrl_name = _register_win_msg('WM_GETCONTROLNAME')
    wm_get_ctrl_type = _register_win_msg('WM_GETCONTROLTYPE')
    re_props = ['class_name', 'name', 'auto_id', 'control_type', 'full_control_type']
    exact_only_props = ['handle', 'pid', 'control_id', 'enabled', 'visible', 'rectangle']
    search_order = ['handle', 'class_name', 'pid', 'control_id', 'visible', 'enabled', 'name', 'auto_id', 'control_type', 'full_control_type', 'rectangle']
    assert set(re_props + exact_only_props) == set(search_order)
    renamed_props = {'title': ('name', None), 'title_re': ('name_re', None), 'process': ('pid', None), 'visible_only': ('visible', {True: True, False: None}), 'enabled_only': ('enabled', {True: True, False: None}), 'top_level_only': ('depth', {True: 1, False: None})}

    def __init__(self, handle=None):
        if False:
            for i in range(10):
                print('nop')
        'Create element by handle (default is root element)'
        self._cache = {}
        if handle is None:
            self._handle = win32functions.GetDesktopWindow()
        else:
            self._handle = handle

    def set_cache_strategy(self, cached):
        if False:
            for i in range(10):
                print('nop')
        'Set a cache strategy for frequently used attributes of the element'
        pass

    @property
    def handle(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the handle of the window'
        return self._handle

    @property
    def rich_text(self):
        if False:
            i = 10
            return i + 15
        'Return the text of the window'
        return handleprops.text(self.handle)
    name = rich_text

    @property
    def control_id(self):
        if False:
            i = 10
            return i + 15
        'Return the ID of the window'
        return handleprops.controlid(self.handle)

    @property
    def process_id(self):
        if False:
            print('Hello World!')
        'Return the ID of process that controls this window'
        return handleprops.processid(self.handle)
    pid = process_id

    @property
    def class_name(self):
        if False:
            return 10
        'Return the class name of the window'
        return handleprops.classname(self.handle)

    @property
    def enabled(self):
        if False:
            print('Hello World!')
        'Return True if the window is enabled'
        return handleprops.isenabled(self.handle)

    @property
    def visible(self):
        if False:
            for i in range(10):
                print('nop')
        'Return True if the window is visible'
        return handleprops.isvisible(self.handle)

    @property
    def parent(self):
        if False:
            print('Hello World!')
        'Return the parent of the window'
        parent_hwnd = handleprops.parent(self.handle)
        if parent_hwnd:
            return HwndElementInfo(parent_hwnd)
        else:
            return None

    def children(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Return a list of immediate children of the window'
        class_name = kwargs.get('class_name', None)
        name = kwargs.get('name', None)
        control_type = kwargs.get('control_type', None)
        process = kwargs.get('process', None)
        child_elements = []

        def enum_window_proc(hwnd, lparam):
            if False:
                return 10
            'Called for each window - adds wrapped elements to a list'
            element = HwndElementInfo(hwnd)
            if process is not None and process != element.pid:
                return True
            if class_name is not None and class_name != element.class_name:
                return True
            if name is not None and name != element.rich_text:
                return True
            if control_type is not None and control_type != element.control_type:
                return True
            child_elements.append(element)
            return True
        enum_win_proc_t = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        proc = enum_win_proc_t(enum_window_proc)
        if self == HwndElementInfo():
            win32functions.EnumWindows(proc, 0)
        else:
            win32functions.EnumChildWindows(self.handle, proc, 0)
        return child_elements

    def iter_children(self, **kwargs):
        if False:
            return 10
        'Return a generator of immediate children of the window'
        for child in self.children(**kwargs):
            yield child

    def descendants(self, **kwargs):
        if False:
            print('Hello World!')
        'Return descendants of the window (all children from sub-tree)'
        if self == HwndElementInfo():
            top_elements = self.children()
            child_elements = self.children(**kwargs)
            for child in top_elements:
                child_elements.extend(child.children(**kwargs))
        else:
            child_elements = self.children(**kwargs)
        depth = kwargs.pop('depth', None)
        child_elements = ElementInfo.filter_with_depth(child_elements, self, depth)
        return child_elements

    @property
    def rectangle(self):
        if False:
            while True:
                i = 10
        'Return rectangle of the element'
        return handleprops.rectangle(self.handle)

    def dump_window(self):
        if False:
            return 10
        'Dump a window as a set of properties'
        return handleprops.dumpwindow(self.handle)

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        "Return a unique hash value based on the element's handle"
        return hash(self.handle)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Check if 2 HwndElementInfo objects describe 1 actual element'
        if not isinstance(other, HwndElementInfo):
            return self.handle == other
        return self.handle == other.handle

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        'Check if two HwndElementInfo objects describe different elements'
        return not self == other

    @property
    def auto_id(self):
        if False:
            print('Hello World!')
        'Return AutomationId of the element'
        textval = ''
        length = 1024
        remote_mem = RemoteMemoryBlock(self, size=length * 2)
        ret = win32gui.SendMessage(self.handle, self.wm_get_ctrl_name, length, remote_mem.mem_address)
        if ret:
            text = ctypes.create_unicode_buffer(length)
            remote_mem.Read(text)
            textval = text.value
        del remote_mem
        return textval

    def __get_control_type(self, full=False):
        if False:
            i = 10
            return i + 15
        'Internal parameterized method to distinguish control_type and full_control_type properties'
        textval = ''
        length = 1024
        remote_mem = RemoteMemoryBlock(self, size=length * 2)
        ret = win32gui.SendMessage(self.handle, self.wm_get_ctrl_type, length, remote_mem.mem_address)
        if ret:
            text = ctypes.create_unicode_buffer(length)
            remote_mem.Read(text)
            textval = text.value
        del remote_mem
        if not full and 'PublicKeyToken' in textval:
            textval = textval.split(', ')[0]
        return textval

    @property
    def control_type(self):
        if False:
            return 10
        'Return control type of the element'
        return self.__get_control_type(full=False)

    @property
    def full_control_type(self):
        if False:
            return 10
        'Return full string of control type of the element'
        return self.__get_control_type(full=True)

    @classmethod
    def from_point(cls, x, y):
        if False:
            for i in range(10):
                print('nop')
        'Return child element at specified point coordinates'
        current_handle = win32gui.WindowFromPoint((x, y))
        child_handle = win32gui.ChildWindowFromPoint(current_handle, (x, y))
        if child_handle:
            return cls(child_handle)
        else:
            return cls(current_handle)

    @classmethod
    def top_from_point(cls, x, y):
        if False:
            while True:
                i = 10
        'Return top level element at specified point coordinates'
        current_elem = cls.from_point(x, y)
        current_parent = current_elem.parent
        while current_parent is not None and current_parent != cls():
            current_elem = current_parent
            current_parent = current_elem.parent
        return current_elem

    @classmethod
    def get_active(cls):
        if False:
            return 10
        'Return current active element'
        gui_info = win32structures.GUITHREADINFO()
        gui_info.cbSize = ctypes.sizeof(gui_info)
        ret = win32functions.GetGUIThreadInfo(0, ctypes.byref(gui_info))
        if not ret:
            raise ctypes.WinError()
        hwndActive = gui_info.hwndActive
        return cls(hwndActive) if hwndActive is not None else None