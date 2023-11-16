"""Wrapper around Menu's and Menu items

These wrappers allow you to work easily with menu items.
You can select or click on items and check if they are
checked or unchecked.
"""
from __future__ import unicode_literals
import ctypes
import ctypes.wintypes
import time
import win32gui
import win32gui_struct
import locale
import six
from functools import wraps
from ..windows import win32defines, win32functions, win32structures
from .. import findbestmatch
from .. import mouse
from ..windows.remote_memory_block import RemoteMemoryBlock
from ..timings import Timings
from .. import deprecated

class MenuItemInfo(object):
    """A holder for Menu Item Info"""

    def __init__(self):
        if False:
            return 10
        self.fType = 0
        self.fState = 0
        self.wID = 0
        self.hSubMenu = 0
        self.hbmpChecked = 0
        self.hbmpUnchecked = 0
        self.dwItemData = 0
        self.text = ''
        self.hbmpItem = 0

class MenuInfo(object):
    """A holder for Menu Info"""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.dwStyle = 0
        self.cyMax = 0
        self.hbrBack = 0
        self.dwContextHelpID = 0
        self.dwMenuData = 0

class MenuItemNotEnabled(RuntimeError):
    """Raised when a menu item is not enabled"""
    pass

class MenuInaccessible(RuntimeError):
    """Raised when a menu has handle but inaccessible."""
    pass

def ensure_accessible(method):
    if False:
        while True:
            i = 10
    'Decorator for Menu instance methods'

    @wraps(method)
    def check(instance, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Check if the instance is accessible'
        if not instance.accessible:
            raise MenuInaccessible
        else:
            return method(instance, *args, **kwargs)
    return check

class MenuItem(object):
    """Wrap a menu item"""

    def __init__(self, ctrl, menu, index, on_main_menu=False):
        if False:
            return 10
        '\n        Initialize the menu item\n\n        * **ctrl**\tThe dialog or control that owns this menu\n        * **menu**\tThe menu that this item is on\n        * **index**\tThe Index of this menu item on the menu\n        * **on_main_menu**\tTrue if the item is on the main menu\n        '
        self._index = index
        self.menu = menu
        self.ctrl = ctrl
        self.on_main_menu = on_main_menu

    def _read_item(self):
        if False:
            for i in range(10):
                print('nop')
        'Read the menu item info\n\n        See https://msdn.microsoft.com/en-us/library/windows/desktop/ms647980.aspx\n        for more information.\n        '
        item_info = MenuItemInfo()
        (buf, extras) = win32gui_struct.EmptyMENUITEMINFO()
        win32gui.GetMenuItemInfo(self.menu.handle, self._index, True, buf)
        (item_info.fType, item_info.fState, item_info.wID, item_info.hSubMenu, item_info.hbmpChecked, item_info.hbmpUnchecked, item_info.dwItemData, item_info.text, item_info.hbmpItem) = win32gui_struct.UnpackMENUITEMINFO(buf)
        return item_info

    def friendly_class_name(self):
        if False:
            return 10
        'Return friendly class name'
        return 'MenuItem'
    FriendlyClassName = deprecated(friendly_class_name)

    def rectangle(self):
        if False:
            while True:
                i = 10
        'Get the rectangle of the menu item'
        rect = win32structures.RECT()
        if self.on_main_menu:
            ctrl = self.ctrl
        else:
            ctrl = 0
        hMenu = ctypes.wintypes.HMENU(self.menu.handle)
        win32functions.GetMenuItemRect(ctrl, hMenu, self._index, ctypes.byref(rect))
        return rect
    Rectangle = deprecated(rectangle)

    def index(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the index of this menu item'
        return self._index
    Index = deprecated(index)

    def state(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the state of this menu item'
        return self._read_item().fState
    State = deprecated(state)

    def item_id(self):
        if False:
            return 10
        'Return the ID of this menu item'
        return self._read_item().wID
    ID = deprecated(item_id, deprecated_name='ID')

    def item_type(self):
        if False:
            return 10
        '\n        Return the Type of this menu item\n\n        Main types are MF_STRING, MF_BITMAP, MF_SEPARATOR.\n\n        See https://msdn.microsoft.com/en-us/library/windows/desktop/ms647980.aspx\n        for further information.\n        '
        return self._read_item().fType
    Type = deprecated(item_type, deprecated_name='Type')

    def text(self):
        if False:
            i = 10
            return i + 15
        'Return the text of this menu item'
        item_info = self._read_item()
        if six.PY2:
            item_info.text = item_info.text.decode(locale.getpreferredencoding())
        if item_info.fType & 256 and (not item_info.text):
            mem = RemoteMemoryBlock(self.ctrl)
            address = item_info.dwItemData
            s = win32structures.LPWSTR()
            mem.Read(s, address)
            address = s
            s = ctypes.create_unicode_buffer(100)
            try:
                mem.Read(s, address)
                item_info.text = s.value
            except Exception:
                item_info.text = '!! non-supported owner drawn item !!'
            del mem
        return item_info.text
    Text = deprecated(text)

    def sub_menu(self):
        if False:
            return 10
        'Return the SubMenu or None if no submenu'
        submenu_handle = self._read_item().hSubMenu
        if submenu_handle:
            win32gui.SendMessageTimeout(self.ctrl.handle, win32defines.WM_INITMENUPOPUP, submenu_handle, self._index, win32defines.SMTO_NORMAL, 0)
            return Menu(self.ctrl, submenu_handle, False, self)
        return None
    SubMenu = deprecated(sub_menu)

    def is_enabled(self):
        if False:
            i = 10
            return i + 15
        'Return True if the item is enabled.'
        return not (self.state() & win32defines.MF_DISABLED or self.state() & win32defines.MF_GRAYED)
    IsEnabled = deprecated(is_enabled)

    def is_checked(self):
        if False:
            while True:
                i = 10
        'Return True if the item is checked.'
        return bool(self.state() & win32defines.MF_CHECKED)
    IsChecked = deprecated(is_checked)

    def click_input(self):
        if False:
            return 10
        "\n        Click on the menu item in a more realistic way\n\n        If the menu is open it will click with the mouse event on the item.\n        If the menu is not open each of it's parent's will be opened\n        until the item is visible.\n        "
        self.ctrl.verify_actionable()
        rect = self.rectangle()
        if not self.is_enabled():
            raise MenuItemNotEnabled('MenuItem {0} is disabled'.format(self.text()))
        if rect == win32structures.RECT(0, 0, 0, 0) and self.menu.owner_item:
            self.menu.owner_item.click_input()
        rect = self.rectangle()
        x_pt = int(float(rect.left + rect.right) / 2.0)
        y_pt = int(float(rect.top + rect.bottom) / 2.0)
        mouse.click(coords=(x_pt, y_pt))
        win32functions.WaitGuiThreadIdle(self.ctrl.handle)
        time.sleep(Timings.after_menu_wait)
    ClickInput = deprecated(click_input)

    def select(self):
        if False:
            print('Hello World!')
        '\n        Select the menu item\n\n        This will send a message to the parent window that the\n        item was picked.\n        '
        if not self.is_enabled():
            raise MenuItemNotEnabled('MenuItem {0} is disabled'.format(self.text()))
        command_id = self.item_id()
        self.ctrl.set_focus()
        self.ctrl.send_message_timeout(self.menu.COMMAND, command_id, timeout=1.0)
        win32functions.WaitGuiThreadIdle(self.ctrl.handle)
        time.sleep(Timings.after_menu_wait)
    click = select
    Click = deprecated(click)
    Select = deprecated(select)

    def get_properties(self):
        if False:
            i = 10
            return i + 15
        "\n        Return the properties for the item as a dict\n\n        If this item opens a sub menu then call Menu.get_properties()\n        to return the list of items in the sub menu. This is avialable\n        under the 'menu_items' key.\n        "
        props = {}
        props['index'] = self.index()
        props['state'] = self.state()
        props['item_type'] = self.item_type()
        props['item_id'] = self.item_id()
        props['text'] = self.text()
        submenu = self.sub_menu()
        if submenu:
            if submenu.accessible:
                props['menu_items'] = submenu.get_properties()
            else:
                props['menu_items'] = []
        return props
    GetProperties = deprecated(get_properties)

    def __repr__(self):
        if False:
            print('Hello World!')
        'Return a representation of the object as a string'
        if six.PY3:
            return '<MenuItem ' + self.text() + '>'
        else:
            return b'<MenuItem {}>'.format(self.text().encode(locale.getpreferredencoding(), errors='backslashreplace'))

class Menu(object):
    """A simple wrapper around a menu handle

    A menu supports methods for querying the menu
    and getting it's menu items.
    """

    def __init__(self, owner_ctrl, menuhandle, is_main_menu=True, owner_item=None):
        if False:
            return 10
        'Initialize the class\n\n        * **owner_ctrl** is the Control that owns this menu\n        * **menuhandle** is the menu handle of the menu\n        * **is_main_menu** we have to track whether it is the main menu\n          or a popup menu\n        * **owner_item** The item that contains this menu - this will be\n          None for the main menu, it will be a MenuItem instance for a\n          submenu.\n        '
        self.ctrl = owner_ctrl
        self.handle = menuhandle
        self.is_main_menu = is_main_menu
        self.owner_item = owner_item
        self._as_parameter_ = self.handle
        self.accessible = True
        if self.is_main_menu:
            self.ctrl.send_message_timeout(win32defines.WM_INITMENU, self.handle)
        menu_info = MenuInfo()
        buf = win32gui_struct.EmptyMENUINFO()
        try:
            win32gui.GetMenuInfo(self.handle, buf)
        except win32gui.error:
            self.accessible = False
        else:
            (menu_info.dwStyle, menu_info.cyMax, menu_info.hbrBack, menu_info.dwContextHelpID, menu_info.dwMenuData) = win32gui_struct.UnpackMENUINFO(buf)
            if menu_info.dwStyle & win32defines.MNS_NOTIFYBYPOS:
                self.COMMAND = win32defines.WM_MENUCOMMAND
            else:
                self.COMMAND = win32defines.WM_COMMAND

    @ensure_accessible
    def item_count(self):
        if False:
            i = 10
            return i + 15
        'Return the count of items in this menu'
        return win32gui.GetMenuItemCount(self.handle)
    ItemCount = deprecated(item_count)

    @ensure_accessible
    def item(self, index, exact=False):
        if False:
            i = 10
            return i + 15
        '\n        Return a specific menu item\n\n        * **index** is the 0 based index or text of the menu item you want.\n        * **exact** is True means exact matching for item text,\n                       False means best matching.\n        '
        if isinstance(index, six.string_types):
            if self.ctrl.appdata is not None:
                menu_appdata = self.ctrl.appdata['menu_items']
            else:
                menu_appdata = None
            return self.get_menu_path(index, appdata=menu_appdata, exact=exact)[-1]
        return MenuItem(self.ctrl, self, index, self.is_main_menu)
    Item = deprecated(item)

    @ensure_accessible
    def items(self):
        if False:
            while True:
                i = 10
        'Return a list of all the items in this menu'
        items = []
        for i in range(0, self.item_count()):
            items.append(self.item(i))
        return items
    Items = deprecated(items)

    @ensure_accessible
    def get_properties(self):
        if False:
            print('Hello World!')
        '\n        Return the properties for the menu as a list of dictionaries\n\n        This method is actually recursive. It calls get_properties() for each\n        of the items. If the item has a sub menu it will call this\n        get_properties to get the sub menu items.\n        '
        item_props = []
        for item in self.items():
            item_props.append(item.get_properties())
        return {'menu_items': item_props}
    GetProperties = deprecated(get_properties)

    @ensure_accessible
    def get_menu_path(self, path, path_items=None, appdata=None, exact=False):
        if False:
            print('Hello World!')
        '\n        Walk the items in this menu to find the item specified by a path\n\n        The path is specified by a list of items separated by \'->\'. Each item\n        can be either a string (can include spaces) e.g. "Save As" or a zero\n        based index of the item to return prefaced by # e.g. #1 or an ID of\n        the item prefaced by $ specifier.\n\n        These can be mixed as necessary. For example:\n            - "#0 -> Save As",\n            - "$23453 -> Save As",\n            - "Tools -> #0 -> Configure"\n\n        Text matching is done using a \'best match\' fuzzy algorithm, so you don\'t\n        have to add all punctuation, ellipses, etc.\n        ID matching is performed against wID field of MENUITEMINFO structure\n        (https://msdn.microsoft.com/en-us/library/windows/desktop/ms647578(v=vs.85).aspx)\n        '
        if path_items is None:
            path_items = []
        parts = [p.strip() for p in path.split('->', 1)]
        current_part = parts[0]
        if current_part.startswith('#'):
            index = int(current_part[1:])
            best_item = self.item(index)
        elif current_part.startswith('$'):
            if appdata is None:
                item_IDs = [item.item_id() for item in self.items()]
            else:
                item_IDs = [item['item_id'] for item in appdata]
            item_id = int(current_part[1:])
            best_item = self.item(item_IDs.index(item_id))
        else:
            if appdata is None:
                item_texts = [item.text() for item in self.items()]
            else:
                item_texts = [item['text'] for item in appdata]
            if exact:
                if current_part not in item_texts:
                    raise IndexError('There are no menu item "' + str(current_part) + '" in ' + str(item_texts))
                best_item = self.items()[item_texts.index(current_part)]
            else:
                best_item = findbestmatch.find_best_match(current_part, item_texts, self.items())
        path_items.append(best_item)
        if parts[1:]:
            if appdata:
                appdata = appdata[best_item.index()]['menu_items']
            if best_item.sub_menu() is not None:
                best_item.sub_menu().get_menu_path('->'.join(parts[1:]), path_items, appdata, exact=exact)
        return path_items
    GetMenuPath = deprecated(get_menu_path)

    def __repr__(self):
        if False:
            while True:
                i = 10
        'Return a simple representation of the menu'
        return '<Menu {0}>'.format(self.handle)