"""Wraps various standard windows controls. To be used with 'win32' backend"""
from __future__ import unicode_literals
import time
import ctypes
import win32gui
import locale
import six
from . import hwndwrapper
from ..windows import win32defines, win32functions, win32structures
from ..timings import Timings
from .. import deprecated

class ButtonWrapper(hwndwrapper.HwndWrapper):
    """Wrap a windows Button control"""
    friendlyclassname = 'Button'
    windowclasses = ['Button', '.*Button', 'WindowsForms\\d*\\.BUTTON\\..*', '.*CheckBox']
    can_be_label = True

    def __init__(self, hwnd):
        if False:
            print('Hello World!')
        'Initialize the control'
        super(ButtonWrapper, self).__init__(hwnd)

    @property
    def _needs_image_prop(self):
        if False:
            i = 10
            return i + 15
        '_needs_image_prop=True if it is an image button'
        style = self.style()
        if self.is_visible() and (style & win32defines.BS_BITMAP or style & win32defines.BS_ICON or style & win32defines.BS_OWNERDRAW):
            return True
        else:
            return False
    _NeedsImageProp = deprecated(_needs_image_prop, deprecated_name='_NeedsImageProp')

    def friendly_class_name(self):
        if False:
            while True:
                i = 10
        '\n        Return the friendly class name of the button\n\n        Windows controls with the class "Button" can look like different\n        controls based on their style. They can look like the following\n        controls:\n\n          - Buttons, this method returns "Button"\n          - CheckBoxes, this method returns "CheckBox"\n          - RadioButtons, this method returns "RadioButton"\n          - GroupBoxes, this method returns "GroupBox"\n        '
        style_lsb = self.style() & 15
        f_class_name = 'Button'
        vb_buttons = {'ThunderOptionButton': 'RadioButton', 'ThunderCheckBox': 'CheckBox', 'ThunderCommandButton': 'Button'}
        if self.class_name() in vb_buttons:
            f_class_name = vb_buttons[self.class_name()]
        if style_lsb in [win32defines.BS_3STATE, win32defines.BS_AUTO3STATE, win32defines.BS_AUTOCHECKBOX, win32defines.BS_CHECKBOX]:
            f_class_name = 'CheckBox'
        elif style_lsb in [win32defines.BS_RADIOBUTTON, win32defines.BS_AUTORADIOBUTTON]:
            f_class_name = 'RadioButton'
        elif style_lsb == win32defines.BS_GROUPBOX:
            f_class_name = 'GroupBox'
        if self.style() & win32defines.BS_PUSHLIKE:
            f_class_name = 'Button'
        return f_class_name

    def get_check_state(self):
        if False:
            while True:
                i = 10
        '\n        Return the check state of the checkbox\n\n        The check state is represented by an integer\n        0 - unchecked\n        1 - checked\n        2 - indeterminate\n\n        The following constants are defined in the win32defines module\n        BST_UNCHECKED = 0\n        BST_CHECKED = 1\n        BST_INDETERMINATE = 2\n        '
        self._ensure_enough_privileges('BM_GETCHECK')
        self.wait_for_idle()
        return self.send_message(win32defines.BM_GETCHECK)
    GetCheckState = deprecated(get_check_state)
    __check_states = {win32defines.BST_UNCHECKED: False, win32defines.BST_CHECKED: True, win32defines.BST_INDETERMINATE: None}

    def is_checked(self):
        if False:
            for i in range(10):
                print('nop')
        'Return True if checked, False if not checked, None if indeterminate'
        return self.__check_states[self.get_check_state()]

    def check(self):
        if False:
            i = 10
            return i + 15
        'Check a checkbox'
        self._ensure_enough_privileges('BM_SETCHECK')
        self.wait_for_idle()
        self.send_message_timeout(win32defines.BM_SETCHECK, win32defines.BST_CHECKED)
        self.wait_for_idle()
        time.sleep(Timings.after_buttoncheck_wait)
        return self
    Check = deprecated(check)

    def uncheck(self):
        if False:
            print('Hello World!')
        'Uncheck a checkbox'
        self._ensure_enough_privileges('BM_SETCHECK')
        self.wait_for_idle()
        self.send_message_timeout(win32defines.BM_SETCHECK, win32defines.BST_UNCHECKED)
        self.wait_for_idle()
        time.sleep(Timings.after_buttoncheck_wait)
        return self
    UnCheck = deprecated(uncheck, deprecated_name='UnCheck')

    def set_check_indeterminate(self):
        if False:
            while True:
                i = 10
        'Set the checkbox to indeterminate'
        self._ensure_enough_privileges('BM_SETCHECK')
        self.wait_for_idle()
        self.send_message_timeout(win32defines.BM_SETCHECK, win32defines.BST_INDETERMINATE)
        self.wait_for_idle()
        time.sleep(Timings.after_buttoncheck_wait)
        return self
    SetCheckIndeterminate = deprecated(set_check_indeterminate)

    def is_dialog(self):
        if False:
            i = 10
            return i + 15
        'Buttons are never dialogs so return False'
        return False

    def click(self, button='left', pressed='', coords=(0, 0), double=False, absolute=False):
        if False:
            i = 10
            return i + 15
        'Click the Button control'
        hwndwrapper.HwndWrapper.click(self, button, pressed, coords, double, absolute)
        time.sleep(Timings.after_button_click_wait)

    def check_by_click(self):
        if False:
            for i in range(10):
                print('nop')
        'Check the CheckBox control by click() method'
        if self.get_check_state() != win32defines.BST_CHECKED:
            self.click()
    CheckByClick = deprecated(check_by_click)

    def uncheck_by_click(self):
        if False:
            for i in range(10):
                print('nop')
        'Uncheck the CheckBox control by click() method'
        if self.get_check_state() != win32defines.BST_UNCHECKED:
            self.click()
    UncheckByClick = deprecated(uncheck_by_click)

    def check_by_click_input(self):
        if False:
            print('Hello World!')
        'Check the CheckBox control by click_input() method'
        if self.get_check_state() != win32defines.BST_CHECKED:
            self.click_input()
    CheckByClickInput = deprecated(check_by_click_input)

    def uncheck_by_click_input(self):
        if False:
            i = 10
            return i + 15
        'Uncheck the CheckBox control by click_input() method'
        if self.get_check_state() != win32defines.BST_UNCHECKED:
            self.click_input()
    UncheckByClickInput = deprecated(uncheck_by_click_input)

def _get_multiple_text_items(wrapper, count_msg, item_len_msg, item_get_msg):
    if False:
        i = 10
        return i + 15
    'Helper function to get multiple text items from a control'
    texts = []
    num_items = wrapper.send_message(count_msg)
    for i in range(0, num_items):
        text_len = wrapper.send_message(item_len_msg, i, 0)
        if six.PY3:
            text = ctypes.create_unicode_buffer(text_len + 1)
        else:
            text = ctypes.create_string_buffer(text_len + 1)
        wrapper.send_message(item_get_msg, i, ctypes.byref(text))
        if six.PY3:
            texts.append(text.value.replace('\u200e', ''))
        else:
            texts.append(text.value.decode(locale.getpreferredencoding(), 'ignore').replace('?', ''))
    return texts

class ComboBoxWrapper(hwndwrapper.HwndWrapper):
    """Wrap a windows ComboBox control"""
    friendlyclassname = 'ComboBox'
    windowclasses = ['ComboBox', 'WindowsForms\\d*\\.COMBOBOX\\..*', '.*ComboBox']
    has_title = False

    def __init__(self, hwnd):
        if False:
            return 10
        'Initialize the control'
        super(ComboBoxWrapper, self).__init__(hwnd)

    @property
    def writable_props(self):
        if False:
            i = 10
            return i + 15
        'Extend default properties list.'
        props = super(ComboBoxWrapper, self).writable_props
        props.extend(['selected_index', 'dropped_rect'])
        return props

    def dropped_rect(self):
        if False:
            return 10
        'Get the dropped rectangle of the combobox'
        dropped_rect = win32structures.RECT()
        self.send_message(win32defines.CB_GETDROPPEDCONTROLRECT, 0, ctypes.byref(dropped_rect))
        dropped_rect -= self.rectangle()
        return dropped_rect
    DroppedRect = deprecated(dropped_rect)

    def item_count(self):
        if False:
            return 10
        'Return the number of items in the combobox'
        self._ensure_enough_privileges('CB_GETCOUNT')
        return self.send_message(win32defines.CB_GETCOUNT)
    ItemCount = deprecated(item_count)

    def selected_index(self):
        if False:
            return 10
        'Return the selected index'
        self._ensure_enough_privileges('CB_GETCURSEL')
        return self.send_message(win32defines.CB_GETCURSEL)
    SelectedIndex = deprecated(selected_index)

    def selected_text(self):
        if False:
            return 10
        'Return the selected text'
        return self.item_texts()[self.selected_index()]
    SelectedText = deprecated(selected_text)

    def _get_item_index(self, ident):
        if False:
            i = 10
            return i + 15
        "Get the index for the item with this 'ident'"
        if isinstance(ident, six.integer_types):
            if ident >= self.item_count():
                raise IndexError(('Combobox has {0} items, you requested ' + 'item {1} (0 based)').format(self.item_count(), ident))
            if ident < 0:
                ident = self.item_count() + ident
        elif isinstance(ident, six.string_types):
            ident = self.item_texts().index(ident)
        return ident

    def item_data(self, item):
        if False:
            return 10
        'Returns the item data associated with the item if any'
        index = self._get_item_index(item)
        return self.send_message(win32defines.CB_GETITEMDATA, index)
    ItemData = deprecated(item_data)

    def item_texts(self):
        if False:
            return 10
        'Return the text of the items of the combobox'
        self._ensure_enough_privileges('CB_GETCOUNT')
        return _get_multiple_text_items(self, win32defines.CB_GETCOUNT, win32defines.CB_GETLBTEXTLEN, win32defines.CB_GETLBTEXT)
    ItemTexts = deprecated(item_texts)

    def texts(self):
        if False:
            print('Hello World!')
        'Return the text of the items in the combobox'
        texts = [self.window_text()]
        texts.extend(self.item_texts())
        return texts

    def get_properties(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the properties of the control as a dictionary'
        props = hwndwrapper.HwndWrapper.get_properties(self)
        return props

    def select(self, item):
        if False:
            for i in range(10):
                print('nop')
        'Select the ComboBox item\n\n        item can be either a 0 based index of the item to select\n        or it can be the string that you want to select\n        '
        self.verify_actionable()
        index = self._get_item_index(item)
        self.send_message_timeout(win32defines.CB_SETCURSEL, index, timeout=0.05)
        self.notify_parent(win32defines.CBN_SELENDOK)
        self.notify_parent(win32defines.CBN_SELCHANGE)
        if self.has_style(win32defines.CBS_DROPDOWN):
            self.notify_parent(win32defines.CBN_CLOSEUP)
        self.wait_for_idle()
        time.sleep(Timings.after_comboboxselect_wait)
        return self
    Select = deprecated(select)

class ListBoxWrapper(hwndwrapper.HwndWrapper):
    """Wrap a windows ListBox control"""
    friendlyclassname = 'ListBox'
    windowclasses = ['ListBox', 'WindowsForms\\d*\\.LISTBOX\\..*', '.*ListBox']
    has_title = False

    def __init__(self, hwnd):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the control'
        super(ListBoxWrapper, self).__init__(hwnd)

    @property
    def writable_props(self):
        if False:
            while True:
                i = 10
        'Extend default properties list.'
        props = super(ListBoxWrapper, self).writable_props
        props.extend(['selected_indices'])
        return props

    def is_single_selection(self):
        if False:
            print('Hello World!')
        'Check whether the listbox has single selection mode.'
        self._ensure_enough_privileges('LB_GETSELCOUNT')
        num_selected = self.send_message(win32defines.LB_GETSELCOUNT)
        return num_selected == win32defines.LB_ERR
    IsSingleSelection = deprecated(is_single_selection)

    def selected_indices(self):
        if False:
            return 10
        'The currently selected indices of the listbox'
        self._ensure_enough_privileges('LB_GETSELCOUNT')
        num_selected = self.send_message(win32defines.LB_GETSELCOUNT)
        if num_selected == win32defines.LB_ERR:
            items = tuple([self.send_message(win32defines.LB_GETCURSEL)])
        else:
            items = (ctypes.c_int * num_selected)()
            self.send_message(win32defines.LB_GETSELITEMS, num_selected, ctypes.byref(items))
            items = tuple(items)
        return items
    SelectedIndices = deprecated(selected_indices)

    def _get_item_index(self, ident):
        if False:
            for i in range(10):
                print('nop')
        "Return the index of the item 'ident'"
        if isinstance(ident, six.integer_types):
            if ident >= self.item_count():
                raise IndexError(('ListBox has {0} items, you requested ' + 'item {1} (0 based)').format(self.item_count(), ident))
            if ident < 0:
                ident = self.item_count() + ident
        elif isinstance(ident, six.string_types):
            ident = self.item_texts().index(ident)
        return ident

    def item_count(self):
        if False:
            i = 10
            return i + 15
        'Return the number of items in the ListBox'
        self._ensure_enough_privileges('LB_GETCOUNT')
        return self.send_message(win32defines.LB_GETCOUNT)
    ItemCount = deprecated(item_count)

    def item_data(self, i):
        if False:
            print('Hello World!')
        'Return the item_data if any associted with the item'
        index = self._get_item_index(i)
        return self.send_message(win32defines.LB_GETITEMDATA, index)
    ItemData = deprecated(item_data)

    def item_texts(self):
        if False:
            i = 10
            return i + 15
        'Return the text of the items of the listbox'
        self._ensure_enough_privileges('LB_GETCOUNT')
        return _get_multiple_text_items(self, win32defines.LB_GETCOUNT, win32defines.LB_GETTEXTLEN, win32defines.LB_GETTEXT)
    ItemTexts = deprecated(item_texts)

    def item_rect(self, item):
        if False:
            for i in range(10):
                print('nop')
        'Return the rect of the item'
        index = self._get_item_index(item)
        rect = win32structures.RECT()
        res = self.send_message(win32defines.LB_GETITEMRECT, index, ctypes.byref(rect))
        if res == win32defines.LB_ERR:
            raise RuntimeError('LB_GETITEMRECT failed')
        return rect
    ItemRect = deprecated(item_rect)

    def texts(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the texts of the control'
        texts = [self.window_text()]
        texts.extend(self.item_texts())
        return texts

    def select(self, item, select=True):
        if False:
            while True:
                i = 10
        'Select the ListBox item\n\n        item can be either a 0 based index of the item to select\n        or it can be the string that you want to select\n        '
        if self.is_single_selection() and isinstance(item, (list, tuple)) and (len(item) > 1):
            raise Exception('Cannot set multiple selection for single-selection listbox!')
        if isinstance(item, (list, tuple)):
            for i in item:
                if i is not None:
                    self.select(i, select)
            return self
        self.verify_actionable()
        index = self._get_item_index(item)
        if self.is_single_selection():
            self.send_message_timeout(win32defines.LB_SETCURSEL, index)
        elif select:
            self.send_message_timeout(win32defines.LB_SETSEL, win32defines.TRUE, index)
        else:
            self.send_message_timeout(win32defines.LB_SETSEL, win32defines.FALSE, index)
        self.notify_parent(win32defines.LBN_SELCHANGE)
        self.wait_for_idle()
        time.sleep(Timings.after_listboxselect_wait)
        return self
    Select = deprecated(select)

    def set_item_focus(self, item):
        if False:
            while True:
                i = 10
        'Set the ListBox focus to the item at index'
        index = self._get_item_index(item)
        self.wait_for_idle()
        if self.has_style(win32defines.LBS_EXTENDEDSEL) or self.has_style(win32defines.LBS_MULTIPLESEL):
            self.send_message_timeout(win32defines.LB_SETCARETINDEX, index)
        else:
            self.send_message_timeout(win32defines.LB_SETCURSEL, index)
        self.wait_for_idle()
        time.sleep(Timings.after_listboxfocuschange_wait)
        return self
    SetItemFocus = deprecated(set_item_focus)

    def get_item_focus(self):
        if False:
            while True:
                i = 10
        'Retrun the index of current selection in a ListBox'
        if self.has_style(win32defines.LBS_EXTENDEDSEL) or self.has_style(win32defines.LBS_MULTIPLESEL):
            return self.send_message(win32defines.LB_GETCARETINDEX)
        else:
            return self.send_message(win32defines.LB_GETCURSEL)
    GetItemFocus = deprecated(get_item_focus)

class EditWrapper(hwndwrapper.HwndWrapper):
    """Wrap a windows Edit control"""
    friendlyclassname = 'Edit'
    windowclasses = ['Edit', '.*Edit', 'TMemo', 'WindowsForms\\d*\\.EDIT\\..*', 'ThunderTextBox', 'ThunderRT6TextBox']
    has_title = False

    def __init__(self, hwnd):
        if False:
            while True:
                i = 10
        'Initialize the control'
        super(EditWrapper, self).__init__(hwnd)

    @property
    def writable_props(self):
        if False:
            for i in range(10):
                print('nop')
        'Extend default properties list.'
        props = super(EditWrapper, self).writable_props
        props.extend(['selection_indices'])
        return props

    def line_count(self):
        if False:
            for i in range(10):
                print('nop')
        'Return how many lines there are in the Edit'
        self._ensure_enough_privileges('EM_GETLINECOUNT')
        return self.send_message(win32defines.EM_GETLINECOUNT)
    LineCount = deprecated(line_count)

    def line_length(self, line_index):
        if False:
            i = 10
            return i + 15
        'Return how many characters there are in the line'
        self._ensure_enough_privileges('EM_LINEINDEX')
        char_index = self.send_message(win32defines.EM_LINEINDEX, line_index)
        return self.send_message(win32defines.EM_LINELENGTH, char_index, 0)
    LineLength = deprecated(line_length)

    def get_line(self, line_index):
        if False:
            while True:
                i = 10
        'Return the line specified'
        text_len = self.line_length(line_index)
        text = ctypes.create_unicode_buffer(text_len + 3)
        text[0] = six.unichr(text_len)
        win32functions.SendMessage(self, win32defines.EM_GETLINE, line_index, ctypes.byref(text))
        return text.value
    GetLine = deprecated(get_line)

    def texts(self):
        if False:
            return 10
        'Get the text of the edit control'
        texts = [self.window_text()]
        for i in range(self.line_count()):
            texts.append(self.get_line(i))
        return texts

    def text_block(self):
        if False:
            i = 10
            return i + 15
        'Get the text of the edit control'
        length = self.send_message(win32defines.WM_GETTEXTLENGTH)
        text = ctypes.create_unicode_buffer(length + 1)
        win32functions.SendMessage(self, win32defines.WM_GETTEXT, length + 1, ctypes.byref(text))
        return text.value
    TextBlock = deprecated(text_block)

    def selection_indices(self):
        if False:
            while True:
                i = 10
        'The start and end indices of the current selection'
        self._ensure_enough_privileges('EM_GETSEL')
        start = ctypes.c_int()
        end = ctypes.c_int()
        self.send_message(win32defines.EM_GETSEL, ctypes.byref(start), ctypes.byref(end))
        return (start.value, end.value)
    SelectionIndices = deprecated(selection_indices)

    def set_window_text(self, text, append=False):
        if False:
            while True:
                i = 10
        'Override set_window_text for edit controls because it should not be\n        used for Edit controls.\n\n        Edit Controls should either use set_edit_text() or type_keys() to modify\n        the contents of the edit control.'
        hwndwrapper.HwndWrapper.set_window_text(self, text, append)
        raise UserWarning('set_window_text() should probably not be called for Edit Controls')

    def set_edit_text(self, text, pos_start=None, pos_end=None):
        if False:
            print('Hello World!')
        'Set the text of the edit control'
        self._ensure_enough_privileges('EM_REPLACESEL')
        self.verify_actionable()
        if pos_start is not None or pos_end is not None:
            (start, end) = self.selection_indices()
            if pos_start is None:
                pos_start = start
            if pos_end is None and (not isinstance(start, six.string_types)):
                pos_end = end
            self.select(pos_start, pos_end)
        else:
            self.select()
        if isinstance(text, six.text_type):
            if six.PY3:
                aligned_text = text
            else:
                aligned_text = text.encode(locale.getpreferredencoding())
        elif isinstance(text, six.binary_type):
            if six.PY3:
                aligned_text = text.decode(locale.getpreferredencoding())
            else:
                aligned_text = text
        elif six.PY3:
            aligned_text = six.text_type(text)
        else:
            aligned_text = six.binary_type(text)
        if isinstance(aligned_text, six.text_type):
            buffer = ctypes.create_unicode_buffer(aligned_text, size=len(aligned_text) + 1)
        else:
            buffer = ctypes.create_string_buffer(aligned_text, size=len(aligned_text) + 1)
        self.send_message(win32defines.EM_REPLACESEL, True, ctypes.byref(buffer))
        if isinstance(aligned_text, six.text_type):
            self.actions.log('Set text to the edit box: ' + aligned_text)
        else:
            self.actions.log(b'Set text to the edit box: ' + aligned_text)
        return self
    set_text = set_edit_text
    SetText = deprecated(set_text)
    SetEditText = deprecated(set_edit_text)

    def select(self, start=0, end=None):
        if False:
            i = 10
            return i + 15
        'Set the edit selection of the edit control'
        self._ensure_enough_privileges('EM_SETSEL')
        self.verify_actionable()
        win32functions.SetFocus(self)
        if isinstance(start, six.text_type):
            string_to_select = start
            start = self.text_block().index(string_to_select)
            if end is None:
                end = start + len(string_to_select)
        elif isinstance(start, six.binary_type):
            string_to_select = start.decode(locale.getpreferredencoding())
            start = self.text_block().index(string_to_select)
            if end is None:
                end = start + len(string_to_select)
        if end is None:
            end = -1
        self.send_message(win32defines.EM_SETSEL, start, end)
        self.wait_for_idle()
        time.sleep(Timings.after_editselect_wait)
        return self
    Select = deprecated(select)

class StaticWrapper(hwndwrapper.HwndWrapper):
    """Wrap a windows Static control"""
    friendlyclassname = 'Static'
    windowclasses = ['Static', 'WindowsForms\\d*\\.STATIC\\..*', 'TPanel', '.*StaticText']
    can_be_label = True

    def __init__(self, hwnd):
        if False:
            print('Hello World!')
        'Initialize the control'
        super(StaticWrapper, self).__init__(hwnd)

    @property
    def _needs_image_prop(self):
        if False:
            for i in range(10):
                print('nop')
        '_needs_image_prop=True if it is an image static'
        if self.is_visible() and (self.has_style(win32defines.SS_ICON) or self.has_style(win32defines.SS_BITMAP) or self.has_style(win32defines.SS_CENTERIMAGE) or self.has_style(win32defines.SS_OWNERDRAW)):
            return True
        else:
            return False
    _NeedsImageProp = deprecated(_needs_image_prop, deprecated_name='_NeedsImageProp')

class PopupMenuWrapper(hwndwrapper.HwndWrapper):
    """Wrap a Popup Menu"""
    friendlyclassname = 'PopupMenu'
    windowclasses = ['#32768']
    has_title = False

    def is_dialog(self):
        if False:
            print('Hello World!')
        'Return whether it is a dialog'
        return True

    def _menu_handle(self):
        if False:
            print('Hello World!')
        'Get the menu handle for the popup menu'
        hMenu = win32gui.SendMessage(self.handle, win32defines.MN_GETHMENU)
        if not hMenu:
            raise ctypes.WinError()
        return (hMenu, False)