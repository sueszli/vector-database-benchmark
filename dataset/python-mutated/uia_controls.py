"""Wrap various UIA windows controls. To be used with 'uia' backend."""
import locale
import time
import comtypes
import six
from . import uiawrapper
from . import win32_controls
from . import common_controls
from .. import findbestmatch
from .. import timings
from ..windows import uia_defines as uia_defs
from ..windows.uia_defines import IUIA
from ..windows.uia_defines import NoPatternInterfaceError
from ..windows.uia_defines import toggle_state_on
from ..windows.uia_defines import get_elem_interface
from ..windows.uia_element_info import UIAElementInfo
from ..windows.uia_element_info import elements_from_uia_array

class WindowWrapper(uiawrapper.UIAWrapper):
    """Wrap a UIA-compatible Window control"""
    _control_types = ['Window']

    def __init__(self, elem):
        if False:
            while True:
                i = 10
        'Initialize the control'
        super(WindowWrapper, self).__init__(elem)

    def move_window(self, x=None, y=None, width=None, height=None):
        if False:
            return 10
        'Move the window to the new coordinates\n\n        * **x** Specifies the new left position of the window.\n                Defaults to the current left position of the window.\n        * **y** Specifies the new top position of the window.\n                Defaults to the current top position of the window.\n        * **width** Specifies the new width of the window.\n                Defaults to the current width of the window.\n        * **height** Specifies the new height of the window.\n                Defaults to the current height of the window.\n        '
        cur_rect = self.rectangle()
        if x is None:
            x = cur_rect.left
        else:
            try:
                y = x.top
                width = x.width()
                height = x.height()
                x = x.left
            except AttributeError:
                pass
        if y is None:
            y = cur_rect.top
        if width is None:
            width = cur_rect.width()
        if height is None:
            height = cur_rect.height()
        self.iface_transform.Move(x, y)
        self.iface_transform.Resize(width, height)
        time.sleep(timings.Timings.after_movewindow_wait)

    def is_dialog(self):
        if False:
            print('Hello World!')
        'Window is always a dialog so return True'
        return True

class ButtonWrapper(uiawrapper.UIAWrapper):
    """Wrap a UIA-compatible Button, CheckBox or RadioButton control"""
    _control_types = ['Button', 'CheckBox', 'RadioButton']

    def __init__(self, elem):
        if False:
            i = 10
            return i + 15
        'Initialize the control'
        super(ButtonWrapper, self).__init__(elem)

    def toggle(self):
        if False:
            print('Hello World!')
        "\n        An interface to Toggle method of the Toggle control pattern.\n\n        Control supporting the Toggle pattern cycles through its\n        toggle states in the following order:\n        ToggleState_On, ToggleState_Off and,\n        if supported, ToggleState_Indeterminate\n\n        Usually applied for the check box control.\n\n        The radio button control does not implement IToggleProvider,\n        because it is not capable of cycling through its valid states.\n        Toggle a state of a check box control. (Use 'select' method instead)\n        Notice, a radio button control isn't supported by UIA.\n        https://msdn.microsoft.com/en-us/library/windows/desktop/ee671290(v=vs.85).aspx\n        "
        name = self.element_info.name
        control_type = self.element_info.control_type
        self.iface_toggle.Toggle()
        if name and control_type:
            self.actions.log('Toggled ' + control_type.lower() + ' "' + name + '"')
        return self

    def get_toggle_state(self):
        if False:
            return 10
        '\n        Get a toggle state of a check box control.\n\n        The toggle state is represented by an integer\n        0 - unchecked\n        1 - checked\n        2 - indeterminate\n\n        The following constants are defined in the uia_defines module\n        toggle_state_off = 0\n        toggle_state_on = 1\n        toggle_state_inderteminate = 2\n        '
        return self.iface_toggle.CurrentToggleState

    def is_dialog(self):
        if False:
            return 10
        'Buttons are never dialogs so return False'
        return False

    def click(self):
        if False:
            print('Hello World!')
        'Click the Button control by using Invoke or Select patterns'
        try:
            self.invoke()
        except NoPatternInterfaceError:
            self.select()
        return self

    def items(self):
        if False:
            for i in range(10):
                print('nop')
        'Find all menu items'
        return self.children(control_type='MenuItem')

class ComboBoxWrapper(uiawrapper.UIAWrapper):
    """Wrap a UIA CoboBox control"""
    _control_types = ['ComboBox']

    def __init__(self, elem):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the control'
        super(ComboBoxWrapper, self).__init__(elem)

    def expand(self):
        if False:
            return 10
        if self.is_expanded():
            return self
        try:
            super(ComboBoxWrapper, self).expand()
        except NoPatternInterfaceError:
            open_buttons = self.children(name='Open', control_type='Button')
            if open_buttons:
                open_buttons[0].invoke()
            else:
                try:
                    self.invoke()
                except NoPatternInterfaceError:
                    raise NoPatternInterfaceError('There is no ExpandCollapsePattern and no "Open" button in .children(). Maybe only .click_input() would help to expand.')
        return self

    def collapse(self):
        if False:
            return 10
        if not self.is_expanded():
            return self
        try:
            super(ComboBoxWrapper, self).collapse()
        except NoPatternInterfaceError:
            close_buttons = self.children(name='Close', control_type='Button')
            if not close_buttons:
                if self.element_info.framework_id == 'WinForm':
                    return self
                else:
                    raise RuntimeError('There is no ExpandCollapsePattern and no "Close" button for the combo box')
            if self.is_editable():
                close_buttons[0].click_input()
            else:
                close_buttons[0].invoke()
        return self

    def is_editable(self):
        if False:
            return 10
        edit_children = self.children(control_type='Edit')
        return len(edit_children) > 0

    def get_expand_state(self):
        if False:
            print('Hello World!')
        try:
            return super(ComboBoxWrapper, self).get_expand_state()
        except NoPatternInterfaceError:
            children_list = self.children(control_type='List')
            if children_list and children_list[0].is_visible():
                if self.element_info.framework_id == 'Qt':
                    return uia_defs.expand_state_collapsed
                return uia_defs.expand_state_expanded
            else:
                return uia_defs.expand_state_collapsed

    def texts(self):
        if False:
            return 10
        'Return the text of the items in the combobox'
        texts = self._texts_from_item_container()
        if len(texts):
            return [t for lst in texts for t in lst]
        try:
            super(ComboBoxWrapper, self).expand()
            for c in self.children():
                texts.append(c.window_text())
        except NoPatternInterfaceError:
            children_lists = self.children(control_type='List')
            if children_lists:
                return children_lists[0].children_texts()
            elif self.handle:
                win32_combo = win32_controls.ComboBoxWrapper(self.handle)
                texts.extend(win32_combo.item_texts())
        else:
            super(ComboBoxWrapper, self).collapse()
        return texts

    def select(self, item):
        if False:
            print('Hello World!')
        '\n        Select the ComboBox item\n\n        The item can be either a 0 based index of the item to select\n        or it can be the string that you want to select\n        '
        self.expand()
        try:
            self._select(item)
        except (IndexError, NoPatternInterfaceError):
            children_lst = self.children(control_type='List')
            if len(children_lst) > 0:
                list_view = children_lst[0]
                list_view.get_item(item).select()
                if isinstance(item, six.string_types):
                    item_wrapper = list_view.children(name=item)[0]
                    item_value = item_wrapper.window_text()
                    if self.element_info.framework_id == 'Win32':
                        if self.selected_text() != item_value:
                            item_wrapper.invoke()
                            if self.selected_text() != item_value:
                                item_wrapper.click_input()
                    elif self.element_info.framework_id == 'Qt':
                        list_view._select(item)
                        if list_view.is_active():
                            item_wrapper.click_input()
                    elif self.selected_text() != item_value:
                        item_wrapper.invoke()
                        if self.selected_text() != item_value:
                            item_wrapper.click_input()
                            if self.selected_text() != item_value:
                                item_wrapper.click_input()
                elif self.selected_index() != item:
                    items = children_lst[0].children(control_type='ListItem')
                    if item < len(items):
                        if self.element_info.framework_id == 'Qt':
                            list_view._select(item)
                            if list_view.is_active():
                                items[item].click_input()
                        else:
                            items[item].invoke()
                    else:
                        raise IndexError('Item number #{} is out of range ({} items in total)'.format(item, len(items)))
            else:
                raise IndexError("item '{0}' not found or can't be accessed".format(item))
        finally:
            self.collapse()
        return self

    def selected_text(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the selected text or None\n\n        Notice, that in case of multi-select it will be only the text from\n        a first selected item\n        '
        try:
            selection = self.get_selection()
            if selection:
                return selection[0].name
            else:
                return None
        except NoPatternInterfaceError:
            return self.iface_value.CurrentValue

    def selected_index(self):
        if False:
            i = 10
            return i + 15
        'Return the selected index'
        try:
            return self.selected_item_index()
        except NoPatternInterfaceError:
            try:
                children_list_element = self.children(control_type='List')[0]
                children_list_element_values = children_list_element.texts()
                if type(children_list_element_values[0]) is list:
                    return children_list_element_values.index(self.selected_text().splitlines())
                else:
                    return children_list_element_values.index(self.selected_text())
            except IndexError:
                return self.texts().index(self.selected_text())

    def item_count(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the number of items in the combobox\n\n        The interface is kept mostly for a backward compatibility with\n        the native ComboBox interface\n        '
        children_list = self.children(control_type='List')
        if children_list:
            return children_list[0].control_count()
        else:
            self.expand()
            try:
                children_list = self.children(control_type='List')
                if children_list:
                    return children_list[0].control_count()
                else:
                    return self.control_count()
            finally:
                self.collapse()

class EditWrapper(uiawrapper.UIAWrapper):
    """Wrap an UIA-compatible Edit control"""
    _control_types = ['Edit']
    has_title = False

    def __init__(self, elem):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the control'
        super(EditWrapper, self).__init__(elem)

    @property
    def writable_props(self):
        if False:
            i = 10
            return i + 15
        'Extend default properties list.'
        props = super(EditWrapper, self).writable_props
        props.extend(['selection_indices'])
        return props

    def line_count(self):
        if False:
            for i in range(10):
                print('nop')
        'Return how many lines there are in the Edit'
        return self.window_text().count('\n') + 1

    def line_length(self, line_index):
        if False:
            while True:
                i = 10
        'Return how many characters there are in the line'
        lines = self.window_text().splitlines()
        if line_index < len(lines):
            return len(lines[line_index])
        elif line_index == self.line_count() - 1:
            return 0
        else:
            raise IndexError('There are only {0} lines but given index is {1}'.format(self.line_count(), line_index))

    def get_line(self, line_index):
        if False:
            while True:
                i = 10
        'Return the line specified'
        lines = self.window_text().splitlines()
        if line_index < len(lines):
            return lines[line_index]
        elif line_index == self.line_count() - 1:
            return ''
        else:
            raise IndexError('There are only {0} lines but given index is {1}'.format(self.line_count(), line_index))

    def get_value(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the current value of the element'
        return self.iface_value.CurrentValue

    def is_editable(self):
        if False:
            print('Hello World!')
        'Return the edit possibility of the element'
        return not self.iface_value.CurrentIsReadOnly

    def texts(self):
        if False:
            i = 10
            return i + 15
        'Get the text of the edit control'
        texts = [self.get_line(i) for i in range(self.line_count())]
        return texts

    def text_block(self):
        if False:
            i = 10
            return i + 15
        'Get the text of the edit control'
        return self.window_text()

    def selection_indices(self):
        if False:
            for i in range(10):
                print('nop')
        'The start and end indices of the current selection'
        selected_text = self.iface_text.GetSelection().GetElement(0).GetText(-1)
        start = self.window_text().find(selected_text)
        end = start + len(selected_text)
        return (start, end)

    def set_window_text(self, text, append=False):
        if False:
            return 10
        'Override set_window_text for edit controls because it should not be\n        used for Edit controls.\n\n        Edit Controls should either use set_edit_text() or type_keys() to modify\n        the contents of the edit control.\n        '
        self.verify_actionable()
        if append:
            text = self.window_text() + text
        self.set_focus()
        self.iface_value.SetValue(text)
        raise UserWarning('set_window_text() should probably not be called for Edit Controls')

    def set_edit_text(self, text, pos_start=None, pos_end=None):
        if False:
            return 10
        'Set the text of the edit control'
        self.verify_actionable()
        if pos_start is not None or pos_end is not None:
            (start, end) = self.selection_indices()
            if pos_start is None:
                pos_start = start
            if pos_end is None and (not isinstance(start, six.string_types)):
                pos_end = end
        else:
            pos_start = 0
            pos_end = len(self.window_text())
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
        current_text = self.window_text()
        new_text = current_text[:pos_start] + aligned_text + current_text[pos_end:]
        self.iface_value.SetValue(new_text)
        if isinstance(aligned_text, six.text_type):
            self.actions.log('Set text to the edit box: ' + aligned_text)
        else:
            self.actions.log(b'Set text to the edit box: ' + aligned_text)
        return self
    set_text = set_edit_text

    def select(self, start=0, end=None):
        if False:
            for i in range(10):
                print('nop')
        'Set the edit selection of the edit control'
        self.verify_actionable()
        self.set_focus()
        string_to_select = None
        if isinstance(start, six.text_type):
            string_to_select = start
        elif isinstance(start, six.binary_type):
            string_to_select = start.decode(locale.getpreferredencoding())
        elif isinstance(start, six.integer_types):
            if isinstance(end, six.integer_types) and start > end:
                (start, end) = (end, start)
            string_to_select = self.window_text()[start:end]
        if string_to_select:
            document_range = self.iface_text.DocumentRange
            search_range = document_range.FindText(string_to_select, False, False)
            try:
                search_range.Select()
            except ValueError:
                raise RuntimeError("Text '{0}' hasn't been found".format(string_to_select))
        return self

class TabControlWrapper(uiawrapper.UIAWrapper):
    """Wrap an UIA-compatible Tab control"""
    _control_types = ['Tab']

    def __init__(self, elem):
        if False:
            while True:
                i = 10
        'Initialize the control'
        super(TabControlWrapper, self).__init__(elem)

    def get_selected_tab(self):
        if False:
            return 10
        'Return an index of a selected tab'
        return self.selected_item_index()

    def tab_count(self):
        if False:
            print('Hello World!')
        'Return a number of tabs'
        return self.control_count()

    def select(self, item):
        if False:
            return 10
        'Select a tab by index or by name'
        self._select(item)
        return self

    def texts(self):
        if False:
            i = 10
            return i + 15
        'Tabs texts'
        return self.children_texts()

class SliderWrapper(uiawrapper.UIAWrapper):
    """Wrap an UIA-compatible Slider control"""
    _control_types = ['Slider']
    has_title = False

    def __init__(self, elem):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the control'
        super(SliderWrapper, self).__init__(elem)

    def min_value(self):
        if False:
            i = 10
            return i + 15
        'Get the minimum value of the Slider'
        return self.iface_range_value.CurrentMinimum

    def max_value(self):
        if False:
            print('Hello World!')
        'Get the maximum value of the Slider'
        return self.iface_range_value.CurrentMaximum

    def small_change(self):
        if False:
            print('Hello World!')
        "\n        Get a small change of slider's thumb\n\n        This change is achieved by pressing left and right arrows\n        when slider's thumb has keyboard focus.\n        "
        return self.iface_range_value.CurrentSmallChange

    def large_change(self):
        if False:
            return 10
        "\n        Get a large change of slider's thumb\n\n        This change is achieved by pressing PgUp and PgDown keys\n        when slider's thumb has keyboard focus.\n        "
        return self.iface_range_value.CurrentLargeChange

    def value(self):
        if False:
            while True:
                i = 10
        "Get a current position of slider's thumb"
        try:
            return self.iface_range_value.CurrentValue
        except NoPatternInterfaceError:
            return self.iface_value.CurrentValue

    def set_value(self, value):
        if False:
            i = 10
            return i + 15
        "Set position of slider's thumb"
        if isinstance(value, float):
            value_to_set = value
        elif isinstance(value, six.integer_types):
            value_to_set = value
        elif isinstance(value, six.text_type):
            value_to_set = float(value)
        else:
            raise ValueError('value should be either string or number')
        try:
            min_value = self.min_value()
            max_value = self.max_value()
            if not min_value <= value_to_set <= max_value:
                raise ValueError('value should be bigger than {0} and smaller than {1}'.format(min_value, max_value))
            self.iface_range_value.SetValue(value_to_set)
        except NoPatternInterfaceError:
            self.iface_value.SetValue(str(value))

class HeaderWrapper(uiawrapper.UIAWrapper):
    """Wrap an UIA-compatible Header control"""
    _control_types = ['Header']

    def __init__(self, elem):
        if False:
            while True:
                i = 10
        'Initialize the control'
        super(HeaderWrapper, self).__init__(elem)

class HeaderItemWrapper(uiawrapper.UIAWrapper):
    """Wrap an UIA-compatible Header Item control"""
    _control_types = ['HeaderItem']

    def __init__(self, elem):
        if False:
            i = 10
            return i + 15
        'Initialize the control'
        super(HeaderItemWrapper, self).__init__(elem)

class ListItemWrapper(uiawrapper.UIAWrapper):
    """Wrap an UIA-compatible ListViewItem control"""
    _control_types = ['DataItem', 'ListItem']

    def __init__(self, elem, container=None):
        if False:
            i = 10
            return i + 15
        'Initialize the control'
        super(ListItemWrapper, self).__init__(elem)
        self.container = container

    def is_checked(self):
        if False:
            return 10
        'Return True if the ListItem is checked\n\n        Only items supporting Toggle pattern should answer.\n        Raise NoPatternInterfaceError if the pattern is not supported\n        '
        return self.iface_toggle.ToggleState_On == toggle_state_on

    def texts(self):
        if False:
            while True:
                i = 10
        'Return a list of item texts'
        content = [ch.window_text() for ch in self.children(content_only=True)]
        if content:
            return content
        else:
            return super(ListItemWrapper, self).texts()

class ListViewWrapper(uiawrapper.UIAWrapper):
    """Wrap an UIA-compatible ListView control"""
    _control_types = ['DataGrid', 'List', 'Table']

    def __init__(self, elem):
        if False:
            while True:
                i = 10
        'Initialize the control'
        super(ListViewWrapper, self).__init__(elem)
        try:
            if self.iface_grid:
                self.iface_grid_support = True
        except NoPatternInterfaceError:
            self.iface_grid_support = False
        self.is_table = not self.iface_grid_support and self.element_info.control_type == 'Table'
        self.row_header = False
        self.col_header = False

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        return self.get_item(key)

    def __raise_not_implemented(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError('This method not work properly for WinForms DataGrid, use cells()')

    def __update_row_header(self):
        if False:
            i = 10
            return i + 15
        try:
            self.row_header = all((isinstance(six.next(row.iter_children()), HeaderWrapper) for row in self.children()))
        except StopIteration:
            self.row_header = False

    def __update_col_header(self):
        if False:
            while True:
                i = 10
        try:
            self.col_header = all((isinstance(col, HeaderWrapper) for col in six.next(self.iter_children()).children()))
        except StopIteration:
            self.col_header = False

    def __resolve_row_index(self, ind):
        if False:
            while True:
                i = 10
        self.__update_col_header()
        return ind + 1 if self.col_header and self.is_table else ind

    def __resolve_col_index(self, ind):
        if False:
            for i in range(10):
                print('nop')
        self.__update_row_header()
        return ind + 1 if self.row_header and self.is_table else ind

    def __resolve_row_count(self, cnt):
        if False:
            i = 10
            return i + 15
        self.__update_col_header()
        return cnt - 1 if self.col_header and self.is_table else cnt

    def item_count(self):
        if False:
            i = 10
            return i + 15
        'A number of items in the ListView'
        if self.iface_grid_support:
            return self.iface_grid.CurrentRowCount
        else:
            return self.__resolve_row_count(len(self.children()))

    def column_count(self):
        if False:
            i = 10
            return i + 15
        'Return the number of columns'
        if self.iface_grid_support:
            return self.iface_grid.CurrentColumnCount
        elif self.is_table:
            self.__raise_not_implemented()
        return 0

    def get_header_controls(self):
        if False:
            while True:
                i = 10
        'Return Header controls associated with the Table'
        return [cell for row in self.children() for cell in row.children() if isinstance(cell, HeaderWrapper)]

    def get_header_control(self):
        if False:
            return 10
        'Return Header control associated with the ListView'
        try:
            hdr = self.children(control_type='Header')[0]
        except (IndexError, NoPatternInterfaceError):
            hdr = None
        return hdr

    def get_column(self, col_index):
        if False:
            i = 10
            return i + 15
        'Get the information for a column of the ListView'
        col = None
        try:
            col = self.columns()[col_index]
        except comtypes.COMError:
            raise IndexError
        return col

    def columns(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the information on the columns of the ListView'
        if self.iface_grid_support:
            arr = self.iface_table.GetCurrentColumnHeaders()
            cols = elements_from_uia_array(arr)
            return [uiawrapper.UIAWrapper(e) for e in cols]
        elif self.is_table:
            self.__raise_not_implemented()
        else:
            return []

    def cells(self):
        if False:
            i = 10
            return i + 15
        'Return list of list of cells for any type of contol'
        row_start_index = self.__resolve_row_index(0)
        col_start_index = self.__resolve_col_index(0)
        rows = self.children(content_only=True)
        return [row.children(content_only=True)[col_start_index:] for row in rows[row_start_index:]]

    def cell(self, row, column):
        if False:
            return 10
        'Return a cell in the ListView control\n\n        Only for controls with Grid pattern support\n\n        * **row** is an index of a row in the list.\n        * **column** is an index of a column in the specified row.\n\n        The returned cell can be of different control types.\n        Mostly: TextBlock, ImageControl, EditControl, DataItem\n        or even another layer of data items (Group, DataGrid)\n        '
        if not isinstance(row, six.integer_types) or not isinstance(column, six.integer_types):
            raise TypeError('row and column must be numbers')
        if self.iface_grid_support:
            try:
                e = self.iface_grid.GetItem(row, column)
                elem_info = UIAElementInfo(e)
                cell_elem = uiawrapper.UIAWrapper(elem_info)
            except (comtypes.COMError, ValueError):
                raise IndexError
        elif self.is_table:
            _row = self.get_item(row)
            cell_elem = _row.children()[self.__resolve_col_index(column)]
        else:
            return None
        return cell_elem

    def get_item(self, row):
        if False:
            return 10
        'Return an item of the ListView control\n\n        * **row** can be either an index of the row or a string\n          with the text of a cell in the row you want returned.\n        '
        if isinstance(row, six.string_types):
            try:
                com_elem = self.iface_item_container.FindItemByProperty(0, IUIA().UIA_dll.UIA_NamePropertyId, row)
                try:
                    get_elem_interface(com_elem, 'VirtualizedItem').Realize()
                    itm = uiawrapper.UIAWrapper(UIAElementInfo(com_elem))
                except NoPatternInterfaceError:
                    itm = uiawrapper.UIAWrapper(UIAElementInfo(com_elem))
            except (NoPatternInterfaceError, ValueError):
                try:
                    itm = self.descendants(name=row)[0]
                    if not isinstance(itm, ListItemWrapper):
                        itm = itm.parent()
                except IndexError:
                    raise ValueError("Element '{0}' not found".format(row))
        elif isinstance(row, six.integer_types):
            try:
                com_elem = 0
                for _ in range(0, self.__resolve_row_index(row) + 1):
                    com_elem = self.iface_item_container.FindItemByProperty(com_elem, 0, uia_defs.vt_empty)
                try:
                    get_elem_interface(com_elem, 'VirtualizedItem').Realize()
                except NoPatternInterfaceError:
                    pass
                itm = uiawrapper.UIAWrapper(UIAElementInfo(com_elem))
            except (NoPatternInterfaceError, ValueError, AttributeError):
                list_items = self.children(content_only=True)
                itm = list_items[self.__resolve_row_index(row)]
        else:
            raise TypeError('String type or integer is expected')
        itm.container = self
        return itm
    item = get_item

    def get_items(self):
        if False:
            print('Hello World!')
        'Return all items of the ListView control'
        return self.children(content_only=True)
    items = get_items

    def get_item_rect(self, item_index):
        if False:
            i = 10
            return i + 15
        'Return the bounding rectangle of the list view item\n\n        The method is kept mostly for a backward compatibility\n        with the native ListViewWrapper interface\n        '
        itm = self.get_item(item_index)
        return itm.rectangle()

    def get_selected_count(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a number of selected items\n\n        The call can be quite expensieve as we retrieve all\n        the selected items in order to count them\n        '
        selection = self.get_selection()
        if selection:
            return len(selection)
        else:
            return 0

    def texts(self):
        if False:
            print('Hello World!')
        'Return a list of item texts'
        return [elem.texts() for elem in self.children(content_only=True)]

    @property
    def writable_props(self):
        if False:
            while True:
                i = 10
        'Extend default properties list.'
        props = super(ListViewWrapper, self).writable_props
        props.extend(['column_count', 'item_count', 'columns'])
        return props

class MenuItemWrapper(uiawrapper.UIAWrapper):
    """Wrap an UIA-compatible MenuItem control"""
    _control_types = ['MenuItem']

    def __init__(self, elem):
        if False:
            while True:
                i = 10
        'Initialize the control'
        super(MenuItemWrapper, self).__init__(elem)

    def items(self):
        if False:
            i = 10
            return i + 15
        'Find all items of the menu item'
        return self.children(control_type='MenuItem')

    def select(self):
        if False:
            for i in range(10):
                print('nop')
        'Apply Select pattern'
        try:
            self.iface_selection_item.Select()
        except NoPatternInterfaceError:
            try:
                self.iface_invoke.Invoke()
            except comtypes.COMError:
                self.click_input()

class MenuWrapper(uiawrapper.UIAWrapper):
    """Wrap an UIA-compatible MenuBar or Menu control"""
    _control_types = ['MenuBar', 'Menu']

    def __init__(self, elem):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the control'
        super(MenuWrapper, self).__init__(elem)

    def items(self):
        if False:
            return 10
        'Find all menu items'
        return self.children(control_type='MenuItem')

    def item_by_index(self, idx):
        if False:
            print('Hello World!')
        'Find a menu item specified by the index'
        item = self.items()[idx]
        return item

    def _activate(self, item, is_last):
        if False:
            while True:
                i = 10
        'Activate the specified item'
        if not item.is_active():
            item.set_focus()
        try:
            item.expand()
        except NoPatternInterfaceError:
            if self.element_info.framework_id == 'WinForm' and (not is_last):
                item.select()

    def _sub_item_by_text(self, menu, name, exact, is_last):
        if False:
            print('Hello World!')
        'Find a menu sub-item by the specified text'
        sub_item = None
        items = menu.items()
        if items:
            if exact:
                for i in items:
                    if name == i.window_text():
                        sub_item = i
                        break
            else:
                texts = []
                for i in items:
                    texts.append(i.window_text())
                sub_item = findbestmatch.find_best_match(name, texts, items)
        self._activate(sub_item, is_last)
        return sub_item

    def _sub_item_by_idx(self, menu, idx, is_last):
        if False:
            while True:
                i = 10
        'Find a menu sub-item by the specified index'
        sub_item = None
        items = menu.items()
        if items:
            sub_item = items[idx]
        self._activate(sub_item, is_last)
        return sub_item

    def item_by_path(self, path, exact=False):
        if False:
            return 10
        'Find a menu item specified by the path\n\n        The full path syntax is specified in:\n        :py:meth:`.controls.menuwrapper.Menu.get_menu_path`\n\n        Note: $ - specifier is not supported\n        '
        menu_items = [p.strip() for p in path.split('->')]
        items_cnt = len(menu_items)
        if items_cnt == 0:
            raise IndexError()
        for item in menu_items:
            if not item:
                raise IndexError("Empty item name between '->' separators")

        def next_level_menu(parent_menu, item_name, is_last):
            if False:
                for i in range(10):
                    print('nop')
            if item_name.startswith('#'):
                return self._sub_item_by_idx(parent_menu, int(item_name[1:]), is_last)
            else:
                return self._sub_item_by_text(parent_menu, item_name, exact, is_last)
        try:
            menu = next_level_menu(self, menu_items[0], items_cnt == 1)
            if items_cnt == 1:
                return menu
            if not menu.items():
                self._activate(menu, False)
                timings.wait_until(timings.Timings.window_find_timeout, timings.Timings.window_find_retry, lambda : len(self.top_level_parent().descendants(control_type='Menu')) > 0)
                menu = self.top_level_parent().descendants(control_type='Menu')[0]
            for i in range(1, items_cnt):
                menu = next_level_menu(menu, menu_items[i], items_cnt == i + 1)
        except AttributeError:
            raise IndexError()
        return menu

class TooltipWrapper(uiawrapper.UIAWrapper):
    """Wrap an UIA-compatible Tooltip control"""
    _control_types = ['ToolTip']

    def __init__(self, elem):
        if False:
            return 10
        'Initialize the control'
        super(TooltipWrapper, self).__init__(elem)

class ToolbarWrapper(uiawrapper.UIAWrapper):
    """Wrap an UIA-compatible ToolBar control

    The control's children usually are: Buttons, SplitButton,
    MenuItems, ThumbControls, TextControls, Separators, CheckBoxes.
    Notice that ToolTip controls are children of the top window and
    not of the toolbar.
    """
    _control_types = ['ToolBar']

    def __init__(self, elem):
        if False:
            return 10
        'Initialize the control'
        super(ToolbarWrapper, self).__init__(elem)
        self.win32_wrapper = None
        if len(self.children()) <= 1 and self.element_info.handle is not None:
            self.win32_wrapper = common_controls.ToolbarWrapper(self.element_info.handle)

    @property
    def writable_props(self):
        if False:
            i = 10
            return i + 15
        'Extend default properties list.'
        props = super(ToolbarWrapper, self).writable_props
        props.extend(['button_count'])
        return props

    def texts(self):
        if False:
            while True:
                i = 10
        'Return texts of the Toolbar'
        return [c.window_text() for c in self.buttons()]

    def button_count(self):
        if False:
            while True:
                i = 10
        'Return a number of buttons on the ToolBar'
        if self.win32_wrapper is not None:
            return len(self.buttons())
        else:
            return len(self.children())

    def buttons(self):
        if False:
            while True:
                i = 10
        'Return all available buttons'
        if self.win32_wrapper is not None:
            cc = []
            btn_count = self.win32_wrapper.button_count()
            if btn_count:
                rectangles = []
                for btn_num in range(btn_count):
                    relative_point = self.win32_wrapper.get_button_rect(btn_num).mid_point()
                    (button_coord_x, button_coord_y) = self.client_to_screen(relative_point)
                    btn_elem_info = UIAElementInfo.from_point(button_coord_x, button_coord_y)
                    button = uiawrapper.UIAWrapper(btn_elem_info)
                    if button.element_info.rectangle not in rectangles:
                        cc.append(button)
                        rectangles.append(button.element_info.rectangle)
            else:
                for btn in self.win32_wrapper.children():
                    cc.append(uiawrapper.UIAWrapper(UIAElementInfo(btn.handle)))
        else:
            cc = self.children()
        return cc

    def button(self, button_identifier, exact=True):
        if False:
            i = 10
            return i + 15
        'Return a button by the specified identifier\n\n        * **button_identifier** can be either an index of a button or\n          a string with the text of the button.\n        * **exact** flag specifies if the exact match for the text look up\n          has to be applied.\n        '
        cc = self.buttons()
        texts = [c.window_text() for c in cc]
        if isinstance(button_identifier, six.string_types):
            self.actions.log('Toolbar buttons: ' + str(texts))
            if exact:
                try:
                    button_index = texts.index(button_identifier)
                except ValueError:
                    raise findbestmatch.MatchError(items=texts, tofind=button_identifier)
            else:
                indices = [i for i in range(0, len(texts))]
                button_index = findbestmatch.find_best_match(button_identifier, texts, indices)
        else:
            button_index = button_identifier
        return cc[button_index]

    def check_button(self, button_identifier, make_checked, exact=True):
        if False:
            i = 10
            return i + 15
        "Find where the button is and toggle it\n\n        * **button_identifier** can be either an index of the button or\n          a string with the text on the button.\n        * **make_checked** specifies the required toggled state of the button.\n          If the button is already in the specified state the state isn't changed.\n        * **exact** flag specifies if the exact match for the text look up\n          has to be applied\n        "
        self.actions.logSectionStart('Checking "' + self.window_text() + '" toolbar button "' + str(button_identifier) + '"')
        button = self.button(button_identifier, exact=exact)
        if make_checked:
            self.actions.log('Pressing down toolbar button "' + str(button_identifier) + '"')
        else:
            self.actions.log('Pressing up toolbar button "' + str(button_identifier) + '"')
        if not button.is_enabled():
            self.actions.log('Toolbar button is not enabled!')
            raise RuntimeError('Toolbar button is not enabled!')
        res = button.get_toggle_state() == toggle_state_on
        if res != make_checked:
            button.toggle()
        self.actions.logSectionEnd()
        return button

    def _activate(self, item, is_last):
        if False:
            i = 10
            return i + 15
        'Activate the specified item'
        if not item.is_active():
            item.set_focus()
        try:
            item.expand()
        except NoPatternInterfaceError:
            if not is_last:
                item.select()

    def _sub_item_by_text(self, menu, name, exact, is_last):
        if False:
            while True:
                i = 10
        'Find a sub-item by the specified text'
        sub_item = None
        items = menu.items()
        if items:
            if exact:
                for i in items:
                    if name == i.window_text():
                        sub_item = i
                        break
            else:
                texts = []
                for i in items:
                    texts.append(i.window_text())
                sub_item = findbestmatch.find_best_match(name, texts, items)
        self._activate(sub_item, is_last)
        return sub_item

    def _sub_item_by_idx(self, menu, idx, is_last):
        if False:
            print('Hello World!')
        'Find a sub-item by the specified index'
        sub_item = None
        items = menu.items()
        if items:
            sub_item = items[idx]
        self._activate(sub_item, is_last)
        return sub_item
    items = buttons

    def item_by_path(self, path, exact=False):
        if False:
            return 10
        '\n        Walk the items in this toolbar to find the item specified by a path\n\n        The path is specified by a list of items separated by \'->\'. Each item\n        can be either a string (can include spaces) e.g. "Save As" or a zero\n        based index of the item to return prefaced by # e.g. #1.\n\n        These can be mixed as necessary. For example:\n            - "#0->Save As",\n            - "Tools->#0->Configure"\n\n        * **path** - Path to the specified item. **Required**.\n        * **exact** - If false, text matching will use a \'best match\' fuzzy algorithm. If true, will try to find the\n                      item with the given name. (Default False). **Optional**\n        '
        toolbar_items = [p.strip() for p in path.split('->')]
        items_cnt = len(toolbar_items)
        if items_cnt == 0:
            raise IndexError('Empty path is not accepted by the method!')
        for item in toolbar_items:
            if not item:
                raise IndexError("Empty item name between '->' separators")

        def next_level_menu(parent_menu, item_name, is_last):
            if False:
                for i in range(10):
                    print('nop')
            if item_name.startswith('#'):
                return self._sub_item_by_idx(parent_menu, int(item_name[1:]), is_last)
            else:
                return self._sub_item_by_text(parent_menu, item_name, exact, is_last)
        if items_cnt == 1:
            menu = next_level_menu(self, toolbar_items[0], True)
            return menu
        menu = self
        new_descendants = []
        for i in range(items_cnt):
            descendants_before = self.top_level_parent().descendants()
            if len(new_descendants) == 0:
                menu = next_level_menu(menu, toolbar_items[i], items_cnt == i + 1)
            else:
                new_descendants.append(menu)
                try:
                    for ctrl in new_descendants[::-1]:
                        try:
                            menu = next_level_menu(ctrl, toolbar_items[i], items_cnt == i + 1)
                        except AttributeError:
                            pass
                except findbestmatch.MatchError:
                    raise AttributeError("Could not find '{}' as a child of one of the following controls: {}".format(toolbar_items[i], new_descendants))
            descendants_after = self.top_level_parent().descendants()
            new_descendants = list(set(descendants_after) - set(descendants_before))
        return menu

class TreeItemWrapper(uiawrapper.UIAWrapper):
    """Wrap an UIA-compatible TreeItem control

    In addition to the provided methods of the wrapper
    additional inherited methods can be especially helpful:
    select(), extend(), collapse(), is_extended(), is_collapsed(),
    click_input(), rectangle() and many others
    """
    _control_types = ['TreeItem']

    def __init__(self, elem):
        if False:
            while True:
                i = 10
        'Initialize the control'
        super(TreeItemWrapper, self).__init__(elem)

    def is_checked(self):
        if False:
            i = 10
            return i + 15
        'Return True if the TreeItem is checked\n\n        Only items supporting Toggle pattern should answer.\n        Raise NoPatternInterfaceError if the pattern is not supported\n        '
        return self.iface_toggle.ToggleState_On == toggle_state_on

    def ensure_visible(self):
        if False:
            return 10
        'Make sure that the TreeView item is visible'
        self.iface_scroll_item.ScrollIntoView()

    def get_child(self, child_spec, exact=False):
        if False:
            return 10
        'Return the child item of this item\n\n        Accepts either a string or an index.\n        If a string is passed then it returns the child item\n        with the best match for the string.\n        '
        cc = self.children(control_type='TreeItem')
        if isinstance(child_spec, six.string_types):
            texts = [c.window_text() for c in cc]
            if exact:
                if child_spec in texts:
                    index = texts.index(child_spec)
                else:
                    raise IndexError('There is no child equal to "' + str(child_spec) + '" in ' + str(texts))
            else:
                indices = range(0, len(texts))
                index = findbestmatch.find_best_match(child_spec, texts, indices, limit_ratio=0.6)
        else:
            index = child_spec
        return cc[index]

    def _calc_click_coords(self):
        if False:
            return 10
        'Override the BaseWrapper helper method\n\n        Try to get coordinates of a text box inside the item.\n        If no text box found just set coordinates\n        close to a left part of the item rectangle\n\n        The returned coordinates are always absolute\n        '
        tt = self.children(control_type='Text')
        if tt:
            point = tt[0].rectangle().mid_point()
            coords = (point.x, point.y)
        else:
            rect = self.rectangle()
            coords = (rect.left + int(float(rect.width()) / 4.0), rect.top + int(float(rect.height()) / 2.0))
        return coords

    def sub_elements(self):
        if False:
            print('Hello World!')
        'Return a list of all visible sub-items of this control'
        return self.descendants(control_type='TreeItem')

class TreeViewWrapper(uiawrapper.UIAWrapper):
    """Wrap an UIA-compatible Tree control"""
    _control_types = ['Tree']

    def __init__(self, elem):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the control'
        super(TreeViewWrapper, self).__init__(elem)

    @property
    def writable_props(self):
        if False:
            i = 10
            return i + 15
        'Extend default properties list.'
        props = super(TreeViewWrapper, self).writable_props
        props.extend(['item_count'])
        return props

    def item_count(self):
        if False:
            return 10
        'Return a number of items in TreeView'
        return len(self.descendants(control_type='TreeItem'))

    def roots(self):
        if False:
            for i in range(10):
                print('nop')
        'Return root elements of TreeView'
        return self.children(control_type='TreeItem')

    def get_item(self, path, exact=False):
        if False:
            while True:
                i = 10
        "Read a TreeView item\n\n        * **path** a path to the item to return. This can be one of\n          the following:\n\n          * A string separated by \\\\ characters. The first character must\n            be \\\\. This string is split on the \\\\ characters and each of\n            these is used to find the specific child at each level. The\n            \\\\ represents the root item - so you don't need to specify the\n            root itself.\n          * A list/tuple of strings - The first item should be the root\n            element.\n          * A list/tuple of integers - The first item the index which root\n            to select. Indexing always starts from zero: get_item((0, 2, 3))\n\n        * **exact** a flag to request exact match of strings in the path\n          or apply a fuzzy logic of best_match thus allowing non-exact\n          path specifiers\n        "
        if not self.item_count():
            return None
        if isinstance(path, six.string_types):
            if not path.startswith('\\'):
                raise RuntimeError('Only absolute paths allowed - please start the path with \\')
            path = path.split('\\')[1:]
        current_elem = None
        if isinstance(path[0], int):
            current_elem = self.roots()[path[0]]
        else:
            roots = self.roots()
            texts = [r.window_text() for r in roots]
            if exact:
                if path[0] in texts:
                    current_elem = roots[texts.index(path[0])]
                else:
                    raise IndexError("There is no root element equal to '{0}'".format(path[0]))
            else:
                try:
                    current_elem = findbestmatch.find_best_match(path[0], texts, roots, limit_ratio=0.6)
                except IndexError:
                    raise IndexError("There is no root element similar to '{0}'".format(path[0]))
        for child_spec in path[1:]:
            try:
                current_elem.expand()
                current_elem = current_elem.get_child(child_spec, exact)
            except IndexError:
                if isinstance(child_spec, six.string_types):
                    raise IndexError("Item '{0}' does not have a child '{1}'".format(current_elem.window_text(), child_spec))
                else:
                    raise IndexError("Item '{0}' does not have {1} children".format(current_elem.window_text(), child_spec + 1))
            except comtypes.COMError:
                raise IndexError("Item '{0}' does not have a child '{1}'".format(current_elem.window_text(), child_spec))
        return current_elem

    def print_items(self):
        if False:
            return 10
        'Print all items with line indents'
        self.text = ''

        def _print_one_level(item, ident):
            if False:
                for i in range(10):
                    print('nop')
            'Get texts for the item and its children'
            self.text += ' ' * ident + item.window_text() + '\n'
            for child in item.children(control_type='TreeItem'):
                _print_one_level(child, ident + 1)
        for root in self.roots():
            _print_one_level(root, 0)
        return self.text

class StaticWrapper(uiawrapper.UIAWrapper):
    """Wrap an UIA-compatible Text control"""
    _control_types = ['Text']
    can_be_label = True

    def __init__(self, elem):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the control'
        super(StaticWrapper, self).__init__(elem)