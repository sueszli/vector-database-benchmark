"""Wrap various Linux ATSPI windows controls. To be used with 'atspi' backend"""
import locale
import six
from . import atspiwrapper
from .. import findbestmatch
from ..linux.atspi_objects import AtspiImage
from ..linux.atspi_objects import AtspiDocument
from ..linux.atspi_objects import AtspiText
from ..linux.atspi_objects import AtspiEditableText

class ButtonWrapper(atspiwrapper.AtspiWrapper):
    """Wrap a Atspi-compatible Button, CheckBox or RadioButton control"""
    _control_types = ['PushButton', 'CheckBox', 'ToggleButton', 'RadioButton']

    def __init__(self, elem):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the control'
        super(ButtonWrapper, self).__init__(elem)
        self.action = self.element_info.get_action()

    def click(self):
        if False:
            print('Hello World!')
        'Click the Button control'
        self.action.do_action_by_name('click')
        return self

    def toggle(self):
        if False:
            print('Hello World!')
        'Method to change toggle button state\n\n        Currently, just a wrapper around the click() method\n        '
        return self.click()

    def get_toggle_state(self):
        if False:
            print('Hello World!')
        'Get a toggle state of a check box control'
        return 'STATE_CHECKED' in self.element_info.get_state_set()

    def is_dialog(self):
        if False:
            while True:
                i = 10
        'Buttons are never dialogs so return False'
        return False

class ComboBoxWrapper(atspiwrapper.AtspiWrapper):
    """Wrap a AT-SPI ComboBox control"""
    _control_types = ['ComboBox']

    def __init__(self, elem):
        if False:
            print('Hello World!')
        'Initialize the control'
        super(ComboBoxWrapper, self).__init__(elem)
        self.action = self.element_info.get_action()

    def _press(self):
        if False:
            i = 10
            return i + 15
        "Perform 'press' action on the control"
        self.action.do_action_by_name('press')

    def expand(self):
        if False:
            while True:
                i = 10
        'Drop down list of items of the control'
        if not self.is_expanded():
            self._press()
        return self

    def collapse(self):
        if False:
            for i in range(10):
                print('nop')
        'Hide list of items of the control'
        if self.is_expanded():
            self._press()
        return self

    def is_expanded(self):
        if False:
            for i in range(10):
                print('nop')
        'Test if the control is expanded'
        return self.children()[0].is_visible()

    def texts(self):
        if False:
            return 10
        'Get texts of all items in the control as list'
        combo_box_container = self.children()[0]
        texts = []
        for el in combo_box_container.children():
            texts.append(el.window_text())
        return texts

    def selected_text(self):
        if False:
            print('Hello World!')
        'Return the selected text'
        return self.window_text()

    def selected_index(self):
        if False:
            while True:
                i = 10
        'Return the selected index'
        return self.texts().index(self.selected_text())

    def item_count(self):
        if False:
            for i in range(10):
                print('nop')
        'Number of items in the control'
        combo_box_container = self.children()[0]
        return combo_box_container.control_count()

    def select(self, item):
        if False:
            for i in range(10):
                print('nop')
        'Select the control item.\n\n        Item can be specified as string or as index\n        '
        self.expand()
        children_lst = self.children(control_type='Menu')
        if len(children_lst) > 0:
            if isinstance(item, six.string_types):
                if self.selected_text() != item:
                    item = children_lst[0].children(name=item)[0]
                    item.click()
            elif self.selected_index() != item:
                items = children_lst[0].children(control_type='MenuItem')
                if item < len(items):
                    items[item].click()
                else:
                    raise IndexError('Item number #{} is out of range ({} items in total)'.format(item, len(items)))
        self.collapse()
        return self

class EditWrapper(atspiwrapper.AtspiWrapper):
    """Wrap single-line and multiline text edit controls"""
    _control_types = ['Text']

    def __init__(self, elem):
        if False:
            print('Hello World!')
        'Initialize the control'
        super(EditWrapper, self).__init__(elem)
        self.text = AtspiText(self.element_info.atspi_accessible.get_text(self))

    def is_editable(self):
        if False:
            return 10
        'Return the edit possibility of the element'
        return 'STATE_EDITABLE' in self.element_info.get_state_set()

    def window_text(self):
        if False:
            print('Hello World!')
        'Window text of the element'
        return self.text.get_whole_text().decode(locale.getpreferredencoding())

    def text_block(self):
        if False:
            while True:
                i = 10
        'Get the text of the edit control\n\n        Currently, only a wrapper around window_text()\n        '
        return self.window_text()

    def line_count(self):
        if False:
            return 10
        'Return how many lines there are in the Edit'
        return self.window_text().count('\n') + 1

    def line_length(self, line_index):
        if False:
            for i in range(10):
                print('nop')
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

    def texts(self):
        if False:
            i = 10
            return i + 15
        'Get the texts of the edit control as a lines array'
        return [self.get_line(i) for i in range(self.line_count())]

    def selection_indices(self):
        if False:
            i = 10
            return i + 15
        'The start and end indices of the current selection'
        return self.text.get_selection()

    def set_edit_text(self, text, pos_start=None, pos_end=None):
        if False:
            i = 10
            return i + 15
        'Set the text of the edit control'
        self.verify_enabled()
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
        editable_text = AtspiEditableText(self.element_info.atspi_accessible.get_editable_text(self))
        editable_text.set_text(new_text.encode(locale.getpreferredencoding()))
        return self
    set_text = set_edit_text

    def select(self, start=0, end=None):
        if False:
            i = 10
            return i + 15
        'Set the edit selection of the edit control'
        self.verify_enabled()
        self.set_focus()
        string_to_select = False
        if isinstance(start, six.text_type):
            string_to_select = start
        elif isinstance(start, six.binary_type):
            string_to_select = start.decode(locale.getpreferredencoding())
        elif isinstance(start, six.integer_types):
            if isinstance(end, six.integer_types) and start > end:
                (start, end) = (end, start)
        if string_to_select:
            start = self.window_text().find(string_to_select)
            if start == -1:
                raise RuntimeError("Text '{0}' hasn't been found".format(string_to_select))
            end = start + len(string_to_select)
        self.text.add_selection(start, end)
        return self

class ImageWrapper(atspiwrapper.AtspiWrapper):
    """Wrap image controls"""
    _control_types = ['Image', 'Icon']

    def __init__(self, elem):
        if False:
            return 10
        'Initialize the control'
        super(ImageWrapper, self).__init__(elem)
        self.image = AtspiImage(self.element_info.atspi_accessible.get_image(self))

    def description(self):
        if False:
            print('Hello World!')
        'Get image description'
        return self.image.get_description().decode(encoding='UTF-8')

    def locale(self):
        if False:
            while True:
                i = 10
        'Get image locale'
        return self.image.get_locale().decode(encoding='UTF-8')

    def size(self):
        if False:
            while True:
                i = 10
        'Get image size. Return a tuple with width and height'
        pnt = self.image.get_size()
        return (pnt.x, pnt.y)

    def bounding_box(self):
        if False:
            i = 10
            return i + 15
        'Get image bounding box'
        return self.image.get_extents()

    def position(self):
        if False:
            i = 10
            return i + 15
        'Get image position coordinates'
        return self.image.get_position()

class DocumentWrapper(atspiwrapper.AtspiWrapper):
    """Wrap document control"""
    _control_types = ['DocumentFrame']

    def __init__(self, elem):
        if False:
            while True:
                i = 10
        'Initialize the control'
        super(DocumentWrapper, self).__init__(elem)
        self.document = AtspiDocument(self.element_info.atspi_accessible.get_document(elem.handle))

    def locale(self):
        if False:
            while True:
                i = 10
        "Return the document's content locale"
        return self.document.get_locale().decode(encoding='UTF-8')

    def attribute_value(self, attrib):
        if False:
            return 10
        "Return the document's attribute value"
        return self.document.get_attribute_value(attrib).decode(encoding='UTF-8')

    def attributes(self):
        if False:
            print('Hello World!')
        "Return the document's constant attributes"
        return self.document.get_attributes()

class MenuWrapper(atspiwrapper.AtspiWrapper):
    """Wrap an Atspi-compatible MenuBar, Menu or MenuItem control"""
    _control_types = ['MenuBar', 'Menu', 'MenuItem']

    def __init__(self, elem):
        if False:
            i = 10
            return i + 15
        'Initialize the control'
        super(MenuWrapper, self).__init__(elem)
        self.action = self.element_info.get_action()
        self.state = self.element_info.get_state_set()

    def items(self):
        if False:
            return 10
        'Find all menu and menu items'
        menus = self.descendants(control_type='Menu')
        menu_items = self.descendants(control_type='MenuItem')
        return menus + menu_items

    def selected_menu_name(self):
        if False:
            while True:
                i = 10
        'Return the selected text'
        return self.element_info.name

    def selected_index(self):
        if False:
            while True:
                i = 10
        'Return the selected index'
        menu_name = self.element_info.name
        par = self.element_info.parent
        children = []
        for child in par.descendants():
            if child.control_type in ['Menu', 'MenuItem']:
                children.append(child)
        for (i, c) in enumerate(children):
            if c.name == menu_name:
                num = i
        return num

    def item_count(self):
        if False:
            print('Hello World!')
        'Number of items in the control'
        children = []
        for child in self.descendants():
            if child.element_info.control_type in ['Menu', 'MenuItem']:
                children.append(child)
        return len(children)

    def click(self):
        if False:
            print('Hello World!')
        'Click the Button control'
        self.action.do_action_by_name('click')
        return self

    def item_by_index(self, idx):
        if False:
            while True:
                i = 10
        'Find a menu item specified by the index'
        item = self.items()[idx]
        return item

    def _activate(self, item):
        if False:
            return 10
        'Activate the specified item'
        if not item.is_active():
            item.set_focus()
        item.action.do_action_by_name('click')

    def _sub_item_by_text(self, menu, name, exact):
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
        self._activate(sub_item)
        return sub_item

    def _sub_item_by_idx(self, menu, idx):
        if False:
            for i in range(10):
                print('nop')
        'Find a menu sub-item by the specified index'
        sub_item = None
        items = menu.items()
        if items:
            sub_item = items[idx]
        self._activate(sub_item)
        return sub_item

    def item_by_path(self, path, exact=False):
        if False:
            i = 10
            return i + 15
        'Find a menu item specified by the path'
        menu_items = [p.strip() for p in path.split('->')]
        items_cnt = len(menu_items)
        if items_cnt == 0:
            raise IndexError()
        for item in menu_items:
            if not item:
                raise IndexError("Empty item name between '->' separators")

        def next_level_menu(parent_menu, item_name):
            if False:
                i = 10
                return i + 15
            if item_name.startswith('#'):
                return self._sub_item_by_idx(parent_menu, int(item_name[1:]))
            else:
                return self._sub_item_by_text(parent_menu, item_name, exact)
        try:
            menu = next_level_menu(self, menu_items[0])
            if items_cnt == 1:
                return menu
            for i in range(1, items_cnt):
                menu = next_level_menu(menu, menu_items[i])
        except AttributeError:
            raise IndexError()
        return menu

class ScrollBarWrapper(atspiwrapper.AtspiWrapper):
    """Wrap an Atspi-compatible Slider control"""
    _control_types = ['ScrollBar']
    has_title = False

    def __init__(self, elem):
        if False:
            return 10
        'Initialize the control'
        super(ScrollBarWrapper, self).__init__(elem)
        self.atspi_value_obj = self.element_info.get_atspi_value_obj()

    def min_value(self):
        if False:
            print('Hello World!')
        'Get the minimum value of the ScrollBar'
        return self.atspi_value_obj.get_minimum_value()

    def max_value(self):
        if False:
            return 10
        'Get the maximum value of the ScrollBar'
        return self.atspi_value_obj.get_maximum_value()

    def min_step(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the minimum step of the ScrollBar'
        return self.atspi_value_obj.get_minimum_increment()

    def get_value(self):
        if False:
            i = 10
            return i + 15
        "Get a current position of slider's thumb"
        return self.atspi_value_obj.get_current_value()

    def set_value(self, value):
        if False:
            print('Hello World!')
        "Set position of slider's thumb"
        if isinstance(value, float):
            value_to_set = value
        elif isinstance(value, six.integer_types):
            value_to_set = value
        elif isinstance(value, six.text_type):
            value_to_set = float(value)
        else:
            raise ValueError('value should be either string or number')
        min_value = self.min_value()
        max_value = self.max_value()
        if not min_value <= value_to_set <= max_value:
            raise ValueError('value should be bigger than {0} and smaller than {1}'.format(min_value, max_value))
        self.atspi_value_obj.set_current_value(value_to_set)