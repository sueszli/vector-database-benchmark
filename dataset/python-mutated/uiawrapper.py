"""Basic wrapping of UI Automation elements"""
from __future__ import unicode_literals
from __future__ import print_function
import six
import time
import warnings
import comtypes
import threading
from .. import backend
from .. import WindowNotFoundError
from ..timings import Timings
from .win_base_wrapper import WinBaseWrapper
from .hwndwrapper import HwndWrapper
from ..base_wrapper import BaseMeta
from ..windows.uia_defines import IUIA
from ..windows import uia_defines as uia_defs
from ..windows.uia_element_info import UIAElementInfo, elements_from_uia_array
AutomationElement = IUIA().ui_automation_client.IUIAutomationElement
DockPattern = IUIA().ui_automation_client.IUIAutomationDockPattern
ExpandCollapsePattern = IUIA().ui_automation_client.IUIAutomationExpandCollapsePattern
GridItemPattern = IUIA().ui_automation_client.IUIAutomationGridItemPattern
GridPattern = IUIA().ui_automation_client.IUIAutomationGridPattern
InvokePattern = IUIA().ui_automation_client.IUIAutomationInvokePattern
ItemContainerPattern = IUIA().ui_automation_client.IUIAutomationItemContainerPattern
LegacyIAccessiblePattern = IUIA().ui_automation_client.IUIAutomationLegacyIAccessiblePattern
MultipleViewPattern = IUIA().ui_automation_client.IUIAutomationMultipleViewPattern
RangeValuePattern = IUIA().ui_automation_client.IUIAutomationRangeValuePattern
ScrollItemPattern = IUIA().ui_automation_client.IUIAutomationScrollItemPattern
ScrollPattern = IUIA().ui_automation_client.IUIAutomationScrollPattern
SelectionItemPattern = IUIA().ui_automation_client.IUIAutomationSelectionItemPattern
SelectionPattern = IUIA().ui_automation_client.IUIAutomationSelectionPattern
SynchronizedInputPattern = IUIA().ui_automation_client.IUIAutomationSynchronizedInputPattern
TableItemPattern = IUIA().ui_automation_client.IUIAutomationTableItemPattern
TablePattern = IUIA().ui_automation_client.IUIAutomationTablePattern
TextPattern = IUIA().ui_automation_client.IUIAutomationTextPattern
TogglePattern = IUIA().ui_automation_client.IUIAutomationTogglePattern
TransformPattern = IUIA().ui_automation_client.IUIAutomationTransformPattern
ValuePattern = IUIA().ui_automation_client.IUIAutomationValuePattern
VirtualizedItemPattern = IUIA().ui_automation_client.IUIAutomationVirtualizedItemPattern
WindowPattern = IUIA().ui_automation_client.IUIAutomationWindowPattern
_friendly_classes = {'Custom': None, 'DataGrid': 'ListView', 'DataItem': 'DataItem', 'Document': None, 'Group': 'GroupBox', 'Header': None, 'HeaderItem': None, 'Hyperlink': None, 'Image': None, 'List': 'ListBox', 'ListItem': 'ListItem', 'MenuBar': 'Menu', 'Menu': 'Menu', 'MenuItem': 'MenuItem', 'Pane': None, 'ProgressBar': 'Progress', 'ScrollBar': None, 'Separator': None, 'Slider': None, 'Spinner': 'UpDown', 'SplitButton': None, 'Tab': 'TabControl', 'Table': None, 'Text': 'Static', 'Thumb': None, 'TitleBar': None, 'ToolBar': 'Toolbar', 'ToolTip': 'ToolTips', 'Tree': 'TreeView', 'TreeItem': 'TreeItem', 'Window': 'Dialog'}

class LazyProperty(object):
    """
    A lazy evaluation of an object attribute.

    The property should represent immutable data, as it replaces itself.
    Provided by: http://stackoverflow.com/a/6849299/1260742
    """

    def __init__(self, fget):
        if False:
            while True:
                i = 10
        'Init the property name and method to calculate the property'
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self, obj, cls):
        if False:
            return 10
        'Replace the property itself on a first access'
        if obj is None:
            return None
        value = self.fget(obj)
        setattr(obj, self.func_name, value)
        return value
lazy_property = LazyProperty

class UiaMeta(BaseMeta):
    """Metaclass for UiaWrapper objects"""
    control_type_to_cls = {}

    def __init__(cls, name, bases, attrs):
        if False:
            while True:
                i = 10
        'Register the control types'
        BaseMeta.__init__(cls, name, bases, attrs)
        for t in cls._control_types:
            UiaMeta.control_type_to_cls[t] = cls

    @staticmethod
    def find_wrapper(element):
        if False:
            print('Hello World!')
        'Find the correct wrapper for this UIA element'
        try:
            wrapper_match = UiaMeta.control_type_to_cls[element.control_type]
        except KeyError:
            wrapper_match = UIAWrapper
        return wrapper_match

@six.add_metaclass(UiaMeta)
class UIAWrapper(WinBaseWrapper):
    """
    Default wrapper for User Interface Automation (UIA) controls.

    All other UIA wrappers are derived from this.

    This class wraps a lot of functionality of underlying UIA features
    for working with windows.

    Most of the methods apply to every single element type. For example
    you can click() on any element.
    """
    _control_types = []

    def __new__(cls, element_info):
        if False:
            for i in range(10):
                print('nop')
        'Construct the control wrapper'
        return super(UIAWrapper, cls)._create_wrapper(cls, element_info, UIAWrapper)

    def __init__(self, element_info):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize the control\n\n        * **element_info** is either a valid UIAElementInfo or it can be an\n          instance or subclass of UIAWrapper.\n        If the handle is not valid then an InvalidWindowHandle error\n        is raised.\n        '
        WinBaseWrapper.__init__(self, element_info, backend.registry.backends['uia'])

    @lazy_property
    def iface_expand_collapse(self):
        if False:
            print('Hello World!')
        "Get the element's ExpandCollapse interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'ExpandCollapse')

    @lazy_property
    def iface_selection(self):
        if False:
            print('Hello World!')
        "Get the element's Selection interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'Selection')

    @lazy_property
    def iface_selection_item(self):
        if False:
            return 10
        "Get the element's SelectionItem interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'SelectionItem')

    @lazy_property
    def iface_invoke(self):
        if False:
            return 10
        "Get the element's Invoke interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'Invoke')

    @lazy_property
    def iface_toggle(self):
        if False:
            print('Hello World!')
        "Get the element's Toggle interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'Toggle')

    @lazy_property
    def iface_text(self):
        if False:
            return 10
        "Get the element's Text interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'Text')

    @lazy_property
    def iface_value(self):
        if False:
            for i in range(10):
                print('nop')
        "Get the element's Value interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'Value')

    @lazy_property
    def iface_range_value(self):
        if False:
            print('Hello World!')
        "Get the element's RangeValue interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'RangeValue')

    @lazy_property
    def iface_grid(self):
        if False:
            print('Hello World!')
        "Get the element's Grid interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'Grid')

    @lazy_property
    def iface_grid_item(self):
        if False:
            while True:
                i = 10
        "Get the element's GridItem interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'GridItem')

    @lazy_property
    def iface_table(self):
        if False:
            print('Hello World!')
        "Get the element's Table interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'Table')

    @lazy_property
    def iface_table_item(self):
        if False:
            return 10
        "Get the element's TableItem interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'TableItem')

    @lazy_property
    def iface_scroll_item(self):
        if False:
            return 10
        "Get the element's ScrollItem interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'ScrollItem')

    @lazy_property
    def iface_scroll(self):
        if False:
            return 10
        "Get the element's Scroll interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'Scroll')

    @lazy_property
    def iface_transform(self):
        if False:
            i = 10
            return i + 15
        "Get the element's Transform interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'Transform')

    @lazy_property
    def iface_transformV2(self):
        if False:
            return 10
        "Get the element's TransformV2 interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'TransformV2')

    @lazy_property
    def iface_window(self):
        if False:
            for i in range(10):
                print('nop')
        "Get the element's Window interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'Window')

    @lazy_property
    def iface_item_container(self):
        if False:
            print('Hello World!')
        "Get the element's ItemContainer interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'ItemContainer')

    @lazy_property
    def iface_virtualized_item(self):
        if False:
            for i in range(10):
                print('nop')
        "Get the element's VirtualizedItem interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'VirtualizedItem')

    @lazy_property
    def iface_legacy_iaccessible(self):
        if False:
            print('Hello World!')
        "Get the element's LegacyIAccessible interface pattern"
        elem = self.element_info.element
        return uia_defs.get_elem_interface(elem, 'LegacyIAccessible')

    @property
    def writable_props(self):
        if False:
            while True:
                i = 10
        'Extend default properties list.'
        props = super(UIAWrapper, self).writable_props
        props.extend(['is_keyboard_focusable', 'has_keyboard_focus', 'automation_id'])
        return props

    def legacy_properties(self):
        if False:
            return 10
        "Get the element's LegacyIAccessible control pattern interface properties"
        elem = self.element_info.element
        impl = uia_defs.get_elem_interface(elem, 'LegacyIAccessible')
        property_name_identifier = 'Current'
        interface_properties = [prop for prop in dir(LegacyIAccessiblePattern) if isinstance(getattr(LegacyIAccessiblePattern, prop), property) and property_name_identifier in prop]
        return {prop.replace(property_name_identifier, ''): getattr(impl, prop) for prop in interface_properties}

    def friendly_class_name(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the friendly class name for the control\n\n        This differs from the class of the control in some cases.\n        class_name() is the actual \'Registered\' window class of the control\n        while friendly_class_name() is hopefully something that will make\n        more sense to the user.\n\n        For example Checkboxes are implemented as Buttons - so the class\n        of a CheckBox is "Button" - but the friendly class is "CheckBox"\n        '
        if self.friendlyclassname is None:
            if self.element_info.control_type not in IUIA().known_control_types.keys():
                self.friendlyclassname = self.element_info.control_type
            else:
                ctrl_type = self.element_info.control_type
                if ctrl_type not in _friendly_classes or _friendly_classes[ctrl_type] is None:
                    self.friendlyclassname = ctrl_type
                else:
                    self.friendlyclassname = _friendly_classes[ctrl_type]
        return self.friendlyclassname

    def automation_id(self):
        if False:
            return 10
        'Return the Automation ID of the control'
        return self.element_info.auto_id

    def is_keyboard_focusable(self):
        if False:
            return 10
        'Return True if the element can be focused with keyboard'
        return self.element_info.element.CurrentIsKeyboardFocusable == 1

    def has_keyboard_focus(self):
        if False:
            print('Hello World!')
        'Return True if the element is focused with keyboard'
        return self.element_info.element.CurrentHasKeyboardFocus == 1

    def set_focus(self):
        if False:
            i = 10
            return i + 15
        'Set the focus to this element'
        try:
            if self.is_minimized():
                if self.was_maximized():
                    self.maximize()
                else:
                    self.restore()
        except uia_defs.NoPatternInterfaceError:
            pass
        try:
            self.element_info.element.SetFocus()
            active_element = UIAElementInfo.get_active()
            if self.element_info != active_element and self.element_info != active_element.top_level_parent:
                if self.handle:
                    warnings.warn('Failed to set focus on element, trying win32 backend', RuntimeWarning)
                    HwndWrapper(self.element_info).set_focus()
                else:
                    warnings.warn("The element has not been focused because UIA SetFocus() failed and we can't use win32 backend instead because the element doesn't have native handle", RuntimeWarning)
        except comtypes.COMError as exc:
            if self.handle:
                warnings.warn('Failed to set focus on element due to COMError: {}, trying win32 backend'.format(exc), RuntimeWarning)
                HwndWrapper(self.element_info).set_focus()
            else:
                warnings.warn("The element has not been focused due to COMError: {}, and we can't use win32 backend instead because the element doesn't have native handle".format(exc), RuntimeWarning)
        return self

    def set_value(self, value):
        if False:
            i = 10
            return i + 15
        'An interface to the SetValue method of the Value control pattern'
        self.iface_value.SetValue(value)
        return self

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Close the window\n\n        Only a control supporting Window pattern should answer.\n        If it doesn\'t (menu shadows, tooltips,...), try to send "Esc" key\n        '
        if not self.is_visible() or not self.is_enabled():
            return
        try:
            name = self.element_info.name
            control_type = self.element_info.control_type
            iface = self.iface_window
            iface.Close()
            if name and control_type:
                self.actions.log('Closed ' + control_type.lower() + ' "' + name + '"')
        except uia_defs.NoPatternInterfaceError:
            try:
                self.type_keys('{ESC}')
            except comtypes.COMError:
                raise WindowNotFoundError

    def minimize(self):
        if False:
            print('Hello World!')
        '\n        Minimize the window\n\n        Only controls supporting Window pattern should answer\n        '
        iface = self.iface_window
        if iface.CurrentCanMinimize:
            iface.SetWindowVisualState(uia_defs.window_visual_state_minimized)
        return self

    def maximize(self):
        if False:
            return 10
        '\n        Maximize the window\n\n        Only controls supporting Window pattern should answer\n        '
        iface = self.iface_window
        if iface.CurrentCanMaximize:
            iface.SetWindowVisualState(uia_defs.window_visual_state_maximized)
        return self

    def restore(self):
        if False:
            return 10
        '\n        Restore the window to normal size\n\n        Only controls supporting Window pattern should answer\n        '
        iface = self.iface_window
        iface.SetWindowVisualState(uia_defs.window_visual_state_normal)
        return self

    def get_show_state(self):
        if False:
            print('Hello World!')
        'Get the show state and Maximized/minimzed/restored state\n\n        Returns values as following\n\n        window_visual_state_normal = 0\n        window_visual_state_maximized = 1\n        window_visual_state_minimized = 2\n        '
        iface = self.iface_window
        ret = iface.CurrentWindowVisualState
        return ret

    def is_minimized(self):
        if False:
            while True:
                i = 10
        'Indicate whether the window is minimized or not'
        return self.get_show_state() == uia_defs.window_visual_state_minimized

    def is_maximized(self):
        if False:
            while True:
                i = 10
        'Indicate whether the window is maximized or not'
        return self.get_show_state() == uia_defs.window_visual_state_maximized

    def is_normal(self):
        if False:
            print('Hello World!')
        'Indicate whether the window is normal (i.e. not minimized and not maximized)'
        return self.get_show_state() == uia_defs.window_visual_state_normal

    def invoke(self):
        if False:
            print('Hello World!')
        'An interface to the Invoke method of the Invoke control pattern'
        name = self.element_info.name
        control_type = self.element_info.control_type
        invoke_pattern_iface = self.iface_invoke

        def watchdog():
            if False:
                while True:
                    i = 10
            thread = threading.Thread(target=invoke_pattern_iface.Invoke)
            thread.daemon = True
            thread.start()
            thread.join(2.0)
            if thread.is_alive():
                warnings.warn('Timeout for InvokePattern.Invoke() call was exceeded', RuntimeWarning)
        watchdog_thread = threading.Thread(target=watchdog)
        watchdog_thread.start()
        watchdog_thread.join(Timings.after_invoke_wait)
        if name and control_type:
            self.actions.log('Invoked ' + control_type.lower() + ' "' + name + '"')
        return self

    def expand(self):
        if False:
            return 10
        '\n        Displays all child nodes, controls, or content of the control\n\n        An interface to Expand method of the ExpandCollapse control pattern.\n        '
        self.iface_expand_collapse.Expand()
        return self

    def collapse(self):
        if False:
            return 10
        '\n        Displays all child nodes, controls, or content of the control\n\n        An interface to Collapse method of the ExpandCollapse control pattern.\n        '
        self.iface_expand_collapse.Collapse()
        return self

    def get_expand_state(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Indicates the state of the control: expanded or collapsed.\n\n        An interface to CurrentExpandCollapseState property of the ExpandCollapse control pattern.\n        Values for enumeration as defined in uia_defines module:\n        expand_state_collapsed = 0\n        expand_state_expanded = 1\n        expand_state_partially = 2\n        expand_state_leaf_node = 3\n        '
        return self.iface_expand_collapse.CurrentExpandCollapseState

    def is_expanded(self):
        if False:
            for i in range(10):
                print('nop')
        'Test if the control is expanded'
        state = self.get_expand_state()
        return state == uia_defs.expand_state_expanded

    def is_collapsed(self):
        if False:
            while True:
                i = 10
        'Test if the control is collapsed'
        state = self.get_expand_state()
        return state == uia_defs.expand_state_collapsed

    def get_selection(self):
        if False:
            i = 10
            return i + 15
        '\n        An interface to GetSelection of the SelectionProvider pattern\n\n        Retrieves a UI Automation provider for each child element\n        that is selected. Builds a list of UIAElementInfo elements\n        from all retrieved providers.\n        '
        ptrs_array = self.iface_selection.GetCurrentSelection()
        return elements_from_uia_array(ptrs_array)

    def selected_item_index(self):
        if False:
            return 10
        'Return the index of a selected item'
        selection = self.get_selection()
        if selection:
            for (i, c) in enumerate(self.children()):
                if c.window_text() == selection[0].name:
                    return i
        return None

    def select(self):
        if False:
            while True:
                i = 10
        'Select the item\n\n        Only items supporting SelectionItem pattern should answer.\n        Raise NoPatternInterfaceError if the pattern is not supported\n\n        Usually applied for controls like: a radio button, a tree view item\n        or a list item.\n        '
        self.iface_selection_item.Select()
        if not self.is_selected():
            warnings.warn('SelectionItem.Select failed, trying LegacyIAccessible.DoDefaultAction', RuntimeWarning)
            self.iface_legacy_iaccessible.DoDefaultAction()
        name = self.element_info.name
        control_type = self.element_info.control_type
        if name and control_type:
            self.actions.log('Selected ' + control_type.lower() + ' "' + name + '"')
        return self

    def is_selected(self):
        if False:
            for i in range(10):
                print('nop')
        'Indicate that the item is selected or not.\n\n        Only items supporting SelectionItem pattern should answer.\n        Raise NoPatternInterfaceError if the pattern is not supported\n\n        Usually applied for controls like: a radio button, a tree view item,\n        a list item.\n        '
        return self.iface_selection_item.CurrentIsSelected

    def children_texts(self):
        if False:
            return 10
        "Get texts of the control's children"
        return [c.window_text() for c in self.children()]

    def can_select_multiple(self):
        if False:
            i = 10
            return i + 15
        '\n        An interface to CanSelectMultiple of the SelectionProvider pattern\n\n        Indicates whether the UI Automation provider allows more than one\n        child element to be selected concurrently.\n        '
        return self.iface_selection.CurrentCanSelectMultiple

    def is_selection_required(self):
        if False:
            i = 10
            return i + 15
        '\n        An interface to IsSelectionRequired property of the SelectionProvider pattern.\n\n        This property can be dynamic. For example, the initial state of\n        a control might not have any items selected by default,\n        meaning that IsSelectionRequired is FALSE. However,\n        after an item is selected the control must always have\n        at least one item selected.\n        '
        return self.iface_selection.CurrentIsSelectionRequired

    def _select(self, item=None):
        if False:
            return 10
        '\n        Find a child item by the name or index and select\n\n        The action can be applied for dirrent controls with items:\n        ComboBox, TreeView, Tab control\n        '
        if isinstance(item, six.integer_types):
            item_index = item
            title = None
        elif isinstance(item, six.string_types):
            item_index = 0
            title = item
        else:
            err_msg = u'unsupported {0} for item {1}'.format(type(item), item)
            raise ValueError(err_msg)
        list_ = self.children(name=title)
        if item_index < len(list_):
            wrp = list_[item_index]
            wrp.iface_selection_item.Select()
            if not wrp.is_selected():
                warnings.warn('SelectionItem.Select failed, trying LegacyIAccessible.DoDefaultAction', RuntimeWarning)
                wrp.iface_legacy_iaccessible.DoDefaultAction()
        else:
            raise IndexError("item '{0}' not found".format(item))

    def is_active(self):
        if False:
            while True:
                i = 10
        'Whether the window is active or not'
        ae = IUIA().get_focused_element()
        focused_wrap = UIAWrapper(UIAElementInfo(ae))
        return focused_wrap.top_level_parent() == self.top_level_parent()

    def is_dialog(self):
        if False:
            while True:
                i = 10
        'Return true if the control is a dialog window (WindowPattern interface is available)'
        try:
            return self.iface_window is not None
        except uia_defs.NoPatternInterfaceError:
            return False

    def menu_select(self, path, exact=False):
        if False:
            i = 10
            return i + 15
        'Select a menu item specified in the path\n\n        The full path syntax is specified in:\n        :py:meth:`pywinauto.menuwrapper.Menu.get_menu_path`\n\n        There are usually at least two menu bars: "System" and "Application"\n        System menu bar is a standard window menu with items like:\n        \'Restore\', \'Move\', \'Size\', \'Minimize\', e.t.c.\n        This menu bar usually has a "Title Bar" control as a parent.\n        Application menu bar is often what we look for. In most cases,\n        its parent is the dialog itself so it should be found among the direct\n        children of the dialog. Notice that we don\'t use "Application"\n        string as a title criteria because it couldn\'t work on applications\n        with a non-english localization.\n        If there is no menu bar has been found we fall back to look up\n        for Menu control. We try to find the control through all descendants\n        of the dialog\n        '
        self.verify_actionable()
        cc = self.children(control_type='MenuBar')
        if not cc:
            cc = self.descendants(control_type='Menu')
            if not cc:
                raise AttributeError
        menu = cc[0]
        menu.item_by_path(path, exact).select()

    def toolbar_select(self, path, exact=False):
        if False:
            print('Hello World!')
        '\n        Select a Toolbar item specified in the path.\n\n        The full path syntax is specified in:\n        :py:meth:`pywinauto.controls.uia_controls.ToolbarWrapper.item_by_path`\n        '
        self.verify_actionable()
        cc = self.children(control_type='ToolBar')
        if not cc:
            cc = self.descendants(control_type='ToolBar')
            if not cc:
                raise AttributeError('Can not find any item with control_type="ToolBar" in children and descendants!')
        menu = cc[0]
        menu.item_by_path(path, exact).select()
    _scroll_types = {'left': {'line': (uia_defs.scroll_small_decrement, uia_defs.scroll_no_amount), 'page': (uia_defs.scroll_large_decrement, uia_defs.scroll_no_amount)}, 'right': {'line': (uia_defs.scroll_small_increment, uia_defs.scroll_no_amount), 'page': (uia_defs.scroll_large_increment, uia_defs.scroll_no_amount)}, 'up': {'line': (uia_defs.scroll_no_amount, uia_defs.scroll_small_decrement), 'page': (uia_defs.scroll_no_amount, uia_defs.scroll_large_decrement)}, 'down': {'line': (uia_defs.scroll_no_amount, uia_defs.scroll_small_increment), 'page': (uia_defs.scroll_no_amount, uia_defs.scroll_large_increment)}}

    def scroll(self, direction, amount, count=1, retry_interval=Timings.scroll_step_wait):
        if False:
            for i in range(10):
                print('nop')
        'Ask the control to scroll itself\n\n        **direction** can be any of "up", "down", "left", "right"\n        **amount** can be only "line" or "page"\n        **count** (optional) the number of times to scroll\n        **retry_interval** (optional) interval between scroll actions\n        '

        def _raise_attrib_err(details):
            if False:
                return 10
            control_type = self.element_info.control_type
            name = self.element_info.name
            msg = ''.join([control_type.lower(), ' "', name, '" ', details])
            raise AttributeError(msg)
        try:
            scroll_if = self.iface_scroll
            if direction.lower() in ('up', 'down'):
                if not scroll_if.CurrentVerticallyScrollable:
                    _raise_attrib_err('is not vertically scrollable')
            elif direction.lower() in ('left', 'right'):
                if not scroll_if.CurrentHorizontallyScrollable:
                    _raise_attrib_err('is not horizontally scrollable')
            (h, v) = self._scroll_types[direction.lower()][amount.lower()]
            for _ in range(count, 0, -1):
                scroll_if.Scroll(h, v)
                time.sleep(retry_interval)
        except uia_defs.NoPatternInterfaceError:
            _raise_attrib_err('is not scrollable')
        except KeyError:
            raise ValueError('Wrong arguments:\n                direction can be any of "up", "down", "left", "right"\n                amount can be only "line" or "page"\n                ')
        return self

    def _texts_from_item_container(self):
        if False:
            print('Hello World!')
        'Get texts through the ItemContainer interface'
        texts = []
        try:
            com_elem = self.iface_item_container.FindItemByProperty(0, 0, uia_defs.vt_empty)
            while com_elem:
                itm = UIAWrapper(UIAElementInfo(com_elem))
                texts.append(itm.texts())
                com_elem = self.iface_item_container.FindItemByProperty(com_elem, 0, uia_defs.vt_empty)
        except uia_defs.NoPatternInterfaceError:
            pass
        return texts

    def move_window(self, x=None, y=None, width=None, height=None):
        if False:
            while True:
                i = 10
        'Move the window to the new coordinates\n        The method should be implemented explicitly by controls that\n        support this action. The most obvious is the Window control.\n        Otherwise the method throws AttributeError\n\n        * **x** Specifies the new left position of the window.\n          Defaults to the current left position of the window.\n        * **y** Specifies the new top position of the window.\n          Defaults to the current top position of the window.\n        * **width** Specifies the new width of the window. Defaults to the\n          current width of the window.\n        * **height** Specifies the new height of the window. Default to the\n          current height of the window.\n        '
        raise AttributeError('This method is not supported for {0}'.format(self))
backend.register('uia', UIAElementInfo, UIAWrapper)