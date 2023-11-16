"""Common UIA definitions and helper functions"""
import comtypes
import comtypes.client
import six
from ..backend import Singleton

@six.add_metaclass(Singleton)
class IUIA(object):
    """Singleton class to store global COM objects from UIAutomationCore.dll"""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.UIA_dll = comtypes.client.GetModule('UIAutomationCore.dll')
        self.ui_automation_client = comtypes.gen.UIAutomationClient
        self.iuia = comtypes.CoCreateInstance(self.ui_automation_client.CUIAutomation().IPersist_GetClassID(), interface=self.ui_automation_client.IUIAutomation, clsctx=comtypes.CLSCTX_INPROC_SERVER)
        self.true_condition = self.iuia.CreateTrueCondition()
        self.tree_scope = {'ancestors': self.UIA_dll.TreeScope_Ancestors, 'children': self.UIA_dll.TreeScope_Children, 'descendants': self.UIA_dll.TreeScope_Descendants, 'element': self.UIA_dll.TreeScope_Element, 'parent': self.UIA_dll.TreeScope_Parent, 'subtree': self.UIA_dll.TreeScope_Subtree}
        self.root = self.iuia.GetRootElement()
        self.raw_tree_walker = self.iuia.RawViewWalker
        self.get_focused_element = self.iuia.GetFocusedElement
        start_len = len('UIA_')
        end_len = len('ControlTypeId')
        self._control_types = [attr[start_len:-end_len] for attr in dir(self.UIA_dll) if attr.endswith('ControlTypeId')]
        self.known_control_types = {'InvalidControlType': 0}
        self.known_control_type_ids = {0: 'InvalidControlType'}
        for ctrl_type in self._control_types:
            type_id_name = 'UIA_' + ctrl_type + 'ControlTypeId'
            type_id = getattr(self.UIA_dll, type_id_name)
            self.known_control_types[ctrl_type] = type_id
            self.known_control_type_ids[type_id] = ctrl_type

    def build_condition(self, process=None, class_name=None, name=None, control_type=None, content_only=None):
        if False:
            i = 10
            return i + 15
        'Build UIA filtering conditions'
        conditions = []
        if process:
            conditions.append(self.iuia.CreatePropertyCondition(self.UIA_dll.UIA_ProcessIdPropertyId, process))
        if class_name:
            conditions.append(self.iuia.CreatePropertyCondition(self.UIA_dll.UIA_ClassNamePropertyId, class_name))
        if control_type:
            if isinstance(control_type, six.string_types):
                control_type = self.known_control_types[control_type]
            elif not isinstance(control_type, int):
                raise TypeError('control_type must be string or integer')
            conditions.append(self.iuia.CreatePropertyCondition(self.UIA_dll.UIA_ControlTypePropertyId, control_type))
        if name:
            conditions.append(self.iuia.CreatePropertyCondition(self.UIA_dll.UIA_NamePropertyId, name))
        if isinstance(content_only, bool):
            conditions.append(self.iuia.CreatePropertyCondition(self.UIA_dll.UIA_IsContentElementPropertyId, content_only))
        if len(conditions) > 1:
            return self.iuia.CreateAndConditionFromArray(conditions)
        if len(conditions) == 1:
            return conditions[0]
        return self.true_condition

def _build_pattern_ids_dic():
    if False:
        return 10
    '\n    A helper procedure to build a registry of control patterns\n    supported on the current system\n    '
    base_names = ['Dock', 'ExpandCollapse', 'GridItem', 'Grid', 'Invoke', 'ItemContainer', 'LegacyIAccessible', 'MulipleView', 'RangeValue', 'ScrollItem', 'Scroll', 'SelectionItem', 'Selection', 'SynchronizedInput', 'TableItem', 'Table', 'Text', 'Toggle', 'VirtualizedItem', 'Value', 'Window', 'Transform', 'Annotation', 'Drag', 'Drop', 'ObjectModel', 'Spreadsheet', 'SpreadsheetItem', 'Styles', 'TextChild', 'TextV2', 'TransformV2', 'TextEdit', 'CustomNavigation']
    ptrn_ids_dic = {}
    for ptrn_name in base_names:
        v2 = ''
        name = ptrn_name
        if ptrn_name.endswith('V2'):
            name = ptrn_name[:-2]
            v2 = '2'
        cls_name = ''.join(['IUIAutomation', name, 'Pattern', v2])
        if hasattr(IUIA().ui_automation_client, cls_name):
            klass = getattr(IUIA().ui_automation_client, cls_name)
            ptrn_id_name = 'UIA_' + name + 'Pattern' + v2 + 'Id'
            ptrn_id = getattr(IUIA().UIA_dll, ptrn_id_name)
            ptrn_ids_dic[ptrn_name] = (ptrn_id, klass)
    return ptrn_ids_dic
pattern_ids = _build_pattern_ids_dic()
toggle_state_off = IUIA().ui_automation_client.ToggleState_Off
toggle_state_on = IUIA().ui_automation_client.ToggleState_On
toggle_state_inderteminate = IUIA().ui_automation_client.ToggleState_Indeterminate

class NoPatternInterfaceError(Exception):
    """There is no such interface for the specified pattern"""
    pass
expand_state_collapsed = IUIA().ui_automation_client.ExpandCollapseState_Collapsed
expand_state_expanded = IUIA().ui_automation_client.ExpandCollapseState_Expanded
expand_state_partially = IUIA().ui_automation_client.ExpandCollapseState_PartiallyExpanded
expand_state_leaf_node = IUIA().ui_automation_client.ExpandCollapseState_LeafNode
window_visual_state_normal = IUIA().ui_automation_client.WindowVisualState_Normal
window_visual_state_maximized = IUIA().ui_automation_client.WindowVisualState_Maximized
window_visual_state_minimized = IUIA().ui_automation_client.WindowVisualState_Minimized
scroll_large_decrement = IUIA().ui_automation_client.ScrollAmount_LargeDecrement
scroll_small_decrement = IUIA().ui_automation_client.ScrollAmount_SmallDecrement
scroll_no_amount = IUIA().ui_automation_client.ScrollAmount_NoAmount
scroll_large_increment = IUIA().ui_automation_client.ScrollAmount_LargeIncrement
scroll_small_increment = IUIA().ui_automation_client.ScrollAmount_SmallIncrement
vt_empty = IUIA().ui_automation_client.VARIANT.empty.vt
vt_null = IUIA().ui_automation_client.VARIANT.null.vt

def get_elem_interface(element_info, pattern_name):
    if False:
        return 10
    'A helper to retrieve an element interface by the specified pattern name\n\n    TODO: handle a wrong pattern name\n    '
    (ptrn_id, cls_name) = pattern_ids[pattern_name]
    try:
        cur_ptrn = element_info.GetCurrentPattern(ptrn_id)
        iface = cur_ptrn.QueryInterface(cls_name)
    except ValueError:
        raise NoPatternInterfaceError()
    return iface