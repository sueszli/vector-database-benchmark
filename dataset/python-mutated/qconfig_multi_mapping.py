from __future__ import annotations
import copy
from typing import Any, Callable, Dict, List, Union
import torch
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig_mapping import _QCONFIG_STYLE_ORDER
from torch.ao.quantization.qconfig import QConfigAny
__all__ = ['QConfigMultiMapping']
_QCONFIG_STYLE_TO_METHOD: Dict[str, str] = {'global_qconfig': 'set_global', 'object_type_qconfigs': 'set_object_type', 'module_name_regex_qconfigs': 'set_module_name_regex', 'module_name_qconfigs': 'set_module_name', 'module_name_object_type_order_qconfigs': 'set_module_name_object_type_order'}

def _remove_duplicates_and_none(qconfig_list: List[QConfigAny]) -> None:
    if False:
        print('Hello World!')
    to_remove = []
    for (index, cur_qconfig) in enumerate(qconfig_list):
        if cur_qconfig is None:
            to_remove.append(index)
            break
        for checked_qconfig in qconfig_list[:index]:
            if torch.ao.quantization.qconfig_equals(cur_qconfig, checked_qconfig):
                to_remove.append(index)
                break
    for index in to_remove[::-1]:
        qconfig_list.pop(index)

class QConfigMultiMapping:
    """
    This class, used with the prepare_n_shadows_model API, stores a list of :class:`torch.ao.quantization.QConfigMapping`s
    so that multiple QConfigs can be specified for each QConfig matching style.

    The user can specify QConfigs using the following methods (in increasing match priority):

        ``set_global`` : sets the global (default) QConfigs

        ``set_object_type`` : sets the QConfigs for a given module type, function, or method name

        ``set_module_name_regex`` : sets the QConfigs for modules matching the given regex string

        ``set_module_name`` : sets the QConfigs for modules matching the given module name

        ``set_module_name_object_type_order`` : sets the QConfigs for modules matching a combination
        of the given module name, object type, and the index at which the module appears

    Note: Usage of set methods is the same as in QConfigMapping except with a passed in list of QConfigs rather than a
    single QConfig.

    Example usage::

        qconfig_mapping = QConfigMultiMapping()
            .set_global([qconfig1, qconfig2])
            .set_object_type(torch.nn.Linear, [qconfig2, qconfig3])
            .set_object_type(torch.nn.ReLU, [qconfig1])
            .set_module_name_regex("foo.*bar.*conv[0-9]+", [qconfig2])
            .set_module_name_regex("foo.*", [qconfig1, qconfig2, qconfig3])
            .set_module_name("module1", [None])
            .set_module_name("module2", [qconfig2])
            .set_module_name_object_type_order("foo.bar", torch.nn.functional.linear, 0, [qconfig3])

    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.qconfig_mappings_list: List[QConfigMapping] = [QConfigMapping()]

    def _handle_list_size_mismatch(self, qconfig_list: List[QConfigAny], style: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        if len(qconfig_list) > len(self.qconfig_mappings_list):
            new_qconfig_mapping = QConfigMapping()
            for qconfig_mapping in self.qconfig_mappings_list:
                for check_style in _QCONFIG_STYLE_ORDER[1:]:
                    qconfigs_dict = getattr(qconfig_mapping, check_style)
                    target_qconfigs_dict = getattr(new_qconfig_mapping, check_style)
                    for key in qconfigs_dict:
                        target_qconfigs_dict[key] = None
                break
            while len(qconfig_list) > len(self.qconfig_mappings_list):
                self.qconfig_mappings_list.append(copy.deepcopy(new_qconfig_mapping))
        else:
            while len(qconfig_list) < len(self.qconfig_mappings_list):
                qconfig_list.append(None)

    def _insert_qconfig_list(self, style: str, args: List[Union[str, int, Callable]], qconfig_list: List[QConfigAny]) -> None:
        if False:
            i = 10
            return i + 15
        _remove_duplicates_and_none(qconfig_list)
        self._handle_list_size_mismatch(qconfig_list, style)
        method_name = _QCONFIG_STYLE_TO_METHOD[style]
        for (qconfig_mapping, qconfig) in zip(self.qconfig_mappings_list, qconfig_list):
            set_method = getattr(qconfig_mapping, method_name)
            set_method(*args, qconfig)

    def set_global(self, global_qconfig_list: List[QConfigAny]) -> QConfigMultiMapping:
        if False:
            print('Hello World!')
        '\n        Set global QConfigs\n        see :func:`~torch.ao.quantization.QConfigMapping.set_global()` for more info\n        '
        self._insert_qconfig_list('global_qconfig', [], global_qconfig_list)
        return self

    def set_object_type(self, object_type: Union[Callable, str], qconfig_list: List[QConfigAny]) -> QConfigMultiMapping:
        if False:
            while True:
                i = 10
        '\n        Set object type QConfigs\n        see :func:`~torch.ao.quantization.QConfigMapping.set_object_type()` for more info\n        '
        self._insert_qconfig_list('object_type_qconfigs', [object_type], qconfig_list)
        return self

    def set_module_name_regex(self, module_name_regex: str, qconfig_list: List[QConfigAny]) -> QConfigMultiMapping:
        if False:
            print('Hello World!')
        '\n        Set module_name_regex QConfigs\n        see :func:`~torch.ao.quantization.QConfigMapping.set_module_name_regex()` for more info\n        '
        self._insert_qconfig_list('module_name_regex_qconfigs', [module_name_regex], qconfig_list)
        return self

    def set_module_name(self, module_name: str, qconfig_list: List[QConfigAny]) -> QConfigMultiMapping:
        if False:
            return 10
        '\n        Set module_name QConfigs\n        see :func:`~torch.ao.quantization.QConfigMapping.set_module_name()` for more info\n        '
        self._insert_qconfig_list('module_name_qconfigs', [module_name], qconfig_list)
        return self

    def set_module_name_object_type_order(self, module_name: str, object_type: Callable, index: int, qconfig_list: List[QConfigAny]) -> QConfigMultiMapping:
        if False:
            for i in range(10):
                print('nop')
        '\n        Set module_name QConfigs\n        see :func:`~torch.ao.quantization.QConfigMapping.set_module_name_object_type_order()` for more info\n        '
        self._insert_qconfig_list('module_name_object_type_order_qconfigs', [module_name, object_type, index], qconfig_list)
        return self

    def __repr__(self):
        if False:
            print('Hello World!')
        return self.__class__.__name__ + ' [' + ''.join((f'\n{qconfig_mapping.__repr__()},' for qconfig_mapping in self.qconfig_mappings_list)) + '\n]'

    @classmethod
    def from_list_qconfig_mapping(cls, qconfig_mapping_list: List[QConfigMapping]) -> QConfigMultiMapping:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a QConfigMultiMapping from a list of QConfigMappings\n        '
        new_qconfig_multi_mapping = cls()
        new_qconfig_multi_mapping.qconfig_mappings_list = copy.deepcopy(qconfig_mapping_list)
        for style in _QCONFIG_STYLE_ORDER[1:]:
            qconfig_dict_list: Dict[Any, List[QConfigAny]] = {}
            for qconfig_mapping in qconfig_mapping_list:
                qconfig_dict = getattr(qconfig_mapping, style)
                for (key, qconfig) in qconfig_dict.items():
                    if key not in qconfig_dict_list:
                        qconfig_dict_list[key] = []
                    qconfig_dict_list[key].append(qconfig)
            set_method_name = _QCONFIG_STYLE_TO_METHOD[style]
            set_method = getattr(new_qconfig_multi_mapping, set_method_name)
            for (key, qconfig_list) in qconfig_dict_list.items():
                if isinstance(key, tuple):
                    set_method(*key, qconfig_list)
                else:
                    set_method(key, qconfig_list)
        return new_qconfig_multi_mapping