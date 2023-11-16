from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import yaml
from torchgen.model import NativeFunction
from torchgen.selective_build.operator import merge_debug_info, merge_operator_dicts, SelectiveBuildOperator, strip_operator_overload_name

@dataclass(frozen=True)
class SelectiveBuilder:
    include_all_operators: bool
    _debug_info: Optional[Tuple[str, ...]]
    operators: Dict[str, SelectiveBuildOperator]
    kernel_metadata: Dict[str, List[str]]
    et_kernel_metadata: Dict[str, List[str]]
    custom_classes: Set[str]
    build_features: Set[str]
    include_all_non_op_selectives: bool

    @staticmethod
    def get_nop_selector() -> 'SelectiveBuilder':
        if False:
            for i in range(10):
                print('nop')
        return SelectiveBuilder.from_yaml_dict({'include_all_operators': True})

    @staticmethod
    def from_yaml_dict(data: Dict[str, object]) -> 'SelectiveBuilder':
        if False:
            i = 10
            return i + 15
        valid_top_level_keys = {'include_all_non_op_selectives', 'include_all_operators', 'debug_info', 'operators', 'kernel_metadata', 'et_kernel_metadata', 'custom_classes', 'build_features'}
        top_level_keys = set(data.keys())
        if len(top_level_keys - valid_top_level_keys) > 0:
            raise Exception('Got unexpected top level keys: {}'.format(','.join(top_level_keys - valid_top_level_keys)))
        include_all_operators = data.get('include_all_operators', False)
        assert isinstance(include_all_operators, bool)
        debug_info = None
        if 'debug_info' in data:
            di_list = data['debug_info']
            assert isinstance(di_list, list)
            debug_info = tuple((str(x) for x in di_list))
        operators = {}
        operators_dict = data.get('operators', {})
        assert isinstance(operators_dict, dict)
        for (k, v) in operators_dict.items():
            operators[k] = SelectiveBuildOperator.from_yaml_dict(k, v)
        kernel_metadata = {}
        kernel_metadata_dict = data.get('kernel_metadata', {})
        assert isinstance(kernel_metadata_dict, dict)
        for (k, v) in kernel_metadata_dict.items():
            kernel_metadata[str(k)] = [str(dtype) for dtype in v]
        et_kernel_metadata = data.get('et_kernel_metadata', {})
        assert isinstance(et_kernel_metadata, dict)
        custom_classes = data.get('custom_classes', [])
        assert isinstance(custom_classes, Iterable)
        custom_classes = set(custom_classes)
        build_features = data.get('build_features', [])
        assert isinstance(build_features, Iterable)
        build_features = set(build_features)
        include_all_non_op_selectives = data.get('include_all_non_op_selectives', False)
        assert isinstance(include_all_non_op_selectives, bool)
        return SelectiveBuilder(include_all_operators, debug_info, operators, kernel_metadata, et_kernel_metadata, custom_classes, build_features, include_all_non_op_selectives)

    @staticmethod
    def from_yaml_str(config_contents: str) -> 'SelectiveBuilder':
        if False:
            i = 10
            return i + 15
        contents = yaml.safe_load(config_contents)
        return SelectiveBuilder.from_yaml_dict(contents)

    @staticmethod
    def from_yaml_path(config_path: str) -> 'SelectiveBuilder':
        if False:
            for i in range(10):
                print('nop')
        with open(config_path) as f:
            contents = yaml.safe_load(f)
            return SelectiveBuilder.from_yaml_dict(contents)

    @staticmethod
    def from_legacy_op_registration_allow_list(allow_list: Set[str], is_root_operator: bool, is_used_for_training: bool) -> 'SelectiveBuilder':
        if False:
            i = 10
            return i + 15
        operators = {}
        for op in allow_list:
            operators[op] = {'name': op, 'is_root_operator': is_root_operator, 'is_used_for_training': is_used_for_training, 'include_all_overloads': True}
        return SelectiveBuilder.from_yaml_dict({'operators': operators, 'include_all_non_op_selectives': True})

    def is_operator_selected(self, name: str) -> bool:
        if False:
            print('Hello World!')
        if self.include_all_operators:
            return True
        if name in self.operators:
            return True
        name = strip_operator_overload_name(name)
        return name in self.operators and self.operators[name].include_all_overloads

    def is_native_function_selected(self, func: NativeFunction) -> bool:
        if False:
            for i in range(10):
                print('nop')
        op_name = op_name_from_native_function(func)
        return self.is_operator_selected(op_name)

    def is_operator_selected_for_training(self, name: str) -> bool:
        if False:
            return 10
        if not self.is_operator_selected(name):
            return False
        if self.include_all_operators:
            return True
        not_training_op = SelectiveBuildOperator(name='', is_root_operator=False, is_used_for_training=False, include_all_overloads=False, _debug_info=None)
        op = not_training_op
        if name in self.operators:
            op = self.operators[name]
        name = strip_operator_overload_name(name)
        base_op = not_training_op
        if name in self.operators:
            base_op = self.operators[name]
        return op.is_used_for_training or (base_op.include_all_overloads and base_op.is_used_for_training)

    def is_native_function_selected_for_training(self, func: NativeFunction) -> bool:
        if False:
            return 10
        op_name = op_name_from_native_function(func)
        return self.is_operator_selected_for_training(op_name)

    def is_root_operator(self, name: str) -> bool:
        if False:
            return 10
        if not self.is_operator_selected(name):
            return False
        if self.include_all_operators:
            return True
        if name in self.operators:
            op: SelectiveBuildOperator = self.operators[name]
            return op.is_root_operator
        name = strip_operator_overload_name(name)
        if name not in self.operators:
            return False
        base_op: SelectiveBuildOperator = self.operators[name]
        return base_op.include_all_overloads and base_op.is_root_operator

    def is_kernel_dtype_selected(self, kernel_tag: str, dtype: str) -> bool:
        if False:
            return 10
        if self.include_all_operators or self.include_all_non_op_selectives:
            return True
        return kernel_tag in self.kernel_metadata and dtype in self.kernel_metadata[kernel_tag]

    def et_get_selected_kernels(self, op_name: str, kernel_key: List[str]) -> List[str]:
        if False:
            print('Hello World!')
        '\n        Return a list of kernel keys that cover the used ops\n        '
        if op_name not in self.et_kernel_metadata:
            return kernel_key if self.include_all_operators else []
        result_set = set()
        for model_kernel_keys in self.et_kernel_metadata[op_name]:
            key_found = False
            for key in kernel_key:
                if key != 'default' and key.split('/')[1] == model_kernel_keys.split('/')[1]:
                    result_set.add(key)
                    key_found = True
                    break
            if not key_found:
                if 'default' not in kernel_key:
                    raise Exception('Missing kernel for the model')
                else:
                    result_set.add('default')
        return list(result_set)

    def to_dict(self) -> Dict[str, object]:
        if False:
            i = 10
            return i + 15
        ret: Dict[str, object] = {'include_all_non_op_selectives': self.include_all_non_op_selectives, 'include_all_operators': self.include_all_operators}
        operators = {}
        for (op_name, op) in self.operators.items():
            operators[op_name] = op.to_dict()
        ret['operators'] = operators
        if self._debug_info is not None:
            ret['debug_info'] = sorted(self._debug_info)
        ret['kernel_metadata'] = {k: sorted(v) for (k, v) in self.kernel_metadata.items()}
        ret['et_kernel_metadata'] = self.et_kernel_metadata
        ret['custom_classes'] = sorted(self.custom_classes)
        ret['build_features'] = sorted(self.build_features)
        return ret

def merge_kernel_metadata(lhs: Dict[str, List[str]], rhs: Dict[str, List[str]]) -> Dict[str, List[str]]:
    if False:
        return 10
    kernel_metadata: Dict[str, List[str]] = {}
    for (tag_name, dtypes) in list(lhs.items()) + list(rhs.items()):
        dtypes_copy = set(dtypes)
        if tag_name in kernel_metadata:
            dtypes_copy |= set(kernel_metadata[tag_name])
        kernel_metadata[tag_name] = list(dtypes_copy)
    return kernel_metadata

def merge_et_kernel_metadata(lhs: Dict[str, List[str]], rhs: Dict[str, List[str]]) -> Dict[str, List[str]]:
    if False:
        print('Hello World!')
    merge_et_kernel_metadata: Dict[str, Set[str]] = defaultdict(set)
    for op in list(lhs.keys()) + list(rhs.keys()):
        merge_et_kernel_metadata[op].update(lhs.get(op, []))
        merge_et_kernel_metadata[op].update(rhs.get(op, []))
    return {op: sorted(val) for (op, val) in merge_et_kernel_metadata.items()}

def combine_selective_builders(lhs: SelectiveBuilder, rhs: SelectiveBuilder) -> SelectiveBuilder:
    if False:
        print('Hello World!')
    include_all_operators = lhs.include_all_operators or rhs.include_all_operators
    debug_info = merge_debug_info(lhs._debug_info, rhs._debug_info)
    operators = merge_operator_dicts(lhs.operators, rhs.operators)
    kernel_metadata = merge_kernel_metadata(lhs.kernel_metadata, rhs.kernel_metadata)
    et_kernel_metadata = merge_et_kernel_metadata(lhs.et_kernel_metadata, rhs.et_kernel_metadata)
    include_all_non_op_selectives = lhs.include_all_non_op_selectives or rhs.include_all_non_op_selectives
    custom_classes = lhs.custom_classes.union(rhs.custom_classes)
    build_features = lhs.build_features.union(rhs.build_features)
    return SelectiveBuilder(include_all_operators, debug_info, operators, kernel_metadata, et_kernel_metadata, custom_classes, build_features, include_all_non_op_selectives)

def op_name_from_native_function(f: NativeFunction) -> str:
    if False:
        for i in range(10):
            print('nop')
    return f'{f.namespace}::{f.func.name}'