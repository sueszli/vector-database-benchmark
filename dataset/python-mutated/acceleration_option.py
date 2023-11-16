from abc import abstractmethod
from typing import Optional, List, Set, Dict
from bigdl.nano.utils.common import _inc_checker, _ipex_checker, _onnxruntime_checker, _openvino_checker
_whole_acceleration_options = ['inc', 'ipex', 'onnxruntime', 'openvino', 'pot', 'bf16', 'fp16', 'jit', 'fx', 'channels_last']

class AccelerationOption(object):
    __slot__ = _whole_acceleration_options

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        '\n        initialize optimization option\n        '
        for option in _whole_acceleration_options:
            setattr(self, option, kwargs.get(option, False))
        self.method = kwargs.get('method', None)

    def get_precision(self):
        if False:
            return 10
        if self.inc or self.pot or self.fx:
            return 'int8'
        if self.bf16:
            return 'bf16'
        if self.fp16:
            return 'fp16'
        return 'fp32'

    def get_accelerator(self):
        if False:
            for i in range(10):
                print('nop')
        if self.onnxruntime:
            return 'onnxruntime'
        if self.openvino:
            return 'openvino'
        if self.jit:
            return 'jit'
        return None

    @abstractmethod
    def optimize(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass

def available_acceleration_combination(excludes: Optional[List[str]], includes: Optional[List[str]], full_methods: Dict[str, AccelerationOption], all_methods: Dict[str, AccelerationOption]=None):
    if False:
        i = 10
        return i + 15
    '\n    :return: a dictionary states the availability (if meet dependencies)\n    '
    dependency_checker = {'inc': _inc_checker, 'ipex': _ipex_checker, 'onnxruntime': _onnxruntime_checker, 'openvino': _openvino_checker, 'pot': _openvino_checker}
    if excludes is None:
        exclude_set: Set[str] = set()
    else:
        exclude_set: Set[str] = set(excludes)
        exclude_set.discard('original')
    if includes is None:
        include_set: Set[str] = set(full_methods.keys())
    else:
        include_set: Set[str] = set(includes)
        include_set.add('original')
        if all_methods is not None:
            for method in include_set:
                if method not in full_methods:
                    full_methods[method] = all_methods[method]
    available_dict = {}
    for (method, option) in full_methods.items():
        if method not in include_set:
            continue
        if method in exclude_set:
            continue
        available_iter = True
        for (name, value) in option.__dict__.items():
            if value is True:
                if name in dependency_checker and (not dependency_checker[name]()):
                    available_iter = False
        available_dict[method] = available_iter
    return available_dict