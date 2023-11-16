import abc
import typing as t
import torch
import torch.fx
from torch.fx._compatibility import compatibility
from .shape_prop import TensorMetadata
from .tools_common import get_node_target, CALLABLE_NODE_OPS
__all__ = ['OperatorSupportBase', 'OperatorSupport', 'create_op_support', 'chain', 'OpSupports', 'any_chain']
TargetTypeName = str
SupportedArgumentDTypes = t.Optional[t.Tuple[t.Sequence[t.Sequence[torch.dtype]], t.Dict[str, t.Sequence[torch.dtype]]]]
SupportDict = t.Mapping[TargetTypeName, SupportedArgumentDTypes]

@compatibility(is_backward_compatible=False)
class OperatorSupportBase(abc.ABC):
    """Interface for determining if a fx.Node is supported by a backend"""

    @abc.abstractmethod
    def is_node_supported(self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

@compatibility(is_backward_compatible=False)
class OperatorSupport(OperatorSupportBase):
    """
    `_support_dict` maps node.target typename to supported inputs dtypes.

    node.target typename is retrieved using helper function `get_node_target()`

    If supported inputs dtypes is None, it means any dtype is supported, else
    we should see a tuple like (([dtypes], ...), {"name":[dtypes], ...}).

    The first tuple ([dtypes], ...) indicates what dtypes are supported for
    inputs in node.args and the second dict {"name": [dtypes], ...} indicates
    what dtypes are supported for inputs in node.kwargs.

    For inputs in args, if we don't want to check it, we can put None there,
    e.g. (None, [torch.float]) indicates that we don't care about the type of
    the first input in args. And for inputs in kwargs, if not listed, will not
    be checked.
    """
    _support_dict: SupportDict

    def __init__(self, support_dict: t.Optional[SupportDict]=None):
        if False:
            i = 10
            return i + 15
        self._support_dict = support_dict or {}

    def is_node_supported(self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
        if False:
            while True:
                i = 10
        "\n        Args:\n            `submodules`: mapping from module name to the module. This can be\n                          retrieved by calling model.named_modules().\n\n            `node`: a Fx node that we want to determine whether it's supported.\n\n        Returns:\n            `is_supported`: whether the arg `node` is supported.\n        "
        if node.op not in CALLABLE_NODE_OPS:
            return True
        target = get_node_target(submodules, node)
        if target not in self._support_dict:
            return False
        if self._support_dict[target] is None:
            return True
        (args_dtypes, kwargs_dtypes) = self._support_dict[target]
        for (i, dtypes) in enumerate(args_dtypes):
            if len(node.args) <= i:
                break
            if dtypes is None:
                continue
            if not isinstance(node.args[i], torch.fx.Node):
                continue
            arg_dtype = _get_arg_dtype(node.args[i])
            if arg_dtype not in dtypes:
                return False
        for (k, dtypes) in kwargs_dtypes.items():
            if k not in node.kwargs:
                continue
            if not isinstance(node.kwargs[k], torch.fx.Node):
                continue
            kwarg_dtype = _get_arg_dtype(node.kwargs[k])
            if kwarg_dtype not in dtypes:
                return False
        return True
IsNodeSupported = t.Callable[[t.Mapping[str, torch.nn.Module], torch.fx.Node], bool]

@compatibility(is_backward_compatible=False)
def create_op_support(is_node_supported: IsNodeSupported) -> OperatorSupportBase:
    if False:
        while True:
            i = 10
    'Wraps a `IsNodeSupported` function into an `OperatorSupportBase` instance\n\n    `IsNodeSupported` has the same call signature as\n    `OperatorSupportBase.is_node_supported`\n    '

    class FunctionalOperatorSupport(OperatorSupportBase):

        def is_node_supported(self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
            if False:
                i = 10
                return i + 15
            return is_node_supported(submodules, node)
    return FunctionalOperatorSupport()

@compatibility(is_backward_compatible=False)
def chain(*op_support: OperatorSupportBase) -> OperatorSupportBase:
    if False:
        while True:
            i = 10
    'Combines a sequence of `OperatorSupportBase` instances to form a single `OperatorSupportBase`\n    instance by evaluating each input `OperatorSupportBase` instance, and returns False if\n    any of it reports False.\n    '

    def _chain(submods, node) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return all((x.is_node_supported(submods, node) for x in op_support))
    return create_op_support(_chain)

@compatibility(is_backward_compatible=False)
def any_chain(*op_support: OperatorSupportBase) -> OperatorSupportBase:
    if False:
        print('Hello World!')
    'Combines a sequence of `OperatorSupportBase` instances to form a single `OperatorSupportBase`\n    instance by evaluating each input `OperatorSupportBase` instance, and returns True if\n    any of it reports True.\n    '

    def _any_chain(submods, node) -> bool:
        if False:
            i = 10
            return i + 15
        return any((x.is_node_supported(submods, node) for x in op_support))
    return create_op_support(_any_chain)

@compatibility(is_backward_compatible=False)
class OpSupports:
    """A set of atomic `OperatorSupportBase` instances that can be combined together
    to form more complex operator support logic.
    """

    @classmethod
    def decline_if_input_dtype(cls, dtype: torch.dtype) -> OperatorSupportBase:
        if False:
            print('Hello World!')
        'Report a node as non-supported, if any of its arguments is of dtype'

        def _decline_if_input_dtype(submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
            if False:
                while True:
                    i = 10
            for arg in node.all_input_nodes:
                if arg.op == 'get_attr':
                    continue
                arg_dtype = _get_arg_dtype(arg)
                if arg_dtype == dtype:
                    return False
            return True
        return create_op_support(_decline_if_input_dtype)

    @classmethod
    def decline_if_node_in_names(cls, disallow_set: t.Set[str]) -> OperatorSupportBase:
        if False:
            for i in range(10):
                print('nop')
        '\n        If a node has a name that is in the disallow set, reported it as non-supported.\n        '

        def _decline_if_node_in_names(submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
            if False:
                return 10
            if node.name in disallow_set:
                return False
            else:
                return True
        return create_op_support(_decline_if_node_in_names)

def _get_arg_dtype(arg: torch.fx.Node) -> t.Any:
    if False:
        while True:
            i = 10
    assert isinstance(arg, torch.fx.Node)
    tensor_meta = arg.meta.get('tensor_meta')
    dtype = tensor_meta.dtype if isinstance(tensor_meta, TensorMetadata) else arg.meta['type']
    return dtype