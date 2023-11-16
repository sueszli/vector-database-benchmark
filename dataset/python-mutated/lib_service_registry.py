import importlib
import inspect
from inspect import Signature
from inspect import _signature_fromstr
from types import BuiltinFunctionType
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
import numpy
from typing_extensions import Self
from .lib_permissions import ALL_EXECUTE
from .lib_permissions import CMPPermission
from .lib_permissions import NONE_EXECUTE
from .signature import get_signature
LIB_IGNORE_ATTRIBUTES = {'os', '__abstractmethods__', '__base__', ' __bases__', '__class__'}

def import_from_path(path: str) -> type:
    if False:
        for i in range(10):
            print('nop')
    if '.' in path:
        (top_level_module, attr_path) = path.split('.', 1)
    else:
        top_level_module = path
        attr_path = ''
    res = importlib.import_module(top_level_module)
    path_parts = [x for x in attr_path.split('.') if x != '']
    for attr in path_parts:
        res = getattr(res, attr)
    return res

class CMPBase:
    """cmp: cascading module permissions"""

    def __init__(self, path: str, children: Optional[Union[List, Dict]]=None, permissions: Optional[CMPPermission]=None, obj: Optional[Any]=None, absolute_path: Optional[str]=None, text_signature: Optional[str]=None):
        if False:
            print('Hello World!')
        self.permissions: Optional[CMPPermission] = permissions
        self.path: str = path
        self.obj: Optional[Any] = obj if obj is not None else None
        self.absolute_path = absolute_path
        self.signature: Optional[Signature] = None
        self.children: Dict[str, CMPBase] = {}
        if isinstance(children, list):
            self.children = {f'{c.path}': c for c in children}
        elif isinstance(children, dict):
            self.children = children
        for c in self.children.values():
            if c.absolute_path is None:
                c.absolute_path = f'{path}.{c.path}'
        if text_signature is not None:
            self.signature = _signature_fromstr(inspect.Signature, obj, text_signature, True)
        self.is_built = False

    def set_signature(self) -> None:
        if False:
            while True:
                i = 10
        pass

    def build(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.obj is None:
            self.obj = import_from_path(self.absolute_path)
        if self.signature is None:
            self.set_signature()
        child_paths = set(self.children.keys())
        for attr_name in getattr(self.obj, '__dict__', {}).keys():
            if attr_name not in LIB_IGNORE_ATTRIBUTES:
                if attr_name in child_paths:
                    child = self.children[attr_name]
                else:
                    try:
                        attr = getattr(self.obj, attr_name)
                    except Exception:
                        continue
                    child = self.init_child(self.obj, f'{self.path}.{attr_name}', attr, f'{self.absolute_path}.{attr_name}')
                if child is not None:
                    child.build()
                    self.children[attr_name] = child

    def __getattr__(self, __name: str) -> Any:
        if False:
            print('Hello World!')
        if __name in self.children:
            return self.children[__name]
        else:
            raise ValueError(f'property {__name} not defined')

    def init_child(self, parent_obj: Union[type, object], child_path: str, child_obj: Union[type, object], absolute_path: str) -> Optional[Self]:
        if False:
            print('Hello World!')
        'Get the child of parent as a CMPBase object\n\n        Args:\n            parent_obj (_type_): parent object\n            child_path (_type_): _description_\n            child_obj (_type_): _description_\n\n        Returns:\n            _type_: _description_\n        '
        parent_is_parent_module = CMPBase.parent_is_parent_module(parent_obj, child_obj)
        if CMPBase.isfunction(child_obj) and parent_is_parent_module:
            return CMPFunction(child_path, permissions=self.permissions, obj=child_obj, absolute_path=absolute_path)
        elif inspect.ismodule(child_obj) and CMPBase.is_submodule(parent_obj, child_obj):
            return CMPModule(child_path, permissions=self.permissions, obj=child_obj, absolute_path=absolute_path)
        elif inspect.isclass(child_obj) and parent_is_parent_module:
            return CMPClass(child_path, permissions=self.permissions, obj=child_obj, absolute_path=absolute_path)
        else:
            return None

    @staticmethod
    def is_submodule(parent: type, child: type) -> bool:
        if False:
            print('Hello World!')
        try:
            if '.' not in child.__package__:
                return False
            else:
                child_parent_module = child.__package__.rsplit('.', 1)[0]
                if parent.__package__ == child_parent_module:
                    return True
                else:
                    return False
        except Exception:
            pass
        return False

    @staticmethod
    def parent_is_parent_module(parent_obj: Any, child_obj: Any) -> Optional[str]:
        if False:
            return 10
        try:
            if hasattr(child_obj, '__module__'):
                return child_obj.__module__ == parent_obj.__name__
            else:
                return child_obj.__class__.__module__ == parent_obj.__name__
        except Exception:
            pass
        return None

    def flatten(self) -> List[Self]:
        if False:
            while True:
                i = 10
        res = [self]
        for c in self.children.values():
            res += c.flatten()
        return res

    @staticmethod
    def isfunction(obj: Callable) -> bool:
        if False:
            return 10
        return inspect.isfunction(obj) or type(obj) == numpy.ufunc or isinstance(obj, BuiltinFunctionType)

    def __repr__(self, indent: int=0, is_last: bool=False, parent_path: str='') -> str:
        if False:
            i = 10
            return i + 15
        'Visualize the tree, e.g.:\n        ├───numpy (ALL_EXECUTE)\n        │    ├───ModuleDeprecationWarning (ALL_EXECUTE)\n        │    ├───VisibleDeprecationWarning (ALL_EXECUTE)\n        │    ├───_CopyMode (ALL_EXECUTE)\n        │    ├───compat (ALL_EXECUTE)\n        │    ├───core (ALL_EXECUTE)\n        │    │    ├───_ufunc_reconstruct (ALL_EXECUTE)\n        │    │    ├───_DType_reconstruct (ALL_EXECUTE)\n        │    │    └───__getattr__ (ALL_EXECUTE)\n        │    ├───char (ALL_EXECUTE)\n        │    │    ├───_use_unicode (ALL_EXECUTE)\n        │    │    ├───_to_string_or_unicode_array (ALL_EXECUTE)\n        │    │    ├───_clean_args (ALL_EXECUTE)\n\n        Args:\n            indent (int, optional): indentation level. Defaults to 0.\n            is_last (bool, optional): is last item of collection. Defaults to False.\n            parent_path (str, optional): path of the parent obj. Defaults to "".\n\n        Returns:\n            str: representation of the CMP\n        '
        (last_idx, c_indent) = (len(self.children) - 1, indent + 1)
        children_string = ''.join([c.__repr__(c_indent, is_last=i == last_idx, parent_path=self.path) for (i, c) in enumerate(sorted(self.children.values(), key=lambda x: x.permissions.permission_string))])
        tree_prefix = '└───' if is_last else '├───'
        indent_str = '│    ' * indent + tree_prefix
        if parent_path != '':
            path = self.path.replace(f'{parent_path}.', '')
        else:
            path = self.path
        return f'{indent_str}{path} ({self.permissions})\n{children_string}'

class CMPModule(CMPBase):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)

class CMPFunction(CMPBase):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self.set_signature()

    @property
    def name(self) -> str:
        if False:
            return 10
        return self.obj.__name__

    def set_signature(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        try:
            self.signature = get_signature(self.obj)
        except Exception:
            pass

class CMPClass(CMPBase):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.set_signature()

    @property
    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.obj.__name__

    def set_signature(self) -> None:
        if False:
            while True:
                i = 10
        try:
            self.signature = get_signature(self.obj)
        except Exception:
            try:
                self.signature = get_signature(self.obj.__init__)
            except Exception:
                pass

class CMPMethod(CMPBase):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)

class CMPProperty(CMPBase):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)

class CMPTree:
    """root node of the Tree(s), with one child per library"""

    def __init__(self, children: List[CMPModule]):
        if False:
            i = 10
            return i + 15
        self.children = {c.path: c for c in children}

    def build(self) -> Self:
        if False:
            i = 10
            return i + 15
        for c in self.children.values():
            c.absolute_path = c.path
            c.build()
        return self

    def flatten(self) -> Sequence[CMPBase]:
        if False:
            i = 10
            return i + 15
        res = []
        for c in self.children.values():
            res += c.flatten()
        return res

    def __getattr__(self, _name: str) -> Any:
        if False:
            while True:
                i = 10
        if _name in self.children:
            return self.children[_name]
        else:
            raise ValueError(f'property {_name} does not exist')

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '\n'.join([c.__repr__() for c in self.children.values()])
action_execute_registry_libs = CMPTree(children=[CMPModule('numpy', permissions=ALL_EXECUTE, children=[CMPFunction('concatenate', permissions=ALL_EXECUTE, text_signature="concatenate(a1,a2, *args,axis=0,out=None,dtype=None,casting='same_kind')"), CMPFunction('source', permissions=NONE_EXECUTE), CMPFunction('fromfile', permissions=NONE_EXECUTE), CMPFunction('set_numeric_ops', permissions=ALL_EXECUTE, text_signature='set_numeric_ops(op1,op2, *args)'), CMPModule('testing', permissions=NONE_EXECUTE)])]).build()