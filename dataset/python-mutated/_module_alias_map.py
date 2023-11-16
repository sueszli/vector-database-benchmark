import importlib
import importlib.util
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from types import ModuleType
from typing import Mapping, Optional, Sequence, Union

def get_meta_path_insertion_index() -> int:
    if False:
        while True:
            i = 10
    for i in range(len(sys.meta_path)):
        finder = sys.meta_path[i]
        if isinstance(finder, type) and issubclass(finder, PathFinder):
            return i
    raise Exception('Could not find the built-in PathFinder in sys.meta_path-- cannot insert the AliasedModuleFinder')

class AliasedModuleFinder(MetaPathFinder):

    def __init__(self, alias_map: Mapping[str, str]):
        if False:
            i = 10
            return i + 15
        self.alias_map = alias_map

    def find_spec(self, fullname: str, _path: Optional[Sequence[Union[bytes, str]]]=None, _target: Optional[ModuleType]=None) -> Optional[ModuleSpec]:
        if False:
            i = 10
            return i + 15
        head = next((k for k in self.alias_map.keys() if fullname.startswith(k)), None)
        if head is not None:
            base_name = self.alias_map[head] + fullname[len(head):]
            base_spec = importlib.util.find_spec(base_name)
            assert base_spec, f'Could not find module spec for {base_name}.'
            return ModuleSpec(fullname, AliasedModuleLoader(fullname, base_spec), origin=base_spec.origin, is_package=base_spec.submodule_search_locations is not None)
        else:
            return None

class AliasedModuleLoader(Loader):

    def __init__(self, alias: str, base_spec: ModuleSpec):
        if False:
            return 10
        self.alias = alias
        self.base_spec = base_spec

    def exec_module(self, _module: ModuleType) -> None:
        if False:
            for i in range(10):
                print('nop')
        base_module = importlib.import_module(self.base_spec.name)
        sys.modules[self.alias] = base_module

    def module_repr(self, module: ModuleType) -> str:
        if False:
            return 10
        assert self.base_spec.loader
        return self.base_spec.loader.module_repr(module)