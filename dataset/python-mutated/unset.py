import warnings
from typing import Any, Dict, Optional, Type
DEPRECATED_NAMES: Dict[str, str] = {'is_unset': '`is_unset` is deprecated use `value is UNSET` instead'}

class UnsetType:
    __instance: Optional['UnsetType'] = None

    def __new__(cls: Type['UnsetType']) -> 'UnsetType':
        if False:
            while True:
                i = 10
        if cls.__instance is None:
            ret = super().__new__(cls)
            cls.__instance = ret
            return ret
        else:
            return cls.__instance

    def __str__(self):
        if False:
            print('Hello World!')
        return ''

    def __repr__(self) -> str:
        if False:
            return 10
        return 'UNSET'

    def __bool__(self):
        if False:
            i = 10
            return i + 15
        return False
UNSET: Any = UnsetType()

def _deprecated_is_unset(value: Any) -> bool:
    if False:
        return 10
    warnings.warn(DEPRECATED_NAMES['is_unset'], DeprecationWarning, stacklevel=2)
    return value is UNSET

def __getattr__(name: str) -> Any:
    if False:
        return 10
    if name in DEPRECATED_NAMES:
        warnings.warn(DEPRECATED_NAMES[name], DeprecationWarning, stacklevel=2)
        return globals()[f'_deprecated_{name}']
    raise AttributeError(f'module {__name__} has no attribute {name}')
__all__ = ['UNSET']