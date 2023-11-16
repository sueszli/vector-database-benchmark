import inspect
from functools import wraps
from typing import Any, Callable, Optional, Type, Union, get_type_hints
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe
from torch.utils.data.datapipes._typing import _DataPipeMeta

class functional_datapipe:
    name: str

    def __init__(self, name: str, enable_df_api_tracing=False) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Define a functional datapipe.\n\n        Args:\n            enable_df_api_tracing - if set, any returned DataPipe would accept\n            DataFrames API in tracing mode.\n        '
        self.name = name
        self.enable_df_api_tracing = enable_df_api_tracing

    def __call__(self, cls):
        if False:
            print('Hello World!')
        if issubclass(cls, IterDataPipe):
            if isinstance(cls, Type):
                if not isinstance(cls, _DataPipeMeta):
                    raise TypeError('`functional_datapipe` can only decorate IterDataPipe')
            elif not isinstance(cls, non_deterministic) and (not (hasattr(cls, '__self__') and isinstance(cls.__self__, non_deterministic))):
                raise TypeError('`functional_datapipe` can only decorate IterDataPipe')
            IterDataPipe.register_datapipe_as_function(self.name, cls, enable_df_api_tracing=self.enable_df_api_tracing)
        elif issubclass(cls, MapDataPipe):
            MapDataPipe.register_datapipe_as_function(self.name, cls)
        return cls
_determinism: bool = False

class guaranteed_datapipes_determinism:
    prev: bool

    def __init__(self) -> None:
        if False:
            return 10
        global _determinism
        self.prev = _determinism
        _determinism = True

    def __enter__(self) -> None:
        if False:
            while True:
                i = 10
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        global _determinism
        _determinism = self.prev

class non_deterministic:
    cls: Optional[Type[IterDataPipe]] = None
    deterministic_fn: Callable[[], bool]

    def __init__(self, arg: Union[Type[IterDataPipe], Callable[[], bool]]) -> None:
        if False:
            i = 10
            return i + 15
        if isinstance(arg, Type):
            if not issubclass(arg, IterDataPipe):
                raise TypeError(f'Only `IterDataPipe` can be decorated with `non_deterministic`, but {arg.__name__} is found')
            self.cls = arg
        elif isinstance(arg, Callable):
            self.deterministic_fn = arg
        else:
            raise TypeError(f'{arg} can not be decorated by non_deterministic')

    def __call__(self, *args, **kwargs):
        if False:
            return 10
        global _determinism
        if self.cls is not None:
            if _determinism:
                raise TypeError("{} is non-deterministic, but you set 'guaranteed_datapipes_determinism'. You can turn off determinism for this DataPipe if that is acceptable for your application".format(self.cls.__name__))
            return self.cls(*args, **kwargs)
        if not (isinstance(args[0], Type) and issubclass(args[0], IterDataPipe)):
            raise TypeError(f'Only `IterDataPipe` can be decorated, but {args[0].__name__} is found')
        self.cls = args[0]
        return self.deterministic_wrapper_fn

    def deterministic_wrapper_fn(self, *args, **kwargs) -> IterDataPipe:
        if False:
            i = 10
            return i + 15
        res = self.deterministic_fn(*args, **kwargs)
        if not isinstance(res, bool):
            raise TypeError(f'deterministic_fn of `non_deterministic` decorator is required to return a boolean value, but {type(res)} is found')
        global _determinism
        if _determinism and res:
            raise TypeError(f"{self.cls.__name__} is non-deterministic with the inputs, but you set 'guaranteed_datapipes_determinism'. You can turn off determinism for this DataPipe if that is acceptable for your application")
        return self.cls(*args, **kwargs)

def argument_validation(f):
    if False:
        return 10
    signature = inspect.signature(f)
    hints = get_type_hints(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        bound = signature.bind(*args, **kwargs)
        for (argument_name, value) in bound.arguments.items():
            if argument_name in hints and isinstance(hints[argument_name], _DataPipeMeta):
                hint = hints[argument_name]
                if not isinstance(value, IterDataPipe):
                    raise TypeError(f"Expected argument '{argument_name}' as a IterDataPipe, but found {type(value)}")
                if not value.type.issubtype(hint.type):
                    raise TypeError(f"Expected type of argument '{argument_name}' as a subtype of hint {hint.type}, but found {value.type}")
        return f(*args, **kwargs)
    return wrapper
_runtime_validation_enabled: bool = True

class runtime_validation_disabled:
    prev: bool

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        global _runtime_validation_enabled
        self.prev = _runtime_validation_enabled
        _runtime_validation_enabled = False

    def __enter__(self) -> None:
        if False:
            return 10
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        global _runtime_validation_enabled
        _runtime_validation_enabled = self.prev

def runtime_validation(f):
    if False:
        while True:
            i = 10
    if f.__name__ != '__iter__':
        raise TypeError(f"Can not decorate function {f.__name__} with 'runtime_validation'")

    @wraps(f)
    def wrapper(self):
        if False:
            while True:
                i = 10
        global _runtime_validation_enabled
        if not _runtime_validation_enabled:
            yield from f(self)
        else:
            it = f(self)
            for d in it:
                if not self.type.issubtype_of_instance(d):
                    raise RuntimeError(f'Expected an instance as subtype of {self.type}, but found {d}({type(d)})')
                yield d
    return wrapper