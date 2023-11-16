"""Distributed engine and memory format configuration."""
import importlib.util
import os
import threading
from collections import defaultdict
from enum import Enum, unique
from functools import wraps
from importlib import reload
from typing import Any, Callable, Dict, Literal, Optional, TypeVar, cast
EngineLiteral = Literal['python', 'ray']
MemoryFormatLiteral = Literal['pandas', 'modin']
FunctionType = TypeVar('FunctionType', bound=Callable[..., Any])
WR_ENGINE: Optional[EngineLiteral] = os.getenv('WR_ENGINE')
WR_MEMORY_FORMAT: Optional[MemoryFormatLiteral] = os.getenv('WR_MEMORY_FORMAT')

@unique
class EngineEnum(Enum):
    """Execution engine enum."""
    RAY = 'ray'
    PYTHON = 'python'

@unique
class MemoryFormatEnum(Enum):
    """Memory format enum."""
    MODIN = 'modin'
    PANDAS = 'pandas'

class Engine:
    """Execution engine configuration class."""
    _engine: Optional[EngineEnum] = EngineEnum[WR_ENGINE.upper()] if WR_ENGINE else None
    _initialized_engine: Optional[EngineEnum] = None
    _registry: Dict[EngineLiteral, Dict[str, Callable[..., Any]]] = defaultdict(dict)
    _lock: threading.RLock = threading.RLock()

    @classmethod
    def get_installed(cls) -> EngineEnum:
        if False:
            for i in range(10):
                print('nop')
        'Get the installed distribution engine.\n\n        This is the engine that can be imported.\n\n        Returns\n        -------\n        EngineEnum\n            The distribution engine installed.\n        '
        if importlib.util.find_spec('ray'):
            return EngineEnum.RAY
        return EngineEnum.PYTHON

    @classmethod
    def get(cls) -> EngineEnum:
        if False:
            while True:
                i = 10
        'Get the configured distribution engine.\n\n        This is the engine currently configured. If None, the installed engine is returned.\n\n        Returns\n        -------\n        str\n            The distribution engine configured.\n        '
        with cls._lock:
            return cls._engine if cls._engine else cls.get_installed()

    @classmethod
    def set(cls, name: EngineLiteral) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the distribution engine.'
        with cls._lock:
            cls._engine = EngineEnum[name.upper()]

    @classmethod
    def dispatch_func(cls, source_func: FunctionType, value: Optional[EngineLiteral]=None) -> FunctionType:
        if False:
            while True:
                i = 10
        'Dispatch a func based on value or the distribution engine and the source function.'
        try:
            with cls._lock:
                return cls._registry[value or cls.get().value][source_func.__name__]
        except KeyError:
            return getattr(source_func, '_source_func', source_func)

    @classmethod
    def register_func(cls, source_func: FunctionType, destination_func: FunctionType) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Register a func based on the distribution engine and source function.'
        with cls._lock:
            cls._registry[cls.get().value][source_func.__name__] = destination_func

    @classmethod
    def dispatch_on_engine(cls, func: FunctionType) -> FunctionType:
        if False:
            print('Hello World!')
        'Dispatch on engine function decorator.'

        @wraps(func)
        def wrapper(*args: Any, **kw: Dict[str, Any]) -> Any:
            if False:
                print('Hello World!')
            cls.initialize(name=cls.get().value)
            return cls.dispatch_func(func)(*args, **kw)
        wrapper._source_func = func
        return wrapper

    @classmethod
    def register(cls, name: Optional[EngineLiteral]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Register the distribution engine dispatch methods.'
        with cls._lock:
            engine_name = cast(EngineLiteral, name or cls.get().value)
            cls.set(engine_name)
            cls._registry.clear()
            if engine_name == EngineEnum.RAY.value:
                from awswrangler.distributed.ray._register import register_ray
                register_ray()

    @classmethod
    def initialize(cls, name: Optional[EngineLiteral]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize the distribution engine.'
        with cls._lock:
            engine_name = cast(EngineLiteral, name or cls.get_installed().value)
            if engine_name == EngineEnum.RAY.value:
                from awswrangler.distributed.ray import initialize_ray
                initialize_ray()
            cls._initialized_engine = EngineEnum[engine_name.upper()]

    @classmethod
    def is_initialized(cls, name: Optional[EngineLiteral]=None) -> bool:
        if False:
            while True:
                i = 10
        'Check if the distribution engine is initialized.'
        with cls._lock:
            engine_name = cast(EngineLiteral, name or cls.get_installed().value)
            return False if not cls._initialized_engine else cls._initialized_engine.value == engine_name

class MemoryFormat:
    """Memory format configuration class."""
    _enum: Optional[MemoryFormatEnum] = MemoryFormatEnum[WR_MEMORY_FORMAT.upper()] if WR_MEMORY_FORMAT else None
    _lock: threading.RLock = threading.RLock()

    @classmethod
    def get_installed(cls) -> MemoryFormatEnum:
        if False:
            for i in range(10):
                print('nop')
        'Get the installed memory format.\n\n        This is the format that can be imported.\n\n        Returns\n        -------\n        Enum\n            The memory format installed.\n        '
        if importlib.util.find_spec('modin'):
            return MemoryFormatEnum.MODIN
        return MemoryFormatEnum.PANDAS

    @classmethod
    def get(cls) -> MemoryFormatEnum:
        if False:
            while True:
                i = 10
        'Get the configured memory format.\n\n        This is the memory format currently configured. If None, the installed memory format is returned.\n\n        Returns\n        -------\n        Enum\n            The memory format configured.\n        '
        with cls._lock:
            return cls._enum if cls._enum else cls.get_installed()

    @classmethod
    def set(cls, name: EngineLiteral) -> None:
        if False:
            return 10
        'Set the memory format.'
        with cls._lock:
            cls._enum = MemoryFormatEnum[name.upper()]
            _reload()

def _reload() -> None:
    if False:
        while True:
            i = 10
    'Reload Pandas proxy module.'
    import awswrangler.pandas
    reload(awswrangler.pandas)
engine: Engine = Engine()
memory_format: MemoryFormat = MemoryFormat()