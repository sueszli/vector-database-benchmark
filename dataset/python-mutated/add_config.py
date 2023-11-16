import builtins
from importlib import import_module
from typing import Callable, Optional
from embedchain.config.base_config import BaseConfig
from embedchain.helper.json_serializable import register_deserializable

@register_deserializable
class ChunkerConfig(BaseConfig):
    """
    Config for the chunker used in `add` method
    """

    def __init__(self, chunk_size: Optional[int]=None, chunk_overlap: Optional[int]=None, length_function: Optional[Callable[[str], int]]=None):
        if False:
            while True:
                i = 10
        self.chunk_size = chunk_size if chunk_size else 2000
        self.chunk_overlap = chunk_overlap if chunk_overlap else 0
        if isinstance(length_function, str):
            self.length_function = self.load_func(length_function)
        else:
            self.length_function = length_function if length_function else len

    def load_func(self, dotpath: str):
        if False:
            return 10
        if '.' not in dotpath:
            return getattr(builtins, dotpath)
        else:
            (module_, func) = dotpath.rsplit('.', maxsplit=1)
            m = import_module(module_)
            return getattr(m, func)

@register_deserializable
class LoaderConfig(BaseConfig):
    """
    Config for the chunker used in `add` method
    """

    def __init__(self):
        if False:
            return 10
        pass

@register_deserializable
class AddConfig(BaseConfig):
    """
    Config for the `add` method.
    """

    def __init__(self, chunker: Optional[ChunkerConfig]=None, loader: Optional[LoaderConfig]=None):
        if False:
            while True:
                i = 10
        '\n        Initializes a configuration class instance for the `add` method.\n\n        :param chunker: Chunker config, defaults to None\n        :type chunker: Optional[ChunkerConfig], optional\n        :param loader: Loader config, defaults to None\n        :type loader: Optional[LoaderConfig], optional\n        '
        self.loader = loader
        self.chunker = chunker