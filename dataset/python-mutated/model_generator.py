import logging
import os
from abc import ABC, abstractmethod
from typing import Callable, Generic, Iterable, TypeVar
from .model import Model
LOG: logging.Logger = logging.getLogger(__name__)

def qualifier(root: str, path: str) -> str:
    if False:
        while True:
            i = 10
    path = os.path.relpath(path, root)
    if path.endswith('.pyi'):
        path = path[:-4]
    elif path.endswith('.py'):
        path = path[:-3]
    qualifier = path.replace('/', '.')
    if qualifier.endswith('.__init__'):
        qualifier = qualifier[:-9]
    return qualifier
T = TypeVar('T', bound=Model, covariant=True)

class ModelGenerator(ABC, Generic[T]):

    @abstractmethod
    def compute_models(self, functions_to_model: Iterable[Callable[..., object]]) -> Iterable[T]:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def gather_functions_to_model(self) -> Iterable[Callable[..., object]]:
        if False:
            while True:
                i = 10
        pass

    def generate_models(self) -> Iterable[T]:
        if False:
            i = 10
            return i + 15
        return self.compute_models(self.gather_functions_to_model())