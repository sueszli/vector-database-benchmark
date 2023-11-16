import logging
from typing import Callable, Iterable, List, TypeVar
from .model import Model
from .model_generator import ModelGenerator
LOG: logging.Logger = logging.getLogger(__name__)
T = TypeVar('T', bound=Model)

class ModelsFilteredByCallableGenerator(ModelGenerator[T]):

    def __init__(self, generator_to_filter: ModelGenerator[T], filter: Callable[[T], bool]) -> None:
        if False:
            print('Hello World!')
        self.generator_to_filter = generator_to_filter
        self.filter = filter

    def gather_functions_to_model(self) -> Iterable[Callable[..., object]]:
        if False:
            while True:
                i = 10
        return self.generator_to_filter.gather_functions_to_model()

    def compute_models(self, functions_to_model: Iterable[Callable[..., object]]) -> List[T]:
        if False:
            i = 10
            return i + 15
        return [model for model in self.generator_to_filter.compute_models(functions_to_model) if self.filter(model)]