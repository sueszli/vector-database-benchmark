import logging
from typing import Callable, Iterable
from .model import Model
from .model_generator import ModelGenerator
LOG: logging.Logger = logging.getLogger(__name__)

class FilteredSourceGenerator(ModelGenerator[Model]):

    def __init__(self, superset_generator: ModelGenerator[Model], subset_generator: ModelGenerator[Model]) -> None:
        if False:
            i = 10
            return i + 15
        self.superset_generator = superset_generator
        self.subset_generator = subset_generator

    def gather_functions_to_model(self) -> Iterable[Callable[..., object]]:
        if False:
            return 10
        return []

    def compute_models(self, functions_to_model: Iterable[Callable[..., object]]) -> Iterable[Model]:
        if False:
            print('Hello World!')
        LOG.info('Computing models for the superset...')
        superset_models = self.superset_generator.generate_models()
        LOG.info('Computing models for the subset...')
        subset_models = self.subset_generator.generate_models()
        return set(superset_models) - set(subset_models)