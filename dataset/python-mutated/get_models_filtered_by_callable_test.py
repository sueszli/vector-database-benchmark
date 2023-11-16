import unittest
from typing import Callable, Iterable, List
from ..get_models_filtered_by_callable import ModelsFilteredByCallableGenerator
from ..model import Model
from ..model_generator import ModelGenerator

class TestModel(Model):

    def __init__(self, index: int) -> None:
        if False:
            while True:
                i = 10
        self.index = index

    def __eq__(self, other: 'TestModel') -> int:
        if False:
            while True:
                i = 10
        return self.index == other.index

    def __hash__(self) -> int:
        if False:
            return 10
        pass

class TestModelGenerator(ModelGenerator[TestModel]):

    def gather_functions_to_model(self) -> Iterable[Callable[..., object]]:
        if False:
            return 10
        return []

    def compute_models(self, functions_to_model: Iterable[Callable[..., object]]) -> List[TestModel]:
        if False:
            for i in range(10):
                print('nop')
        return [TestModel(0), TestModel(1), TestModel(2)]

def is_even_index(model: TestModel) -> bool:
    if False:
        return 10
    return model.index % 2 == 0

class ModelsFilteredByCallableGeneratorTest(unittest.TestCase):

    def test_compute_models(self) -> None:
        if False:
            print('Hello World!')
        generator = ModelsFilteredByCallableGenerator(generator_to_filter=TestModelGenerator(), filter=is_even_index)
        self.assertListEqual(generator.compute_models([]), [TestModel(0), TestModel(2)])