import unittest
from unittest.mock import MagicMock
from ..get_constructor_initialized_attribute_sources import ConstructorInitializedAttributeSourceGenerator
from .test_functions import __name__ as qualifier, TestChildClassB, TestGrandChildClassA

class ConstructorInitializedAttributeSourceGeneratorTest(unittest.TestCase):

    def test_compute_models(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pyre_connection = MagicMock()
        pyre_connection.query_server.return_value = {'response': [{'response': {'attributes': [{'name': '__init__', 'annotation': f'BoundMethod[typing.Callable({qualifier}.TestGrandChildClassA.__init__)[[Named(self, {qualifier}.TestGrandChildClassA), Named(x, int)], typing.Any], {qualifier}.TestGrandChildClassA]', 'kind': 'regular', 'final': False}, {'name': 'x', 'annotation': 'int', 'kind': 'regular', 'final': False}]}}, {'response': {'attributes': [{'name': '__init__', 'annotation': f'BoundMethod[typing.Callable({qualifier}.TestChildClassB.__init__)[[Named(self, {qualifier}.TestChildClassB), Named(x, int)], typing.Any], {qualifier}.TestChildClassB]', 'kind': 'regular', 'final': False}, {'name': 'x', 'annotation': 'int', 'kind': 'regular', 'final': False}]}}]}
        self.assertEqual(set(map(str, ConstructorInitializedAttributeSourceGenerator(classes_to_taint=[f'{qualifier}.TestClass'], pyre_connection=pyre_connection, taint_annotation='Taint').compute_models([TestGrandChildClassA.__init__, TestChildClassB.__init__]))), {f'{qualifier}.TestGrandChildClassA.x: Taint = ...', f'{qualifier}.TestChildClassB.x: Taint = ...'})

    def test_gather_functions_to_model(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertEqual(set(ConstructorInitializedAttributeSourceGenerator(classes_to_taint=[f'{qualifier}.TestClass'], pyre_connection=MagicMock(), taint_annotation='Taint').gather_functions_to_model()), {TestGrandChildClassA.__init__, TestChildClassB.__init__})

    def test_filter(self) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(set(ConstructorInitializedAttributeSourceGenerator(classes_to_taint=[f'{qualifier}.TestClass'], pyre_connection=MagicMock(), filter_classes_by=lambda module: not module.__name__ == 'TestChildClassB', taint_annotation='Taint').gather_functions_to_model()), {TestGrandChildClassA.__init__})