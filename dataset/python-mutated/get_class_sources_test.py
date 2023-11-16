import unittest
from ..get_class_sources import ClassSourceGenerator
from .test_functions import __name__ as qualifier, TestChildClassB, TestGrandChildClassA

class GetClassSourcesTest(unittest.TestCase):

    def test_gather_functions_to_model(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(set(ClassSourceGenerator(classes_to_taint=[f'{qualifier}.TestClass']).gather_functions_to_model()), {TestChildClassB.__init__, TestGrandChildClassA.__init__})