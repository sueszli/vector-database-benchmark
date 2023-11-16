import ast
import unittest
from unittest.mock import MagicMock, patch
from ..generator_specifications import AllParametersAnnotation
from ..get_REST_api_sources import RESTApiSourceGenerator
from ..get_undecorated_sources import __name__ as undecorated_source_name, UndecoratedSourceGenerator
from ..model import CallableModel, FunctionDefinitionModel
from .test_functions import all_functions, testA, testB, TestClass

class GetUndecoratedSourcesTest(unittest.TestCase):

    @patch.object(RESTApiSourceGenerator, 'generate_models')
    @patch('{}.AnnotatedFreeFunctionWithDecoratorGenerator'.format(undecorated_source_name))
    def test_compute_models(self, mock_annotated_decorator: MagicMock, mock_RESTapi_decorator_generate_models: MagicMock) -> None:
        if False:
            while True:
                i = 10
        mock_RESTapi_decorator_generate_models.return_value = {CallableModel(testA, parameter_annotation=AllParametersAnnotation(arg='TaintSource[UserControlled]', vararg='TaintSource[UserControlled]', kwarg='TaintSource[UserControlled]')), CallableModel(testB, parameter_annotation=AllParametersAnnotation(arg='TaintSource[UserControlled]', vararg='TaintSource[UserControlled]', kwarg='TaintSource[UserControlled]')), CallableModel(TestClass().methodA, parameter_annotation=AllParametersAnnotation(arg='TaintSource[UserControlled]', vararg='TaintSource[UserControlled]', kwarg='TaintSource[UserControlled]'))}
        generator_instance = MagicMock()
        generator_instance.generate_models.return_value = {FunctionDefinitionModel(ast.parse('def testA(): pass').body[0], parameter_annotation=AllParametersAnnotation(arg='TaintSource[UserControlled]', vararg='TaintSource[UserControlled]', kwarg='TaintSource[UserControlled]'), qualifier='tools.pyre.tools.generate_taint_models.tests.test_functions')}
        mock_annotated_decorator.side_effect = [generator_instance]
        self.maxDiff = None
        self.assertEqual({*map(str, UndecoratedSourceGenerator(source_generator=RESTApiSourceGenerator(django_urls=MagicMock()), root='/root', decorators_to_filter=[]).compute_models(all_functions))}, {'def tools.pyre.tools.generate_taint_models.tests.test_functions.TestClass.methodA(self: TaintSource[UserControlled], x: TaintSource[UserControlled]): ...', 'def tools.pyre.tools.generate_taint_models.tests.test_functions.testB(x: TaintSource[UserControlled]): ...'})