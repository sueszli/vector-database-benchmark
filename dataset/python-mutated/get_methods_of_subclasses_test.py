import unittest
from unittest.mock import MagicMock, patch
from ....api import query
from .. import get_methods_of_subclasses
from ..generator_specifications import AllParametersAnnotation, AnnotationSpecification, WhitelistSpecification

class MethodsOfSubclassesGeneratorTest(unittest.TestCase):

    @patch.object(get_methods_of_subclasses, 'get_all_subclass_defines_from_pyre')
    def test_compute_models(self, get_all_subclass_defines_from_pyre_mock: MagicMock) -> None:
        if False:
            for i in range(10):
                print('nop')
        annotations = AnnotationSpecification(parameter_annotation=AllParametersAnnotation(arg='ArgAnnotation'), returns='ReturnAnnotation')
        whitelist = WhitelistSpecification(parameter_name={'ignored_arg'})
        generator = get_methods_of_subclasses.MethodsOfSubclassesGenerator(base_classes=['WantedParent1', 'WantedParent2'], pyre_connection=None, annotations=annotations, whitelist=whitelist)
        get_all_subclass_defines_from_pyre_mock.return_value = {'WantedParent1': [query.Define(name='WantedChild1.target_fn1', parameters=[query.DefineParameter(name='arg1', annotation=''), query.DefineParameter(name='ignored_arg', annotation='')], return_annotation=''), query.Define(name='WantedChild1.target_fn2', parameters=[], return_annotation='')], 'WantedParent2': [query.Define(name='WantedChild2.target_fn', parameters=[query.DefineParameter(name='arg', annotation=''), query.DefineParameter(name='*vararg', annotation=''), query.DefineParameter(name='**kwarg', annotation='')], return_annotation='')]}
        models = [str(model) for model in generator.compute_models(None)]
        self.assertListEqual(models, ['def WantedChild1.target_fn1(arg1: ArgAnnotation,' + ' ignored_arg) -> ReturnAnnotation: ...', 'def WantedChild1.target_fn2() -> ReturnAnnotation: ...', 'def WantedChild2.target_fn(arg: ArgAnnotation,' + ' *vararg, **kwarg) -> ReturnAnnotation: ...'])