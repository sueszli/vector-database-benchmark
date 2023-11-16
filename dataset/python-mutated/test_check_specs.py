import abc
import time
import unittest
from typing import Dict, Any, Type
import numpy as np
import torch
from ray.rllib.core.models.specs.checker import SpecCheckingError
from ray.rllib.core.models.specs.checker import convert_to_canonical_format, check_input_specs, check_output_specs
from ray.rllib.core.models.specs.specs_base import TensorSpec, TypeSpec
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.utils.annotations import override
from ray.rllib.utils.nested_dict import NestedDict
ONLY_ONE_KEY_ALLOWED = 'Only one key is allowed in the data dict.'

class AbstractInterfaceClass(abc.ABC):
    """An abstract class that has a couple of methods, each having their own
    input/output constraints."""

    @property
    @abc.abstractmethod
    def input_specs(self) -> SpecDict:
        if False:
            for i in range(10):
                print('nop')
        pass

    @property
    @abc.abstractmethod
    def output_specs(self) -> SpecDict:
        if False:
            print('Hello World!')
        pass

    @check_input_specs('input_specs', filter=True, cache=False, only_check_on_retry=False)
    @check_output_specs('output_specs', cache=False)
    def check_input_and_output(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return self._check_input_and_output(input_dict)

    @abc.abstractmethod
    def _check_input_and_output(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        pass

    @check_input_specs('input_specs', filter=True, cache=False, only_check_on_retry=False)
    def check_only_input(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'should not override this method'
        return self._check_only_input(input_dict)

    @abc.abstractmethod
    def _check_only_input(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        pass

    @check_output_specs('output_specs', cache=False)
    def check_only_output(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'should not override this method'
        return self._check_only_output(input_dict)

    @abc.abstractmethod
    def _check_only_output(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        pass

    @check_input_specs('input_specs', filter=True, cache=True, only_check_on_retry=False)
    @check_output_specs('output_specs', cache=True)
    def check_input_and_output_with_cache(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        'should not override this method'
        return self._check_input_and_output(input_dict)

    @check_input_specs('input_specs', filter=False, cache=False, only_check_on_retry=False)
    @check_output_specs('output_specs', cache=False)
    def check_input_and_output_wo_filter(self, input_dict) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'should not override this method'
        return self._check_input_and_output(input_dict)

class InputNumberOutputFloat(AbstractInterfaceClass):
    """This is an abstract class enforcing a contraint on input/output"""

    @property
    def input_specs(self) -> SpecDict:
        if False:
            print('Hello World!')
        return SpecDict({'input': (float, int)})

    @property
    def output_specs(self) -> SpecDict:
        if False:
            while True:
                i = 10
        return SpecDict({'output': float})

class CorrectImplementation(InputNumberOutputFloat):

    def run(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            return 10
        output = float(input_dict['input']) * 2
        return {'output': output}

    @override(AbstractInterfaceClass)
    def _check_input_and_output(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        if len(input_dict) > 1 or 'input' not in input_dict:
            raise ValueError(ONLY_ONE_KEY_ALLOWED)
        return self.run(input_dict)

    @override(AbstractInterfaceClass)
    def _check_only_input(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        if len(input_dict) > 1 or 'input' not in input_dict:
            raise ValueError(ONLY_ONE_KEY_ALLOWED)
        out = self.run(input_dict)
        return {'output': str(out)}

    @override(AbstractInterfaceClass)
    def _check_only_output(self, input_dict) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        if 'input' in input_dict:
            raise ValueError('input_dict should not have `input` key in check_only_output')
        return self.run({'input': input_dict['not_input']})

class IncorrectImplementation(CorrectImplementation):

    @override(CorrectImplementation)
    def run(self, input_dict) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        output = str(input_dict['input'] * 2)
        return {'output': output}

class TestCheckSpecs(unittest.TestCase):

    def test_check_input_and_output(self):
        if False:
            return 10
        correct_module = CorrectImplementation()
        output = correct_module.check_input_and_output({'input': 2})
        correct_module.output_specs.validate(NestedDict(output))
        self.assertRaises(SpecCheckingError, lambda : correct_module.check_input_and_output({'not_input': 2}))

    def test_check_only_input(self):
        if False:
            for i in range(10):
                print('nop')
        correct_module = CorrectImplementation()
        output = correct_module.check_only_input({'input': 2})
        self.assertRaises(ValueError, lambda : correct_module.output_specs.validate(NestedDict(output)))

    def test_check_only_output(self):
        if False:
            print('Hello World!')
        correct_module = CorrectImplementation()
        output = correct_module.check_only_output({'not_input': 2})
        correct_module.output_specs.validate(NestedDict(output))

    def test_incorrect_implementation(self):
        if False:
            for i in range(10):
                print('nop')
        incorrect_module = IncorrectImplementation()
        self.assertRaises(SpecCheckingError, lambda : incorrect_module.check_input_and_output({'input': 2}))
        incorrect_module.check_only_input({'input': 2})
        self.assertRaises(SpecCheckingError, lambda : incorrect_module.check_only_output({'not_input': 2}))

    def test_filter(self):
        if False:
            return 10
        input_dict = NestedDict({'input': 2})
        for i in range(100):
            inds = (str(i),) + tuple((str(j) for j in range(i + 1, i + 11)))
            input_dict[inds] = i
        correct_module = CorrectImplementation()
        correct_module.check_input_and_output(input_dict)
        self.assertRaises(ValueError, lambda : correct_module.check_input_and_output_wo_filter(input_dict))

    def test_cache(self):
        if False:
            return 10
        input_dict = NestedDict({'input': 2})
        for i in range(100):
            inds = (str(i),) + tuple((str(j) for j in range(i + 1, i + 11)))
            input_dict[inds] = i
        N = 500
        (time1, time2) = ([], [])
        for _ in range(N):
            module = CorrectImplementation()
            fn = getattr(module, 'check_input_and_output_with_cache')
            start = time.time()
            fn(input_dict)
            end = time.time()
            time1.append(end - start)
            start = time.time()
            fn(input_dict)
            end = time.time()
            time2.append(end - start)
        lower_bound_time1 = np.mean(time1)
        upper_bound_time2 = np.mean(time2)
        print(f'time1: {np.mean(time1)}')
        print(f'time2: {np.mean(time2)}')
        self.assertGreater(lower_bound_time1, upper_bound_time2)

    def test_tensor_specs(self):
        if False:
            while True:
                i = 10

        class ClassWithTensorSpec:

            @property
            def input_spec1(self) -> TensorSpec:
                if False:
                    while True:
                        i = 10
                return TensorSpec('b, h', h=4, framework='torch')

            @check_input_specs('input_spec1', cache=False, only_check_on_retry=False)
            def forward(self, input_data) -> Any:
                if False:
                    return 10
                return input_data
        module = ClassWithTensorSpec()
        module.forward(torch.rand(2, 4))
        self.assertRaises(SpecCheckingError, lambda : module.forward(torch.rand(2, 3)))

    def test_type_specs(self):
        if False:
            print('Hello World!')

        class SpecialOutputType:
            pass

        class WrongOutputType:
            pass

        class ClassWithTypeSpec:

            @property
            def output_specs(self) -> Type:
                if False:
                    print('Hello World!')
                return SpecialOutputType

            @check_output_specs('output_specs', cache=False)
            def forward_pass(self, input_data) -> Any:
                if False:
                    return 10
                return SpecialOutputType()

            @check_output_specs('output_specs', cache=False)
            def forward_fail(self, input_data) -> Any:
                if False:
                    print('Hello World!')
                return WrongOutputType()
        module = ClassWithTypeSpec()
        output = module.forward_pass(torch.rand(2, 4))
        self.assertIsInstance(output, SpecialOutputType)
        self.assertRaises(SpecCheckingError, lambda : module.forward_fail(torch.rand(2, 3)))

    def test_convert_to_canonical_format(self):
        if False:
            i = 10
            return i + 15
        self.assertDictEqual(convert_to_canonical_format(['foo', 'bar']).asdict(), SpecDict({'foo': None, 'bar': None}).asdict())
        self.assertDictEqual(convert_to_canonical_format(['foo', ('bar', 'jar')]).asdict(), SpecDict({'foo': None, 'bar': {'jar': None}}).asdict())
        returned = convert_to_canonical_format({'foo': {'bar': TensorSpec('b', framework='torch')}, 'jar': {'tar': int, 'car': None}})
        self.assertIsInstance(returned, SpecDict)
        self.assertDictEqual(returned.asdict(), SpecDict({'foo': {'bar': TensorSpec('b', framework='torch')}, 'jar': {'tar': TypeSpec(int), 'car': None}}).asdict())
        returned = convert_to_canonical_format(SpecDict({'foo': {'bar': TensorSpec('b', framework='torch')}, 'jar': {'tar': int}}))
        self.assertIsInstance(returned, SpecDict)
        self.assertDictEqual(returned.asdict(), SpecDict({'foo': {'bar': TensorSpec('b', framework='torch')}, 'jar': {'tar': TypeSpec(int)}}).asdict())
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))