import unittest
from ray.rllib.models.base_model import UnrollOutputType, Model, RecurrentModel, ForwardOutputType
import numpy as np
from ray.rllib.models.temp_spec_classes import TensorDict, SpecDict
from ray.rllib.utils.annotations import override
from ray.rllib.utils.test_utils import check

class NpRecurrentModelImpl(RecurrentModel):
    """A numpy recurrent model for checking:
    (1) initial states
    (2) that model in/out is as expected
    (3) unroll logic
    (4) spec checking"""

    def __init__(self, input_check=None, output_check=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.input_check = input_check
        self.output_check = output_check

    @property
    @override(RecurrentModel)
    def input_specs(self):
        if False:
            print('Hello World!')
        return SpecDict({'in': 'h'}, h=3)

    @property
    @override(RecurrentModel)
    def output_specs(self):
        if False:
            while True:
                i = 10
        return SpecDict({'out': 'o'}, o=2)

    @property
    @override(RecurrentModel)
    def next_state_spec(self):
        if False:
            i = 10
            return i + 15
        return SpecDict({'out': 'i'}, i=4)

    @property
    @override(RecurrentModel)
    def prev_state_spec(self):
        if False:
            return 10
        return SpecDict({'in': 'o'}, o=1)

    @override(RecurrentModel)
    def _update_inputs_and_prev_state(self, inputs, states):
        if False:
            while True:
                i = 10
        if self.input_check:
            self.input_check(inputs, states)
        return (inputs, states)

    @override(RecurrentModel)
    def _update_outputs_and_next_state(self, outputs, states):
        if False:
            for i in range(10):
                print('nop')
        if self.output_check:
            self.output_check(outputs, states)
        return (outputs, states)

    @override(RecurrentModel)
    def _initial_state(self):
        if False:
            i = 10
            return i + 15
        return TensorDict({'in': np.arange(1)})

    @override(RecurrentModel)
    def _unroll(self, inputs: TensorDict, prev_state: TensorDict) -> UnrollOutputType:
        if False:
            for i in range(10):
                print('nop')
        check(inputs['in'], np.arange(3))
        check(prev_state['in'], np.arange(1))
        return (TensorDict({'out': np.arange(2)}), TensorDict({'out': np.arange(4)}))

class NpModelImpl(Model):
    """Non-recurrent extension of NPRecurrentModelImpl

    For testing:
    (1) rollout and forward_ logic
    (2) spec checking
    """

    def __init__(self, input_check=None, output_check=None):
        if False:
            while True:
                i = 10
        super().__init__()
        self.input_check = input_check
        self.output_check = output_check

    @property
    @override(Model)
    def input_specs(self):
        if False:
            for i in range(10):
                print('nop')
        return SpecDict({'in': 'h'}, h=3)

    @property
    @override(Model)
    def output_specs(self):
        if False:
            i = 10
            return i + 15
        return SpecDict({'out': 'o'}, o=2)

    @override(Model)
    def _update_inputs(self, inputs):
        if False:
            return 10
        if self.input_check:
            return self.input_check(inputs)
        return inputs

    @override(Model)
    def _update_outputs(self, outputs):
        if False:
            for i in range(10):
                print('nop')
        if self.output_check:
            self.output_check(outputs)
        return outputs

    @override(Model)
    def _forward(self, inputs: TensorDict) -> ForwardOutputType:
        if False:
            print('Hello World!')
        check(inputs['in'], np.arange(3))
        return TensorDict({'out': np.arange(2)})

class TestRecurrentModel(unittest.TestCase):

    def test_initial_state(self):
        if False:
            while True:
                i = 10
        'Check that the _initial state is corrected called by initial_state\n        and outputs correct values.'
        output = NpRecurrentModelImpl().initial_state()
        desired = TensorDict({'in': np.arange(1)})
        for k in output.flatten().keys() | desired.flatten().keys():
            check(output[k], desired[k])

    def test_unroll(self):
        if False:
            while True:
                i = 10
        'Test that _unroll is correctly called by unroll and outputs are the\n        correct values'
        (out, out_state) = NpRecurrentModelImpl().unroll(inputs=TensorDict({'in': np.arange(3)}), prev_state=TensorDict({'in': np.arange(1)}))
        (desired, desired_state) = (TensorDict({'out': np.arange(2)}), TensorDict({'out': np.arange(4)}))
        for k in out.flatten().keys() | desired.flatten().keys():
            check(out[k], desired[k])
        for k in out_state.flatten().keys() | desired_state.flatten().keys():
            check(out_state[k], desired_state[k])

    def test_unroll_filter(self):
        if False:
            return 10
        'Test that unroll correctly filters unused data'

        def in_check(inputs, states):
            if False:
                while True:
                    i = 10
            assert 'bork' not in inputs.keys() and 'borkbork' not in states.keys()
            return (inputs, states)
        m = NpRecurrentModelImpl(input_check=in_check)
        (out, state) = m.unroll(inputs=TensorDict({'in': np.arange(3), 'bork': np.zeros(1)}), prev_state=TensorDict({'in': np.arange(1), 'borkbork': np.zeros(1)}))

    def test_hooks(self):
        if False:
            return 10
        'Test that _update_inputs_and_prev_state and _update_outputs_and_prev_state\n        are called during unroll'

        class MyException(Exception):
            pass

        def exc(a, b):
            if False:
                return 10
            raise MyException()
        with self.assertRaises(MyException):
            m = NpRecurrentModelImpl(input_check=exc)
            m.unroll(inputs=TensorDict({'in': np.arange(3)}), prev_state=TensorDict({'in': np.arange(1)}))
        with self.assertRaises(MyException):
            m = NpRecurrentModelImpl(output_check=exc)
            m.unroll(inputs=TensorDict({'in': np.arange(3)}), prev_state=TensorDict({'in': np.arange(1)}))

class TestModel(unittest.TestCase):

    def test_unroll(self):
        if False:
            return 10
        'Test that unroll correctly calls _forward. The outputs\n        should be as expected.'
        m = NpModelImpl()
        (output, nullstate) = m.unroll(inputs=TensorDict({'in': np.arange(3)}), prev_state=TensorDict())
        self.assertEqual(nullstate, TensorDict())
        check(output['out'], np.arange(2))

    def test_hooks(self):
        if False:
            return 10
        'Test that unroll correctly calls the filter functions\n        before _forward'

        class MyException(Exception):
            pass

        def exc(a):
            if False:
                for i in range(10):
                    print('nop')
            raise MyException()
        with self.assertRaises(MyException):
            NpModelImpl(input_check=exc).unroll(inputs=TensorDict({'in': np.arange(3)}), prev_state=TensorDict())
        with self.assertRaises(MyException):
            NpModelImpl(output_check=exc).unroll(inputs=TensorDict({'in': np.arange(3)}), prev_state=TensorDict())
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))