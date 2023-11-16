import unittest
import torch
from nni.nas.space.pytorch.graph import PytorchGraphModelSpace
from nni.nas.utils import original_state_dict_hooks

class ConvertMixin:

    def tensor_equal(self, x, y):
        if False:
            while True:
                i = 10
        if not isinstance(x, torch.Tensor):
            return x == y
        return torch.allclose(x.float().nan_to_num(42), y.float().nan_to_num(42), rtol=0.001, atol=0.0001)

    def checkExportImport(self, model, input, check_value=True, strict_load=True):
        if False:
            print('Hello World!')
        model_ir = self._convert_model(model, input)
        converted_model = model_ir.executable_model()
        with original_state_dict_hooks(converted_model):
            converted_model.load_state_dict(model.state_dict(), strict=strict_load)
        with torch.no_grad():
            expected_output = model.eval()(*input)
            converted_output = converted_model.eval()(*input)
        if check_value:
            if isinstance(expected_output, (list, tuple)):
                for (e, c) in zip(expected_output, converted_output):
                    self.assertTrue(self.tensor_equal(e, c), msg=f'{e} != {c}')
            else:
                self.assertTrue(self.tensor_equal(expected_output, converted_output), msg=f'{expected_output} != {converted_output}')
        return converted_model

    def run_test(self, *args, **kwargs):
        if False:
            return 10
        return self.checkExportImport(*args, **kwargs)

    @staticmethod
    def _convert_model(model, input):
        if False:
            for i in range(10):
                print('nop')
        return PytorchGraphModelSpace.from_model(model)

class ConvertWithShapeMixin(ConvertMixin):

    @staticmethod
    def _convert_model(model, input):
        if False:
            while True:
                i = 10
        return PytorchGraphModelSpace.from_model(model, dummy_input=input)