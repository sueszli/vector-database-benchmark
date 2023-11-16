from unittest import TestCase
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from bigdl.nano.pytorch import InferenceOptimizer
from torchvision.models.resnet import resnet18
from unittest.mock import PropertyMock, patch
from bigdl.nano.utils.common import _avx512_checker
import tempfile
from typing import List
from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_2_0
from bigdl.nano.utils.common import compare_version, _avx512_checker, _avx2_checker
import operator
import numpy as np

class CaseWithoutAVX512:

    def test_unsupported_HW_or_OS(self):
        if False:
            print('Hello World!')
        model = resnet18(num_classes=10)
        with pytest.raises(RuntimeError, match='Applying IPEX BF16 optimization needs the cpu support avx512.'):
            bf16_model = InferenceOptimizer.quantize(model, precision='bf16', use_ipex=True)

class DummyMultiInputModel(nn.Module):
    """
    A simple model for test various inputs of channels last format
    """

    def __init__(self):
        if False:
            return 10
        super(DummyMultiInputModel, self).__init__()

    def forward(self, x1, x2, x3: List[float]):
        if False:
            i = 10
            return i + 15
        return (x1, x2, x3)

class DummyModelWith3d(nn.Module):
    """
    A simple model for test various inputs of channels last format
    """

    def __init__(self):
        if False:
            return 10
        super(DummyModelWith3d, self).__init__()
        self.conv3d_1 = nn.Conv3d(3, 33, 3, stride=2)

    def forward(self, x1, x2: int):
        if False:
            i = 10
            return i + 15
        return (self.conv3d_1(x1), x2)

class MultipleInputNet(nn.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dense1 = nn.Linear(10, 1)
        self.dense2 = nn.Linear(10, 1)

    def forward(self, x1, x2):
        if False:
            while True:
                i = 10
        return self.dense1(x1) + self.dense2(x2)

class MultipleInputWithKwargsNet(nn.Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.dense1 = nn.Linear(10, 1)
        self.dense2 = nn.Linear(10, 1)

    def forward(self, x1, x2, x3=10):
        if False:
            i = 10
            return i + 15
        return self.dense1(x1) + self.dense2(x2) + x3

class JumpInputNet(nn.Module):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.dense1 = nn.Linear(10, 1)
        self.dense2 = nn.Linear(10, 1)

    def forward(self, x1, x2=None, x3=None):
        if False:
            while True:
                i = 10
        if x3 is not None:
            return self.dense1(x1) + self.dense2(x3)
        else:
            return self.dense1(x1)

class Pytorch1_11:

    @patch('bigdl.nano.deps.ipex.ipex_inference_bf16_model.PytorchIPEXJITBF16Model._check_cpu_isa', new_callable=PropertyMock)
    def test_unsupported_HW_or_OS(self, mocked_check_cpu_isa):
        if False:
            return 10
        mocked_check_cpu_isa.return_value = False
        model = resnet18(num_classes=10)
        with pytest.raises(RuntimeError, match='Applying IPEX BF16 optimization needs the cpu support avx512.'):
            bf16_model = InferenceOptimizer.quantize(model, precision='bf16', use_ipex=True)

    def test_bf16_inference_with_jit(self):
        if False:
            i = 10
            return i + 15
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16', accelerator='jit', input_sample=x)
        with InferenceOptimizer.get_context(bf16_model):
            for i in range(10):
                y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(load_model):
            for i in range(10):
                y_hat_ = load_model(x)
        assert y_hat_.shape == (10, 10) and y_hat_.dtype == torch.bfloat16
        assert y_hat.equal(y_hat_)

    def test_bf16_ipex_with_avx512_core(self):
        if False:
            while True:
                i = 10
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16', use_ipex=True)
        with InferenceOptimizer.get_context(bf16_model):
            y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16

    def test_bf16_ipex_save_load(self):
        if False:
            print('Hello World!')
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16', use_ipex=True)
        with InferenceOptimizer.get_context(bf16_model):
            y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)
        with InferenceOptimizer.get_context(load_model):
            y_hat_ = load_model(x)
        assert y_hat_.shape == (10, 10) and y_hat_.dtype == torch.bfloat16
        assert y_hat.equal(y_hat_)

    def test_bf16_ipex_jit_save_load(self):
        if False:
            return 10
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16', use_ipex=True, accelerator='jit', input_sample=x)
        with InferenceOptimizer.get_context(bf16_model):
            y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(load_model):
            y_hat_ = load_model(x)
        assert y_hat_.shape == (10, 10) and y_hat_.dtype == torch.bfloat16
        assert y_hat.equal(y_hat_)

    def test_bf16_ipex_jit_additional_attrs(self):
        if False:
            print('Hello World!')
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        model.channels = 3

        def hello():
            if False:
                print('Hello World!')
            print('hello world!')
        model.hello = hello
        new_model = InferenceOptimizer.quantize(model, precision='bf16', accelerator='jit', use_ipex=True, input_sample=x)
        with InferenceOptimizer.get_context(new_model):
            new_model(x)
        assert new_model.channels == 3
        new_model.hello()
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(new_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model=model)
        assert load_model.channels == 3
        load_model.hello()
        with pytest.raises(AttributeError, match="'PytorchIPEXJITBF16Model' object has no attribute 'strange_call'"):
            load_model.strange_call()
        new_model = InferenceOptimizer.quantize(model, precision='bf16', accelerator='jit', input_sample=x)
        with InferenceOptimizer.get_context(new_model):
            new_model(x)
        assert new_model.channels == 3
        new_model.hello()
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(new_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model=model)
        assert load_model.channels == 3
        load_model.hello()
        with pytest.raises(AttributeError, match="'PytorchIPEXJITBF16Model' object has no attribute 'strange_call'"):
            load_model.strange_call()
        new_model = InferenceOptimizer.quantize(model, precision='bf16', use_ipex=True)
        with InferenceOptimizer.get_context(new_model):
            new_model(x)
        assert new_model.channels == 3
        new_model.hello()
        with pytest.raises(AttributeError):
            new_model.width
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(new_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model=model)
        assert load_model.channels == 3
        load_model.hello()
        with pytest.raises(AttributeError, match="'PytorchIPEXJITBF16Model' object has no attribute 'strange_call'"):
            load_model.strange_call()

    def test_bf16_ipex_jit_method(self):
        if False:
            return 10

        class Net(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()

            def forward(self, x):
                if False:
                    return 10
                return torch.arange(len(x))
        model = Net()
        input_sample = torch.rand(1, 3, 1, 1)
        input = torch.rand(5, 3, 1, 1)
        expected_output_len = 5
        accmodel = InferenceOptimizer.quantize(model, precision='bf16', accelerator='jit', use_ipex=True, input_sample=input_sample, jit_method='script')
        with InferenceOptimizer.get_context(accmodel):
            output = accmodel(input)
        assert output.shape[0] == expected_output_len
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(accmodel, tmp_dir_name)
            loaded_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(loaded_model):
            output = loaded_model(input)
        assert output.shape[0] == expected_output_len
        assert loaded_model.jit_method == 'script'
        accmodel = InferenceOptimizer.quantize(model, precision='bf16', accelerator='jit', use_ipex=True, input_sample=input_sample, jit_method='trace')
        with InferenceOptimizer.get_context(accmodel):
            output = accmodel(input)
        assert output.shape[0] != expected_output_len
        accmodel = InferenceOptimizer.quantize(model, precision='bf16', accelerator='jit', input_sample=input_sample)
        with InferenceOptimizer.get_context(accmodel):
            output = accmodel(input)
        assert output.shape[0] != expected_output_len
        with pytest.raises(RuntimeError):
            InferenceOptimizer.quantize(model, precision='bf16', accelerator='jit', input_sample=input_sample, jit_method='scriptttt')

    def test_ipex_jit_inference_weights_prepack(self):
        if False:
            while True:
                i = 10
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        model = InferenceOptimizer.quantize(model, precision='bf16', accelerator='jit', use_ipex=True, input_sample=x, weights_prepack=False)
        with InferenceOptimizer.get_context(model):
            model(x)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(new_model):
            new_model(x)
            assert new_model.weights_prepack is False

    def test_bf16_ipex_channels_last_various_input_sample(self):
        if False:
            print('Hello World!')
        model = DummyMultiInputModel()
        x1 = torch.rand(1, 8, 8)
        x2 = torch.rand(1, 3, 8, 8)
        x3 = [1, 2, 3, 4]
        bf16_ipex_channels_last_model = InferenceOptimizer.quantize(model, precision='bf16', channels_last=True, use_ipex=True)
        with InferenceOptimizer.get_context(bf16_ipex_channels_last_model):
            bf16_ipex_channels_last_model(x1, x2, x3)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_ipex_channels_last_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)
            load_model(x1, x2, x3)

    def test_bf16_jit_channels_last_various_input_sample(self):
        if False:
            i = 10
            return i + 15
        model = DummyMultiInputModel()
        x1 = torch.rand(1, 8, 8)
        x2 = torch.rand(1, 3, 8, 8)
        x3 = [1, 2, 3, 4]
        bf16_jit_channels_last_model = InferenceOptimizer.quantize(model, precision='bf16', channels_last=True, accelerator='jit')
        with InferenceOptimizer.get_context(bf16_jit_channels_last_model):
            bf16_jit_channels_last_model(x1, x2, x3)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_jit_channels_last_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)
            load_model(x1, x2, x3)

    def test_bf16_ipex_jit_channels_last_various_input_sample(self):
        if False:
            for i in range(10):
                print('nop')
        model = DummyMultiInputModel()
        x1 = torch.rand(1, 8, 8)
        x2 = torch.rand(1, 3, 8, 8)
        x3 = [1, 2, 3, 4]
        bf16_ipex_jit_channels_last_model = InferenceOptimizer.quantize(model, precision='bf16', channels_last=True, use_ipex=True, accelerator='jit')
        with InferenceOptimizer.get_context(bf16_ipex_jit_channels_last_model):
            bf16_ipex_jit_channels_last_model(x1, x2, x3)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_ipex_jit_channels_last_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)
            load_model(x1, x2, x3)

    def test_ipex_jit_inference_onednn(self):
        if False:
            for i in range(10):
                print('nop')
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        model = InferenceOptimizer.quantize(model, precision='bf16', accelerator='jit', use_ipex=True, input_sample=x, enable_onednn=True)
        with InferenceOptimizer.get_context(model):
            model(x)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(new_model):
            new_model(x)
            assert new_model.enable_onednn is True
        model = InferenceOptimizer.quantize(model, precision='bf16', accelerator='jit', use_ipex=True, input_sample=x, enable_onednn=False)
        with InferenceOptimizer.get_context(model):
            model(x)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(new_model):
            new_model(x)
            assert new_model.enable_onednn is False

    def test_ipex_jit_channels_last_3d_inference(self):
        if False:
            i = 10
            return i + 15
        model = DummyModelWith3d()
        x1 = torch.rand(32, 3, 3, 224, 224)
        x2 = 3
        ipex_jit_channels_last_model = InferenceOptimizer.quantize(model, accelerator='jit', use_ipex=True, precision='bf16', input_sample=(x1, x2), enable_onednn=True, channels_last=True)
        with InferenceOptimizer.get_context(ipex_jit_channels_last_model):
            ipex_jit_channels_last_model(x1, x2)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(ipex_jit_channels_last_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
            with InferenceOptimizer.get_context(load_model):
                load_model(x1, x2)

    def test_ipex_jit_keyword_argument(self):
        if False:
            return 10
        net = MultipleInputNet()
        x1 = torch.randn(32, 10)
        x2 = torch.randn(32, 10)
        y = torch.randn(32, 1)
        dataloader = DataLoader(TensorDataset(x1, x2, y), batch_size=1)
        model = InferenceOptimizer.quantize(net, precision='bf16', accelerator=None, use_ipex=True, calib_data=dataloader)
        with InferenceOptimizer.get_context(model):
            model(x1, x2)
            model(x1, x2=x2)
            model(x1=x1, x2=x2)
        model = InferenceOptimizer.quantize(net, precision='bf16', accelerator='jit', use_ipex=True, calib_data=dataloader)
        with InferenceOptimizer.get_context(model):
            model(x1=x1, x2=x2)

    @pytest.mark.skipif(compare_version('torch', operator.lt, '2.0'), reason='example_kwarg_inputs is only supported when torch>=2.0')
    def test_bf16_jit_ipex_jump_input(self):
        if False:
            for i in range(10):
                print('nop')
        model = JumpInputNet()
        x1 = torch.randn(1, 10)
        x3 = torch.randn(1, 10)
        target = model(x1, None, x3)
        with pytest.raises(RuntimeError):
            opt_model = InferenceOptimizer.quantize(model, precision='bf16', accelerator='jit', input_sample=(x1, None, x3), jit_method='trace')
        opt_model = InferenceOptimizer.quantize(model, accelerator='jit', precision='bf16', input_sample=None, example_kwarg_inputs={'x1': x1, 'x3': x3})
        output1 = opt_model(x1, x3)
        np.testing.assert_allclose(output1.detach().numpy(), target.detach().numpy(), atol=0.01)
        with tempfile.TemporaryDirectory() as tmp_dir:
            InferenceOptimizer.save(opt_model, tmp_dir)
            loaded_model = InferenceOptimizer.load(tmp_dir)
        output2 = loaded_model(x1, x3)
        np.testing.assert_allclose(output2.detach().numpy(), output1.detach().numpy(), atol=1e-05)
        opt_model = InferenceOptimizer.quantize(model, accelerator='jit', precision='bf16', use_ipex=True, input_sample=None, example_kwarg_inputs={'x1': x1, 'x3': x3}, jit_method='trace')
        output1 = opt_model(x1, x3)
        np.testing.assert_allclose(output1.detach().numpy(), target.detach().numpy(), atol=0.01)
        with tempfile.TemporaryDirectory() as tmp_dir:
            InferenceOptimizer.save(opt_model, tmp_dir)
            loaded_model = InferenceOptimizer.load(tmp_dir)
        output2 = loaded_model(x1, x3)
        np.testing.assert_allclose(output2.detach().numpy(), output1.detach().numpy(), atol=1e-05)
TORCH_VERSION_CLS = Pytorch1_11
if not _avx512_checker():
    print('IPEX BF16 Inference Model Without AVX512')
    TORCH_VERSION_CLS = CaseWithoutAVX512

class TestIPEXBF16(TORCH_VERSION_CLS, TestCase):
    pass
if __name__ == '__main__':
    pytest.main([__file__])