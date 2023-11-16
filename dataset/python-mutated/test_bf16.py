from unittest import TestCase
import pytest
import torch
from bigdl.nano.pytorch import InferenceOptimizer
from torchvision.models.resnet import resnet18
from unittest.mock import MagicMock, PropertyMock, patch
from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_1_12
import tempfile

class Pytorch1_11:
    """
    Pytorch version >= 1.10 and <1.12, bfloat16 precision is supported.
    But there is no optimization on bfloat16.
    """

    def test_bf16_pytorch_1_11(self):
        if False:
            while True:
                i = 10
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        with pytest.raises(RuntimeError, match='Require torch>=1.12 to obtain bfloat16 acceleration.'):
            bf16_model = InferenceOptimizer.quantize(model, precision='bf16')

class Pytorch1_12:
    """
    Pytorch version >= 1.10 and <1.12, bfloat16 precision is supported.
    But there is no optimization on bfloat16.
    """

    @patch('bigdl.nano.pytorch.amp.bfloat16.BF16Model._has_bf16_isa', new_callable=PropertyMock)
    def test_unsupported_HW_or_OS(self, mocked_has_bf16_isa):
        if False:
            for i in range(10):
                print('nop')
        mocked_has_bf16_isa.return_value = False
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16')
        with InferenceOptimizer.get_context(bf16_model):
            y_hat = bf16_model(x)

    @patch('bigdl.nano.pytorch.amp.bfloat16.BF16Model._max_bf16_isa', return_value=None)
    @patch('bigdl.nano.pytorch.amp.bfloat16.BF16Model._has_bf16_isa', new_callable=PropertyMock)
    @pytest.mark.skip(reason='Disable dnnl log check if torch==1.12')
    def test_not_executed_on_bf16(self, mocked_has_bf16_isa, mocked_max_bf16_isa):
        if False:
            i = 10
            return i + 15
        '\n        Pytorch version is correct and bf16 instructions are detected.\n        But somehow in the run, there is no bf16 instructions used.\n        '
        mocked_has_bf16_isa.return_value = True
        mocked_max_bf16_isa.return_value = None
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16')
        with InferenceOptimizer.get_context(bf16_model):
            bf16_model(x)

    @patch.dict('os.environ', {'ALLOW_NON_BF16_ISA': '1'})
    def test_bf16_common(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Debug mode. Allow run bf16 forward without bf16 instruction support.\n        '
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16')
        with InferenceOptimizer.get_context(bf16_model):
            y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16

    def test_bf16_with_amx_bf16(self):
        if False:
            i = 10
            return i + 15
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16')
        with patch.object(type(bf16_model), '_has_bf16_isa', PropertyMock(return_value=True)):
            bf16_model._max_bf16_isa = MagicMock(return_value='AMX')
            with InferenceOptimizer.get_context(bf16_model):
                y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16

    def test_bf16_with_avx512_bf16(self):
        if False:
            for i in range(10):
                print('nop')
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16')
        with patch.object(type(bf16_model), '_has_bf16_isa', PropertyMock(return_value=True)):
            bf16_model._max_bf16_isa = MagicMock(return_value='AVX512')
            with InferenceOptimizer.get_context(bf16_model):
                y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16

    def test_bf16_save_and_load(self):
        if False:
            print('Hello World!')
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16')
        with InferenceOptimizer.get_context(bf16_model):
            y_hat1 = bf16_model(x)
        assert y_hat1.shape == (10, 10) and y_hat1.dtype == torch.bfloat16
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)
        with InferenceOptimizer.get_context(load_model):
            y_hat2 = load_model(x)
        assert y_hat2.shape == (10, 10) and y_hat2.dtype == torch.bfloat16
        assert y_hat1.equal(y_hat2)
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16', channels_last=True)
        with InferenceOptimizer.get_context(bf16_model):
            y_hat1 = bf16_model(x)
        assert y_hat1.shape == (10, 10) and y_hat1.dtype == torch.bfloat16
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)
        with InferenceOptimizer.get_context(load_model):
            y_hat2 = load_model(x)
        assert y_hat2.shape == (10, 10) and y_hat2.dtype == torch.bfloat16
        assert y_hat1.equal(y_hat2)

    def test_bf16_additional_attrs(self):
        if False:
            return 10
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        model.channels = 3

        def hello():
            if False:
                i = 10
                return i + 15
            print('hello world!')
        model.hello = hello
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16')
        with InferenceOptimizer.get_context(bf16_model):
            y_hat1 = bf16_model(x)
        assert y_hat1.shape == (10, 10) and y_hat1.dtype == torch.bfloat16
        assert bf16_model.channels == 3
        bf16_model.hello()
        with pytest.raises(AttributeError, match="'ResNet' object has no attribute 'strange_call'"):
            bf16_model.strange_call()
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16', channels_last=True)
        with InferenceOptimizer.get_context(bf16_model):
            y_hat1 = bf16_model(x)
        assert y_hat1.shape == (10, 10) and y_hat1.dtype == torch.bfloat16
        assert bf16_model.channels == 3
        bf16_model.hello()
        with pytest.raises(AttributeError):
            bf16_model.width
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model=model)
        assert load_model.channels == 3
        load_model.hello()
        with pytest.raises(AttributeError, match="'ResNet' object has no attribute 'strange_call'"):
            load_model.strange_call()

    def test_bf16_channels_last_various_input_sample(self):
        if False:
            print('Hello World!')

        class DummyModel(torch.nn.Module):
            """
            A simple model for test various inputs of channels last format
            """

            def __init__(self):
                if False:
                    return 10
                super(DummyModel, self).__init__()

            def forward(self, x1, x2, x3):
                if False:
                    for i in range(10):
                        print('nop')
                return (x1, x2, x3)
        model = DummyModel()
        x1 = torch.rand(10, 256, 256)
        x2 = torch.rand(10, 3, 256, 256)
        x3 = x2.tolist()
        bf16_channels_last_model = InferenceOptimizer.quantize(model, precision='bf16', channels_last=True)
        with InferenceOptimizer.get_context(bf16_channels_last_model):
            bf16_channels_last_model(x1, x2, x3)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_channels_last_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)
            load_model(x1, x2, x3)

    def test_bf16_channels_last_3d_various_input_sample(self):
        if False:
            i = 10
            return i + 15
        import torch.nn as nn

        class DummyModelWith3d(nn.Module):
            """
            A simple model for test various inputs of channels last format
            """

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super(DummyModelWith3d, self).__init__()
                self.conv3d_1 = nn.Conv3d(3, 33, 3, stride=2)

            def forward(self, x1, x2: int):
                if False:
                    return 10
                return (self.conv3d_1(x1), x2)
        model = DummyModelWith3d()
        x1 = torch.rand(32, 3, 3, 224, 224)
        x2 = 3
        bf16_channels_3d_last_model = InferenceOptimizer.quantize(model, precision='bf16', channels_last=True)
        with InferenceOptimizer.get_context(bf16_channels_3d_last_model):
            bf16_channels_3d_last_model(x1, x2)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_channels_3d_last_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)
            load_model(x1, x2)
TORCH_VERSION_CLS = Pytorch1_12
if TORCH_VERSION_LESS_1_12:
    print('1.11')
    TORCH_VERSION_CLS = Pytorch1_11

class TestBF16(TORCH_VERSION_CLS, TestCase):
    pass
if __name__ == '__main__':
    pytest.main([__file__])