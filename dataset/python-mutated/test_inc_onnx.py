import pytest
import os
from unittest import TestCase
import tempfile
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchmetrics
import numpy as np
from bigdl.nano.pytorch import Trainer
from bigdl.nano.pytorch import InferenceOptimizer
from bigdl.nano.pytorch.vision.models import vision
batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), 'data')

class ResNet18(nn.Module):

    def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
        if False:
            return 10
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.model = nn.Sequential(backbone, head)

    def forward(self, x):
        if False:
            print('Hello World!')
        return self.model(x)

class MultiInputModel(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(28 * 28, 128)
        self.layer_3 = nn.Linear(256, 2)

    def forward(self, x1, x2):
        if False:
            while True:
                i = 10
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        return self.layer_3(x)

def customized_collate_fn(batch):
    if False:
        i = 10
        return i + 15
    (batch, targets) = zip(*batch)
    batch = torch.stack(batch, dim=0)
    targets = torch.stack(targets, dim=0)
    batch = batch.permute(0, 3, 1, 2).contiguous()
    return (batch, targets)

class TestOnnx(TestCase):

    def test_trainer_compile_with_onnx_quantize(self):
        if False:
            for i in range(10):
                print('nop')
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = Trainer(max_epochs=1)
        pl_model = Trainer.compile(model, loss, optimizer)
        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)
        ds = TensorDataset(x, y)
        train_loader = DataLoader(ds, batch_size=2)
        trainer.fit(pl_model, train_loader)
        onnx_model = InferenceOptimizer.quantize(pl_model, accelerator='onnxruntime', method='qlinear', calib_data=train_loader)
        for (x, y) in train_loader:
            forward_res = onnx_model(x).numpy()
        x = torch.rand((1, 3, 256, 256))
        y = torch.ones((1,), dtype=torch.long)
        onnx_model = InferenceOptimizer.quantize(pl_model, accelerator='onnxruntime', method='qlinear', calib_data=(x, y))
        pl_model.eval()
        onnx_model = InferenceOptimizer.quantize(pl_model, accelerator='onnxruntime', method='qlinear', calib_data=train_loader, metric=torchmetrics.F1Score('multiclass', num_classes=10), accuracy_criterion={'relative': 0.99, 'higher_is_better': True})
        for (x, y) in train_loader:
            forward_res = onnx_model(x).numpy()
        trainer.validate(onnx_model, train_loader)
        trainer.test(onnx_model, train_loader)
        trainer.predict(onnx_model, train_loader)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            loaded_onnx_model = InferenceOptimizer.load(tmp_dir_name)
        for (x, y) in train_loader:
            forward_res = loaded_onnx_model(x)

    def test_trainer_compile_with_onnx_quantize_customized_collate_fn(self):
        if False:
            for i in range(10):
                print('nop')
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = Trainer(max_epochs=1)
        pl_model = Trainer.compile(model, loss, optimizer)
        x = torch.rand((10, 256, 256, 3))
        y = torch.ones((10,), dtype=torch.long)
        ds = TensorDataset(x, y)
        train_loader = DataLoader(ds, batch_size=2, collate_fn=customized_collate_fn)
        trainer.fit(pl_model, train_loader)
        onnx_model = InferenceOptimizer.quantize(pl_model, accelerator='onnxruntime', method='qlinear', calib_data=train_loader)
        for (x, y) in train_loader:
            forward_res = onnx_model(x).numpy()
        pl_model.eval()
        onnx_model = InferenceOptimizer.quantize(pl_model, accelerator='onnxruntime', method='qlinear', calib_data=train_loader, metric=torchmetrics.F1Score('multiclass', num_classes=10), accuracy_criterion={'relative': 0.99, 'higher_is_better': True})
        for (x, y) in train_loader:
            forward_res = onnx_model(x).numpy()
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            loaded_onnx_model = InferenceOptimizer.load(tmp_dir_name)
        for (x, y) in train_loader:
            forward_res = loaded_onnx_model(x)

    def test_trainer_compile_with_onnx_quantize_context_manager(self):
        if False:
            i = 10
            return i + 15
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = Trainer(max_epochs=1)
        pl_model = Trainer.compile(model, loss, optimizer)
        x = torch.rand((10, 256, 256, 3))
        y = torch.ones((10,), dtype=torch.long)
        ds = TensorDataset(x, y)
        train_loader = DataLoader(ds, batch_size=2, collate_fn=customized_collate_fn)
        trainer.fit(pl_model, train_loader)
        onnx_model = InferenceOptimizer.quantize(pl_model, accelerator='onnxruntime', method='qlinear', calib_data=train_loader, thread_num=2)
        with InferenceOptimizer.get_context(onnx_model):
            assert torch.get_num_threads() == 2
            x = torch.rand((2, 3, 256, 256))
            output = onnx_model(x)
            assert output.shape == (2, 10)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(model):
            assert torch.get_num_threads() == 2
            output = model(x)

    def test_trainer_compile_with_onnx_quantize_additional_attributes(self):
        if False:
            while True:
                i = 10
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = Trainer(max_epochs=1)
        pl_model = Trainer.compile(model, loss, optimizer)
        x = torch.rand((10, 256, 256, 3))
        y = torch.ones((10,), dtype=torch.long)
        ds = TensorDataset(x, y)
        train_loader = DataLoader(ds, batch_size=2, collate_fn=customized_collate_fn)
        trainer.fit(pl_model, train_loader)
        pl_model.channels = 3

        def hello():
            if False:
                while True:
                    i = 10
            print('hello world!')
        pl_model.hello = hello
        onnx_model = InferenceOptimizer.quantize(pl_model, accelerator='onnxruntime', method='qlinear', calib_data=train_loader, thread_num=2)
        with InferenceOptimizer.get_context(onnx_model):
            assert torch.get_num_threads() == 2
            x = torch.rand((2, 3, 256, 256))
            output = onnx_model(x)
            assert output.shape == (2, 10)
        assert onnx_model.channels == 3
        onnx_model.hello()
        with pytest.raises(AttributeError, match="'PytorchONNXRuntimeModel' object has no attribute 'width'"):
            onnx_model.width
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
        with pytest.raises(AttributeError, match="'PytorchONNXRuntimeModel' object has no attribute 'channels'"):
            load_model.channels
        with pytest.raises(AttributeError):
            load_model.hello()
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model=pl_model)
        assert load_model.channels == 3
        load_model.hello()
        with pytest.raises(AttributeError, match="'PytorchONNXRuntimeModel' object has no attribute 'width'"):
            onnx_model.width

    def test_onnx_quantize_dynamic_axes(self):
        if False:
            return 10

        class CustomModel(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.pool = nn.AvgPool2d(kernel_size=3, stride=3)

            def forward(self, x):
                if False:
                    return 10
                return self.pool(x)
        model = CustomModel()
        x1 = torch.rand(1, 3, 14, 14)
        x2 = torch.rand(4, 3, 14, 14)
        x3 = torch.rand(1, 3, 12, 12)
        accmodel = InferenceOptimizer.quantize(model, accelerator='onnxruntime', method='qlinear', calib_data=torch.rand(1, 3, 14, 14))
        accmodel(x1)
        accmodel(x2)
        try:
            accmodel(x3)
        except Exception as e:
            assert e
        accmodel = InferenceOptimizer.quantize(model, accelerator='onnxruntime', calib_data=torch.rand(1, 3, 14, 14), dynamic_axes={'x': [0, 2, 3]})
        accmodel(x1)
        accmodel(x2)
        accmodel(x3)

    def test_onnx_inc_default_values(self):
        if False:
            return 10
        from bigdl.nano.utils.common import compare_version
        import operator
        if not compare_version('neural_compressor', operator.ge, '1.14.0'):
            return

        class Net(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()

            def forward(self, x, a=True, b=False):
                if False:
                    return 10
                if a:
                    return x + 1
                if b:
                    return x - 1
                return x
        model = Net()
        data = torch.zeros(1, 3, 1, 1) - 1
        result_true = model(data)
        accmodel = InferenceOptimizer.quantize(model, accelerator='onnxruntime', calib_data=torch.rand(2, 3, 1, 1))
        result_m = accmodel(data)
        assert abs(torch.sum(result_m).item()) < 1e-05
        accmodel = InferenceOptimizer.quantize(model, accelerator='onnxruntime', calib_data=torch.rand(2, 3, 1, 1), input_sample=(torch.rand(2, 3, 1, 1), False, True))
        data = torch.zeros(1, 3, 1, 1) + 1
        result_m = accmodel(data)
        assert abs(torch.sum(result_m).item()) < 1e-05

        class Net(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()

            def forward(self, x, a=3):
                if False:
                    while True:
                        i = 10
                return x + a
        model = Net()
        accmodel = InferenceOptimizer.quantize(model, accelerator='onnxruntime', calib_data=((torch.rand(2, 3, 1, 1), 5), torch.zeros(2, 3, 1, 1)))
        data = torch.zeros(1, 3, 1, 1) - 5
        result_m = accmodel(data, 5)
        assert abs(torch.sum(result_m).item()) < 1e-05
        accmodel = InferenceOptimizer.quantize(model, accelerator='onnxruntime', calib_data=(torch.rand(2, 3, 1, 1), 5))
        data = torch.zeros(1, 3, 1, 1) - 5
        result_m = accmodel(data, 5)
        assert abs(torch.sum(result_m).item()) < 1e-05

        class Net(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()

            def forward(self, x, a=None):
                if False:
                    for i in range(10):
                        print('nop')
                if a is None:
                    return x
                else:
                    return x + 1
        model = Net()
        data = torch.zeros(1, 3, 1, 1)
        accmodel = InferenceOptimizer.quantize(model, accelerator='onnxruntime', calib_data=torch.rand(2, 3, 1, 1))
        result_m = accmodel(data)
        assert abs(torch.sum(result_m).item()) < 1e-05

        class Net(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()

            def forward(self, x1, x2, x3, x4, a=None, b=None, c=None):
                if False:
                    while True:
                        i = 10
                output = x1 + x2 + x3 + x4
                if a is not None:
                    output += a
                if b is not None:
                    output += b
                if c is not None:
                    output += c
                return output
        model = Net()
        x1 = torch.zeros(1, 3, 1, 1)
        x2 = torch.zeros(1, 3, 1, 1)
        x3 = torch.zeros(1, 3, 1, 1)
        x4 = torch.zeros(1, 3, 1, 1)
        accmodel = InferenceOptimizer.quantize(model, accelerator='onnxruntime', calib_data=((x1, x2, x3, x4), torch.zeros(1, 3, 1, 1)))
        result_m = accmodel(x1, x2, x3, x4)
        assert abs(torch.sum(result_m).item()) < 1e-05
        accmodel = InferenceOptimizer.quantize(model, accelerator='onnxruntime', calib_data=((x1, x2, x3, x4, 1, 1), torch.zeros(1, 3, 1, 1)))
        result_m = accmodel(x1, x2, x3, x4, 1, 1)
        assert abs(torch.sum(result_m).item()) < 1e-05 + 6
        accmodel = InferenceOptimizer.quantize(model, accelerator='onnxruntime', calib_data=(x1, x2, x3, x4))
        result_m = accmodel(x1, x2, x3, x4)
        assert abs(torch.sum(result_m).item()) < 1e-05
        accmodel = InferenceOptimizer.quantize(model, accelerator='onnxruntime', calib_data=(x1, x2, x3, x4, 1))
        result_m = accmodel(x1, x2, x3, x4, 2)
        assert abs(torch.sum(result_m).item()) < 1e-05 + 6

    def test_onnx_quantize_output_tensors(self):
        if False:
            print('Hello World!')
        model = ResNet18(10, pretrained=True, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        pl_model = Trainer.compile(model, loss, optimizer)
        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)
        ds = TensorDataset(x, y)
        train_loader = DataLoader(ds, batch_size=2)
        onnx_model = InferenceOptimizer.quantize(pl_model, accelerator='onnxruntime', method='qlinear', calib_data=train_loader)
        test_onnx_model = InferenceOptimizer.quantize(pl_model, accelerator='onnxruntime', method='qlinear', calib_data=train_loader, output_tensors=False)
        for (x, y) in train_loader:
            forward_res_tensor = onnx_model(x).numpy()
            forward_res_numpy = test_onnx_model(x)
            assert isinstance(forward_res_numpy, np.ndarray)
            np.testing.assert_almost_equal(forward_res_tensor, forward_res_numpy, decimal=5)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(test_onnx_model, tmp_dir_name)
            test_load_model = InferenceOptimizer.load(tmp_dir_name)
        for (x, y) in train_loader:
            forward_res_tensor = load_model(x).numpy()
            forward_res_numpy = test_load_model(x)
            assert isinstance(forward_res_numpy, np.ndarray)
            np.testing.assert_almost_equal(forward_res_tensor, forward_res_numpy, decimal=5)

    def test_onnx_quantize_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        model = MultiInputModel()
        x1 = torch.randn(100, 28 * 28)
        x2 = torch.randn(100, 28 * 28)
        target = model(x1, x2)

        class CustomDataset(Dataset):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.x1 = x1
                self.x2 = x2

            def __len__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 100

            def __getitem__(self, idx):
                if False:
                    i = 10
                    return i + 15
                return ((self.x1[idx], self.x2[idx]), target)
        dataset = CustomDataset()
        loader = DataLoader(dataset, batch_size=1)
        with torch.no_grad():
            onnx_model = InferenceOptimizer.quantize(model, accelerator='onnxruntime', precision='int8', input_sample=(x1, x2), calib_data=loader)
        with InferenceOptimizer.get_context(onnx_model):
            output1 = onnx_model(x1, x2)
            np.testing.assert_almost_equal(target.numpy(), output1.numpy(), decimal=0)
            output2 = onnx_model(x1, x2=x2)
            np.testing.assert_almost_equal(output1.numpy(), output2.numpy(), decimal=5)
            output3 = onnx_model(x1=x1, x2=x2)
            np.testing.assert_almost_equal(output1.numpy(), output3.numpy(), decimal=5)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(load_model):
            output4 = load_model(x1=x1, x2=x2)
            np.testing.assert_almost_equal(output4.numpy(), output4.numpy(), decimal=5)
if __name__ == '__main__':
    pytest.main([__file__])