import pytest
import os
from unittest import TestCase
import tempfile
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from bigdl.nano.pytorch import Trainer, InferenceOptimizer
from bigdl.nano.pytorch.vision.models import vision
batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), 'data')

class ResNet18(nn.Module):

    def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
        if False:
            i = 10
            return i + 15
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.model = nn.Sequential(backbone, head)

    def forward(self, x):
        if False:
            return 10
        return self.model(x)

class MultiInputModel(nn.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(28 * 28, 128)
        self.layer_3 = nn.Linear(256, 2)

    def forward(self, x1, x2):
        if False:
            print('Hello World!')
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        return self.layer_3(x)

class TupleInputModel(nn.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(28 * 28, 128)
        self.layer_3 = nn.Linear(256, 1)

    def forward(self, x1, x2, x3):
        if False:
            print('Hello World!')
        x1 = self.layer_1(x1)
        x2_ = None
        for x in x2:
            if x2_ is None:
                x2_ = self.layer_2(x)
            else:
                x2_ += self.layer_2(x)
        x = torch.cat([x1, x2_], axis=1)
        return self.layer_3(x) + x3

class DictOutputModel(nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)

    def forward(self, x1, x2):
        if False:
            for i in range(10):
                print('nop')
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        x3 = self.layer_3(x)
        output = {'x1': x1, 'x2': x2, 'x3': x3}
        return output

class DictOutputModel2(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)

    def forward(self, x1, x2):
        if False:
            return 10
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        x3 = self.layer_3(x)
        output = {'x3': x3, 'x1': x1, 'x2': x2}
        return output

class DictTensorOutputModel1(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)

    def forward(self, x1, x2):
        if False:
            return 10
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        x3 = self.layer_3(x)
        output = {'x1': x1, 'x2': x2}
        return (output, x3)

class MultiDictOutputModel(nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)

    def forward(self, x1, x2):
        if False:
            for i in range(10):
                print('nop')
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        x3 = self.layer_3(x)
        output1 = {'x1': x1, 'x2': x2, 'x3': x3}
        output2 = {'x3': x3, 'x1': x1, 'x2': x2}
        return (output1, output2)

class MultiDictTensorOutputModel(nn.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)

    def forward(self, x1, x2):
        if False:
            i = 10
            return i + 15
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        x3 = self.layer_3(x)
        output1 = {'x1': x1, 'x2': x2, 'x3': x3}
        output2 = {'x3': x3, 'x1': x1, 'x2': x2}
        return (x3, output1, output2)

class ListOutputModel(nn.Module):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)

    def forward(self, x1, x2):
        if False:
            print('Hello World!')
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        output = self.layer_3(x)
        return [x1, x2, output]

class TupleTensorOutputModel(nn.Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)

    def forward(self, x1, x2):
        if False:
            print('Hello World!')
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        output = self.layer_3(x)
        return (output, (x1, x2, x))

class MultiTupleOutputModel(nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)

    def forward(self, x1, x2):
        if False:
            for i in range(10):
                print('nop')
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        output = self.layer_3(x)
        return ([x1, x2], (output, x))

class MultiTupleTensorOutputModel(nn.Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)

    def forward(self, x1, x2):
        if False:
            print('Hello World!')
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        output = self.layer_3(x)
        return (output, [x1, x2], (output, x))

class TupleDictOutputModel(nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)

    def forward(self, x1, x2):
        if False:
            i = 10
            return i + 15
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        output = self.layer_3(x)
        return ([x1, x2], {'x': x, 'output': output})

class TupleDictOutputModel2(nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)

    def forward(self, x1, x2):
        if False:
            for i in range(10):
                print('nop')
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        output = self.layer_3(x)
        return [x1, x2, {'x': x, 'output': output}]

class TupleDictOutputModel3(nn.Module):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 12)
        self.layer_2 = nn.Linear(28 * 28, 12)
        self.layer_3 = nn.Linear(24, 1)

    def forward(self, x1, x2):
        if False:
            while True:
                i = 10
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)
        output = self.layer_3(x)
        return {'intermediate': [x1, x2], 'x': x, 'output': output}

class TestOnnx(TestCase):

    def test_trace_onnx(self):
        if False:
            return 10
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
        onnx_model = InferenceOptimizer.trace(pl_model, accelerator='onnxruntime', input_sample=train_loader)
        for (x, y) in train_loader:
            model.eval()
            with torch.no_grad():
                forward_res_pytorch = pl_model(x).numpy()
            forward_res_onnx = onnx_model(x).numpy()
            np.testing.assert_almost_equal(forward_res_onnx, forward_res_pytorch, decimal=5)
        trainer.validate(onnx_model, train_loader)
        trainer.test(onnx_model, train_loader)

    def test_trace_multiple_input_onnx(self):
        if False:
            i = 10
            return i + 15
        model = MultiInputModel()
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = Trainer(max_epochs=1)
        pl_model = Trainer.compile(model, loss, optimizer)
        x1 = torch.randn(100, 28 * 28)
        x2 = torch.randn(100, 28 * 28)
        y = torch.zeros(100).long()
        y[0:50] = 1
        train_loader = DataLoader(TensorDataset(x1, x2, y), batch_size=32, shuffle=True)
        trainer.fit(pl_model, train_loader)
        onnx_model = InferenceOptimizer.trace(pl_model, accelerator='onnxruntime', input_sample=train_loader)
        for (x1, x2, y) in train_loader:
            model.eval()
            with torch.no_grad():
                forward_res_pytorch = pl_model(x1, x2).numpy()
            forward_res_onnx = onnx_model(x1, x2).numpy()
            np.testing.assert_almost_equal(forward_res_onnx, forward_res_pytorch, decimal=5)
        trainer.validate(onnx_model, train_loader)
        trainer.test(onnx_model, train_loader)
        trainer.predict(onnx_model, train_loader)

    def test_onnx_save_load(self):
        if False:
            print('Hello World!')
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
        onnx_model = InferenceOptimizer.trace(pl_model, accelerator='onnxruntime', input_sample=train_loader, thread_num=1)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            onnx_model_new = InferenceOptimizer.load(tmp_dir_name)
        assert onnx_model_new.session_options.intra_op_num_threads == 1
        assert onnx_model_new.session_options.inter_op_num_threads == 1
        for (x, y) in train_loader:
            forward_res_onnx = onnx_model(x).numpy()
            forward_res_onnx_new = onnx_model_new(x).numpy()
            np.testing.assert_almost_equal(forward_res_onnx, forward_res_onnx_new, decimal=5)

    def test_onnx_context_manager(self):
        if False:
            print('Hello World!')
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
        onnx_model = InferenceOptimizer.trace(pl_model, accelerator='onnxruntime', input_sample=train_loader, thread_num=1)
        with InferenceOptimizer.get_context(onnx_model):
            assert torch.get_num_threads() == 1
            output = onnx_model(x)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(model):
            assert torch.get_num_threads() == 1
            output = onnx_model(x)

    def test_onnx_additional_attributes(self):
        if False:
            while True:
                i = 10
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
        pl_model.channels = 3

        def hello():
            if False:
                for i in range(10):
                    print('nop')
            print('hello world!')
        pl_model.hello = hello
        onnx_model = InferenceOptimizer.trace(pl_model, accelerator='onnxruntime', input_sample=train_loader, thread_num=1)
        with InferenceOptimizer.get_context(onnx_model):
            assert torch.get_num_threads() == 1
            output = onnx_model(x)
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
            load_model.width

    def test_onnx_default_values(self):
        if False:
            i = 10
            return i + 15

        class Net(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()

            def forward(self, x, a=True, b=False):
                if False:
                    for i in range(10):
                        print('nop')
                if a:
                    return x + 1
                if b:
                    return x - 1
                return x
        model = Net()
        data = torch.rand(1, 3, 1, 1)
        result_true = model(data)
        accmodel = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=(torch.rand(2, 3, 1, 1),))
        result_m = accmodel(data)
        assert torch.equal(result_true, result_m)
        accmodel = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=torch.rand(2, 3, 1, 1))
        result_m = accmodel(data)
        assert torch.equal(result_true, result_m)
        data = torch.rand(1, 3, 1, 1)
        result_true = model(data, False, True)
        accmodel = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=(torch.rand(2, 3, 1, 1), False, True))
        result_m = accmodel(data)
        assert torch.equal(result_true, result_m)

        class Net(nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()

            def forward(self, x: torch.Tensor, y: int=3):
                if False:
                    return 10
                return x + y
        model = Net()
        x = torch.rand(1, 3, 1, 1)
        y = 3
        result_true = model(x, y)
        accmodel = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=torch.rand(2, 3, 1, 1))
        result_m = accmodel(x, y)
        assert torch.equal(result_true, result_m)
        accmodel = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=(torch.rand(2, 3, 1, 1), 3))
        result_m = accmodel(x, y)
        assert torch.equal(result_true, result_m)

    def test_onnx_dynamic_axes(self):
        if False:
            i = 10
            return i + 15

        class CustomModel(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.pool = nn.AvgPool2d(kernel_size=3, stride=3)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.pool(x)
        model = CustomModel()
        x1 = torch.rand(1, 3, 14, 14)
        x2 = torch.rand(4, 3, 14, 14)
        x3 = torch.rand(1, 3, 12, 12)
        accmodel = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=torch.rand(1, 3, 14, 14))
        accmodel(x1)
        accmodel(x2)
        try:
            accmodel(x3)
        except Exception as e:
            assert e
        accmodel = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=torch.rand(1, 3, 14, 14), dynamic_axes={'x': [0, 2, 3]})
        accmodel(x1)
        accmodel(x2)
        accmodel(x3)

    def test_onnx_trace_output_tensors(self):
        if False:
            i = 10
            return i + 15
        model = ResNet18(10, pretrained=True, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        pl_model = Trainer.compile(model, loss, optimizer)
        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)
        ds = TensorDataset(x, y)
        train_loader = DataLoader(ds, batch_size=2)
        onnx_model = InferenceOptimizer.trace(pl_model, accelerator='onnxruntime', input_sample=train_loader)
        test_onnx_model = InferenceOptimizer.trace(pl_model, accelerator='onnxruntime', input_sample=train_loader, output_tensors=False)
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
        forward_res_tensor = load_model(x).numpy()
        forward_res_numpy = test_load_model(x)
        assert isinstance(forward_res_numpy, np.ndarray)
        np.testing.assert_almost_equal(forward_res_tensor, forward_res_numpy, decimal=5)

    def test_onnx_tuple_input(self):
        if False:
            for i in range(10):
                print('nop')
        model = TupleInputModel()
        x1 = torch.randn(100, 28 * 28)
        x2 = [torch.randn(100, 28 * 28), torch.randn(100, 28 * 28), torch.randn(100, 28 * 28)]
        x3 = 5
        target = model(x1, x2, x3)
        onnx_model = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=(x1, x2, x3))
        with InferenceOptimizer.get_context(onnx_model):
            output1 = onnx_model(x1, x2, x3)
            np.testing.assert_almost_equal(target.numpy(), output1.numpy(), decimal=5)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(load_model):
            output2 = load_model(x1, x2, x3)
            np.testing.assert_almost_equal(output1.numpy(), output2.numpy(), decimal=5)

    def test_onnx_kwargs(self):
        if False:
            print('Hello World!')
        model = MultiInputModel()
        x1 = torch.randn(100, 28 * 28)
        x2 = torch.randn(100, 28 * 28)
        target = model(x1, x2)
        onnx_model = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(onnx_model):
            output1 = onnx_model(x1, x2)
            np.testing.assert_almost_equal(target.numpy(), output1.numpy(), decimal=5)
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

    def test_onnxruntime_dict_output(self):
        if False:
            i = 10
            return i + 15
        x1 = torch.randn(10, 28 * 28)
        x2 = torch.randn(10, 28 * 28)
        for Model in [DictOutputModel, DictOutputModel2]:
            model = Model()
            output = model(x1, x2)
            assert isinstance(output, dict)
            onnx_model = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=(x1, x2))
            with InferenceOptimizer.get_context(onnx_model):
                output1 = onnx_model(x1, x2)
            assert output.keys() == output1.keys()
            for k in output.keys():
                np.testing.assert_almost_equal(output[k].detach().numpy(), output1[k].detach().numpy(), decimal=5)
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                InferenceOptimizer.save(onnx_model, tmp_dir_name)
                load_model = InferenceOptimizer.load(tmp_dir_name)
            with InferenceOptimizer.get_context(load_model):
                output2 = load_model(x1, x2)
            assert output.keys() == output2.keys()
            for k in output.keys():
                np.testing.assert_almost_equal(output[k].detach().numpy(), output2[k].detach().numpy(), decimal=5)
        model = DictTensorOutputModel1()
        (dic, out) = model(x1, x2)
        assert isinstance(dic, dict)
        onnx_model = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(onnx_model):
            (dic1, out1) = onnx_model(x1, x2)
        assert dic1.keys() == dic.keys()
        np.testing.assert_almost_equal(out.detach().numpy(), out1.detach().numpy(), decimal=5)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(load_model):
            (dic2, out2) = load_model(x1, x2)
        assert dic2.keys() == dic1.keys()
        np.testing.assert_almost_equal(out2.detach().numpy(), out1.detach().numpy(), decimal=5)
        model = MultiDictOutputModel()
        (dic1, dic2) = model(x1, x2)
        assert isinstance(dic1, dict)
        assert isinstance(dic2, dict)
        ov_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(ov_model):
            (output_dic1, output_dic2) = ov_model(x1, x2)
        assert dic1.keys() == output_dic1.keys()
        assert dic2.keys() == output_dic2.keys()
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(ov_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(load_model):
            (output2_dic1, output2_dic2) = load_model(x1, x2)
        assert dic1.keys() == output2_dic1.keys()
        assert dic2.keys() == output2_dic2.keys()
        model = MultiDictTensorOutputModel()
        (output, dic1, dic2) = model(x1, x2)
        assert isinstance(dic1, dict)
        assert isinstance(dic2, dict)
        ov_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(ov_model):
            (output1, output1_dic1, output1_dic2) = ov_model(x1, x2)
        assert dic1.keys() == output_dic1.keys()
        assert dic2.keys() == output_dic2.keys()
        np.testing.assert_almost_equal(output.detach().numpy(), output1.detach().numpy(), decimal=5)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(ov_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(load_model):
            (output2, output2_dic1, output2_dic2) = load_model(x1, x2)
        assert dic1.keys() == output2_dic1.keys()
        assert dic2.keys() == output2_dic2.keys()
        np.testing.assert_almost_equal(output.detach().numpy(), output1.detach().numpy(), decimal=5)

    def test_onnxruntime_list_output(self):
        if False:
            for i in range(10):
                print('nop')
        x1 = torch.randn(10, 28 * 28)
        x2 = torch.randn(10, 28 * 28)
        model = ListOutputModel()
        output = model(x1, x2)
        assert isinstance(output, list)
        onnx_model = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(onnx_model):
            output1 = onnx_model(x1, x2)
        assert len(output) == len(output1)
        for k in range(len(output)):
            np.testing.assert_almost_equal(output[k].detach().numpy(), output1[k].detach().numpy(), decimal=5)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(load_model):
            output2 = load_model(x1, x2)
        assert len(output) == len(output2)
        for k in range(len(output)):
            np.testing.assert_almost_equal(output[k].detach().numpy(), output2[k].detach().numpy(), decimal=5)
        model = TupleTensorOutputModel()
        (out, list_out) = model(x1, x2)
        assert isinstance(list_out, tuple)
        onnx_model = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(onnx_model):
            (out1, list_out1) = onnx_model(x1, x2)
        assert len(list_out) == len(list_out1)
        np.testing.assert_almost_equal(out.detach().numpy(), out1.detach().numpy(), decimal=5)
        for k in range(len(list_out)):
            np.testing.assert_almost_equal(list_out[k].detach().numpy(), list_out1[k].detach().numpy(), decimal=5)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(load_model):
            (out2, list_out2) = load_model(x1, x2)
        assert len(list_out) == len(list_out2)
        np.testing.assert_almost_equal(out2.detach().numpy(), out1.detach().numpy(), decimal=5)
        for k in range(len(list_out1)):
            np.testing.assert_almost_equal(list_out1[k].detach().numpy(), list_out2[k].detach().numpy(), decimal=5)
        model = MultiTupleOutputModel()
        (list1, list2) = model(x1, x2)
        assert isinstance(list1, list)
        assert isinstance(list2, tuple)
        onnx_model = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(onnx_model):
            (output_list1, output_list2) = onnx_model(x1, x2)
        assert len(output_list1) == len(list1)
        assert len(output_list2) == len(list2)
        for k in range(len(list1)):
            np.testing.assert_almost_equal(list1[k].detach().numpy(), output_list1[k].detach().numpy(), decimal=5)
        for k in range(len(list2)):
            np.testing.assert_almost_equal(list2[k].detach().numpy(), output_list2[k].detach().numpy(), decimal=5)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(load_model):
            (output2_list1, output2_list2) = load_model(x1, x2)
        assert len(output2_list1) == len(list1)
        assert len(output2_list2) == len(list2)
        for k in range(len(list1)):
            np.testing.assert_almost_equal(list1[k].detach().numpy(), output2_list1[k].detach().numpy(), decimal=5)
        for k in range(len(list2)):
            np.testing.assert_almost_equal(list2[k].detach().numpy(), output2_list2[k].detach().numpy(), decimal=5)
        model = MultiTupleTensorOutputModel()
        (output, list1, list2) = model(x1, x2)
        assert isinstance(list1, list)
        assert isinstance(list2, tuple)
        onnx_model = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(onnx_model):
            (output1, output1_list1, output1_list2) = onnx_model(x1, x2)
        assert len(output1_list1) == len(list1)
        assert len(output1_list2) == len(list2)
        for k in range(len(list1)):
            np.testing.assert_almost_equal(list1[k].detach().numpy(), output1_list1[k].detach().numpy(), decimal=5)
        for k in range(len(list2)):
            np.testing.assert_almost_equal(list2[k].detach().numpy(), output1_list2[k].detach().numpy(), decimal=5)
        np.testing.assert_almost_equal(output.detach().numpy(), output1.detach().numpy(), decimal=5)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(load_model):
            (output2, output2_list1, output2_list2) = load_model(x1, x2)
        assert len(output2_list1) == len(list1)
        assert len(output2_list2) == len(list2)
        for k in range(len(list1)):
            np.testing.assert_almost_equal(list1[k].detach().numpy(), output2_list1[k].detach().numpy(), decimal=5)
        for k in range(len(list2)):
            np.testing.assert_almost_equal(list2[k].detach().numpy(), output2_list2[k].detach().numpy(), decimal=5)
        np.testing.assert_almost_equal(output2.detach().numpy(), output1.detach().numpy(), decimal=5)

    def test_onnxruntime_list_dict_output(self):
        if False:
            return 10
        x1 = torch.randn(10, 28 * 28)
        x2 = torch.randn(10, 28 * 28)
        model = TupleDictOutputModel()
        (output, dic) = model(x1, x2)
        assert isinstance(output, list)
        assert isinstance(dic, dict)
        onnx_model = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(onnx_model):
            (output1, dic1) = onnx_model(x1, x2)
        assert isinstance(output1, list)
        assert isinstance(dic1, dict)
        assert dic.keys() == dic1.keys()
        for k in range(len(output)):
            np.testing.assert_almost_equal(output[k].detach().numpy(), output1[k].detach().numpy(), decimal=5)
        for k in dic.keys():
            np.testing.assert_almost_equal(dic[k].detach().numpy(), dic1[k].detach().numpy(), decimal=5)
        model = TupleDictOutputModel2()
        output = model(x1, x2)
        assert isinstance(output, list)
        assert isinstance(output[2], dict)
        onnx_model = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(onnx_model):
            output1 = onnx_model(x1, x2)
        assert isinstance(output1, list)
        assert isinstance(output1[2], dict)
        assert len(output1) == len(output)
        for i in range(2):
            np.testing.assert_almost_equal(output[i].detach().numpy(), output1[i].detach().numpy(), decimal=5)
        assert output[2].keys() == output1[2].keys()
        for k in output[2].keys():
            np.testing.assert_almost_equal(output[2][k].detach().numpy(), output1[2][k].detach().numpy(), decimal=5)
        model = TupleDictOutputModel3()
        output = model(x1, x2)
        assert isinstance(output, dict)
        assert isinstance(output['intermediate'], list)
        onnx_model = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=(x1, x2))
        with InferenceOptimizer.get_context(onnx_model):
            output1 = onnx_model(x1, x2)
        assert isinstance(output1, dict)
        assert isinstance(output1['intermediate'], list)
        assert output1.keys() == output.keys()
        for k in output.keys():
            if k != 'intermediate':
                np.testing.assert_almost_equal(output[k].detach().numpy(), output1[k].detach().numpy(), decimal=5)
            else:
                for i in range(2):
                    np.testing.assert_almost_equal(output['intermediate'][i].detach().numpy(), output1['intermediate'][i].detach().numpy(), decimal=5)
if __name__ == '__main__':
    pytest.main([__file__])