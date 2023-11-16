from typing import Dict, Optional
import os
import tempfile
import tarfile
import pytest
import torch
from allennlp.nn import InitializerApplicator, Initializer
from allennlp.nn.initializers import PretrainedModelInitializer
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.params import Params

class _Net1(torch.nn.Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 10)
        self.linear_2 = torch.nn.Linear(10, 5)
        self.scalar = torch.nn.Parameter(torch.rand(()))

    def forward(self, inputs):
        if False:
            print('Hello World!')
        pass

class _Net2(torch.nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 10)
        self.linear_3 = torch.nn.Linear(10, 5)
        self.scalar = torch.nn.Parameter(torch.rand(()))

    def forward(self, inputs):
        if False:
            print('Hello World!')
        pass

class TestPretrainedModelInitializer(AllenNlpTestCase):

    def setup_method(self):
        if False:
            print('Hello World!')
        super().setup_method()
        self.net1 = _Net1()
        self.net2 = _Net2()
        self.temp_file = self.TEST_DIR / 'weights.th'
        torch.save(self.net2.state_dict(), self.temp_file)

    def _are_equal(self, linear1: torch.nn.Linear, linear2: torch.nn.Linear) -> bool:
        if False:
            i = 10
            return i + 15
        return torch.equal(linear1.weight, linear2.weight) and torch.equal(linear1.bias, linear2.bias)

    def _get_applicator(self, regex: str, weights_file_path: str, parameter_name_overrides: Optional[Dict[str, str]]=None) -> InitializerApplicator:
        if False:
            print('Hello World!')
        initializer = PretrainedModelInitializer(weights_file_path, parameter_name_overrides)
        return InitializerApplicator([(regex, initializer)])

    def test_random_initialization(self):
        if False:
            while True:
                i = 10
        assert not self._are_equal(self.net1.linear_1, self.net2.linear_1)
        assert not self._are_equal(self.net1.linear_2, self.net2.linear_3)

    def test_from_params(self):
        if False:
            for i in range(10):
                print('nop')
        params = Params({'type': 'pretrained', 'weights_file_path': self.temp_file})
        initializer = Initializer.from_params(params)
        assert initializer.weights
        assert initializer.parameter_name_overrides == {}
        name_overrides = {'a': 'b', 'c': 'd'}
        params = Params({'type': 'pretrained', 'weights_file_path': self.temp_file, 'parameter_name_overrides': name_overrides})
        initializer = Initializer.from_params(params)
        assert initializer.weights
        assert initializer.parameter_name_overrides == name_overrides

    def test_from_params_tar_gz(self):
        if False:
            while True:
                i = 10
        with tempfile.NamedTemporaryFile(suffix='.tar.gz') as f:
            with tarfile.open(fileobj=f, mode='w:gz') as archive:
                archive.add(self.temp_file, arcname=os.path.basename(self.temp_file))
            f.flush()
            params = Params({'type': 'pretrained', 'weights_file_path': f.name})
            initializer = Initializer.from_params(params)
        assert initializer.weights
        assert initializer.parameter_name_overrides == {}
        for (name, parameter) in self.net2.state_dict().items():
            assert torch.equal(parameter, initializer.weights[name])

    def test_default_parameter_names(self):
        if False:
            i = 10
            return i + 15
        applicator = self._get_applicator('linear_1.weight|linear_1.bias', self.temp_file)
        applicator(self.net1)
        assert self._are_equal(self.net1.linear_1, self.net2.linear_1)
        assert not self._are_equal(self.net1.linear_2, self.net2.linear_3)

    def test_parameter_name_overrides(self):
        if False:
            while True:
                i = 10
        name_overrides = {'linear_2.weight': 'linear_3.weight', 'linear_2.bias': 'linear_3.bias'}
        applicator = self._get_applicator('linear_*', self.temp_file, name_overrides)
        applicator(self.net1)
        assert self._are_equal(self.net1.linear_1, self.net2.linear_1)
        assert self._are_equal(self.net1.linear_2, self.net2.linear_3)

    def test_size_mismatch(self):
        if False:
            while True:
                i = 10
        name_overrides = {'linear_1.weight': 'linear_3.weight'}
        applicator = self._get_applicator('linear_1.*', self.temp_file, name_overrides)
        with pytest.raises(ConfigurationError):
            applicator(self.net1)

    def test_zero_dim_tensor(self):
        if False:
            return 10
        applicator = self._get_applicator('scalar', self.temp_file)
        applicator(self.net1)
        assert torch.equal(self.net1.scalar, self.net2.scalar)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='No CUDA device registered.')
    def test_load_to_gpu_from_gpu(self):
        if False:
            return 10
        self.net1.cuda(device=0)
        self.net2.cuda(device=0)
        assert self.net1.linear_1.weight.is_cuda is True
        assert self.net1.linear_1.bias.is_cuda is True
        assert self.net2.linear_1.weight.is_cuda is True
        assert self.net2.linear_1.bias.is_cuda is True
        temp_file = self.TEST_DIR / 'gpu_weights.th'
        torch.save(self.net2.state_dict(), temp_file)
        applicator = self._get_applicator('linear_1.*', temp_file)
        applicator(self.net1)
        assert self.net1.linear_1.weight.is_cuda is True
        assert self.net1.linear_1.bias.is_cuda is True
        assert self.net2.linear_1.weight.is_cuda is True
        assert self.net2.linear_1.bias.is_cuda is True
        assert self._are_equal(self.net1.linear_1, self.net2.linear_1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='No CUDA device registered.')
    def test_load_to_cpu_from_gpu(self):
        if False:
            print('Hello World!')
        self.net2.cuda(device=0)
        assert self.net2.linear_1.weight.is_cuda is True
        assert self.net2.linear_1.bias.is_cuda is True
        temp_file = self.TEST_DIR / 'gpu_weights.th'
        torch.save(self.net2.state_dict(), temp_file)
        applicator = self._get_applicator('linear_1.*', temp_file)
        applicator(self.net1)
        assert self.net1.linear_1.weight.is_cuda is False
        assert self.net1.linear_1.bias.is_cuda is False
        assert self._are_equal(self.net1.linear_1, self.net2.linear_1.cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='No CUDA device registered.')
    def test_load_to_gpu_from_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.net1.cuda(device=0)
        assert self.net1.linear_1.weight.is_cuda is True
        assert self.net1.linear_1.bias.is_cuda is True
        applicator = self._get_applicator('linear_1.*', self.temp_file)
        applicator(self.net1)
        assert self.net1.linear_1.weight.is_cuda is True
        assert self.net1.linear_1.bias.is_cuda is True
        assert self._are_equal(self.net1.linear_1.cpu(), self.net2.linear_1)