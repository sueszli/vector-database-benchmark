import pytest
import torch
from pyro.contrib.oed.eig import xexpx

@pytest.mark.parametrize('argument,output', [(torch.tensor([float('-inf')]), torch.tensor([0.0])), (torch.tensor([0.0]), torch.tensor([0.0])), (torch.tensor([1.0]), torch.exp(torch.tensor([1.0])))])
def test_xexpx(argument, output):
    if False:
        return 10
    assert xexpx(argument) == output