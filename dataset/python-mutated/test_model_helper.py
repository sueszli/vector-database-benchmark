import pytest
import torch
from ding.torch_utils.model_helper import get_num_params

@pytest.mark.unittest
class TestModelHelper:

    def test_model_helper(self):
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Test the model helper.\n        '
        net = torch.nn.Linear(3, 4, bias=False)
        assert get_num_params(net) == 12
        net = torch.nn.Conv2d(3, 3, kernel_size=3, bias=False)
        assert get_num_params(net) == 81