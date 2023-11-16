"""
NNI test for ModelSpeedup
In this test, we detect whether the masks can propagate correctly in log_softmax.

"""
import unittest
import torch
import torch.nn.functional as F
from nni.compression.utils.counter import count_flops_params
from nni.compression.speedup import ModelSpeedup
from nni.compression.pruning import L1NormPruner

class NaiveModel(torch.nn.Module):

    def __init__(self, acti, acti_kw):
        if False:
            return 10
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
        self.fc2 = torch.nn.Linear(500, 10)
        self.relu1 = torch.nn.ReLU6()
        self.relu2 = torch.nn.ReLU6()
        self.acti = acti
        self.acti_kw = acti_kw
        self.max_pool1 = torch.nn.MaxPool2d(2, 2)
        self.max_pool2 = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        x = self.relu1(self.conv1(x))
        x = self.max_pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(-1, x.size()[1:].numel())
        x = self.acti(self.fc1(x), **self.acti_kw)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class SpeedupSoftmaxTestCase(unittest.TestCase):

    def do_test(self, acti, actikw):
        if False:
            for i in range(10):
                print('nop')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NaiveModel(acti, actikw).to(device)
        dummy_input = torch.randn([1000, 1, 28, 28]).to(device)
        config_list = [{'sparsity': 0.5, 'op_types': ['Linear'], 'op_names': ['fc1']}]
        pruner = L1NormPruner(model=model, config_list=config_list)
        (_, masks) = pruner.compress()
        pruner.unwrap_model()
        speedup_model = ModelSpeedup(model, dummy_input, masks).speedup_model()
        speedup_model(dummy_input)
        print('model before speedup', repr(model))
        (flops, params, _) = count_flops_params(model, (1, 1, 28, 28), verbose=False)
        print(f'Pretrained model FLOPs {flops / 1000000.0:.2f} M, #Params: {params / 1000000.0:.2f}M')
        print('model after speedup', repr(speedup_model))
        (flops, params, _) = count_flops_params(speedup_model, dummy_input, verbose=False)
        print(f'Pruned model FLOPs {flops / 1000000.0:.2f} M, #Params: {params / 1000000.0:.2f}M')

    def test_function(self):
        if False:
            return 10
        return self.do_test(F.softmax, {'dim': 1})

    def test_module(self):
        if False:
            i = 10
            return i + 15
        return self.do_test(torch.nn.Softmax(dim=1), {})
if __name__ == '__main__':
    unittest.main()