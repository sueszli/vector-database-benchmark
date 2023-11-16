import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.utils.bundled_inputs
from torch.ao.quantization import default_qconfig, float_qparams_weight_only_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.testing._internal.common_quantization import NodeSpec as ns
from torch.testing._internal.common_quantization import QuantizationLiteTestCase, LinearModelWithSubmodule

class TestLiteFuseFx(QuantizationLiteTestCase):

    def test_embedding(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)

            def forward(self, indices):
                if False:
                    for i in range(10):
                        print('nop')
                return self.emb(indices)
        model = M().eval()
        indices = torch.randint(low=0, high=10, size=(20,))
        quantized_node = ns.call_module(nnq.Embedding)
        configs = [(float_qparams_weight_only_qconfig, ns.call_module(nnq.Embedding)), (None, ns.call_module(nn.Embedding)), (default_qconfig, ns.call_module(nn.Embedding))]
        for (qconfig, node) in configs:
            qconfig_dict = {'': qconfig}
            m = prepare_fx(model, qconfig_dict, example_inputs=torch.randint(low=0, high=10, size=(20,)))
            m = convert_fx(m)
            self._compare_script_and_mobile(m, input=indices)

    def test_conv2d(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.conv1(x)
                x = self.conv2(x)
                return x
        m = M().eval()
        qconfig_dict = {'': default_qconfig, 'module_name': [('conv1', None)]}
        m = prepare_fx(m, qconfig_dict, example_inputs=torch.randn(1, 1, 1, 1))
        data = torch.randn(1, 1, 1, 1)
        m = convert_fx(m)
        self._compare_script_and_mobile(m, input=data)

    def test_submodule(self):
        if False:
            while True:
                i = 10
        configs = [{}, {'module_name': [('subm', None)]}, {'module_name': [('fc', None)]}]
        for config in configs:
            model = LinearModelWithSubmodule().eval()
            qconfig_dict = {'': torch.ao.quantization.get_default_qconfig('qnnpack'), **config}
            model = prepare_fx(model, qconfig_dict, example_inputs=torch.randn(5, 5))
            quant = convert_fx(model)
            x = torch.randn(5, 5)
            self._compare_script_and_mobile(quant, input=x)
if __name__ == '__main__':
    run_tests()