import torch
from ptflops import get_model_complexity_info

class FlopsEst(object):

    def __init__(self, model, input_shape=(2, 3, 224, 224), device='cpu'):
        if False:
            for i in range(10):
                print('nop')
        self.block_num = len(model.blocks)
        self.choice_num = len(model.blocks[0])
        self.flops_dict = {}
        self.params_dict = {}
        if device == 'cpu':
            model = model.cpu()
        else:
            model = model.cuda()
        self.params_fixed = 0
        self.flops_fixed = 0
        input = torch.randn(input_shape)
        (flops, params) = get_model_complexity_info(model.conv_stem, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
        self.params_fixed += params / 1000000.0
        self.flops_fixed += flops / 1000000.0
        input = model.conv_stem(input)
        for (block_id, block) in enumerate(model.blocks):
            self.flops_dict[block_id] = {}
            self.params_dict[block_id] = {}
            for (module_id, module) in enumerate(block):
                (flops, params) = get_model_complexity_info(module, tuple(input.shape[1:]), as_strings=False, print_per_layer_stat=False)
                self.flops_dict[block_id][module_id] = flops / 1000000.0
                self.params_dict[block_id][module_id] = params / 1000000.0
            input = module(input)
        (flops, params) = get_model_complexity_info(model.global_pool, tuple(input.shape[1:]), as_strings=False, print_per_layer_stat=False)
        self.params_fixed += params / 1000000.0
        self.flops_fixed += flops / 1000000.0
        input = model.global_pool(input)
        (flops, params) = get_model_complexity_info(model.conv_head, tuple(input.shape[1:]), as_strings=False, print_per_layer_stat=False)
        self.params_fixed += params / 1000000.0
        self.flops_fixed += flops / 1000000.0

    def get_params(self, arch):
        if False:
            for i in range(10):
                print('nop')
        params = 0
        for (block_id, block) in enumerate(arch):
            if block == -1:
                continue
            params += self.params_dict[block_id][block]
        return params + self.params_fixed

    def get_flops(self, arch):
        if False:
            i = 10
            return i + 15
        flops = 0
        for (block_id, block) in enumerate(arch):
            if block == 'LayerChoice1' or block_id == 'LayerChoice23':
                continue
            for (idx, choice) in enumerate(arch[block]):
                flops += self.flops_dict[block_id][idx] * (1 if choice else 0)
        return flops + self.flops_fixed