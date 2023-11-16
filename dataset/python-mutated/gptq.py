import torch
from peft.tuners.lora.layer import LoraLayer

class QuantLinear(torch.nn.Module, LoraLayer):

    def __init__(self, adapter_name, quant_linear_module, r: int=0, lora_alpha: int=1, lora_dropout: float=0.0, **kwargs):
        if False:
            while True:
                i = 10
        torch.nn.Module.__init__(self)
        LoraLayer.__init__(self, in_features=quant_linear_module.infeatures, out_features=quant_linear_module.outfeatures)
        self.quant_linear_module = quant_linear_module
        self.weight = quant_linear_module.qweight
        init_lora_weights = kwargs.pop('init_lora_weights', True)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.set_adapter(adapter_name)

    def forward(self, x: torch.Tensor):
        if False:
            return 10
        result = self.quant_linear_module(x)
        if self.disable_adapters:
            return result
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(lora_A.weight.dtype)
            output = lora_B(lora_A(dropout(x)))
            if requires_conversion:
                output = output.to(expected_dtype)
            output = output * scaling
            result += output
        return result