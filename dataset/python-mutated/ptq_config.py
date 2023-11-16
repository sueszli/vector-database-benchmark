import copy
from .ptq_quantizer import SUPPORT_ACT_QUANTIZERS, SUPPORT_WT_QUANTIZERS, KLQuantizer, PerChannelAbsmaxQuantizer

class PTQConfig:
    """
    The PTQ config shows how to quantize the inputs and outputs.
    """

    def __init__(self, activation_quantizer, weight_quantizer):
        if False:
            print('Hello World!')
        '\n        Constructor.\n\n        Args:\n            activation_quantizer(BaseQuantizer): The activation quantizer.\n                It should be the instance of BaseQuantizer.\n            weight_quantizer(BaseQuantizer): The weight quantizer.\n                It should be the instance of BaseQuantizer.\n        '
        super().__init__()
        assert isinstance(activation_quantizer, tuple(SUPPORT_ACT_QUANTIZERS))
        assert isinstance(weight_quantizer, tuple(SUPPORT_WT_QUANTIZERS))
        self.in_act_quantizer = copy.deepcopy(activation_quantizer)
        self.out_act_quantizer = copy.deepcopy(activation_quantizer)
        self.wt_quantizer = copy.deepcopy(weight_quantizer)
        self.quant_hook_handle = None
        self.enable_in_act_quantizer = False
default_ptq_config = PTQConfig(KLQuantizer(), PerChannelAbsmaxQuantizer())