import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from .conv_tbc import ConvTBC
from typing import Dict, Optional
from torch import Tensor

@with_incremental_state
class LinearizedConvolution(ConvTBC):
    """An optimized version of nn.Conv1d.

    At training time, this module uses ConvTBC, which is an optimized version
    of Conv1d. At inference time, it optimizes incremental generation (i.e.,
    one time step at a time) by replacing the convolutions with linear layers.
    Note that the input order changes from training to inference.
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self._linearized_weight = None
        self.register_backward_hook(self._clear_linearized_weight)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if False:
            return 10
        state = ConvTBC.state_dict(self, destination, prefix, keep_vars=keep_vars)
        if prefix + '_linearized_weight' in state:
            del state[prefix + '_linearized_weight']
        return state

    def upgrade_state_dict_named(self, state_dict, name):
        if False:
            for i in range(10):
                print('nop')
        prefix = name + '.' if name != '' else ''
        if prefix + '_linearized_weight' in state_dict:
            del state_dict[prefix + '_linearized_weight']

    @torch.jit.export
    def forward(self, input, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]=None):
        if False:
            print('Hello World!')
        '\n        Args:\n            incremental_state: Used to buffer signal; if not None, then input is\n                expected to contain a single frame. If the input order changes\n                between time steps, call reorder_incremental_state.\n        Input:\n            Time x Batch x Channel during training\n            Batch x Time x Channel during inference\n        '
        if incremental_state is None:
            output = self.conv_tbc(input)
            if self.kernel_size[0] > 1 and self.padding[0] > 0:
                output = output[:-self.padding[0], :, :]
            return output
        weight = self._get_linearized_weight()
        kw = self.kernel_size[0]
        bsz = input.size(0)
        if kw > 1:
            input = input.data
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is None:
                input_buffer = input.new(bsz, kw, input.size(2)).zero_()
                self._set_input_buffer(incremental_state, input_buffer)
            else:
                input_buffer[:, :-1, :] = input_buffer[:, 1:, :].clone()
            input_buffer[:, -1, :] = input[:, -1, :]
            input = input_buffer
        with torch.no_grad():
            output = F.linear(input.view(bsz, -1), weight, self.bias)
        return output.view(bsz, 1, -1)

    @torch.jit.unused
    def reorder_incremental_state(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]], new_order):
        if False:
            return 10
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    @torch.jit.unused
    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]):
        if False:
            while True:
                i = 10
        return utils.get_incremental_state(self, incremental_state, 'input_buffer')

    @torch.jit.unused
    def _set_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]], new_buffer):
        if False:
            while True:
                i = 10
        return utils.set_incremental_state(self, incremental_state, 'input_buffer', new_buffer)

    @torch.jit.unused
    def _get_linearized_weight(self):
        if False:
            return 10
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            return weight.view(self.out_channels, -1)
        return self._linearized_weight

    @torch.jit.unused
    def _clear_linearized_weight(self, *args):
        if False:
            return 10
        self._linearized_weight = None