from typing import Dict, List, NamedTuple, Optional
import torch
import torch.nn as nn
from torch import Tensor
EncoderOut = NamedTuple('EncoderOut', [('encoder_out', Tensor), ('encoder_padding_mask', Optional[Tensor]), ('encoder_embedding', Optional[Tensor]), ('encoder_states', Optional[List[Tensor]]), ('src_tokens', Optional[Tensor]), ('src_lengths', Optional[Tensor])])

class FairseqEncoder(nn.Module):
    """Base class for encoders."""

    def __init__(self, dictionary):
        if False:
            print('Hello World!')
        super().__init__()
        self.dictionary = dictionary

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Args:\n            src_tokens (LongTensor): tokens in the source language of shape\n                `(batch, src_len)`\n            src_lengths (LongTensor): lengths of each source sentence of shape\n                `(batch)`\n        '
        raise NotImplementedError

    def forward_torchscript(self, net_input: Dict[str, Tensor]):
        if False:
            print('Hello World!')
        'A TorchScript-compatible version of forward.\n\n        Encoders which use additional arguments may want to override\n        this method for TorchScript compatibility.\n        '
        if torch.jit.is_scripting():
            return self.forward(src_tokens=net_input['src_tokens'], src_lengths=net_input['src_lengths'])
        else:
            return self.forward_non_torchscript(net_input)

    @torch.jit.unused
    def forward_non_torchscript(self, net_input: Dict[str, Tensor]):
        if False:
            print('Hello World!')
        encoder_input = {k: v for (k, v) in net_input.items() if k != 'prev_output_tokens'}
        return self.forward(**encoder_input)

    def reorder_encoder_out(self, encoder_out, new_order):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reorder encoder output according to `new_order`.\n\n        Args:\n            encoder_out: output from the ``forward()`` method\n            new_order (LongTensor): desired order\n\n        Returns:\n            `encoder_out` rearranged according to `new_order`\n        '
        raise NotImplementedError

    def max_positions(self):
        if False:
            i = 10
            return i + 15
        'Maximum input length supported by the encoder.'
        return 1000000.0

    def upgrade_state_dict_named(self, state_dict, name):
        if False:
            return 10
        'Upgrade old state dicts to work with newer code.'
        return state_dict

    def set_num_updates(self, num_updates):
        if False:
            return 10
        'State from trainer to pass along to model at every update.'

        def _apply(m):
            if False:
                for i in range(10):
                    print('nop')
            if hasattr(m, 'set_num_updates') and m != self:
                m.set_num_updates(num_updates)
        self.apply(_apply)