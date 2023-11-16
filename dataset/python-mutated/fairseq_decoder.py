from typing import Dict, List, Optional, Tuple
import torch.nn as nn
from fairseq import utils
from torch import Tensor

class FairseqDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self, dictionary):
        if False:
            return 10
        super().__init__()
        self.dictionary = dictionary
        self.onnx_trace = False
        self.adaptive_softmax = None

    def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
        if False:
            return 10
        "\n        Args:\n            prev_output_tokens (LongTensor): shifted output tokens of shape\n                `(batch, tgt_len)`, for teacher forcing\n            encoder_out (dict, optional): output from the encoder, used for\n                encoder-side attention\n\n        Returns:\n            tuple:\n                - the decoder's output of shape `(batch, tgt_len, vocab)`\n                - a dictionary with any model-specific outputs\n        "
        (x, extra) = self.extract_features(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        x = self.output_layer(x)
        return (x, extra)

    def extract_features(self, prev_output_tokens, encoder_out=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns:\n            tuple:\n                - the decoder's features of shape `(batch, tgt_len, embed_dim)`\n                - a dictionary with any model-specific outputs\n        "
        raise NotImplementedError

    def output_layer(self, features, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Project features to the default output size, e.g., vocabulary size.\n\n        Args:\n            features (Tensor): features returned by *extract_features*.\n        '
        raise NotImplementedError

    def get_normalized_probs(self, net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]], log_probs: bool, sample: Optional[Dict[str, Tensor]]=None):
        if False:
            return 10
        "Get normalized probabilities (or log probs) from a net's output."
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    def get_normalized_probs_scriptable(self, net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]], log_probs: bool, sample: Optional[Dict[str, Tensor]]=None):
        if False:
            print('Hello World!')
        "Get normalized probabilities (or log probs) from a net's output."
        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            if sample is not None:
                assert 'target' in sample
                target = sample['target']
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out
        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    def max_positions(self):
        if False:
            return 10
        'Maximum input length supported by the decoder.'
        return 1000000.0

    def upgrade_state_dict_named(self, state_dict, name):
        if False:
            for i in range(10):
                print('nop')
        'Upgrade old state dicts to work with newer code.'
        return state_dict

    def prepare_for_onnx_export_(self):
        if False:
            while True:
                i = 10
        self.onnx_trace = True