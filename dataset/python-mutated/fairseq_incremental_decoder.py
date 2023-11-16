import logging
from typing import Dict, Optional
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.models import FairseqDecoder
from torch import Tensor
logger = logging.getLogger(__name__)

@with_incremental_state
class FairseqIncrementalDecoder(FairseqDecoder):
    """Base class for incremental decoders.

    Incremental decoding is a special mode at inference time where the Model
    only receives a single timestep of input corresponding to the previous
    output token (for teacher forcing) and must produce the next output
    *incrementally*. Thus the model must cache any long-term state that is
    needed about the sequence, e.g., hidden states, convolutional states, etc.

    Compared to the standard :class:`FairseqDecoder` interface, the incremental
    decoder interface allows :func:`forward` functions to take an extra keyword
    argument (*incremental_state*) that can be used to cache state across
    time-steps.

    The :class:`FairseqIncrementalDecoder` interface also defines the
    :func:`reorder_incremental_state` method, which is used during beam search
    to select and reorder the incremental state based on the selection of beams.

    To learn more about how incremental decoding works, refer to `this blog
    <http://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq/>`_.
    """

    def __init__(self, dictionary):
        if False:
            print('Hello World!')
        super().__init__(dictionary)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        if False:
            return 10
        "\n        Args:\n            prev_output_tokens (LongTensor): shifted output tokens of shape\n                `(batch, tgt_len)`, for teacher forcing\n            encoder_out (dict, optional): output from the encoder, used for\n                encoder-side attention\n            incremental_state (dict, optional): dictionary used for storing\n                state during :ref:`Incremental decoding`\n\n        Returns:\n            tuple:\n                - the decoder's output of shape `(batch, tgt_len, vocab)`\n                - a dictionary with any model-specific outputs\n        "
        raise NotImplementedError

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        if False:
            print('Hello World!')
        "\n        Returns:\n            tuple:\n                - the decoder's features of shape `(batch, tgt_len, embed_dim)`\n                - a dictionary with any model-specific outputs\n        "
        raise NotImplementedError

    def reorder_incremental_state(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor):
        if False:
            while True:
                i = 10
        'Reorder incremental state.\n\n        This will be called when the order of the input has changed from the\n        previous time step. A typical use case is beam search, where the input\n        order changes between time steps based on the selection of beams.\n        '
        pass

    def reorder_incremental_state_scripting(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor):
        if False:
            return 10
        'Main entry point for reordering the incremental state.\n\n        Due to limitations in TorchScript, we call this function in\n        :class:`fairseq.sequence_generator.SequenceGenerator` instead of\n        calling :func:`reorder_incremental_state` directly.\n        '
        for module in self.modules():
            if hasattr(module, 'reorder_incremental_state'):
                result = module.reorder_incremental_state(incremental_state, new_order)
                if result is not None:
                    incremental_state = result

    def set_beam_size(self, beam_size):
        if False:
            for i in range(10):
                print('nop')
        'Sets the beam size in the decoder and all children.'
        if getattr(self, '_beam_size', -1) != beam_size:
            seen = set()

            def apply_set_beam_size(module):
                if False:
                    return 10
                if module != self and hasattr(module, 'set_beam_size') and (module not in seen):
                    seen.add(module)
                    module.set_beam_size(beam_size)
            self.apply(apply_set_beam_size)
            self._beam_size = beam_size