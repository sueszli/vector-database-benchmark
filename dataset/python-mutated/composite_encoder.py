from .fairseq_encoder import FairseqEncoder

class CompositeEncoder(FairseqEncoder):
    """
    A wrapper around a dictionary of :class:`FairseqEncoder` objects.

    We run forward on each encoder and return a dictionary of outputs. The first
    encoder's dictionary is used for initialization.

    Args:
        encoders (dict): a dictionary of :class:`FairseqEncoder` objects.
    """

    def __init__(self, encoders):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(next(iter(encoders.values())).dictionary)
        self.encoders = encoders
        for key in self.encoders:
            self.add_module(key, self.encoders[key])

    def forward(self, src_tokens, src_lengths):
        if False:
            print('Hello World!')
        '\n        Args:\n            src_tokens (LongTensor): tokens in the source language of shape\n                `(batch, src_len)`\n            src_lengths (LongTensor): lengths of each source sentence of shape\n                `(batch)`\n\n        Returns:\n            dict:\n                the outputs from each Encoder\n        '
        encoder_out = {}
        for key in self.encoders:
            encoder_out[key] = self.encoders[key](src_tokens, src_lengths)
        return encoder_out

    def reorder_encoder_out(self, encoder_out, new_order):
        if False:
            print('Hello World!')
        'Reorder encoder output according to new_order.'
        for key in self.encoders:
            encoder_out[key] = self.encoders[key].reorder_encoder_out(encoder_out[key], new_order)
        return encoder_out

    def max_positions(self):
        if False:
            return 10
        return min((self.encoders[key].max_positions() for key in self.encoders))

    def upgrade_state_dict(self, state_dict):
        if False:
            for i in range(10):
                print('nop')
        for key in self.encoders:
            self.encoders[key].upgrade_state_dict(state_dict)
        return state_dict