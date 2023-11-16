from fairseq.models import FairseqDecoder, FairseqEncoder

class MultiModalityEncoder(FairseqEncoder):

    def __init__(self, dictionary):
        if False:
            while True:
                i = 10
        super().__init__(dictionary)

    def select_encoder(self, mode, **kwargs):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Model must implement the select_encoder method')
        return (None, kwargs)

    def forward(self, src_tokens, src_lengths=None, mode='', **kwargs):
        if False:
            print('Hello World!')
        (encoder, kwargs) = self.select_encoder(mode, **kwargs)
        return encoder(src_tokens, src_lengths, **kwargs)

class MultiInputDecoder(FairseqDecoder):

    def __init__(self, dictionary):
        if False:
            i = 10
            return i + 15
        super().__init__(dictionary)

    def select_decoder(self, mode, **kwargs):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Model must implement the select_decoder method')
        return (None, kwargs)

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None, mode='', **kwargs):
        if False:
            i = 10
            return i + 15
        (decoder, kwargs) = self.select_decoder(mode, **kwargs)
        return decoder(prev_output_tokens, encoder_out, incremental_state=incremental_state, **kwargs)