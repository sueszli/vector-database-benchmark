from fairseq.data.encoders import register_bpe
from fairseq.data.encoders.byte_utils import SPACE, SPACE_ESCAPE, byte_encode, smart_byte_decode

@register_bpe('bytes')
class Bytes(object):

    def __init__(self, *unused):
        if False:
            for i in range(10):
                print('nop')
        pass

    @staticmethod
    def add_args(parser):
        if False:
            while True:
                i = 10
        pass

    @staticmethod
    def encode(x: str) -> str:
        if False:
            while True:
                i = 10
        encoded = byte_encode(x)
        escaped = encoded.replace(SPACE, SPACE_ESCAPE)
        return SPACE.join(list(escaped))

    @staticmethod
    def decode(x: str) -> str:
        if False:
            while True:
                i = 10
        unescaped = x.replace(SPACE, '').replace(SPACE_ESCAPE, SPACE)
        return smart_byte_decode(unescaped)