import re
from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass

@register_tokenizer('space', dataclass=FairseqDataclass)
class SpaceTokenizer(object):

    def __init__(self, *unused):
        if False:
            while True:
                i = 10
        self.space_tok = re.compile('\\s+')

    def encode(self, x: str) -> str:
        if False:
            while True:
                i = 10
        return self.space_tok.sub(' ', x)

    def decode(self, x: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return x