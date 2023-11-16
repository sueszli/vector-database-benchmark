from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass

@register_tokenizer('nltk', dataclass=FairseqDataclass)
class NLTKTokenizer(object):

    def __init__(self, *unused):
        if False:
            for i in range(10):
                print('nop')
        try:
            from nltk.tokenize import word_tokenize
            self.word_tokenize = word_tokenize
        except ImportError:
            raise ImportError('Please install nltk with: pip install nltk')

    def encode(self, x: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return ' '.join(self.word_tokenize(x))

    def decode(self, x: str) -> str:
        if False:
            while True:
                i = 10
        return x