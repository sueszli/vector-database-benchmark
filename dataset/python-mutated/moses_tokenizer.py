from dataclasses import dataclass, field
from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass

@dataclass
class MosesTokenizerConfig(FairseqDataclass):
    source_lang: str = field(default='en', metadata={'help': 'source language'})
    target_lang: str = field(default='en', metadata={'help': 'target language'})
    moses_no_dash_splits: bool = field(default=False, metadata={'help': "don't apply dash split rules"})
    moses_no_escape: bool = field(default=False, metadata={'help': "don't perform HTML escaping on apostrophe, quotes, etc."})

@register_tokenizer('moses', dataclass=MosesTokenizerConfig)
class MosesTokenizer(object):

    def __init__(self, cfg: MosesTokenizerConfig):
        if False:
            i = 10
            return i + 15
        self.cfg = cfg
        try:
            from sacremoses import MosesTokenizer, MosesDetokenizer
            self.tok = MosesTokenizer(cfg.source_lang)
            self.detok = MosesDetokenizer(cfg.target_lang)
        except ImportError:
            raise ImportError('Please install Moses tokenizer with: pip install sacremoses')

    def encode(self, x: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.tok.tokenize(x, aggressive_dash_splits=not self.cfg.moses_no_dash_splits, return_str=True, escape=not self.cfg.moses_no_escape)

    def decode(self, x: str) -> str:
        if False:
            while True:
                i = 10
        return self.detok.detokenize(x.split())