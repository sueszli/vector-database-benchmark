from dataclasses import dataclass, field
from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass
from fairseq import file_utils

@dataclass
class HuggingFaceByteLevelBPEConfig(FairseqDataclass):
    bpe_merges: str = field(default='???', metadata={'help': 'path to merges.txt'})
    bpe_vocab: str = field(default='???', metadata={'help': 'path to vocab.json'})
    bpe_add_prefix_space: bool = field(default=False, metadata={'help': 'add prefix space before encoding'})

@register_bpe('hf_byte_bpe', dataclass=HuggingFaceByteLevelBPEConfig)
class HuggingFaceByteLevelBPE(object):

    def __init__(self, cfg):
        if False:
            for i in range(10):
                print('nop')
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ImportError:
            raise ImportError('Please install huggingface/tokenizers with: pip install tokenizers')
        bpe_vocab = file_utils.cached_path(cfg.bpe_vocab)
        bpe_merges = file_utils.cached_path(cfg.bpe_merges)
        self.bpe = ByteLevelBPETokenizer(bpe_vocab, bpe_merges, add_prefix_space=cfg.bpe_add_prefix_space)

    def encode(self, x: str) -> str:
        if False:
            return 10
        return ' '.join(map(str, self.bpe.encode(x).ids))

    def decode(self, x: str) -> str:
        if False:
            i = 10
            return i + 15
        return self.bpe.decode([int(tok) if tok not in {'<unk>', '<mask>'} else tok for tok in x.split()])

    def is_beginning_of_word(self, x: str) -> bool:
        if False:
            i = 10
            return i + 15
        return self.decode(x).startswith(' ')