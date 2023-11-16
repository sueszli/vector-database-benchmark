from dataclasses import dataclass, field
from typing import Optional
from fairseq import file_utils
from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass

@dataclass
class SentencepieceConfig(FairseqDataclass):
    sentencepiece_model: str = field(default='???', metadata={'help': 'path to sentencepiece model'})
    sentencepiece_enable_sampling: bool = field(default=False, metadata={'help': 'enable sampling'})
    sentencepiece_alpha: Optional[float] = field(default=None, metadata={'help': 'soothing parameter for unigram sampling, and merge probability for BPE-dropout'})

@register_bpe('sentencepiece', dataclass=SentencepieceConfig)
class SentencepieceBPE(object):

    def __init__(self, cfg):
        if False:
            for i in range(10):
                print('nop')
        self.enable_sampling = cfg.sentencepiece_enable_sampling
        self.alpha = cfg.sentencepiece_alpha
        sentencepiece_model = file_utils.cached_path(cfg.sentencepiece_model)
        try:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(sentencepiece_model)
        except ImportError:
            raise ImportError('Please install sentencepiece with: pip install sentencepiece')

    def encode(self, x: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return ' '.join(self.sp.Encode(x, out_type=str, enable_sampling=self.enable_sampling, alpha=self.alpha))

    def decode(self, x: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return x.replace(' ', '').replace('▁', ' ').strip()

    def is_beginning_of_word(self, x: str) -> bool:
        if False:
            i = 10
            return i + 15
        if x in ['<unk>', '<s>', '</s>', '<pad>']:
            return True
        return x.startswith('▁')