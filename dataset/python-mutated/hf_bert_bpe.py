from dataclasses import dataclass, field
from typing import Optional
from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass

@dataclass
class BertBPEConfig(FairseqDataclass):
    bpe_cased: bool = field(default=False, metadata={'help': 'set for cased BPE'})
    bpe_vocab_file: Optional[str] = field(default=None, metadata={'help': 'bpe vocab file'})

@register_bpe('bert', dataclass=BertBPEConfig)
class BertBPE(object):

    def __init__(self, cfg):
        if False:
            while True:
                i = 10
        try:
            from transformers import BertTokenizer
        except ImportError:
            raise ImportError('Please install transformers with: pip install transformers')
        if cfg.bpe_vocab_file:
            self.bert_tokenizer = BertTokenizer(cfg.bpe_vocab_file, do_lower_case=not cfg.bpe_cased)
        else:
            vocab_file_name = 'bert-base-cased' if cfg.bpe_cased else 'bert-base-uncased'
            self.bert_tokenizer = BertTokenizer.from_pretrained(vocab_file_name)

    def encode(self, x: str) -> str:
        if False:
            while True:
                i = 10
        return ' '.join(self.bert_tokenizer.tokenize(x))

    def decode(self, x: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.bert_tokenizer.clean_up_tokenization(self.bert_tokenizer.convert_tokens_to_string(x.split(' ')))

    def is_beginning_of_word(self, x: str) -> bool:
        if False:
            print('Hello World!')
        return not x.startswith('##')