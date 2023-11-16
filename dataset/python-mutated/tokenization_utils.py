"""Tokenization utils for RoFormer."""
from typing import List
from tokenizers import NormalizedString, PreTokenizedString, normalizers

class JiebaPreTokenizer:

    def __init__(self, vocab) -> None:
        if False:
            while True:
                i = 10
        self.vocab = vocab
        self.normalizers = normalizers.BertNormalizer(clean_text=False, handle_chinese_chars=True, strip_accents=False, lowercase=False)
        try:
            import rjieba
        except ImportError:
            raise ImportError('You need to install rjieba to use RoFormerTokenizer. See https://pypi.org/project/rjieba/ for installation.')
        self.jieba = rjieba

    def jieba_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        if False:
            for i in range(10):
                print('nop')
        splits = []
        for (token, start, end) in self.jieba.tokenize(str(normalized_string), hmm=False):
            if token in self.vocab:
                splits.append(normalized_string[start:end])
            else:
                token_list = self.normalizers.normalize_str(token).split()
                for token in token_list:
                    if token:
                        end = start + len(token)
                        splits.append(normalized_string[start:end])
                        start = end
        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):
        if False:
            while True:
                i = 10
        pretok.split(self.jieba_split)