from __future__ import annotations
import cudf
from cudf._lib.nvtext.tokenize import TokenizeVocabulary as cpp_tokenize_vocabulary, tokenize_with_vocabulary as cpp_tokenize_with_vocabulary

class TokenizeVocabulary:
    """
    A vocabulary object used to tokenize input text.

    Parameters
    ----------
    vocabulary : str
        Strings column of vocabulary terms
    """

    def __init__(self, vocabulary: 'cudf.Series'):
        if False:
            for i in range(10):
                print('nop')
        self.vocabulary = cpp_tokenize_vocabulary(vocabulary._column)

    def tokenize(self, text, delimiter: str='', default_id: int=-1):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        text : cudf string series\n            The strings to be tokenized.\n        delimiter : str\n            Delimiter to identify tokens. Default is whitespace.\n        default_id : int\n            Value to use for tokens not found in the vocabulary.\n            Default is -1.\n\n        Returns\n        -------\n        Tokenized strings\n        '
        if delimiter is None:
            delimiter = ''
        delim = cudf.Scalar(delimiter, dtype='str')
        result = cpp_tokenize_with_vocabulary(text._column, self.vocabulary, delim, default_id)
        return cudf.Series(result)