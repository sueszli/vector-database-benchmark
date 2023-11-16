from __future__ import annotations
import cudf
from cudf._lib.nvtext.byte_pair_encode import BPEMergePairs as cpp_merge_pairs, byte_pair_encoding as cpp_byte_pair_encoding

class BytePairEncoder:
    """
    Given a merge pairs strings series, performs byte pair encoding on
    a strings series using the provided separator.

    Parameters
    ----------
    merges_pairs : str
        Strings column of merge pairs

    Returns
    -------
    BytePairEncoder
    """

    def __init__(self, merges_pair: 'cudf.Series'):
        if False:
            while True:
                i = 10
        self.merge_pairs = cpp_merge_pairs(merges_pair._column)

    def __call__(self, text, separator: str=' '):
        if False:
            return 10
        '\n\n        Parameters\n        ----------\n        text : cudf string series\n            The strings to be encoded.\n\n        Returns\n        -------\n        Encoded strings\n\n        Examples\n        --------\n        >>> import cudf\n        >>> from cudf.core.byte_pair_encoding import BytePairEncoder\n        >>> mps = cudf.Series(["e n", "i t", "i s", "e s", "en t",\n        ...                    "c e", "es t", "en ce", "T h", "Th is",\n        ...                    "t est", "s ent", "t h", "th is"])\n        >>> bpe = BytePairEncoder(mps)\n        >>> str_series = cudf.Series([\'This is the sentence\', \'thisisit\'])\n        >>> bpe(str_series)\n        0    This is a sent ence\n        1             this is it\n        dtype: object\n        '
        sep = cudf.Scalar(separator, dtype='str')
        result = cpp_byte_pair_encoding(text._column, self.merge_pairs, sep)
        return cudf.Series(result)