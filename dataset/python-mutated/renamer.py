from __future__ import annotations
from river import base
__all__ = ['Renamer', 'Prefixer', 'Suffixer']

class Renamer(base.Transformer):
    """Renames features following substitution rules.

    Parameters
    ----------
    mapping
        Dictionnary describing substitution rules. Keys in `mapping` that are not a feature's name are silently ignored.

    Examples
    --------

    >>> from river import compose

    >>> mapping = {'a': 'v', 'c': 'o'}
    >>> x = {'a': 42, 'b': 12}
    >>> compose.Renamer(mapping).transform_one(x)
    {'b': 12, 'v': 42}

    """

    def __init__(self, mapping: dict[str, str]):
        if False:
            return 10
        self.mapping = mapping

    def transform_one(self, x):
        if False:
            return 10
        for (old_key, new_key) in self.mapping.items():
            try:
                x[new_key] = x.pop(old_key)
            except KeyError:
                pass
        return x

class Prefixer(base.Transformer):
    """Prepends a prefix on features names.

    Parameters
    ----------
    prefix

    Examples
    --------

    >>> from river import compose

    >>> x = {'a': 42, 'b': 12}
    >>> compose.Prefixer('prefix_').transform_one(x)
    {'prefix_a': 42, 'prefix_b': 12}

    """

    def __init__(self, prefix: str):
        if False:
            print('Hello World!')
        self.prefix = prefix

    def _rename(self, s: str) -> str:
        if False:
            return 10
        return f'{self.prefix}{s}'

    def transform_one(self, x):
        if False:
            i = 10
            return i + 15
        return {self._rename(i): xi for (i, xi) in x.items()}

class Suffixer(base.Transformer):
    """Appends a suffix on features names.

    Parameters
    ----------
    suffix

    Examples
    --------

    >>> from river import compose

    >>> x = {'a': 42, 'b': 12}
    >>> compose.Suffixer('_suffix').transform_one(x)
    {'a_suffix': 42, 'b_suffix': 12}

    """

    def __init__(self, suffix: str):
        if False:
            print('Hello World!')
        self.suffix = suffix

    def _rename(self, s: str) -> str:
        if False:
            while True:
                i = 10
        return f'{s}{self.suffix}'

    def transform_one(self, x):
        if False:
            print('Hello World!')
        return {self._rename(i): xi for (i, xi) in x.items()}