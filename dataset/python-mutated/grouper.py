from __future__ import annotations
import collections
import copy
import functools
from river import base
__all__ = ['Grouper']

class Grouper(base.Transformer):
    """Applies a transformer within different groups.

    This transformer allows you to split your data into groups and apply a transformer within each
    group. This happens in a streaming manner, which means that the groups are discovered online.
    A separate copy of the provided transformer is made whenever a new group appears. The groups
    are defined according to one or more keys.

    Parameters
    ----------
    transformer
    by
        The field on which to group the data. This can either by a single value, or a list of
        values.

    """

    def __init__(self, transformer: base.Transformer, by: base.typing.FeatureName | list[base.typing.FeatureName]):
        if False:
            i = 10
            return i + 15
        self.transformer = transformer
        self.by = by if isinstance(by, list) else [by]
        self.transformers: collections.defaultdict = collections.defaultdict(functools.partial(copy.deepcopy, transformer))

    def _get_key(self, x):
        if False:
            while True:
                i = 10
        return '_'.join((str(x[k]) for k in self.by))

    def learn_one(self, x):
        if False:
            for i in range(10):
                print('nop')
        key = self._get_key(x)
        self.transformers[key].learn_one(x)
        return self

    def transform_one(self, x):
        if False:
            i = 10
            return i + 15
        key = self._get_key(x)
        return self.transformers[key].transform_one(x)