import random
from typing import Any
from typing import Iterable
from typing import Tuple
from typing import TypeVar
import numpy as np
import pandas as pd
Frame = TypeVar('Frame', bound=pd.core.generic.NDFrame)

class Partitioning(object):
    """A class representing a (consistent) partitioning of dataframe objects.
  """

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__class__.__name__

    def is_subpartitioning_of(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Returns whether self is a sub-partition of other.\n\n    Specifically, returns whether something partitioned by self is necissarily\n    also partitioned by other.\n    '
        raise NotImplementedError

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        return self != other and self <= other

    def __le__(self, other):
        if False:
            print('Hello World!')
        return not self.is_subpartitioning_of(other)

    def partition_fn(self, df, num_partitions):
        if False:
            while True:
                i = 10
        'A callable that actually performs the partitioning of a Frame df.\n\n    This will be invoked via a FlatMap in conjunction with a GroupKey to\n    achieve the desired partitioning.\n    '
        raise NotImplementedError

    def test_partition_fn(self, df):
        if False:
            return 10
        return self.partition_fn(df, 5)

class Index(Partitioning):
    """A partitioning by index (either fully or partially).

  If the set of "levels" of the index to consider is not specified, the entire
  index is used.

  These form a partial order, given by

      Singleton() < Index([i]) < Index([i, j]) < ... < Index() < Arbitrary()

  The ordering is implemented via the is_subpartitioning_of method, where the
  examples on the right are subpartitionings of the examples on the left above.
  """

    def __init__(self, levels=None):
        if False:
            i = 10
            return i + 15
        self._levels = levels

    def __repr__(self):
        if False:
            while True:
                i = 10
        if self._levels:
            return 'Index%s' % self._levels
        else:
            return 'Index'

    def __eq__(self, other):
        if False:
            return 10
        return type(self) == type(other) and self._levels == other._levels

    def __hash__(self):
        if False:
            print('Hello World!')
        if self._levels:
            return hash(tuple(sorted(self._levels)))
        else:
            return hash(type(self))

    def is_subpartitioning_of(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, Singleton):
            return True
        elif isinstance(other, Index):
            if self._levels is None:
                return True
            elif other._levels is None:
                return False
            else:
                return all((level in self._levels for level in other._levels))
        elif isinstance(other, (Arbitrary, JoinIndex)):
            return False
        else:
            raise ValueError(f'Encountered unknown type {other!r}')

    def _hash_index(self, df):
        if False:
            while True:
                i = 10
        if self._levels is None:
            levels = list(range(df.index.nlevels))
        else:
            levels = self._levels
        return sum((pd.util.hash_array(np.asarray(df.index.get_level_values(level))) for level in levels))

    def partition_fn(self, df, num_partitions):
        if False:
            while True:
                i = 10
        hashes = self._hash_index(df)
        for key in range(num_partitions):
            yield (key, df[hashes % num_partitions == key])

    def check(self, dfs):
        if False:
            i = 10
            return i + 15
        dfs = [df for df in dfs if len(df)]
        if not len(dfs):
            return True

        def apply_consistent_order(dfs):
            if False:
                for i in range(10):
                    print('nop')
            return sorted((df.sort_index() for df in dfs if len(df)), key=lambda df: sum(self._hash_index(df)))
        dfs = apply_consistent_order(dfs)
        repartitioned_dfs = apply_consistent_order((df for (_, df) in self.test_partition_fn(pd.concat(dfs))))
        for (df, repartitioned_df) in zip(dfs, repartitioned_dfs):
            if not df.index.equals(repartitioned_df.index):
                return False
        return True

class Singleton(Partitioning):
    """A partitioning of all the data into a single partition.
  """

    def __init__(self, reason=None):
        if False:
            return 10
        self._reason = reason

    @property
    def reason(self):
        if False:
            print('Hello World!')
        return self._reason

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return type(self) == type(other)

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(type(self))

    def is_subpartitioning_of(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, Singleton)

    def partition_fn(self, df, num_partitions):
        if False:
            i = 10
            return i + 15
        yield (None, df)

    def check(self, dfs):
        if False:
            return 10
        return len(dfs) <= 1

class JoinIndex(Partitioning):
    """A partitioning that lets two frames be joined.
  This can either be a hash partitioning on the full index, or a common
  ancestor with no intervening re-indexing/re-partitioning.

  It fits into the partial ordering as

      Index() < JoinIndex(x) < JoinIndex() < Arbitrary()

  with

      JoinIndex(x) and JoinIndex(y)

  being incomparable for nontrivial x != y.

  Expressions desiring to make use of this index should simply declare a
  requirement of JoinIndex().
  """

    def __init__(self, ancestor=None):
        if False:
            print('Hello World!')
        self._ancestor = ancestor

    def __repr__(self):
        if False:
            return 10
        if self._ancestor:
            return 'JoinIndex[%s]' % self._ancestor
        else:
            return 'JoinIndex'

    def __eq__(self, other):
        if False:
            return 10
        if type(self) != type(other):
            return False
        elif self._ancestor is None:
            return other._ancestor is None
        elif other._ancestor is None:
            return False
        else:
            return self._ancestor == other._ancestor

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash((type(self), self._ancestor))

    def is_subpartitioning_of(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, Arbitrary):
            return False
        elif isinstance(other, JoinIndex):
            return self._ancestor is None or self == other
        else:
            return True

    def test_partition_fn(self, df):
        if False:
            print('Hello World!')
        return Index().test_partition_fn(df)

    def check(self, dfs):
        if False:
            print('Hello World!')
        return True

class Arbitrary(Partitioning):
    """A partitioning imposing no constraints on the actual partitioning.
  """

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return type(self) == type(other)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(type(self))

    def is_subpartitioning_of(self, other):
        if False:
            print('Hello World!')
        return True

    def test_partition_fn(self, df):
        if False:
            while True:
                i = 10
        num_partitions = 10

        def shuffled(seq):
            if False:
                for i in range(10):
                    print('nop')
            seq = list(seq)
            random.shuffle(seq)
            return seq
        part = pd.Series(shuffled(range(len(df))), index=df.index) % num_partitions
        for k in range(num_partitions):
            yield (k, df[part == k])

    def check(self, dfs):
        if False:
            return 10
        return True