"""Transform Beam PTransforms into Dask Bag operations.

A minimum set of operation substitutions, to adap Beam's PTransform model
to Dask Bag functions.

TODO(alxr): Translate ops from https://docs.dask.org/en/latest/bag-api.html.
"""
import abc
import dataclasses
import typing as t
import apache_beam
import dask.bag as db
from apache_beam.pipeline import AppliedPTransform
from apache_beam.runners.dask.overrides import _Create
from apache_beam.runners.dask.overrides import _Flatten
from apache_beam.runners.dask.overrides import _GroupByKeyOnly
OpInput = t.Union[db.Bag, t.Sequence[db.Bag], None]

@dataclasses.dataclass
class DaskBagOp(abc.ABC):
    applied: AppliedPTransform

    @property
    def transform(self):
        if False:
            for i in range(10):
                print('nop')
        return self.applied.transform

    @abc.abstractmethod
    def apply(self, input_bag: OpInput) -> db.Bag:
        if False:
            print('Hello World!')
        pass

class NoOp(DaskBagOp):

    def apply(self, input_bag: OpInput) -> db.Bag:
        if False:
            print('Hello World!')
        return input_bag

class Create(DaskBagOp):

    def apply(self, input_bag: OpInput) -> db.Bag:
        if False:
            print('Hello World!')
        assert input_bag is None, 'Create expects no input!'
        original_transform = t.cast(_Create, self.transform)
        items = original_transform.values
        return db.from_sequence(items)

class ParDo(DaskBagOp):

    def apply(self, input_bag: db.Bag) -> db.Bag:
        if False:
            while True:
                i = 10
        transform = t.cast(apache_beam.ParDo, self.transform)
        return input_bag.map(transform.fn.process, *transform.args, **transform.kwargs).flatten()

class Map(DaskBagOp):

    def apply(self, input_bag: db.Bag) -> db.Bag:
        if False:
            for i in range(10):
                print('nop')
        transform = t.cast(apache_beam.Map, self.transform)
        return input_bag.map(transform.fn.process, *transform.args, **transform.kwargs)

class GroupByKey(DaskBagOp):

    def apply(self, input_bag: db.Bag) -> db.Bag:
        if False:
            for i in range(10):
                print('nop')

        def key(item):
            if False:
                print('Hello World!')
            return item[0]

        def value(item):
            if False:
                return 10
            (k, v) = item
            return (k, [elm[1] for elm in v])
        return input_bag.groupby(key).map(value)

class Flatten(DaskBagOp):

    def apply(self, input_bag: OpInput) -> db.Bag:
        if False:
            return 10
        assert type(input_bag) is list, 'Must take a sequence of bags!'
        return db.concat(input_bag)
TRANSLATIONS = {_Create: Create, apache_beam.ParDo: ParDo, apache_beam.Map: Map, _GroupByKeyOnly: GroupByKey, _Flatten: Flatten}