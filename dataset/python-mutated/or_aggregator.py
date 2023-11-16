"""
OR Aggregator
-------------

Aggregator that identifies a time step as anomalous if any of the components
is flagged as anomalous (logical OR).
"""
from typing import Sequence
from darts import TimeSeries
from darts.ad.aggregators.aggregators import NonFittableAggregator

class OrAggregator(NonFittableAggregator):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'OrAggregator'

    def _predict_core(self, series: Sequence[TimeSeries]) -> Sequence[TimeSeries]:
        if False:
            print('Hello World!')
        return [s.sum(axis=1).map(lambda x: (x > 0).astype(s.dtype)) for s in series]