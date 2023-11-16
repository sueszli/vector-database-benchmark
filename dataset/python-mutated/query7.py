"""
Query 7, 'Highest Bid'. Select the bids with the highest bid price in the
last minute. In CQL syntax::

  SELECT Rstream(B.auction, B.price, B.bidder)
  FROM Bid [RANGE 1 MINUTE SLIDE 1 MINUTE] B
  WHERE B.price = (SELECT MAX(B1.price)
                   FROM BID [RANGE 1 MINUTE SLIDE 1 MINUTE] B1);

We will use a shorter window to help make testing easier. We'll also
implement this using a side-input in order to exercise that functionality.
(A combiner, as used in Query 5, is a more efficient approach.).
"""
import apache_beam as beam
from apache_beam.testing.benchmarks.nexmark.queries import nexmark_query_util
from apache_beam.transforms import window

def load(events, metadata=None, pipeline_options=None):
    if False:
        for i in range(10):
            print('nop')
    sliding_bids = events | nexmark_query_util.JustBids() | beam.WindowInto(window.FixedWindows(metadata.get('window_size_sec')))
    max_prices = sliding_bids | beam.Map(lambda bid: bid.price) | beam.CombineGlobally(max).without_defaults()
    return sliding_bids | 'select_bids' >> beam.ParDo(SelectMaxBidFn(), beam.pvalue.AsSingleton(max_prices))

class SelectMaxBidFn(beam.DoFn):

    def process(self, element, max_bid_price):
        if False:
            return 10
        if element.price == max_bid_price:
            yield element