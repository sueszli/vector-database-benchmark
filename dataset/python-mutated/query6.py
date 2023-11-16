"""
Query 6, 'Average Selling Price by Seller'. Select the average selling price
over the last 10 closed auctions by the same seller. In CQL syntax::

  SELECT Istream(AVG(Q.final), Q.seller)
  FROM (SELECT Rstream(MAX(B.price) AS final, A.seller)
    FROM Auction A [ROWS UNBOUNDED], Bid B [ROWS UNBOUNDED]
    WHERE A.id=B.auction
      AND B.datetime < A.expires AND A.expires < CURRENT_TIME
    GROUP BY A.id, A.seller) [PARTITION BY A.seller ROWS 10] Q
  GROUP BY Q.seller;
"""
import apache_beam as beam
from apache_beam.testing.benchmarks.nexmark.queries import nexmark_query_util
from apache_beam.testing.benchmarks.nexmark.queries import winning_bids
from apache_beam.testing.benchmarks.nexmark.queries.nexmark_query_util import ResultNames
from apache_beam.transforms import trigger
from apache_beam.transforms import window

def load(events, metadata=None, pipeline_options=None):
    if False:
        i = 10
        return i + 15
    return events | beam.Filter(nexmark_query_util.auction_or_bid) | winning_bids.WinningBids() | beam.Map(lambda auc_bid: (auc_bid.auction.seller, auc_bid.bid)) | beam.WindowInto(window.GlobalWindows(), trigger=trigger.Repeatedly(trigger.AfterCount(1)), accumulation_mode=trigger.AccumulationMode.ACCUMULATING, allowed_lateness=0) | beam.CombinePerKey(MovingMeanSellingPriceFn(10)) | beam.Map(lambda t: {ResultNames.SELLER: t[0], ResultNames.PRICE: t[1]})

class MovingMeanSellingPriceFn(beam.CombineFn):
    """
  Combiner to keep track of up to max_num_bids of the most recent wining
  bids and calculate their average selling price.
  """

    def __init__(self, max_num_bids):
        if False:
            return 10
        self.max_num_bids = max_num_bids

    def create_accumulator(self):
        if False:
            print('Hello World!')
        return []

    def add_input(self, accumulator, element):
        if False:
            while True:
                i = 10
        accumulator.append(element)
        new_accu = sorted(accumulator, key=lambda bid: (-bid.date_time, -bid.price))
        if len(new_accu) > self.max_num_bids:
            del new_accu[self.max_num_bids]
        return new_accu

    def merge_accumulators(self, accumulators):
        if False:
            while True:
                i = 10
        new_accu = []
        for accumulator in accumulators:
            new_accu += accumulator
        new_accu.sort(key=lambda bid: (bid.date_time, bid.price))
        return new_accu[-self.max_num_bids:]

    def extract_output(self, accumulator):
        if False:
            while True:
                i = 10
        if len(accumulator) == 0:
            return 0
        sum_price = sum((bid.price for bid in accumulator))
        return int(sum_price / len(accumulator))