"""
Query 5, 'Hot Items'. Which auctions have seen the most bids in the last hour
(updated every minute). In CQL syntax::

  SELECT Rstream(auction)
  FROM (SELECT B1.auction, count(*) AS num
        FROM Bid [RANGE 60 MINUTE SLIDE 1 MINUTE] B1
        GROUP BY B1.auction)
  WHERE num >= ALL (SELECT count(*)
                    FROM Bid [RANGE 60 MINUTE SLIDE 1 MINUTE] B2
                    GROUP BY B2.auction);

To make things a bit more dynamic and easier to test we use much shorter
windows, and we'll also preserve the bid counts.
"""
import apache_beam as beam
from apache_beam.testing.benchmarks.nexmark.queries import nexmark_query_util
from apache_beam.testing.benchmarks.nexmark.queries.nexmark_query_util import ResultNames
from apache_beam.transforms import window

def load(events, metadata=None, pipeline_options=None):
    if False:
        while True:
            i = 10
    return events | nexmark_query_util.JustBids() | 'query5_sliding_window' >> beam.WindowInto(window.SlidingWindows(metadata.get('window_size_sec'), metadata.get('window_period_sec'))) | 'extract_bid_auction' >> beam.Map(lambda bid: bid.auction) | 'bid_count_per_auction' >> beam.combiners.Count.PerElement() | 'bid_max_count' >> beam.CombineGlobally(MostBidCombineFn()).without_defaults() | beam.FlatMap(lambda auc_count: [{ResultNames.AUCTION_ID: auction, ResultNames.NUM: auc_count[1]} for auction in auc_count[0]])

class MostBidCombineFn(beam.CombineFn):
    """
  combiner function to find auctions with most bid counts
  """

    def create_accumulator(self):
        if False:
            print('Hello World!')
        return ([], 0)

    def add_input(self, accumulator, element):
        if False:
            return 10
        (accu_list, accu_count) = accumulator
        (auction, count) = element
        if accu_count < count:
            return ([auction], count)
        elif accu_count > count:
            return (accu_list, accu_count)
        else:
            accu_list_new = accu_list.copy()
            accu_list_new.append(auction)
            return (accu_list_new, accu_count)

    def merge_accumulators(self, accumulators):
        if False:
            print('Hello World!')
        max_list = []
        max_count = 0
        for (accu_list, count) in accumulators:
            if count == max_count:
                max_list = max_list + accu_list
            elif count < max_count:
                continue
            else:
                max_list = accu_list
                max_count = count
        return (max_list, max_count)

    def extract_output(self, accumulator):
        if False:
            return 10
        return accumulator