"""
Query 9: Winning-bids: extract the most recent of the highest bids
See winning_bids.py for detailed documentation
"""
import apache_beam as beam
from apache_beam.testing.benchmarks.nexmark.queries import nexmark_query_util
from apache_beam.testing.benchmarks.nexmark.queries import winning_bids

def load(events, metadata=None, pipeline_options=None):
    if False:
        return 10
    return events | beam.Filter(nexmark_query_util.auction_or_bid) | winning_bids.WinningBids()