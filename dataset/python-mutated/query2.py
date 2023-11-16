"""
Query 2: Find bids with specific auction ids and show their bid price

This query selects Bids that have a particular auctiuon id, and output their
auction id with bid price.
It illustrates a simple filter.
"""
import apache_beam as beam
from apache_beam.testing.benchmarks.nexmark.queries import nexmark_query_util
from apache_beam.testing.benchmarks.nexmark.queries.nexmark_query_util import ResultNames

def load(events, metadata=None, pipeline_options=None):
    if False:
        return 10
    return events | nexmark_query_util.JustBids() | 'filter_by_skip' >> beam.Filter(lambda bid: bid.auction % metadata.get('auction_skip') == 0) | 'project' >> beam.Map(lambda bid: {ResultNames.AUCTION_ID: bid.auction, ResultNames.PRICE: bid.price})