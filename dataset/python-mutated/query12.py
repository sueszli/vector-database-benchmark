"""
Query 12, How many bids does a user make within a fixed processing time limit
(Not in original suite.)

Group bids by the same user into processing time windows of window_size_sec.
Emit the count of bids per window.
"""
import apache_beam as beam
from apache_beam.testing.benchmarks.nexmark.queries import nexmark_query_util
from apache_beam.testing.benchmarks.nexmark.queries.nexmark_query_util import ResultNames
from apache_beam.transforms import trigger
from apache_beam.transforms import window

def load(events, metadata=None, pipeline_options=None):
    if False:
        while True:
            i = 10
    return events | nexmark_query_util.JustBids() | 'query12_extract_bidder' >> beam.Map(lambda bid: bid.bidder) | beam.WindowInto(window.GlobalWindows(), trigger=trigger.Repeatedly(trigger.AfterProcessingTime(metadata.get('window_size_sec'))), accumulation_mode=trigger.AccumulationMode.DISCARDING, allowed_lateness=0) | 'query12_bid_count' >> beam.combiners.Count.PerElement() | 'query12_output' >> beam.Map(lambda t: {ResultNames.BIDDER_ID: t[0], ResultNames.BID_COUNT: t[1]})