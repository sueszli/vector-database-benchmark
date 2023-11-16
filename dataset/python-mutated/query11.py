"""
Query 11, How many bids did a user make in each session he was active?
(Not in original suite.)

Group bids by the same user into sessions with window_size_sec max gap.
However limit the session to at most max_log_events. Emit the number of
bids per session.
"""
import apache_beam as beam
from apache_beam.testing.benchmarks.nexmark.queries import nexmark_query_util
from apache_beam.testing.benchmarks.nexmark.queries.nexmark_query_util import ResultNames
from apache_beam.transforms import trigger
from apache_beam.transforms import window

def load(events, metadata=None, pipeline_options=None):
    if False:
        return 10
    return events | nexmark_query_util.JustBids() | 'query11_extract_bidder' >> beam.Map(lambda bid: bid.bidder) | 'query11_session_window' >> beam.WindowInto(window.Sessions(metadata.get('window_size_sec')), trigger=trigger.AfterWatermark(early=trigger.AfterCount(metadata.get('max_log_events'))), accumulation_mode=trigger.AccumulationMode.DISCARDING, allowed_lateness=metadata.get('occasional_delay_sec') // 2) | beam.combiners.Count.PerElement() | beam.Map(lambda bidder_count: {ResultNames.BIDDER_ID: bidder_count[0], ResultNames.BID_COUNT: bidder_count[1]})