"""Nexmark Query 1: Convert bid prices from dollars to euros.

The Nexmark suite is a series of queries (streaming pipelines) performed
on a simulation of auction events.

This query converts bid prices from dollars to euros.
It illustrates a simple map.
"""
import apache_beam as beam
from apache_beam.testing.benchmarks.nexmark.models import nexmark_model
from apache_beam.testing.benchmarks.nexmark.queries import nexmark_query_util
USD_TO_EURO = 0.89

def load(events, metadata=None, pipeline_options=None):
    if False:
        i = 10
        return i + 15
    return events | nexmark_query_util.JustBids() | 'ConvertToEuro' >> beam.Map(lambda bid: nexmark_model.Bid(bid.auction, bid.bidder, bid.price * USD_TO_EURO, bid.date_time, bid.extra))