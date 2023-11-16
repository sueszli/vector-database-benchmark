"""
Query 8, 'Monitor New Users'. Select people who have entered the system and
created auctions in the last 12 hours, updated every 12 hours. In CQL syntax::

  SELECT Rstream(P.id, P.name, A.reserve)
  FROM Person [RANGE 12 HOUR] P, Auction [RANGE 12 HOUR] A
  WHERE P.id = A.seller;

To make things a bit more dynamic and easier to test we'll use a much
shorter window.
"""
import apache_beam as beam
from apache_beam.testing.benchmarks.nexmark.queries import nexmark_query_util
from apache_beam.testing.benchmarks.nexmark.queries.nexmark_query_util import ResultNames
from apache_beam.transforms import window

def load(events, metadata=None, pipeline_options=None):
    if False:
        print('Hello World!')
    persons_by_id = events | nexmark_query_util.JustPerson() | 'query8_window_person' >> beam.WindowInto(window.FixedWindows(metadata.get('window_size_sec'))) | 'query8_person_by_id' >> beam.ParDo(nexmark_query_util.PersonByIdFn())
    auctions_by_seller = events | nexmark_query_util.JustAuctions() | 'query8_window_auction' >> beam.WindowInto(window.FixedWindows(metadata.get('window_size_sec'))) | 'query8_auction_by_seller' >> beam.ParDo(nexmark_query_util.AuctionBySellerFn())
    return {nexmark_query_util.PERSON_TAG: persons_by_id, nexmark_query_util.AUCTION_TAG: auctions_by_seller} | beam.CoGroupByKey() | 'query8_join' >> beam.ParDo(JoinPersonAuctionFn())

class JoinPersonAuctionFn(beam.DoFn):

    def process(self, element):
        if False:
            print('Hello World!')
        (_, group) = element
        persons = group[nexmark_query_util.PERSON_TAG]
        person = persons[0] if persons else None
        if person is None:
            return
        for auction in group[nexmark_query_util.AUCTION_TAG]:
            yield {ResultNames.ID: person.id, ResultNames.NAME: person.name, ResultNames.RESERVE: auction.reserve}