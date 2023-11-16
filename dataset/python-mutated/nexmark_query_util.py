"""Utilities for working with NEXmark data stream."""
import apache_beam as beam
from apache_beam.testing.benchmarks.nexmark.models import nexmark_model
AUCTION_TAG = 'auctions'
BID_TAG = 'bids'
PERSON_TAG = 'person'

class ResultNames:
    SELLER = 'seller'
    PRICE = 'price'
    NAME = 'name'
    CITY = 'city'
    STATE = 'state'
    AUCTION_ID = 'auction_id'
    ID = 'id'
    RESERVE = 'reserve'
    CATEGORY = 'category'
    IS_LAST = 'is_last'
    BIDDER_ID = 'bidder_id'
    BID_COUNT = 'bid_count'
    NUM = 'num'

def is_bid(event):
    if False:
        print('Hello World!')
    return isinstance(event, nexmark_model.Bid)

def is_auction(event):
    if False:
        print('Hello World!')
    return isinstance(event, nexmark_model.Auction)

def is_person(event):
    if False:
        return 10
    return isinstance(event, nexmark_model.Person)

def auction_or_bid(event):
    if False:
        return 10
    return isinstance(event, (nexmark_model.Auction, nexmark_model.Bid))

class JustBids(beam.PTransform):

    def expand(self, pcoll):
        if False:
            print('Hello World!')
        return pcoll | 'IsBid' >> beam.Filter(is_bid)

class JustAuctions(beam.PTransform):

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        return pcoll | 'IsAuction' >> beam.Filter(is_auction)

class JustPerson(beam.PTransform):

    def expand(self, pcoll):
        if False:
            i = 10
            return i + 15
        return pcoll | 'IsPerson' >> beam.Filter(is_person)

class AuctionByIdFn(beam.DoFn):

    def process(self, element):
        if False:
            return 10
        yield (element.id, element)

class BidByAuctionIdFn(beam.DoFn):

    def process(self, element):
        if False:
            i = 10
            return i + 15
        yield (element.auction, element)

class PersonByIdFn(beam.DoFn):

    def process(self, element):
        if False:
            return 10
        yield (element.id, element)

class AuctionBySellerFn(beam.DoFn):

    def process(self, element):
        if False:
            for i in range(10):
                print('nop')
        yield (element.seller, element)