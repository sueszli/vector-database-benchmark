"""Result of WinningBid transform."""
from apache_beam.coders import coder_impl
from apache_beam.coders.coders import FastCoder
from apache_beam.testing.benchmarks.nexmark import nexmark_util
from apache_beam.testing.benchmarks.nexmark.models import nexmark_model

class AuctionBidCoder(FastCoder):

    def to_type_hint(self):
        if False:
            while True:
                i = 10
        return AuctionBid

    def _create_impl(self):
        if False:
            print('Hello World!')
        return AuctionBidCoderImpl()

    def is_deterministic(self):
        if False:
            while True:
                i = 10
        return True

class AuctionBid(object):
    CODER = AuctionBidCoder()

    def __init__(self, auction, bid):
        if False:
            while True:
                i = 10
        self.auction = auction
        self.bid = bid

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return nexmark_util.model_to_json(self)

class AuctionBidCoderImpl(coder_impl.StreamCoderImpl):
    _auction_coder_impl = nexmark_model.AuctionCoderImpl()
    _bid_coder_Impl = nexmark_model.BidCoderImpl()

    def encode_to_stream(self, value, stream, nested):
        if False:
            return 10
        self._auction_coder_impl.encode_to_stream(value.auction, stream, True)
        self._bid_coder_Impl.encode_to_stream(value.bid, stream, True)

    def decode_from_stream(self, stream, nested):
        if False:
            while True:
                i = 10
        auction = self._auction_coder_impl.decode_from_stream(stream, True)
        bid = self._bid_coder_Impl.decode_from_stream(stream, True)
        return AuctionBid(auction, bid)