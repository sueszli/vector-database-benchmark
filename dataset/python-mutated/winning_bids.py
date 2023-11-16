"""
A transform to find winning bids for each closed auction. In pseudo CQL syntax:

SELECT Rstream(A.*, B.auction, B.bidder, MAX(B.price), B.dateTime)
FROM Auction A [ROWS UNBOUNDED], Bid B [ROWS UNBOUNDED]
WHERE A.id = B.auction AND B.datetime < A.expires AND A.expires < CURRENT_TIME
GROUP BY A.id

We will also check that the winning bid is above the auction reserve. Note that
we ignore the auction opening bid value since it has no impact on which bid
eventually wins, if any.

Our implementation will use a custom windowing function in order to bring bids
and auctions together without requiring global state.
"""
import apache_beam as beam
from apache_beam.coders import coder_impl
from apache_beam.coders.coders import FastCoder
from apache_beam.testing.benchmarks.nexmark.models import auction_bid
from apache_beam.testing.benchmarks.nexmark.models import nexmark_model
from apache_beam.testing.benchmarks.nexmark.queries import nexmark_query_util
from apache_beam.transforms.window import IntervalWindow
from apache_beam.transforms.window import WindowFn
from apache_beam.utils.timestamp import Duration

class AuctionOrBidWindow(IntervalWindow):
    """Windows for open auctions and bids."""

    def __init__(self, start, end, auction_id, is_auction_window):
        if False:
            print('Hello World!')
        super().__init__(start, end)
        self.auction = auction_id
        self.is_auction_window = is_auction_window

    @staticmethod
    def for_auction(timestamp, auction):
        if False:
            print('Hello World!')
        return AuctionOrBidWindow(timestamp, auction.expires, auction.id, True)

    @staticmethod
    def for_bid(expected_duration_micro, timestamp, bid):
        if False:
            for i in range(10):
                print('nop')
        return AuctionOrBidWindow(timestamp, timestamp + Duration(micros=expected_duration_micro * 2), bid.auction, False)

    def is_auction_window_fn(self):
        if False:
            return 10
        return self.is_auction_window

    def __str__(self):
        if False:
            return 10
        return 'AuctionOrBidWindow{start:%s; end:%s; auction:%d; isAuctionWindow:%s}' % (self.start, self.end, self.auction, self.is_auction_window)

class AuctionOrBidWindowCoder(FastCoder):

    def _create_impl(self):
        if False:
            for i in range(10):
                print('nop')
        return AuctionOrBidWindowCoderImpl()

    def is_deterministic(self):
        if False:
            while True:
                i = 10
        return True

class AuctionOrBidWindowCoderImpl(coder_impl.StreamCoderImpl):
    _super_coder_impl = coder_impl.IntervalWindowCoderImpl()
    _id_coder_impl = coder_impl.VarIntCoderImpl()
    _bool_coder_impl = coder_impl.BooleanCoderImpl()

    def encode_to_stream(self, value, stream, nested):
        if False:
            for i in range(10):
                print('nop')
        self._super_coder_impl.encode_to_stream(value, stream, True)
        self._id_coder_impl.encode_to_stream(value.auction, stream, True)
        self._bool_coder_impl.encode_to_stream(value.is_auction_window, stream, True)

    def decode_from_stream(self, stream, nested):
        if False:
            return 10
        super_window = self._super_coder_impl.decode_from_stream(stream, True)
        auction = self._id_coder_impl.decode_from_stream(stream, True)
        is_auction = self._bool_coder_impl.decode_from_stream(stream, True)
        return AuctionOrBidWindow(super_window.start, super_window.end, auction, is_auction)

class AuctionOrBidWindowFn(WindowFn):

    def __init__(self, expected_duration_micro):
        if False:
            for i in range(10):
                print('nop')
        self.expected_duration = expected_duration_micro

    def assign(self, assign_context):
        if False:
            for i in range(10):
                print('nop')
        event = assign_context.element
        if isinstance(event, nexmark_model.Auction):
            return [AuctionOrBidWindow.for_auction(assign_context.timestamp, event)]
        elif isinstance(event, nexmark_model.Bid):
            return [AuctionOrBidWindow.for_bid(self.expected_duration, assign_context.timestamp, event)]
        else:
            raise ValueError('%s can only assign windows to auctions and bids, but received %s' % (self.__class__.__name__, event))

    def merge(self, merge_context):
        if False:
            print('Hello World!')
        auction_id_to_auction_window = {}
        auction_id_to_bid_window = {}
        for window in merge_context.windows:
            if window.is_auction_window_fn():
                auction_id_to_auction_window[window.auction] = window
            else:
                if window.auction not in auction_id_to_bid_window:
                    auction_id_to_bid_window[window.auction] = []
                auction_id_to_bid_window[window.auction].append(window)
        for (auction, auction_window) in auction_id_to_auction_window.items():
            bid_window_list = auction_id_to_bid_window.get(auction)
            if bid_window_list is not None:
                to_merge = []
                for bid_window in bid_window_list:
                    if bid_window.start < auction_window.end:
                        to_merge.append(bid_window)
                if len(to_merge) > 0:
                    to_merge.append(auction_window)
                    merge_context.merge(to_merge, auction_window)

    def get_window_coder(self):
        if False:
            print('Hello World!')
        return AuctionOrBidWindowCoder()

    def get_transformed_output_time(self, window, input_timestamp):
        if False:
            while True:
                i = 10
        return window.max_timestamp()

class JoinAuctionBidFn(beam.DoFn):

    @staticmethod
    def higher_bid(bid, other):
        if False:
            print('Hello World!')
        if bid.price > other.price:
            return True
        elif bid.price < other.price:
            return False
        else:
            return bid.date_time < other.date_time

    def process(self, element):
        if False:
            i = 10
            return i + 15
        (_, group) = element
        auctions = group[nexmark_query_util.AUCTION_TAG]
        auction = auctions[0] if auctions else None
        if auction is None:
            return
        best_bid = None
        for bid in group[nexmark_query_util.BID_TAG]:
            if bid.price < auction.reserve:
                continue
            if best_bid is None or JoinAuctionBidFn.higher_bid(bid, best_bid):
                best_bid = bid
        if best_bid:
            yield auction_bid.AuctionBid(auction, best_bid)

class WinningBids(beam.PTransform):

    def __init__(self):
        if False:
            print('Hello World!')
        expected_duration = 16667000
        self.auction_or_bid_windowFn = AuctionOrBidWindowFn(expected_duration)

    def expand(self, pcoll):
        if False:
            print('Hello World!')
        events = pcoll | beam.WindowInto(self.auction_or_bid_windowFn)
        auction_by_id = events | nexmark_query_util.JustAuctions() | 'auction_by_id' >> beam.ParDo(nexmark_query_util.AuctionByIdFn())
        bids_by_auction_id = events | nexmark_query_util.JustBids() | 'bid_by_auction' >> beam.ParDo(nexmark_query_util.BidByAuctionIdFn())
        return {nexmark_query_util.AUCTION_TAG: auction_by_id, nexmark_query_util.BID_TAG: bids_by_auction_id} | beam.CoGroupByKey() | beam.ParDo(JoinAuctionBidFn())