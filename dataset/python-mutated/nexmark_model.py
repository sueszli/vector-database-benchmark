"""Nexmark model.

The nexmark suite is a series of queries (streaming pipelines) performed
on a simulation of auction events. The model includes the three roles that
generate events:

  - The person who starts and auction or makes a bid (Person).
  - The auction item (Auction).
  - The bid on an item for auction (Bid).

"""
from apache_beam.coders import coder_impl
from apache_beam.coders.coders import FastCoder
from apache_beam.coders.coders import StrUtf8Coder
from apache_beam.testing.benchmarks.nexmark import nexmark_util

class PersonCoder(FastCoder):

    def to_type_hint(self):
        if False:
            return 10
        return Person

    def _create_impl(self):
        if False:
            print('Hello World!')
        return PersonCoderImpl()

    def is_deterministic(self):
        if False:
            return 10
        return True

class Person(object):
    """Author of an auction or a bid."""
    CODER = PersonCoder()

    def __init__(self, id, name, email, credit_card, city, state, date_time, extra=None):
        if False:
            while True:
                i = 10
        self.id = id
        self.name = name
        self.email_address = email
        self.credit_card = credit_card
        self.city = city
        self.state = state
        self.date_time = date_time
        self.extra = extra

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return nexmark_util.model_to_json(self)

class AuctionCoder(FastCoder):

    def to_type_hint(self):
        if False:
            while True:
                i = 10
        return Auction

    def _create_impl(self):
        if False:
            while True:
                i = 10
        return AuctionCoderImpl()

    def is_deterministic(self):
        if False:
            while True:
                i = 10
        return True

class Auction(object):
    """Item for auction."""
    CODER = AuctionCoder()

    def __init__(self, id, item_name, description, initial_bid, reserve_price, date_time, expires, seller, category, extra=None):
        if False:
            print('Hello World!')
        self.id = id
        self.item_name = item_name
        self.description = description
        self.initial_bid = initial_bid
        self.reserve = reserve_price
        self.date_time = date_time
        self.expires = expires
        self.seller = seller
        self.category = category
        self.extra = extra

    def __repr__(self):
        if False:
            while True:
                i = 10
        return nexmark_util.model_to_json(self)

class BidCoder(FastCoder):

    def to_type_hint(self):
        if False:
            while True:
                i = 10
        return Bid

    def _create_impl(self):
        if False:
            for i in range(10):
                print('nop')
        return BidCoderImpl()

    def is_deterministic(self):
        if False:
            for i in range(10):
                print('nop')
        return True

class Bid(object):
    """A bid for an item for auction."""
    CODER = BidCoder()

    def __init__(self, auction, bidder, price, date_time, extra=None):
        if False:
            print('Hello World!')
        self.auction = auction
        self.bidder = bidder
        self.price = price
        self.date_time = date_time
        self.extra = extra

    def __repr__(self):
        if False:
            print('Hello World!')
        return nexmark_util.model_to_json(self)

class AuctionCoderImpl(coder_impl.StreamCoderImpl):
    _int_coder_impl = coder_impl.VarIntCoderImpl()
    _str_coder_impl = StrUtf8Coder().get_impl()
    _time_coder_impl = coder_impl.TimestampCoderImpl()

    def encode_to_stream(self, value, stream, nested):
        if False:
            for i in range(10):
                print('nop')
        self._int_coder_impl.encode_to_stream(value.id, stream, True)
        self._str_coder_impl.encode_to_stream(value.item_name, stream, True)
        self._str_coder_impl.encode_to_stream(value.description, stream, True)
        self._int_coder_impl.encode_to_stream(value.initial_bid, stream, True)
        self._int_coder_impl.encode_to_stream(value.reserve, stream, True)
        self._time_coder_impl.encode_to_stream(value.date_time, stream, True)
        self._time_coder_impl.encode_to_stream(value.expires, stream, True)
        self._int_coder_impl.encode_to_stream(value.seller, stream, True)
        self._int_coder_impl.encode_to_stream(value.category, stream, True)
        self._str_coder_impl.encode_to_stream(value.extra, stream, True)

    def decode_from_stream(self, stream, nested):
        if False:
            print('Hello World!')
        id = self._int_coder_impl.decode_from_stream(stream, True)
        item_name = self._str_coder_impl.decode_from_stream(stream, True)
        description = self._str_coder_impl.decode_from_stream(stream, True)
        initial_bid = self._int_coder_impl.decode_from_stream(stream, True)
        reserve = self._int_coder_impl.decode_from_stream(stream, True)
        date_time = self._time_coder_impl.decode_from_stream(stream, True)
        expires = self._time_coder_impl.decode_from_stream(stream, True)
        seller = self._int_coder_impl.decode_from_stream(stream, True)
        category = self._int_coder_impl.decode_from_stream(stream, True)
        extra = self._str_coder_impl.decode_from_stream(stream, True)
        return Auction(id, item_name, description, initial_bid, reserve, date_time, expires, seller, category, extra)

class BidCoderImpl(coder_impl.StreamCoderImpl):
    _int_coder_impl = coder_impl.VarIntCoderImpl()
    _str_coder_impl = StrUtf8Coder().get_impl()
    _time_coder_impl = coder_impl.TimestampCoderImpl()

    def encode_to_stream(self, value, stream, nested):
        if False:
            while True:
                i = 10
        self._int_coder_impl.encode_to_stream(value.auction, stream, True)
        self._int_coder_impl.encode_to_stream(value.bidder, stream, True)
        self._int_coder_impl.encode_to_stream(value.price, stream, True)
        self._time_coder_impl.encode_to_stream(value.date_time, stream, True)
        self._str_coder_impl.encode_to_stream(value.extra, stream, True)

    def decode_from_stream(self, stream, nested):
        if False:
            while True:
                i = 10
        auction = self._int_coder_impl.decode_from_stream(stream, True)
        bidder = self._int_coder_impl.decode_from_stream(stream, True)
        price = self._int_coder_impl.decode_from_stream(stream, True)
        date_time = self._time_coder_impl.decode_from_stream(stream, True)
        extra = self._str_coder_impl.decode_from_stream(stream, True)
        return Bid(auction, bidder, price, date_time, extra)

class PersonCoderImpl(coder_impl.StreamCoderImpl):
    _int_coder_impl = coder_impl.VarIntCoderImpl()
    _str_coder_impl = StrUtf8Coder().get_impl()
    _time_coder_impl = coder_impl.TimestampCoderImpl()

    def encode_to_stream(self, value, stream, nested):
        if False:
            return 10
        self._int_coder_impl.encode_to_stream(value.id, stream, True)
        self._str_coder_impl.encode_to_stream(value.name, stream, True)
        self._str_coder_impl.encode_to_stream(value.email_address, stream, True)
        self._str_coder_impl.encode_to_stream(value.credit_card, stream, True)
        self._str_coder_impl.encode_to_stream(value.city, stream, True)
        self._str_coder_impl.encode_to_stream(value.state, stream, True)
        self._time_coder_impl.encode_to_stream(value.date_time, stream, True)
        self._str_coder_impl.encode_to_stream(value.extra, stream, True)

    def decode_from_stream(self, stream, nested):
        if False:
            print('Hello World!')
        id = self._int_coder_impl.decode_from_stream(stream, True)
        name = self._str_coder_impl.decode_from_stream(stream, True)
        email = self._str_coder_impl.decode_from_stream(stream, True)
        credit_card = self._str_coder_impl.decode_from_stream(stream, True)
        city = self._str_coder_impl.decode_from_stream(stream, True)
        state = self._str_coder_impl.decode_from_stream(stream, True)
        date_time = self._time_coder_impl.decode_from_stream(stream, True)
        extra = self._str_coder_impl.decode_from_stream(stream, True)
        return Person(id, name, email, credit_card, city, state, date_time, extra)