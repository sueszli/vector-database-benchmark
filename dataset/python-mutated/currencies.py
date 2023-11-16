"""Supported currencies."""
import enum
from tf_quant_finance.experimental.pricing_platform.instrument_protos import currencies_pb2
Currency = enum.Enum('Currency', zip(currencies_pb2.Currency.keys(), currencies_pb2.Currency.keys()))
Currency.__doc__ = 'Supported currencies.'
Currency.__repr__ = lambda self: self.value
Currency.__call__ = lambda self: self.value
CurrencyProtoType = Currency.mro()[0]

def from_proto_value(value: int) -> CurrencyProtoType:
    if False:
        i = 10
        return i + 15
    'Creates Currency from a proto field value.'
    return Currency(currencies_pb2.Currency.Name(value))
__all__ = ['Currency', 'from_proto_value']