"""Supported day count conventions."""
import enum
from tf_quant_finance.experimental.pricing_platform.instrument_protos import daycount_conventions_pb2
DayCountConventions = enum.Enum('DayCountConventions', zip(daycount_conventions_pb2.DayCountConvention.keys(), daycount_conventions_pb2.DayCountConvention.keys()))
DayCountConventions.__doc__ = 'Supported day count conventions.'
DayCountConventions.__repr__ = lambda self: self.value
DayCountConventions.__call__ = lambda self: self.value
DayCountConventionsProtoType = DayCountConventions.mro()[0]

def from_proto_value(value: int) -> DayCountConventionsProtoType:
    if False:
        print('Hello World!')
    'Creates DayCountConventions from a proto field value.'
    return DayCountConventions(daycount_conventions_pb2.DayCountConvention.Name(value))
__all__ = ['DayCountConventions', 'from_proto_value']