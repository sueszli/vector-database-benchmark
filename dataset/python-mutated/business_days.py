"""Supported bank holidays and business day conventions."""
import enum
from tf_quant_finance.experimental.pricing_platform.instrument_protos import business_days_pb2
__all__ = ['BusinessDayConvention', 'convention_from_proto_value', 'BankHolidays', 'holiday_from_proto_value']
BusinessDayConvention = enum.Enum('BusinessDayConvention', zip(business_days_pb2.BusinessDayConvention.keys(), business_days_pb2.BusinessDayConvention.keys()))
BusinessDayConvention.__doc__ = 'Supported business day conventions.'
BusinessDayConvention.__repr__ = lambda self: self.value
BusinessDayConvention.__call__ = lambda self: self.value
BusinessDayConventionProtoType = BusinessDayConvention.mro()[0]

def convention_from_proto_value(value: int) -> BusinessDayConventionProtoType:
    if False:
        return 10
    'Creates BusinessDayConvention from a proto field value.'
    return BusinessDayConvention(business_days_pb2.BusinessDayConvention.Name(value))
BankHolidays = enum.Enum('BankHolidays', zip(business_days_pb2.BankHolidays.keys(), business_days_pb2.BankHolidays.keys()))
BankHolidays.__doc__ = 'Supported bank holidays.'
BankHolidays.__repr__ = lambda self: self.value
BankHolidays.__call__ = lambda self: self.value
BankHolidaysProtoType = BankHolidays.mro()[0]

def holiday_from_proto_value(value: int) -> BankHolidaysProtoType:
    if False:
        i = 10
        return i + 15
    'Creates BankHolidays from a proto field value.'
    return BankHolidays(business_days_pb2.BankHolidays.Name(value))