"""Common rates related utilities."""
import enum
from typing import Optional
import tensorflow.compat.v2 as tf
from tf_quant_finance import datetime as dates
from tf_quant_finance import types
from tf_quant_finance import utils as tff_utils

class AverageType(enum.Enum):
    """Averaging types."""
    COMPOUNDING = 1
    ARITHMETIC_AVERAGE = 2

class DayCountConvention(enum.Enum):
    """Day count conventions for accrual."""
    ACTUAL_360 = 1
    ACTUAL_365 = 2
    THIRTY_360_ISDA = 3

class RateIndexType(enum.Enum):
    """Interest rate indexes."""
    LIBOR = 1
    SWAP = 2

class InterestRateModelType(enum.Enum):
    """Models for pricing interest rate derivatives."""
    LOGNORMAL_RATE = 1
    NORMAL_RATE = 2
    LOGNORMAL_SMILE_CONSISTENT_REPLICATION = 3
    NORMAL_SMILE_CONSISTENT_REPLICATION = 4

@tff_utils.dataclass
class InterestRateMarket:
    """InterestRateMarket data."""
    reference_curve: Optional['RateCurve'] = None
    discount_curve: Optional['RateCurve'] = None
    libor_rate: Optional[types.RealTensor] = None
    swap_rate: Optional[types.RealTensor] = None
    volatility_curve: Optional['VolatiltyCube'] = None

@tff_utils.dataclass
class FixedCouponSpecs:
    """FixedCouponSpecs data."""
    coupon_frequency: types.RealTensor
    currency: str
    notional: types.RealTensor
    coupon_rate: types.RealTensor
    daycount_convention: types.RealTensor
    businessday_rule: dates.BusinessDayConvention

@tff_utils.dataclass
class FloatCouponSpecs:
    """FloatCouponSpecs data."""
    coupon_frequency: types.RealTensor
    reference_rate_term: types.RealTensor
    reset_frequency: types.RealTensor
    currency: str
    notional: types.RealTensor
    daycount_convention: DayCountConvention
    businessday_rule: dates.BusinessDayConvention
    coupon_basis: types.RealTensor
    coupon_multiplier: types.RealTensor

@tff_utils.dataclass
class CMSCouponSpecs:
    """CMSCouponSpecs data."""
    coupon_frequency: dates.PeriodTensor
    tenor: dates.PeriodTensor
    float_leg: FloatCouponSpecs
    fixed_leg: FixedCouponSpecs
    notional: types.RealTensor
    daycount_convention: DayCountConvention
    coupon_basis: types.RealTensor
    coupon_multiplier: types.RealTensor
    businessday_rule: dates.BusinessDayConvention

def elapsed_time(date_1, date_2, dtype):
    if False:
        return 10
    'Computes elapsed time between two date tensors.'
    days_in_year = 365.0
    return tf.cast(date_1.days_until(date_2), dtype=dtype) / days_in_year

def get_daycount_fraction(date_start, date_end, convention, dtype):
    if False:
        i = 10
        return i + 15
    'Return the day count fraction between two dates.'
    if convention == DayCountConvention.ACTUAL_365:
        return dates.daycount_actual_365_fixed(start_date=date_start, end_date=date_end, dtype=dtype)
    elif convention == DayCountConvention.ACTUAL_360:
        return dates.daycount_actual_360(start_date=date_start, end_date=date_end, dtype=dtype)
    elif convention == DayCountConvention.THIRTY_360_ISDA:
        return dates.daycount_thirty_360_isda(start_date=date_start, end_date=date_end, dtype=dtype)
    else:
        raise ValueError('Daycount convention not implemented.')

def get_rate_index(market, valuation_date, rate_type=None, currency=None, dtype=None):
    if False:
        return 10
    'Return the relevant rate from the market data.'
    del currency
    if rate_type == RateIndexType.LIBOR:
        rate = market.libor_rate or tf.zeros(valuation_date.shape, dtype=dtype)
    elif rate_type == RateIndexType.SWAP:
        rate = market.swap_rate or tf.zeros(valuation_date.shape, dtype=dtype)
    else:
        raise ValueError('Unrecognized rate type.')
    return rate

def get_implied_volatility_data(market, valuation_date=None, volatility_type=None, currency=None):
    if False:
        print('Hello World!')
    'Return the implied volatility data from the market data.'
    del valuation_date, volatility_type, currency
    vol_data = market.volatility_curve
    return vol_data