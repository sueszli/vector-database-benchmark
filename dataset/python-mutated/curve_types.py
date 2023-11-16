"""Curve types."""
from typing import Union
import dataclasses
from tf_quant_finance.experimental.pricing_platform.framework.core import currencies
from tf_quant_finance.experimental.pricing_platform.framework.core import rate_indices

def _init_currency(currency: Union[currencies.CurrencyProtoType, str]) -> currencies.CurrencyProtoType:
    if False:
        for i in range(10):
            print('nop')
    'Converts input to a currency object.'
    if isinstance(currency, str):
        try:
            return getattr(currencies.Currency, currency)
        except KeyError:
            raise ValueError(f'{currency} is not a valid currency')
    return currency

@dataclasses.dataclass
class RiskFreeCurve:
    """Risk free curve description."""
    currency: Union[currencies.CurrencyProtoType, str]

    def __post_init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.currency = _init_currency(self.currency)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash((self.currency,))

@dataclasses.dataclass
class RateIndexCurve:
    """Rate index curve description."""
    currency: currencies.CurrencyProtoType
    index: rate_indices.RateIndex

    def __post_init__(self):
        if False:
            i = 10
            return i + 15
        self.currency = _init_currency(self.currency)

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash((self.currency, self.index.type))
CurveType = Union[RiskFreeCurve, RateIndexCurve]
__all__ = ['CurveType', 'RiskFreeCurve', 'RateIndexCurve']