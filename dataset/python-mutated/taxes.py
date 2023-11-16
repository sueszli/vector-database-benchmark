from dataclasses import dataclass
from decimal import Decimal
from prices import Money, TaxedMoney

class TaxError(Exception):
    """Default tax error."""

def zero_money(currency: str) -> Money:
    if False:
        return 10
    "Return a money object set to zero.\n\n    This is a function used as a model's default.\n    "
    return Money(0, currency)

def zero_taxed_money(currency: str) -> TaxedMoney:
    if False:
        return 10
    zero = zero_money(currency)
    return TaxedMoney(net=zero, gross=zero)

@dataclass(frozen=True)
class TaxType:
    """Dataclass for unifying tax type object that comes from tax gateway."""
    code: str
    description: str

@dataclass(frozen=True)
class TaxLineData:
    tax_rate: Decimal
    total_gross_amount: Decimal
    total_net_amount: Decimal

@dataclass(frozen=True)
class TaxData:
    shipping_price_gross_amount: Decimal
    shipping_price_net_amount: Decimal
    shipping_tax_rate: Decimal
    lines: list[TaxLineData]