from math import ceil
from freqtrade.exceptions import OperationalException
from freqtrade.util import FtPrecise
one = FtPrecise(1.0)
four = FtPrecise(4.0)
twenty_four = FtPrecise(24.0)

def interest(exchange_name: str, borrowed: FtPrecise, rate: FtPrecise, hours: FtPrecise) -> FtPrecise:
    if False:
        for i in range(10):
            print('nop')
    '\n    Equation to calculate interest on margin trades\n\n    :param exchange_name: The exchanged being trading on\n    :param borrowed: The amount of currency being borrowed\n    :param rate: The rate of interest (i.e daily interest rate)\n    :param hours: The time in hours that the currency has been borrowed for\n\n    Raises:\n        OperationalException: Raised if freqtrade does\n        not support margin trading for this exchange\n\n    Returns: The amount of interest owed (currency matches borrowed)\n    '
    exchange_name = exchange_name.lower()
    if exchange_name == 'binance':
        return borrowed * rate * FtPrecise(ceil(hours)) / twenty_four
    elif exchange_name == 'kraken':
        return borrowed * rate * (one + FtPrecise(ceil(hours / four)))
    else:
        raise OperationalException(f'Leverage not available on {exchange_name} with freqtrade')