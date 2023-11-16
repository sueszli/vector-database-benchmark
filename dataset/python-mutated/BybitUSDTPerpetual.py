from .BybitUSDTPerpetualMain import BybitUSDTPerpetualMain
from jesse.enums import exchanges

class BybitUSDTPerpetual(BybitUSDTPerpetualMain):

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__(name=exchanges.BYBIT_USDT_PERPETUAL, rest_endpoint='https://api.bybit.com')