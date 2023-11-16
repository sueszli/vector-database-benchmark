from .DydxPerpetualMain import DydxPerpetualMain
from jesse.enums import exchanges

class DydxPerpetual(DydxPerpetualMain):

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        from jesse.modes.import_candles_mode.drivers.Bitfinex.BitfinexSpot import BitfinexSpot
        super().__init__(name=exchanges.DYDX_PERPETUAL, rest_endpoint='https://api.dydx.exchange', backup_exchange_class=BitfinexSpot)