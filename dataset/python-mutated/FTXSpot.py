from .FTXMain import FTXMain
from jesse.enums import exchanges

class FTXSpot(FTXMain):

    def __init__(self) -> None:
        if False:
            return 10
        from jesse.modes.import_candles_mode.drivers.Bitfinex.BitfinexSpot import BitfinexSpot
        super().__init__(name=exchanges.FTX_SPOT, rest_endpoint='https://ftx.com', backup_exchange_class=BitfinexSpot)