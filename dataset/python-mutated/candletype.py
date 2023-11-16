from enum import Enum

class CandleType(str, Enum):
    """Enum to distinguish candle types"""
    SPOT = 'spot'
    FUTURES = 'futures'
    MARK = 'mark'
    INDEX = 'index'
    PREMIUMINDEX = 'premiumIndex'
    FUNDING_RATE = 'funding_rate'

    def __str__(self):
        if False:
            while True:
                i = 10
        return f'{self.name.lower()}'

    @staticmethod
    def from_string(value: str) -> 'CandleType':
        if False:
            i = 10
            return i + 15
        if not value:
            return CandleType.SPOT
        return CandleType(value)

    @staticmethod
    def get_default(trading_mode: str) -> 'CandleType':
        if False:
            i = 10
            return i + 15
        if trading_mode == 'futures':
            return CandleType.FUTURES
        return CandleType.SPOT