from enum import Enum

class MarketDirection(Enum):
    """
    Enum for various market directions.
    """
    LONG = 'long'
    SHORT = 'short'
    EVEN = 'even'
    NONE = 'none'

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.value