import pytest
from jesse import research

def test_store_candles():
    if False:
        i = 10
        return i + 15
    "\n    for now, don't actually store it in the db. But test validations, etc\n    "
    with pytest.raises(TypeError):
        research.store_candles({}, 'Test Exchange', 'BTC-USDT')
    with pytest.raises(TypeError):
        research.store_candles([], 'Test Exchange', 'BTC-USDT')
    with pytest.raises(ValueError):
        close_prices = [10, 11]
        np_candles = research.candles_from_close_prices(close_prices)
        np_candles[1][0] += 300000
        research.store_candles(np_candles, 'Test Exchange', 'BTC-USDT')
    close_prices = [10, 11, 12, 12, 11, 13, 14, 12, 11, 15]
    np_candles = research.candles_from_close_prices(close_prices)
    research.store_candles(np_candles, 'Test Exchange', 'BTC-USDT')