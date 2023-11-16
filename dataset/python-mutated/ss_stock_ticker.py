import pytest
from libqtile.widget import stock_ticker
RESPONSE = {'Meta Data': {'1. Information': 'Intraday (1min) open, high, low, close prices and volume', '2. Symbol': 'QTIL', '3. Last Refreshed': '2021-07-30 19:09:00', '4. Interval': '1min', '5. Output Size': 'Compact', '6. Time Zone': 'US/Eastern'}, 'Time Series (1min)': {'2021-07-30 19:09:00': {'1. open': '140.9800', '2. high': '140.9800', '3. low': '140.9800', '4. close': '140.9800', '5. volume': '527'}, '2021-07-30 17:27:00': {'1. open': '141.1900', '2. high': '141.1900', '3. low': '141.1900', '4. close': '141.1900', '5. volume': '300'}, '2021-07-30 16:44:00': {'1. open': '141.0000', '2. high': '141.0000', '3. low': '141.0000', '4. close': '141.0000', '5. volume': '482'}, '2021-07-30 16:26:00': {'1. open': '141.0000', '2. high': '141.0000', '3. low': '141.0000', '4. close': '141.0000', '5. volume': '102'}}}

@pytest.fixture
def widget(monkeypatch):
    if False:
        for i in range(10):
            print('nop')

    def result(self):
        if False:
            i = 10
            return i + 15
        return RESPONSE
    monkeypatch.setattr('libqtile.widget.stock_ticker.StockTicker.fetch', result)
    yield stock_ticker.StockTicker

@pytest.mark.parametrize('screenshot_manager', [{'symbol': 'QTIL'}], indirect=True)
def ss_stock_ticker(screenshot_manager):
    if False:
        return 10
    screenshot_manager.take_screenshot()