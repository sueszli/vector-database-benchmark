__author__ = 'saeedamen'
import pandas
from findatapy.util.loggermanager import LoggerManager

class MarketLiquidity(object):
    """Calculates spread between bid/ask and also tick count.

    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.logger = LoggerManager().getLogger(__name__)
        return

    def calculate_spreads(self, data_frame, asset, bid_field='bid', ask_field='ask'):
        if False:
            i = 10
            return i + 15
        if isinstance(asset, str):
            asset = [asset]
        cols = [x + '.spread' for x in asset]
        data_frame_spreads = pandas.DataFrame(index=data_frame.index, columns=cols)
        for a in asset:
            data_frame_spreads[a + '.spread'] = data_frame[a + '.' + ask_field] - data_frame[a + '.' + bid_field]
        return data_frame_spreads

    def calculate_tick_count(self, data_frame, asset, freq='1h'):
        if False:
            while True:
                i = 10
        if isinstance(asset, str):
            asset = [asset]
        data_frame_tick_count = data_frame.resample(freq, how='count').dropna()
        data_frame_tick_count = data_frame_tick_count[[0]]
        data_frame_tick_count.columns = [x + '.event' for x in asset]
        return data_frame_tick_count
if __name__ == '__main__':
    pass