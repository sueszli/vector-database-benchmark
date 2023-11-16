__author__ = 'saeedamen'
import abc

class AbstractCurve(object):
    """Abstract class for creating total return indices and curves, which is for example implemented by FXSpotCurve
    and could be implemented by other asset classes.

    """

    @abc.abstractmethod
    def generate_key(self):
        if False:
            while True:
                i = 10
        return

    @abc.abstractmethod
    def fetch_continuous_time_series(self, md_request, market_data_generator):
        if False:
            while True:
                i = 10
        return

    @abc.abstractmethod
    def construct_total_returns_index(self):
        if False:
            i = 10
            return i + 15
        return