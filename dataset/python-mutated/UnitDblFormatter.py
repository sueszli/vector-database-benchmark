"""UnitDblFormatter module containing class UnitDblFormatter."""
import matplotlib.ticker as ticker
__all__ = ['UnitDblFormatter']

class UnitDblFormatter(ticker.ScalarFormatter):
    """
    The formatter for UnitDbl data types.

    This allows for formatting with the unit string.
    """

    def __call__(self, x, pos=None):
        if False:
            i = 10
            return i + 15
        if len(self.locs) == 0:
            return ''
        else:
            return f'{x:.12}'

    def format_data_short(self, value):
        if False:
            print('Hello World!')
        return f'{value:.12}'

    def format_data(self, value):
        if False:
            return 10
        return f'{value:.12}'