from abc import abstractproperty, abstractmethod
from zipline.data.bar_reader import BarReader

class SessionBarReader(BarReader):
    """
    Reader for OHCLV pricing data at a session frequency.
    """

    @property
    def data_frequency(self):
        if False:
            print('Hello World!')
        return 'session'

    @abstractproperty
    def sessions(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns\n        -------\n        sessions : DatetimeIndex\n           All session labels (unioning the range for all assets) which the\n           reader can provide.\n        '

class CurrencyAwareSessionBarReader(SessionBarReader):

    @abstractmethod
    def currency_codes(self, sids):
        if False:
            while True:
                i = 10
        "\n        Get currencies in which prices are quoted for the requested sids.\n\n        Assumes that a sid's prices are always quoted in a single currency.\n\n        Parameters\n        ----------\n        sids : np.array[int64]\n            Array of sids for which currencies are needed.\n\n        Returns\n        -------\n        currency_codes : np.array[object]\n            Array of currency codes for listing currencies of\n            ``sids``. Implementations should return None for sids whose\n            currency is unknown.\n        "