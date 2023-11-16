from .alphavantage import AlphaVantage as av

class SectorPerformances(av):
    """This class implements all the sector performance api calls
    """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Inherit AlphaVantage base class with its default arguments\n        '
        super(SectorPerformances, self).__init__(*args, **kwargs)
        self._append_type = False
        if self.output_format.lower() == 'csv':
            raise ValueError('Output format {} is not comatible with the SectorPerformances class'.format(self.output_format.lower()))

    def percentage_to_float(self, val):
        if False:
            while True:
                i = 10
        ' Transform a string of the form f.f% into f.f/100\n\n        Keyword Arguments:\n            val: The string to convert\n        '
        return float(val.strip('%')) / 100

    @av._output_format_sector
    @av._call_api_on_func
    def get_sector(self):
        if False:
            i = 10
            return i + 15
        'This API returns the realtime and historical sector performances\n        calculated from S&P500 incumbents.\n\n        Returns:\n            A pandas or a dictionary with the results from the api call\n        '
        _FUNCTION_KEY = 'SECTOR'
        _DATA_KEYS = ['Rank A: Real-Time Performance', 'Rank B: 1 Day Performance', 'Rank C: 5 Day Performance', 'Rank D: 1 Month Performance', 'Rank E: 3 Month Performance', 'Rank F: Year-to-Date (YTD) Performance', 'Rank G: 1 Year Performance', 'Rank H: 3 Year Performance', 'Rank I: 5 Year Performance', 'Rank J: 10 Year Performance']
        return (_FUNCTION_KEY, _DATA_KEYS, 'Meta Data')