from coalib.parsing.filters import available_filters

class InvalidFilterException(LookupError):

    def __init__(self, filter_name):
        if False:
            for i in range(10):
                print('nop')
        joined_available_filters = ', '.join(sorted(available_filters))
        super().__init__(f'{filter_name!r} is an invalid filter. Available filters: {joined_available_filters}')