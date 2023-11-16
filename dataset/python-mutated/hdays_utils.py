from typing import Iterable, Optional, Union
import holidays

def get_country_holidays(country: str, years: Optional[Union[int, Iterable[int]]]=None):
    if False:
        while True:
            i = 10
    '\n    Helper function to get holidays for a country.\n\n    Parameters\n    ----------\n        country : str\n            Country name to retrieve country specific holidays\n        years : int, list\n            Year or list of years to retrieve holidays for\n\n    Returns\n    -------\n        set\n            All possible holiday dates and names of given country\n\n    '
    substitutions = {'TU': 'TR'}
    country = substitutions.get(country, country)
    if not hasattr(holidays, country):
        raise AttributeError(f'Holidays in {country} are not currently supported!')
    return getattr(holidays, country)(years=years)