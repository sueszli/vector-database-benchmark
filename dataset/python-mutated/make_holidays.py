from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import holidays

def get_country_holidays_class(country):
    if False:
        while True:
            i = 10
    'Get class for a supported country.\n\n    Parameters\n    ----------\n    country: country code\n\n    Returns\n    -------\n    A valid country holidays class\n    '
    substitutions = {'TU': 'TR'}
    country = substitutions.get(country, country)
    if not hasattr(holidays, country):
        raise AttributeError(f'Holidays in {country} are not currently supported!')
    return getattr(holidays, country)

def get_holiday_names(country):
    if False:
        print('Hello World!')
    'Return all possible holiday names of given country\n\n    Parameters\n    ----------\n    country: country name\n\n    Returns\n    -------\n    A set of all possible holiday names of given country\n    '
    country_holidays = get_country_holidays_class(country)
    return set(country_holidays(language='en_US', years=np.arange(1995, 2045)).values())

def make_holidays_df(year_list, country, province=None, state=None):
    if False:
        return 10
    "Make dataframe of holidays for given years and countries\n\n    Parameters\n    ----------\n    year_list: a list of years\n    country: country name\n\n    Returns\n    -------\n    Dataframe with 'ds' and 'holiday', which can directly feed\n    to 'holidays' params in Prophet\n    "
    country_holidays = get_country_holidays_class(country)
    holidays = country_holidays(expand=False, language='en_US', subdiv=province, years=year_list)
    holidays_df = pd.DataFrame([(date, holidays.get_list(date)) for date in holidays], columns=['ds', 'holiday'])
    holidays_df = holidays_df.explode('holiday')
    holidays_df.reset_index(inplace=True, drop=True)
    holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
    return holidays_df