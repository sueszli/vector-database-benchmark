from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
import unicodedata
import pandas as pd
import numpy as np
from holidays import list_supported_countries
from prophet.make_holidays import make_holidays_df

def utf8_to_ascii(text: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Holidays often have utf-8 characters. These are not allowed in R package data (they generate a NOTE).\n    TODO: revisit whether we want to do this lossy conversion.\n    '
    ascii_text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('ascii')
    ascii_text = re.sub('\\(\\)$', '', ascii_text).strip()
    if sum((1 for x in ascii_text if x not in [' ', '(', ')', ','])) == 0:
        return 'FAILED_TO_PARSE'
    else:
        return ascii_text

def generate_holidays_df() -> pd.DataFrame:
    if False:
        for i in range(10):
            print('nop')
    'Generate csv file of all possible holiday names, ds, and countries, year combination.'
    country_codes = set(list_supported_countries().keys())
    country_codes.add('TU')
    all_holidays = []
    for country_code in country_codes:
        df = make_holidays_df(year_list=np.arange(1995, 2045, 1).tolist(), country=country_code)
        df['country'] = country_code
        all_holidays.append(df)
    generated_holidays = pd.concat(all_holidays, axis=0, ignore_index=True)
    generated_holidays['year'] = generated_holidays.ds.dt.year
    generated_holidays.sort_values(['country', 'ds', 'holiday'], inplace=True)
    generated_holidays['holiday'] = generated_holidays['holiday'].apply(utf8_to_ascii)
    failed_countries = generated_holidays.loc[generated_holidays['holiday'] == 'FAILED_TO_PARSE', 'country'].unique()
    if len(failed_countries) > 0:
        print('Failed to convert UTF-8 holidays for:')
        print('\n'.join(failed_countries))
    assert 'FAILED_TO_PARSE' not in generated_holidays['holiday'].unique()
    return generated_holidays
if __name__ == '__main__':
    import argparse
    import pathlib
    if not pathlib.Path.cwd().stem == 'python':
        raise RuntimeError('Run script from prophet/python directory')
    OUT_CSV_PATH = pathlib.Path('.') / '..' / 'R/data-raw/generated_holidays.csv'
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', default=OUT_CSV_PATH)
    args = parser.parse_args()
    df = generate_holidays_df()
    df.to_csv(args.outfile, index=False)