import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
from ydata_profiling.model import BaseDescription

def test_example(get_data_file, test_output_dir):
    if False:
        for i in range(10):
            print('nop')
    file_name = get_data_file('meteorites.csv', 'https://data.nasa.gov/api/views/gh4g-9sfh/rows.csv?accessType=DOWNLOAD')
    np.random.seed(7331)
    df = pd.read_csv(file_name)
    df['year'] = pd.to_datetime(df['year'], errors='coerce')
    df['source'] = 'NASA'
    df['boolean'] = np.random.choice([True, False], df.shape[0])
    df['mixed'] = np.random.choice([1, 'A'], df.shape[0])
    df['reclat_city'] = df['reclat'] + np.random.normal(scale=5, size=len(df))
    duplicates_to_add = pd.DataFrame(df.iloc[0:10].copy())
    df = pd.concat([df, duplicates_to_add], ignore_index=True)
    output_file = test_output_dir / 'profile.html'
    profile = ProfileReport(df, title='NASA Meteorites', samples={'head': 5, 'tail': 5}, duplicates={'head': 10}, minimal=True)
    profile.to_file(output_file)
    assert (test_output_dir / 'profile.html').exists(), 'Output file does not exist'
    assert type(profile.get_description()) == BaseDescription, 'Unexpected result'
    assert '<span class=badge>9</span>' in profile.to_html()