import os
import pandas as pd

def load_prices_from_csv(filepath, identifier_col, tz='UTC'):
    if False:
        return 10
    data = pd.read_csv(filepath, index_col=identifier_col)
    data.index = pd.DatetimeIndex(data.index, tz=tz)
    data.sort_index(inplace=True)
    return data

def load_prices_from_csv_folder(folderpath, identifier_col, tz='UTC'):
    if False:
        for i in range(10):
            print('nop')
    data = None
    for file in os.listdir(folderpath):
        if '.csv' not in file:
            continue
        raw = load_prices_from_csv(os.path.join(folderpath, file), identifier_col, tz)
        if data is None:
            data = raw
        else:
            data = pd.concat([data, raw], axis=1)
    return data