import os

def read_csv_as_df(csv_path):
    if False:
        print('Hello World!')
    import pandas as pd
    data = pd.read_csv(csv_path)
    return data

def get_csv():
    if False:
        return 10
    csv_path = os.path.join(os.path.dirname(__file__), '../IF1706_20161108.csv')
    return read_csv_as_df(csv_path)