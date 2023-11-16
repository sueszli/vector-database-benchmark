from io import StringIO
import pandas as pd

def read_csv(data):
    if False:
        i = 10
        return i + 15
    return pd.read_csv(StringIO(data), sep=';')