iris_data = None
import pandas as pd
from dagster import asset

@asset
def iris_setosa(iris_data: pd.DataFrame) -> pd.DataFrame:
    if False:
        i = 10
        return i + 15
    return iris_data[iris_data['species'] == 'Iris-setosa']