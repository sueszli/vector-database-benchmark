import random
from typing import List, Union
import numpy as np
import pandas as pd

class _SnowparkDataLikeBaseClass:

    def __init__(self, is_map: bool=False, num_of_rows: int=50000, num_of_cols: int=4):
        if False:
            i = 10
            return i + 15
        self._data: Union[pd.DataFrame, List[List[int]], None] = None
        self._is_map = is_map
        self._num_of_rows = num_of_rows
        self._num_of_cols = num_of_cols

    def count(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._num_of_rows

    def take(self, n: int):
        if False:
            for i in range(10):
                print('nop')
        'Returns n element of fake Data like, which imitates take of snowflake.snowpark.dataframe.DataFrame'
        self._lazy_evaluation()
        if n > self._num_of_rows:
            n = self._num_of_rows
        assert self._data is not None
        return self._data[:n]

    def collect(self) -> List[List[int]]:
        if False:
            return 10
        'Returns fake Data like, which imitates collection of snowflake.snowpark.dataframe.DataFrame'
        self._lazy_evaluation()
        assert self._data is not None
        return self._data

    def _lazy_evaluation(self):
        if False:
            print('Hello World!')
        "Sometimes we don't need data inside Data like class, so we populate it once and only when necessary"
        if self._data is None:
            if self._is_map:
                self._data = pd.DataFrame(np.random.randn(self._num_of_rows, 2) / [50, 50] + [37.76, -122.4], columns=['lat', 'lon'])
            else:
                random.seed(0)
                self._data = self._random_data()

    def _random_data(self) -> List[List[int]]:
        if False:
            for i in range(10):
                print('nop')
        data: List[List[int]] = []
        for _ in range(0, self._num_of_rows):
            data.append(self._random_row())
        return data

    def _random_row(self) -> List[int]:
        if False:
            while True:
                i = 10
        row: List[int] = []
        for _ in range(0, self._num_of_cols):
            row.append(random.randint(1, 1000000))
        return row

class DataFrame(_SnowparkDataLikeBaseClass):
    """This is dummy DataFrame class,
    which imitates snowflake.snowpark.dataframe.DataFrame class
    for testing purposes."""
    __module__ = 'snowflake.snowpark.dataframe'

    def __init__(self, is_map: bool=False, num_of_rows: int=50000, num_of_cols: int=4):
        if False:
            return 10
        super(DataFrame, self).__init__(is_map=is_map, num_of_rows=num_of_rows, num_of_cols=num_of_cols)

class Table(_SnowparkDataLikeBaseClass):
    """This is dummy Table class,
    which imitates snowflake.snowpark.dataframe.DataFrame class
    for testing purposes."""
    __module__ = 'snowflake.snowpark.table'

    def __init__(self, is_map: bool=False, num_of_rows: int=50000, num_of_cols: int=4):
        if False:
            return 10
        super(Table, self).__init__(is_map=is_map, num_of_rows=num_of_rows, num_of_cols=num_of_cols)

class Row:
    """This is dummy Row class,
    which imitates snowflake.snowpark.row.Row class
    for testing purposes."""
    __module__ = 'snowflake.snowpark.row'