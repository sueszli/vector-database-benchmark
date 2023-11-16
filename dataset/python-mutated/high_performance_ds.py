from __future__ import annotations
import inspect
import logging
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Text, Union, cast
import numpy as np
import pandas as pd
import qlib.utils.index_data as idd
from ..log import get_module_logger
from ..utils.index_data import IndexData, SingleData
from ..utils.resam import resam_ts_data, ts_data_last
from ..utils.time import Freq, is_single_value

class BaseQuote:

    def __init__(self, quote_df: pd.DataFrame, freq: str) -> None:
        if False:
            i = 10
            return i + 15
        self.logger = get_module_logger('online operator', level=logging.INFO)

    def get_all_stock(self) -> Iterable:
        if False:
            print('Hello World!')
        'return all stock codes\n\n        Return\n        ------\n        Iterable\n            all stock codes\n        '
        raise NotImplementedError(f'Please implement the `get_all_stock` method')

    def get_data(self, stock_id: str, start_time: Union[pd.Timestamp, str], end_time: Union[pd.Timestamp, str], field: Union[str], method: Optional[str]=None) -> Union[None, int, float, bool, IndexData]:
        if False:
            i = 10
            return i + 15
        'get the specific field of stock data during start time and end_time,\n           and apply method to the data.\n\n           Example:\n            .. code-block::\n                                        $close      $volume\n                instrument  datetime\n                SH600000    2010-01-04  86.778313   16162960.0\n                            2010-01-05  87.433578   28117442.0\n                            2010-01-06  85.713585   23632884.0\n                            2010-01-07  83.788803   20813402.0\n                            2010-01-08  84.730675   16044853.0\n\n                SH600655    2010-01-04  2699.567383  158193.328125\n                            2010-01-08  2612.359619   77501.406250\n                            2010-01-11  2712.982422  160852.390625\n                            2010-01-12  2788.688232  164587.937500\n                            2010-01-13  2790.604004  145460.453125\n\n                this function is used for three case:\n\n                1. method is not None. It returns int/float/bool/None.\n                    - It will return None in one case, the method return None\n\n                    print(get_data(stock_id="SH600000", start_time="2010-01-04", end_time="2010-01-06", field="$close", method="last"))\n\n                    85.713585\n\n                2. method is None. It returns IndexData.\n                    print(get_data(stock_id="SH600000", start_time="2010-01-04", end_time="2010-01-06", field="$close", method=None))\n\n                    IndexData([86.778313, 87.433578, 85.713585], [2010-01-04, 2010-01-05, 2010-01-06])\n\n        Parameters\n        ----------\n        stock_id: str\n        start_time : Union[pd.Timestamp, str]\n            closed start time for backtest\n        end_time : Union[pd.Timestamp, str]\n            closed end time for backtest\n        field : str\n            the columns of data to fetch\n        method : Union[str, None]\n            the method apply to data.\n            e.g [None, "last", "all", "sum", "mean", "ts_data_last"]\n\n        Return\n        ----------\n        Union[None, int, float, bool, IndexData]\n            it will return None in following cases\n            - There is no stock data which meet the query criterion from data source.\n            - The `method` returns None\n        '
        raise NotImplementedError(f'Please implement the `get_data` method')

class PandasQuote(BaseQuote):

    def __init__(self, quote_df: pd.DataFrame, freq: str) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(quote_df=quote_df, freq=freq)
        quote_dict = {}
        for (stock_id, stock_val) in quote_df.groupby(level='instrument'):
            quote_dict[stock_id] = stock_val.droplevel(level='instrument')
        self.data = quote_dict

    def get_all_stock(self):
        if False:
            i = 10
            return i + 15
        return self.data.keys()

    def get_data(self, stock_id, start_time, end_time, field, method=None):
        if False:
            i = 10
            return i + 15
        if method == 'ts_data_last':
            method = ts_data_last
        stock_data = resam_ts_data(self.data[stock_id][field], start_time, end_time, method=method)
        if stock_data is None:
            return None
        elif isinstance(stock_data, (bool, np.bool_, int, float, np.number)):
            return stock_data
        elif isinstance(stock_data, pd.Series):
            return idd.SingleData(stock_data)
        else:
            raise ValueError(f'stock data from resam_ts_data must be a number, pd.Series or pd.DataFrame')

class NumpyQuote(BaseQuote):

    def __init__(self, quote_df: pd.DataFrame, freq: str, region: str='cn') -> None:
        if False:
            i = 10
            return i + 15
        'NumpyQuote\n\n        Parameters\n        ----------\n        quote_df : pd.DataFrame\n            the init dataframe from qlib.\n        self.data : Dict(stock_id, IndexData.DataFrame)\n        '
        super().__init__(quote_df=quote_df, freq=freq)
        quote_dict = {}
        for (stock_id, stock_val) in quote_df.groupby(level='instrument'):
            quote_dict[stock_id] = idd.MultiData(stock_val.droplevel(level='instrument'))
            quote_dict[stock_id].sort_index()
        self.data = quote_dict
        (n, unit) = Freq.parse(freq)
        if unit in Freq.SUPPORT_CAL_LIST:
            self.freq = Freq.get_timedelta(1, unit)
        else:
            raise ValueError(f'{freq} is not supported in NumpyQuote')
        self.region = region

    def get_all_stock(self):
        if False:
            i = 10
            return i + 15
        return self.data.keys()

    @lru_cache(maxsize=512)
    def get_data(self, stock_id, start_time, end_time, field, method=None):
        if False:
            print('Hello World!')
        if stock_id not in self.get_all_stock():
            return None
        if is_single_value(start_time, end_time, self.freq, self.region):
            try:
                return self.data[stock_id].loc[start_time, field]
            except KeyError:
                return None
        else:
            data = self.data[stock_id].loc[start_time:end_time, field]
            if data.empty:
                return None
            if method is not None:
                data = self._agg_data(data, method)
            return data

    @staticmethod
    def _agg_data(data: IndexData, method: str) -> Union[IndexData, np.ndarray, None]:
        if False:
            print('Hello World!')
        'Agg data by specific method.'
        if method == 'sum':
            return np.nansum(data)
        elif method == 'mean':
            return np.nanmean(data)
        elif method == 'last':
            return data[-1]
        elif method == 'all':
            return data.all()
        elif method == 'ts_data_last':
            valid_data = data.loc[~data.isna().data.astype(bool)]
            if len(valid_data) == 0:
                return None
            else:
                return valid_data.iloc[-1]
        else:
            raise ValueError(f'{method} is not supported')

class BaseSingleMetric:
    """
    The data structure of the single metric.
    The following methods are used for computing metrics in one indicator.
    """

    def __init__(self, metric: Union[dict, pd.Series]):
        if False:
            while True:
                i = 10
        'Single data structure for each metric.\n\n        Parameters\n        ----------\n        metric : Union[dict, pd.Series]\n            keys/index is stock_id, value is the metric value.\n            for example:\n                SH600068    NaN\n                SH600079    1.0\n                SH600266    NaN\n                           ...\n                SZ300692    NaN\n                SZ300719    NaN,\n        '
        raise NotImplementedError(f'Please implement the `__init__` method')

    def __add__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        if False:
            print('Hello World!')
        raise NotImplementedError(f'Please implement the `__add__` method')

    def __radd__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        if False:
            while True:
                i = 10
        return self + other

    def __sub__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError(f'Please implement the `__sub__` method')

    def __rsub__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        if False:
            while True:
                i = 10
        raise NotImplementedError(f'Please implement the `__rsub__` method')

    def __mul__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError(f'Please implement the `__mul__` method')

    def __truediv__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        if False:
            return 10
        raise NotImplementedError(f'Please implement the `__truediv__` method')

    def __eq__(self, other: object) -> BaseSingleMetric:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError(f'Please implement the `__eq__` method')

    def __gt__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        if False:
            return 10
        raise NotImplementedError(f'Please implement the `__gt__` method')

    def __lt__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        if False:
            while True:
                i = 10
        raise NotImplementedError(f'Please implement the `__lt__` method')

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError(f'Please implement the `__len__` method')

    def sum(self) -> float:
        if False:
            while True:
                i = 10
        raise NotImplementedError(f'Please implement the `sum` method')

    def mean(self) -> float:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError(f'Please implement the `mean` method')

    def count(self) -> int:
        if False:
            return 10
        'Return the count of the single metric, NaN is not included.'
        raise NotImplementedError(f'Please implement the `count` method')

    def abs(self) -> BaseSingleMetric:
        if False:
            while True:
                i = 10
        raise NotImplementedError(f'Please implement the `abs` method')

    @property
    def empty(self) -> bool:
        if False:
            print('Hello World!')
        'If metric is empty, return True.'
        raise NotImplementedError(f'Please implement the `empty` method')

    def add(self, other: BaseSingleMetric, fill_value: float=None) -> BaseSingleMetric:
        if False:
            print('Hello World!')
        'Replace np.NaN with fill_value in two metrics and add them.'
        raise NotImplementedError(f'Please implement the `add` method')

    def replace(self, replace_dict: dict) -> BaseSingleMetric:
        if False:
            for i in range(10):
                print('nop')
        'Replace the value of metric according to replace_dict.'
        raise NotImplementedError(f'Please implement the `replace` method')

    def apply(self, func: Callable) -> BaseSingleMetric:
        if False:
            while True:
                i = 10
        'Replace the value of metric with func (metric).\n        Currently, the func is only qlib/backtest/order/Order.parse_dir.\n        '
        raise NotImplementedError(f"Please implement the 'apply' method")

class BaseOrderIndicator:
    """
    The data structure of order indicator.
    !!!NOTE: There are two ways to organize the data structure. Please choose a better way.
        1. One way is using BaseSingleMetric to represent each metric. For example, the data
        structure of PandasOrderIndicator is Dict[str, PandasSingleMetric]. It uses
        PandasSingleMetric based on pd.Series to represent each metric.
        2. The another way doesn't use BaseSingleMetric to represent each metric. The data
        structure of PandasOrderIndicator is a whole matrix. It means you are not necessary
        to inherit the BaseSingleMetric.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.data = {}
        self.logger = get_module_logger('online operator')

    def assign(self, col: str, metric: Union[dict, pd.Series]) -> None:
        if False:
            return 10
        'assign one metric.\n\n        Parameters\n        ----------\n        col : str\n            the metric name of one metric.\n        metric : Union[dict, pd.Series]\n            one metric with stock_id index, such as deal_amount, ffr, etc.\n            for example:\n                SH600068    NaN\n                SH600079    1.0\n                SH600266    NaN\n                           ...\n                SZ300692    NaN\n                SZ300719    NaN,\n        '
        raise NotImplementedError(f"Please implement the 'assign' method")

    def transfer(self, func: Callable, new_col: str=None) -> Optional[BaseSingleMetric]:
        if False:
            for i in range(10):
                print('nop')
        'compute new metric with existing metrics.\n\n        Parameters\n        ----------\n        func : Callable\n            the func of computing new metric.\n            the kwargs of func will be replaced with metric data by name in this function.\n            e.g.\n                def func(pa):\n                    return (pa > 0).sum() / pa.count()\n        new_col : str, optional\n            New metric will be assigned in the data if new_col is not None, by default None.\n\n        Return\n        ----------\n        BaseSingleMetric\n            new metric.\n        '
        func_sig = inspect.signature(func).parameters.keys()
        func_kwargs = {sig: self.data[sig] for sig in func_sig}
        tmp_metric = func(**func_kwargs)
        if new_col is not None:
            self.data[new_col] = tmp_metric
            return None
        else:
            return tmp_metric

    def get_metric_series(self, metric: str) -> pd.Series:
        if False:
            print('Hello World!')
        'return the single metric with pd.Series format.\n\n        Parameters\n        ----------\n        metric : str\n            the metric name.\n\n        Return\n        ----------\n        pd.Series\n            the single metric.\n            If there is no metric name in the data, return pd.Series().\n        '
        raise NotImplementedError(f"Please implement the 'get_metric_series' method")

    def get_index_data(self, metric: str) -> SingleData:
        if False:
            while True:
                i = 10
        'get one metric with the format of SingleData\n\n        Parameters\n        ----------\n        metric : str\n            the metric name.\n\n        Return\n        ------\n        IndexData.Series\n            one metric with the format of SingleData\n        '
        raise NotImplementedError(f"Please implement the 'get_index_data' method")

    @staticmethod
    def sum_all_indicators(order_indicator: BaseOrderIndicator, indicators: List[BaseOrderIndicator], metrics: Union[str, List[str]], fill_value: float=0) -> None:
        if False:
            return 10
        'sum indicators with the same metrics.\n        and assign to the order_indicator(BaseOrderIndicator).\n        NOTE: indicators could be a empty list when orders in lower level all fail.\n\n        Parameters\n        ----------\n        order_indicator : BaseOrderIndicator\n            the order indicator to assign.\n        indicators : List[BaseOrderIndicator]\n            the list of all inner indicators.\n        metrics : Union[str, List[str]]\n            all metrics needs to be sumed.\n        fill_value : float, optional\n            fill np.NaN with value. By default None.\n        '
        raise NotImplementedError(f"Please implement the 'sum_all_indicators' method")

    def to_series(self) -> Dict[Text, pd.Series]:
        if False:
            return 10
        'return the metrics as pandas series\n\n        for example: { "ffr":\n                SH600068    NaN\n                SH600079    1.0\n                SH600266    NaN\n                           ...\n                SZ300692    NaN\n                SZ300719    NaN,\n                ...\n         }\n        '
        raise NotImplementedError(f'Please implement the `to_series` method')

class SingleMetric(BaseSingleMetric):

    def __init__(self, metric):
        if False:
            i = 10
            return i + 15
        self.metric = metric

    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, (int, float)):
            return self.__class__(self.metric + other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric + other.metric)
        else:
            return NotImplemented

    def __sub__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, (int, float)):
            return self.__class__(self.metric - other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric - other.metric)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if False:
            return 10
        if isinstance(other, (int, float)):
            return self.__class__(other - self.metric)
        elif isinstance(other, self.__class__):
            return self.__class__(other.metric - self.metric)
        else:
            return NotImplemented

    def __mul__(self, other):
        if False:
            return 10
        if isinstance(other, (int, float)):
            return self.__class__(self.metric * other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric * other.metric)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, (int, float)):
            return self.__class__(self.metric / other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric / other.metric)
        else:
            return NotImplemented

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, (int, float)):
            return self.__class__(self.metric == other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric == other.metric)
        else:
            return NotImplemented

    def __gt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, (int, float)):
            return self.__class__(self.metric > other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric > other.metric)
        else:
            return NotImplemented

    def __lt__(self, other):
        if False:
            return 10
        if isinstance(other, (int, float)):
            return self.__class__(self.metric < other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric < other.metric)
        else:
            return NotImplemented

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.metric)

class PandasSingleMetric(SingleMetric):
    """Each SingleMetric is based on pd.Series."""

    def __init__(self, metric: Union[dict, pd.Series]={}):
        if False:
            while True:
                i = 10
        if isinstance(metric, dict):
            self.metric = pd.Series(metric)
        elif isinstance(metric, pd.Series):
            self.metric = metric
        else:
            raise ValueError(f'metric must be dict or pd.Series')

    def sum(self):
        if False:
            return 10
        return self.metric.sum()

    def mean(self):
        if False:
            while True:
                i = 10
        return self.metric.mean()

    def count(self):
        if False:
            return 10
        return self.metric.count()

    def abs(self):
        if False:
            while True:
                i = 10
        return self.__class__(self.metric.abs())

    @property
    def empty(self):
        if False:
            print('Hello World!')
        return self.metric.empty

    @property
    def index(self):
        if False:
            return 10
        return list(self.metric.index)

    def add(self, other: BaseSingleMetric, fill_value: float=None) -> PandasSingleMetric:
        if False:
            i = 10
            return i + 15
        other = cast(PandasSingleMetric, other)
        return self.__class__(self.metric.add(other.metric, fill_value=fill_value))

    def replace(self, replace_dict: dict) -> PandasSingleMetric:
        if False:
            i = 10
            return i + 15
        return self.__class__(self.metric.replace(replace_dict))

    def apply(self, func: Callable) -> PandasSingleMetric:
        if False:
            for i in range(10):
                print('nop')
        return self.__class__(self.metric.apply(func))

    def reindex(self, index: Any, fill_value: float) -> PandasSingleMetric:
        if False:
            i = 10
            return i + 15
        return self.__class__(self.metric.reindex(index, fill_value=fill_value))

    def __repr__(self):
        if False:
            return 10
        return repr(self.metric)

class PandasOrderIndicator(BaseOrderIndicator):
    """
    The data structure is OrderedDict(str: PandasSingleMetric).
    Each PandasSingleMetric based on pd.Series is one metric.
    Str is the name of metric.
    """

    def __init__(self) -> None:
        if False:
            return 10
        super(PandasOrderIndicator, self).__init__()
        self.data: Dict[str, PandasSingleMetric] = OrderedDict()

    def assign(self, col: str, metric: Union[dict, pd.Series]) -> None:
        if False:
            print('Hello World!')
        self.data[col] = PandasSingleMetric(metric)

    def get_index_data(self, metric: str) -> SingleData:
        if False:
            return 10
        if metric in self.data:
            return idd.SingleData(self.data[metric].metric)
        else:
            return idd.SingleData()

    def get_metric_series(self, metric: str) -> Union[pd.Series]:
        if False:
            i = 10
            return i + 15
        if metric in self.data:
            return self.data[metric].metric
        else:
            return pd.Series()

    def to_series(self):
        if False:
            print('Hello World!')
        return {k: v.metric for (k, v) in self.data.items()}

    @staticmethod
    def sum_all_indicators(order_indicator: BaseOrderIndicator, indicators: List[BaseOrderIndicator], metrics: Union[str, List[str]], fill_value: float=0) -> None:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(metrics, str):
            metrics = [metrics]
        for metric in metrics:
            tmp_metric = PandasSingleMetric({})
            for indicator in indicators:
                tmp_metric = tmp_metric.add(indicator.data[metric], fill_value)
            order_indicator.assign(metric, tmp_metric.metric)

    def __repr__(self):
        if False:
            print('Hello World!')
        return repr(self.data)

class NumpyOrderIndicator(BaseOrderIndicator):
    """
    The data structure is OrderedDict(str: SingleData).
    Each idd.SingleData is one metric.
    Str is the name of metric.
    """

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super(NumpyOrderIndicator, self).__init__()
        self.data: Dict[str, SingleData] = OrderedDict()

    def assign(self, col: str, metric: dict) -> None:
        if False:
            while True:
                i = 10
        self.data[col] = idd.SingleData(metric)

    def get_index_data(self, metric: str) -> SingleData:
        if False:
            for i in range(10):
                print('nop')
        if metric in self.data:
            return self.data[metric]
        else:
            return idd.SingleData()

    def get_metric_series(self, metric: str) -> Union[pd.Series]:
        if False:
            while True:
                i = 10
        return self.data[metric].to_series()

    def to_series(self) -> Dict[str, pd.Series]:
        if False:
            while True:
                i = 10
        tmp_metric_dict = {}
        for metric in self.data:
            tmp_metric_dict[metric] = self.get_metric_series(metric)
        return tmp_metric_dict

    @staticmethod
    def sum_all_indicators(order_indicator: BaseOrderIndicator, indicators: List[BaseOrderIndicator], metrics: Union[str, List[str]], fill_value: float=0) -> None:
        if False:
            while True:
                i = 10
        stock_set: set = set()
        for indicator in indicators:
            stock_set = stock_set | set(indicator.data[metrics[0]].index.tolist())
        stocks = sorted(list(stock_set))
        if isinstance(metrics, str):
            metrics = [metrics]
        for metric in metrics:
            order_indicator.data[metric] = idd.sum_by_index([indicator.data[metric] for indicator in indicators], stocks, fill_value)

    def __repr__(self):
        if False:
            return 10
        return repr(self.data)