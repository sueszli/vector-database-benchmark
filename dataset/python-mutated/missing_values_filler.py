"""
Missing Values Filler
---------------------
"""
from typing import Any, Mapping, Union
from darts import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not
from darts.utils.missing_values import fill_missing_values
from .base_data_transformer import BaseDataTransformer
logger = get_logger(__name__)

class MissingValuesFiller(BaseDataTransformer):

    def __init__(self, fill: Union[str, float]='auto', name: str='MissingValuesFiller', n_jobs: int=1, verbose: bool=False):
        if False:
            i = 10
            return i + 15
        "Data transformer to fill missing values from a (sequence of) deterministic ``TimeSeries``.\n\n        Parameters\n        ----------\n        fill\n            The value used to replace the missing values.\n            If set to 'auto', will auto-fill missing values using the :func:`pd.Dataframe.interpolate()` method.\n        name\n            A specific name for the transformer\n        n_jobs\n            The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is\n            passed as input to a method, parallelising operations regarding different ``TimeSeries``. Defaults to `1`\n            (sequential). Setting the parameter to `-1` means using all the available processors.\n            Note: for a small amount of data, the parallelisation overhead could end up increasing the total\n            required amount of time.\n        verbose\n            Optionally, whether to print operations progress\n\n        Examples\n        --------\n        >>> import numpy as np\n        >>> from darts import TimeSeries\n        >>> from darts.dataprocessing.transformers import MissingValuesFiller\n        >>> values = np.arange(start=0, stop=1, step=0.1)\n        >>> values[5:8] = np.nan\n        >>> series = TimeSeries.from_values(values)\n        >>> transformer = MissingValuesFiller()\n        >>> series_filled = transformer.transform(series)\n        >>> print(series_filled)\n        <TimeSeries (DataArray) (time: 10, component: 1, sample: 1)>\n        array([[[0. ]],\n            [[0.1]],\n            [[0.2]],\n            [[0.3]],\n            [[0.4]],\n            [[0.5]],\n            [[0.6]],\n            [[0.7]],\n            [[0.8]],\n            [[0.9]]])\n        Coordinates:\n        * time       (time) int64 0 1 2 3 4 5 6 7 8 9\n        * component  (component) object '0'\n        Dimensions without coordinates: sample\n        "
        raise_if_not(isinstance(fill, str) or isinstance(fill, float), '`fill` should either be a string or a float', logger)
        raise_if(isinstance(fill, str) and fill != 'auto', "invalid string for `fill`: can only be set to 'auto'", logger)
        self._fill = fill
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose)

    @staticmethod
    def ts_transform(series: TimeSeries, params: Mapping[str, Any], **kwargs) -> TimeSeries:
        if False:
            i = 10
            return i + 15
        return fill_missing_values(series, params['fixed']['_fill'], **kwargs)