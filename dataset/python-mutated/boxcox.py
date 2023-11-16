"""
Box-Cox Transformer
-------------------
"""
from typing import Any, Mapping, Optional, Sequence, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import boxcox, boxcox_normmax
from darts.logging import get_logger, raise_if
from darts.timeseries import TimeSeries
from .fittable_data_transformer import FittableDataTransformer
from .invertible_data_transformer import InvertibleDataTransformer
logger = get_logger(__name__)

class BoxCox(FittableDataTransformer, InvertibleDataTransformer):

    def __init__(self, name: str='BoxCox', lmbda: Optional[Union[float, Sequence[float], Sequence[Sequence[float]]]]=None, optim_method: Literal['mle', 'pearsonr']='mle', global_fit: bool=False, n_jobs: int=1, verbose: bool=False):
        if False:
            print('Hello World!')
        "Box-Cox data transformer.\n\n        See [1]_ for more information about Box-Cox transforms.\n\n        The transformation is applied independently for each dimension (component) of the time series.\n        For stochastic series, it is done jointly over all samples, effectively merging all samples of\n        a component in order to compute the transform.\n\n        Notes\n        -----\n        The scaler will not scale the series' static covariates. This has to be done either before constructing the\n        series, or later on by extracting the covariates, transforming the values and then reapplying them to the\n        series. For this, see TimeSeries properties `TimeSeries.static_covariates` and method\n        `TimeSeries.with_static_covariates()`\n\n        Parameters\n        ----------\n        name\n            A specific name for the transformer\n        lmbda\n            The parameter :math:`\\lambda` of the Box-Cox transform. If a single float is given, the same\n            :math:`\\lambda` value will be used for all dimensions of the series, for all the series.\n            If a sequence is given, there is one value per component in the series. If a sequence of sequence\n            is given, there is one value per component for all series.\n            If `None` given, will automatically find an optimal value of :math:`\\lambda` (for each dimension\n            of the time series, for each time series) using :func:`scipy.stats.boxcox_normmax`\n            with ``method=optim_method``.\n        optim_method\n            Specifies which method to use to find an optimal value for the lmbda parameter.\n            Either 'mle' or 'pearsonr'. Ignored if `lmbda` is not `None`.\n        global_fit\n            Optionally, whether all of the `TimeSeries` passed to the `fit()` method should be used to fit\n            a *single* set of parameters, or if a different set of parameters should be independently fitted\n            to each provided `TimeSeries`. If `True`, then a `Sequence[TimeSeries]` is passed to `ts_fit`\n            and a single set of parameters is fitted using all of the provided `TimeSeries`. If `False`, then\n            each `TimeSeries` is individually passed to `ts_fit`, and a different set of fitted parameters\n            if yielded for each of these fitting operations. See `FittableDataTransformer` documentation for\n            further details.\n        n_jobs\n            The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is\n            passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`\n            (sequential). Setting the parameter to `-1` means using all the available processors.\n            Note: for a small amount of data, the parallelisation overhead could end up increasing the total\n            required amount of time.\n        verbose\n            Whether to print operations progress\n\n        Examples\n        --------\n        >>> from darts.datasets import AirPassengersDataset\n        >>> from darts.dataprocessing.transformers import BoxCox\n        >>> series = AirPassengersDataset().load()\n        >>> transformer = BoxCox(lmbda=0.2)\n        >>> series_transformed = transformer.fit_transform(series)\n        >>> print(series_transformed.head())\n        <TimeSeries (DataArray) (Month: 5, component: 1, sample: 1)>\n        array([[[7.84735157]],\n            [[7.98214351]],\n            [[8.2765364 ]],\n            [[8.21563229]],\n            [[8.04749318]]])\n        Coordinates:\n        * Month      (Month) datetime64[ns] 1949-01-01 1949-02-01 ... 1949-05-01\n        * component  (component) object '#Passengers'\n        Dimensions without coordinates: sample\n\n        References\n        ----------\n        .. [1] https://otexts.com/fpp2/transformations.html#mathematical-transformations\n        "
        raise_if(not isinstance(optim_method, str) or optim_method not in ['mle', 'pearsonr'], "optim_method parameter must be either 'mle' or 'pearsonr'", logger)
        self._lmbda = lmbda
        self._optim_method = optim_method
        if isinstance(lmbda, Sequence) and isinstance(lmbda[0], Sequence):
            parallel_params = ('_lmbda',)
        else:
            parallel_params = False
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose, parallel_params=parallel_params, mask_components=True, global_fit=global_fit)

    @staticmethod
    def ts_fit(series: Union[TimeSeries, Sequence[TimeSeries]], params: Mapping[str, Any], *args, **kwargs) -> Union[Sequence[float], pd.core.series.Series]:
        if False:
            return 10
        (lmbda, method) = (params['fixed']['_lmbda'], params['fixed']['_optim_method'])
        if isinstance(series, TimeSeries):
            series = [series]
        if lmbda is None:
            vals = np.concatenate([BoxCox.stack_samples(ts) for ts in series], axis=0)
            lmbda = np.apply_along_axis(boxcox_normmax, axis=0, arr=vals, method=method)
        elif isinstance(lmbda, Sequence):
            raise_if(len(lmbda) != series[0].width, 'lmbda should have one value per dimension (ie. column or variable) of the time series', logger)
        else:
            lmbda = [lmbda] * series[0].width
        return lmbda

    @staticmethod
    def ts_transform(series: TimeSeries, params: Mapping[str, Any], **kwargs) -> TimeSeries:
        if False:
            for i in range(10):
                print('nop')
        lmbda = params['fitted']
        vals = BoxCox.stack_samples(series)
        transformed_vals = np.stack([boxcox(vals[:, i], lmbda=lmbda[i]) for i in range(series.width)], axis=1)
        transformed_vals = BoxCox.unstack_samples(transformed_vals, series=series)
        return series.with_values(transformed_vals)

    @staticmethod
    def ts_inverse_transform(series: TimeSeries, params: Mapping[str, Any], **kwargs) -> TimeSeries:
        if False:
            print('Hello World!')
        lmbda = params['fitted']
        vals = BoxCox.stack_samples(series)
        inv_transformed_vals = np.stack([inv_boxcox(vals[:, i], lmbda[i]) for i in range(series.width)], axis=1)
        inv_transformed_vals = BoxCox.unstack_samples(inv_transformed_vals, series=series)
        return series.with_values(inv_transformed_vals)