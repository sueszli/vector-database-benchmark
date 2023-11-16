"""
Fittable Data Transformer Base Class
------------------------------------
"""
from abc import abstractmethod
from typing import Any, Generator, List, Mapping, Optional, Sequence, Union
import numpy as np
from darts import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not
from darts.utils import _build_tqdm_iterator, _parallel_apply
from .base_data_transformer import BaseDataTransformer
logger = get_logger(__name__)

class FittableDataTransformer(BaseDataTransformer):

    def __init__(self, name: str='FittableDataTransformer', n_jobs: int=1, verbose: bool=False, parallel_params: Union[bool, Sequence[str]]=False, mask_components: bool=True, global_fit: bool=False):
        if False:
            return 10
        "Base class for fittable transformers.\n\n        All the deriving classes have to implement the static methods\n        :func:`ts_transform()` and :func:`ts_fit()`. The fitting and transformation functions must\n        be passed during the transformer's initialization. This class takes care of parallelizing\n        operations involving multiple ``TimeSeries`` when possible.\n\n        Parameters\n        ----------\n        name\n            The data transformer's name\n        n_jobs\n            The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is\n            passed as input to a method, parallelising operations regarding different `TimeSeries`. Defaults to `1`\n            (sequential). Setting the parameter to `-1` means using all the available processors.\n            Note: for a small amount of data, the parallelisation overhead could end up increasing the total\n            required amount of time.\n        verbose\n            Optionally, whether to print operations progress\n        parallel_params\n            Optionally, specifies which fixed parameters (i.e. the attributes initialized in the child-most\n            class's `__init__`) take on different values for different parallel jobs. Fixed parameters specified\n            by `parallel_params` are assumed to be a `Sequence` of values that should be used for that parameter\n            in each parallel job; the length of this `Sequence` should equal the number of parallel jobs. If\n            `parallel_params=True`, every fixed parameter will take on a different value for each\n            parallel job. If `parallel_params=False`, every fixed parameter will take on the same value for\n            each parallel job. If `parallel_params` is a `Sequence` of fixed attribute names, only those\n            attribute names specified will take on different values between different parallel jobs.\n        mask_components\n            Optionally, whether or not to automatically apply any provided `component_mask`s to the\n            `TimeSeries` inputs passed to `transform`, `fit`, `inverse_transform`, or `fit_transform`.\n            If `True`, any specified `component_mask` will be applied to each input timeseries\n            before passing them to the called method; the masked components will also be automatically\n            'unmasked' in the returned `TimeSeries`. If `False`, then `component_mask` (if provided) will\n            be passed as a keyword argument, but won't automatically be applied to the input timeseries.\n            See `apply_component_mask` method of `BaseDataTransformer` for further details.\n        global_fit\n            Optionally, whether all of the `TimeSeries` passed to the `fit()` method should be used to fit\n            a *single* set of parameters, or if a different set of parameters should be independently fitted\n            to each provided `TimeSeries`. If `True`, then a `Sequence[TimeSeries]` is passed to `ts_fit`\n            and a single set of parameters is fitted using all of the provided `TimeSeries`. If `False`, then\n            each `TimeSeries` is individually passed to `ts_fit`, and a different set of fitted parameters\n            if yielded for each of these fitting operations. See `ts_fit` for further details.\n\n        Notes\n        -----\n        If `global_fit` is `False` and `fit` is called with a `Sequence` containing `n` different `TimeSeries`,\n        then `n` sets of parameters will be fitted. When `transform` and/or `inverse_transform` is subsequently\n        called with a `Series[TimeSeries]`, the `i`th set of fitted parameter values will be passed to\n        `ts_transform`/`ts_inverse_transform` to transform the `i`th `TimeSeries` in this sequence. Conversely,\n        if `global_fit` is `True`, then only a single set of fitted values will be produced when `fit` is\n        provided with a `Sequence[TimeSeries]`. Consequently, if a `Sequence[TimeSeries]` is then passed to\n        `transform`/`inverse_transform`, each of these `TimeSeries` will be transformed using the exact same set\n        of fitted parameters.\n\n        Note that if an invertible *and* fittable data transformer is to be globally fitted, the data transformer\n        class should first inherit from `FittableDataTransformer` and then from `InveritibleDataTransformer`. In\n        other words, `MyTransformer(FittableDataTransformer, InveritibleDataTransformer)` is correct, but\n        `MyTransformer(InveritibleDataTransformer, FittableDataTransformer)` is **not**. If this is not implemented\n        correctly, then the `global_fit` parameter will not be correctly passed to `FittableDataTransformer`'s\n        constructor.\n\n        The :func:`ts_transform()` and :func:`ts_fit()` methods are designed to be static methods instead of instance\n        methods to allow an efficient parallelisation also when the scaler instance is storing a non-negligible\n        amount of data. Using instance methods would imply copying the instance's data through multiple processes, which\n        can easily introduce a bottleneck and nullify parallelisation benefits.\n\n        Example\n        --------\n        >>> from darts.dataprocessing.transformers import FittableDataTransformer\n        >>> from darts.utils.timeseries_generation import linear_timeseries\n        >>>\n        >>> class SimpleRangeScaler(FittableDataTransformer):\n        >>>\n        >>>     def __init__(self, scale, position):\n        >>>         self._scale = scale\n        >>>         self._position = position\n        >>>         super().__init__()\n        >>>\n        >>>     @staticmethod\n        >>>     def ts_transform(series, params):\n        >>>         vals = series.all_values(copy=False)\n        >>>         fit_params = params['fitted']\n        >>>         unit_scale = (vals - fit_params['position'])/fit_params['scale']\n        >>>         fix_params = params['fixed']\n        >>>         rescaled = fix_params['_scale'] * unit_scale + fix_params['_position']\n        >>>         return series.from_values(rescaled)\n        >>>\n        >>>     @staticmethod\n        >>>     def ts_fit(series, params):\n        >>>         vals = series.all_values(copy=False)\n        >>>         scale = vals.max() - vals.min()\n        >>>         position = vals[0]\n        >>>         return {'scale': scale, 'position': position}\n        >>>\n        >>> series = linear_timeseries(length=5, start_value=1, end_value=5)\n        >>> print(series)\n        <TimeSeries (DataArray) (time: 5, component: 1, sample: 1)>\n        array([[[1.]],\n\n            [[2.]],\n\n            [[3.]],\n\n            [[4.]],\n\n            [[5.]]])\n        Coordinates:\n        * time       (time) datetime64[ns] 2000-01-01 2000-01-02 ... 2000-01-05\n        * component  (component) object 'linear'\n        Dimensions without coordinates: sample\n        Attributes:\n            static_covariates:  None\n            hierarchy:          None\n        >>> series = SimpleRangeScaler(scale=2, position=-1).fit_transform(series)\n        >>> print(series)\n        <TimeSeries (DataArray) (time: 5, component: 1, sample: 1)>\n        array([[[-1. ]],\n\n            [[-0.5]],\n\n            [[ 0. ]],\n\n            [[ 0.5]],\n\n            [[ 1. ]]])\n        Coordinates:\n        * time       (time) int64 0 1 2 3 4\n        * component  (component) <U1 '0'\n        Dimensions without coordinates: sample\n        Attributes:\n            static_covariates:  None\n            hierarchy:          None\n        "
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose, parallel_params=parallel_params, mask_components=mask_components)
        self._fit_called = False
        self._fitted_params = None
        self._global_fit = global_fit

    @staticmethod
    @abstractmethod
    def ts_fit(series: Union[TimeSeries, Sequence[TimeSeries]], params: Mapping[str, Any], *args, **kwargs):
        if False:
            print('Hello World!')
        "The function that will be applied to each series when :func:`fit` is called.\n\n        If the `global_fit` attribute is set to `False`, then `ts_fit` should accept a `TimeSeries` as a first\n        argument and return a set of parameters that are fitted to this individual `TimeSeries`. Conversely, if the\n        `global_fit` attribute is set to `True`, then `ts_fit` should accept a `Sequence[TimeSeries]` and\n        return a set of parameters that are fitted to *all* of the provided `TimeSeries`. All these parameters will\n        be stored in ``self._fitted_params``, which can be later used during the transformation step.\n\n        Regardless of whether the `global_fit` attribute is set to `True` or `False`, `ts_fit` should also accept\n        a dictionary of fixed parameter values as a second argument (i.e. `params['fixed'] contains the fixed\n        parameters of the data transformer).\n\n        Any additional positional and/or keyword arguments passed to the `fit` method will be passed as\n        positional/keyword arguments to `ts_fit`.\n\n        This method is not implemented in the base class and must be implemented in the deriving classes.\n\n        If more parameters are added as input in the derived classes, :func:`_fit_iterator()`\n        should be redefined accordingly, to yield the necessary arguments to this function (See\n        :func:`_fit_iterator()` for further details)\n\n        Parameters\n        ----------\n        series (Union[TimeSeries, Sequence[TimeSeries]])\n            `TimeSeries` against which the scaler will be fit.\n\n        Notes\n        -----\n        This method is designed to be a static method instead of instance methods to allow an efficient\n        parallelisation also when the scaler instance is storing a non-negligible amount of data. Using instance\n        methods would imply copying the instance's data through multiple processes, which can easily introduce a\n        bottleneck and nullify parallelisation benefits.\n        "
        pass

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]], *args, component_mask: Optional[np.array]=None, **kwargs) -> 'FittableDataTransformer':
        if False:
            while True:
                i = 10
        'Fits transformer to a (sequence of) `TimeSeries` by calling the user-implemented `ts_fit` method.\n\n        The fitted parameters returned by `ts_fit` are stored in the ``self._fitted_params`` attribute.\n        If a `Sequence[TimeSeries]` is passed as the `series` data, then one of two outcomes will occur:\n            1. If the `global_fit` attribute was set to `False`, then a different set of parameters will be\n            individually fitted to each `TimeSeries` in the `Sequence`. In this case, this function automatically\n            parallelises this fitting process over all of the multiple `TimeSeries` that have been passed.\n            2. If the `global_fit` attribute was set to `True`, then all of the `TimeSeries` objects will be used\n            fit a single set of parameters.\n\n        Parameters\n        ----------\n        series\n            (sequence of) series to fit the transformer on.\n        args\n            Additional positional arguments for the :func:`ts_fit` method\n        component_mask : Optional[np.ndarray] = None\n            Optionally, a 1-D boolean np.ndarray of length ``series.n_components`` that specifies which\n            components of the underlying `series` the transform should be fitted to.\n        kwargs\n            Additional keyword arguments for the :func:`ts_fit` method\n\n        Returns\n        -------\n        FittableDataTransformer\n            Fitted transformer.\n        '
        self._fit_called = True
        desc = f'Fitting ({self._name})'
        if isinstance(series, TimeSeries):
            data = [series]
        else:
            data = series
        if self._mask_components:
            data = [self.apply_component_mask(ts, component_mask, return_ts=True) for ts in data]
        else:
            kwargs['component_mask'] = component_mask
        params_iterator = self._get_params(n_timeseries=len(data), calling_fit=True)
        fit_iterator = zip(data, params_iterator) if not self._global_fit else zip([data], params_iterator)
        n_jobs = len(data) if not self._global_fit else 1
        input_iterator = _build_tqdm_iterator(fit_iterator, verbose=self._verbose, desc=desc, total=n_jobs)
        self._fitted_params = _parallel_apply(input_iterator, self.__class__.ts_fit, self._n_jobs, args, kwargs)
        return self

    def fit_transform(self, series: Union[TimeSeries, Sequence[TimeSeries]], *args, component_mask: Optional[np.array]=None, **kwargs) -> Union[TimeSeries, List[TimeSeries]]:
        if False:
            while True:
                i = 10
        'Fit the transformer to the (sequence of) series and return the transformed input.\n\n        Parameters\n        ----------\n        series\n            the (sequence of) series to transform.\n        args\n            Additional positional arguments passed to the :func:`ts_transform` and :func:`ts_fit` methods.\n        component_mask : Optional[np.ndarray] = None\n            Optionally, a 1-D boolean np.ndarray of length ``series.n_components`` that specifies which\n            components of the underlying `series` the transform should be fitted and applied to.\n        kwargs\n            Additional keyword arguments passed to the :func:`ts_transform` and :func:`ts_fit` methods.\n\n        Returns\n        -------\n        Union[TimeSeries, Sequence[TimeSeries]]\n            Transformed data.\n        '
        return self.fit(series, *args, component_mask=component_mask, **kwargs).transform(series, *args, component_mask=component_mask, **kwargs)

    def _get_params(self, n_timeseries: int, calling_fit: bool=False) -> Generator[Mapping[str, Any], None, None]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overrides `_get_params` of `BaseDataTransformer`. Creates generator of dictionaries containing\n        both the fixed parameter values (i.e. attributes defined in the child-most class), as well as\n        the fitted parameter values (only if `calling_fit = False`). Those fixed parameters\n        specified by `parallel_params` are given different values over each of the parallel jobs;\n        every fitted parameter is given a different value for each parallel job (since `self.fit`\n        returns a `Sequence` containing one fitted parameter value of each parallel job). Called by\n        `transform` and `inverse_transform`.\n        '
        self._check_fixed_params(n_timeseries)
        fitted_params = self._get_fitted_params(n_timeseries, calling_fit)

        def params_generator(n_jobs, fixed_params, fitted_params, parallel_params, global_fit):
            if False:
                i = 10
                return i + 15
            fixed_params_copy = fixed_params.copy()
            for i in range(n_jobs):
                for key in parallel_params:
                    fixed_params_copy[key] = fixed_params[key][i]
                params = {}
                if fixed_params_copy:
                    params['fixed'] = fixed_params_copy
                if fitted_params:
                    params['fitted'] = fitted_params[0] if global_fit else fitted_params[i]
                if not params:
                    params = None
                yield params
        n_jobs = n_timeseries if not (calling_fit and self._global_fit) else 1
        return params_generator(n_jobs, self._fixed_params, fitted_params, self._parallel_params, self._global_fit)

    def _get_fitted_params(self, n_timeseries: int, calling_fit: bool) -> Sequence[Any]:
        if False:
            while True:
                i = 10
        '\n        Returns `self._fitted_params` if `calling_fit = False`, otherwise returns an empty\n        tuple. If `calling_fit = False`, also checks that `self._fitted_params`, which is a\n        sequence of values, contains exactly `n_timeseries` values; if not, a `ValueError` is thrown.\n        '
        if not calling_fit:
            raise_if_not(self._fit_called, 'Must call `fit` before calling `transform`/`inverse_transform`.')
            fitted_params = self._fitted_params
        else:
            fitted_params = tuple()
        if not self._global_fit and fitted_params:
            raise_if(n_timeseries > len(fitted_params), f'{n_timeseries} TimeSeries were provided but only {len(fitted_params)} TimeSeries were specified upon training {self.name}.')
        return fitted_params