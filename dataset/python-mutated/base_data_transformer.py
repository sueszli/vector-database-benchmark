"""
Data Transformer Base Class
---------------------------
"""
from abc import ABC, abstractmethod
from typing import Any, Generator, List, Mapping, Optional, Sequence, Union
import numpy as np
import xarray as xr
from darts import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not
from darts.utils import _build_tqdm_iterator, _parallel_apply
logger = get_logger(__name__)

class BaseDataTransformer(ABC):

    def __init__(self, name: str='BaseDataTransformer', n_jobs: int=1, verbose: bool=False, parallel_params: Union[bool, Sequence[str]]=False, mask_components: bool=True):
        if False:
            i = 10
            return i + 15
        "Abstract class for data transformers.\n\n        All the deriving classes have to implement the static method :func:`ts_transform`; this implemented\n        method can then be applied to ``TimeSeries`` or ``Sequence[TimeSeries]`` inputs by calling the\n        :func:`transform()` method. Internally, :func:`transform()` parallelizes func:`ts_transform` over all\n        of the ``TimeSeries`` inputs passed to it. See the func:`ts_transform` method documentation\n        for further details on how to implement this method in a user-defined class.\n\n        Data transformers requiring to be fit first before calling :func:`transform` should derive\n        from :class:`.FittableDataTransformer` instead. Data transformers that are invertible should derive\n        from :class:`.InvertibleDataTransformer` instead. Transformers which are both fittable and invertible\n        should inherit from both :class:`.FittableDataTransformer` and :class:`.InvertibleDataTransformer`.\n\n        All Data Transformers can store *fixed parameters* that are automatically passed to func:`ts_transform`;\n        the fixed parameters of a data transformer object are taken to be all those attributes defined\n        in the `__init__` method of the child-most class *before* `super().__init__` is called. The fixed parameter\n        values can then be accessed within the func:`ts_transform` method through the `params` dictionary\n        argument. More specifically, `params['fixed']` stores a dictionary with all of the fixed parameter\n        values, where the keys are simply the attribute names of each fixed parameter (e.g. the `self._my_param`\n        fixed parameter attribute is accessed through `params['fixed']['_my_param']`).\n\n        Data Transformers which inherit from :class:`.FittableDataTransformer` can also store fitted parameters\n        alongside fixed parameters; please refer to the :class:`.FittableDataTransformer` documentation for\n        further details.\n\n        Note: the :func:`ts_transform` method is designed to be a static method instead of a instance method to allow an\n        efficient parallelisation also when the scaler instance is storing a non-negligible amount of data. Using\n        an instance method would imply copying the instance's data through multiple processes, which can easily\n        introduce a bottleneck and nullify parallelisation benefits.\n\n        Parameters\n        ----------\n        name\n            The data transformer's name\n        n_jobs\n            The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]``\n            is passed as input to a method, parallelising operations regarding different TimeSeries.\n            Defaults to `1` (sequential). Setting the parameter to `-1` means using all the available processors.\n            Note: for a small amount of data, the parallelisation overhead could end up increasing the total\n            required amount of time.\n        verbose\n            Optionally, whether to print operations progress\n        parallel_params\n            Optionally, specifies which fixed parameters (i.e. the attributes initialized in the child-most\n            class's `__init__`) take on different values for different parallel jobs. Fixed parameters specified\n            by `parallel_params` are assumed to be a `Sequence` of values that should be used for that parameter\n            in each parallel job; the length of this `Sequence` should equal the number of parallel jobs. If\n            `parallel_params=True`, every fixed parameter will take on a different value for each\n            parallel job. If `parallel_params=False`, every fixed parameter will take on the same value for\n            each parallel job. If `parallel_params` is a `Sequence` of fixed attribute names, only those\n            attribute names specified will take on different values between different parallel jobs.\n        mask_components\n            Optionally, whether or not to automatically apply any provided `component_mask`s to the\n            `TimeSeries` inputs passed to `transform`, `fit`, `inverse_transform`, or `fit_transform`.\n            If `True`, any specified `component_mask` will be applied to each input timeseries\n            before passing them to the called method; the masked components will also be automatically\n            'unmasked' in the returned `TimeSeries`. If `False`, then `component_mask` (if provided) will\n            be passed as a keyword argument, but won't automatically be applied to the input timeseries.\n            See `apply_component_mask` for further details.\n\n        Example\n        --------\n        >>> from darts.dataprocessing.transformers import BaseDataTransformer\n        >>> from darts.utils.timeseries_generation import linear_timeseries\n        >>>\n        >>> class SimpleTransform(BaseDataTransformer):\n        >>>\n        >>>         def __init__(self, a):\n        >>>             self._a = a\n        >>>             super().__init__()\n        >>>\n        >>>         @staticmethod\n        >>>         def ts_transform(series, params, **kwargs):\n        >>>             a = params['fixed']['_a']\n        >>>             b = kwargs.pop('b')\n        >>>             return a*series + b\n        >>>\n        >>> series = linear_timeseries(length=5)\n        >>> print(series)\n        <TimeSeries (DataArray) (time: 5, component: 1, sample: 1)>\n        array([[[0.  ]],\n\n            [[0.25]],\n\n            [[0.5 ]],\n\n            [[0.75]],\n\n            [[1.  ]]])\n        Coordinates:\n        * time       (time) datetime64[ns] 2000-01-01 2000-01-02 ... 2000-01-05\n        * component  (component) object 'linear'\n        Dimensions without coordinates: sample\n        Attributes:\n            static_covariates:  None\n            hierarchy:          None\n        >>> series = SimpleTransform(a=2).transform(series, b=3)\n        >>> print(series)\n        <TimeSeries (DataArray) (time: 5, component: 1, sample: 1)>\n        array([[[3. ]],\n\n            [[3.5]],\n\n            [[4. ]],\n\n            [[4.5]],\n\n            [[5. ]]])\n        Coordinates:\n        * time       (time) datetime64[ns] 2000-01-01 2000-01-02 ... 2000-01-05\n        * component  (component) object 'linear'\n        Dimensions without coordinates: sample\n        Attributes:\n            static_covariates:  None\n            hierarchy:          None\n        "
        self._fixed_params = vars(self).copy()
        if parallel_params and (not isinstance(parallel_params, Sequence)):
            parallel_params = self._fixed_params.keys()
        elif not parallel_params:
            parallel_params = tuple()
        self._parallel_params = parallel_params
        self._mask_components = mask_components
        self._name = name
        self._verbose = verbose
        self._n_jobs = n_jobs

    def set_verbose(self, value: bool):
        if False:
            print('Hello World!')
        "Set the verbosity status.\n\n        `True` for enabling the detailed report about scaler's operation progress,\n        `False` for no additional information.\n\n        Parameters\n        ----------\n        value\n            New verbosity status\n        "
        raise_if_not(isinstance(value, bool), 'Verbosity status must be a boolean.')
        self._verbose = value

    def set_n_jobs(self, value: int):
        if False:
            while True:
                i = 10
        'Set the number of processors to be used by the transformer while processing multiple ``TimeSeries``.\n\n        Parameters\n        ----------\n        value\n            New n_jobs value.  Set to `-1` for using all the available cores.\n        '
        raise_if_not(isinstance(value, int), 'n_jobs must be an integer')
        self._n_jobs = value

    @staticmethod
    @abstractmethod
    def ts_transform(series: TimeSeries, params: Mapping[str, Any]) -> TimeSeries:
        if False:
            return 10
        "The function that will be applied to each series when :func:`transform()` is called.\n\n        This method is not implemented in the base class and must be implemented in the deriving classes.\n\n        The function must take as first argument a ``TimeSeries`` object and, as a second argument, a\n        dictionary containing the fixed and/or fitted parameters of the transformation; this function\n        should then return a transformed ``TimeSeries`` object.\n\n        The `params` dictionary *can* contain up to two keys:\n            1. `params['fixed']` stores the fixed parameters of the transformation (i.e. attributed\n            defined in the `__init__` method of the child-most class *before* `super().__init__` is called);\n            `params['fixed']` is a dictionary itself, whose keys are the names of the fixed parameter\n            attributes. For example, if `_my_fixed_param` is defined as an attribute in the child-most\n            class, then this fixed parameter value can be accessed through `params['fixed']['_my_fixed_param']`.\n            2. If the transform inherits from the :class:`.FittableDataTransformer` class, then `params['fitted']`\n            will store the fitted parameters of the transformation; the fitted parameters are simply the output(s)\n            returned by the `ts_fit` function, whatever those output(s) may be. See :class:`.FittableDataTransformer`\n            for further details about fitted parameters.\n\n        Any positional/keyword argument supplied to the `transform` method are passed as positional/keyword arguments\n        to `ts_transform`; hence, `ts_transform` should also accept `*args` and/or `**kwargs` if positional/keyword\n        arguments are passed to `transform`. Note that if the `mask_components` attribute of `BaseDataTransformer`\n        is set to `False`, then the `component_mask` provided to `transform` will be passed as an additional keyword\n        argument to `ts_transform`.\n\n        The `BaseDataTransformer` class includes some helper methods which may prove useful when implementing a\n        `ts_transform` function:\n            1. The `apply_component_mask` and `unapply_component_mask` methods, which apply and 'unapply'\n            `component_mask`s to a `TimeSeries` respectively; these methods are automatically called in `transform`\n            if the `mask_component` attribute of `BaseDataTransformer` is set to `True`, but you may want to manually\n            call them if you set `mask_components` to `False` and wish to manually specify how `component_mask`s are\n            applied to a `TimeSeries`.\n            2. The `stack_samples` method, which stacks all the samples in a `TimeSeries` along\n            the component axis, so that the `TimeSeries` goes from shape `(n_timesteps, n_components, n_samples)` to\n            shape `(n_timesteps, n_components * n_samples)`. This stacking is useful if a pointwise transform is being\n            implemented (i.e. transforming the value at time `t` depends only on the value of the series at that\n            time `t`). Once transformed, the stacked `TimeSeries` can be 'unstacked' using the `unstack_samples` method.\n\n        Parameters\n        ----------\n        series\n            series to be transformed.\n        params\n            Dictionary containing the parameters of the transformation function. Fixed parameters\n            (i.e. attributes defined in the child-most class of the transformation prior to\n            calling `super.__init__()`) are stored under the `'fixed'` key. If the transformation\n            inherits from the `FittableDataTransformer` class, then the fitted parameters of the\n            transformation (i.e. the values returned by `ts_fit`) are stored under the\n            `'fitted'` key.\n        args\n            Any poisitional arguments provided in addition to `series` when\n        kwargs\n            Any additional keyword arguments provided to `transform`. Note that if the `mask_component`\n            attribute of `BaseDataTransformer` is set to `False`, then `component_mask` will\n            be passed as a keyword argument.\n\n        Notes\n        -----\n        This method is designed to be a static method instead of instance method to allow an efficient\n        parallelisation also when the scaler instance is storing a non-negligible amount of data. Using instance\n        methods would imply copying the instance's data through multiple processes, which can easily introduce a\n        bottleneck and nullify parallelisation benefits.\n        "
        pass

    def transform(self, series: Union[TimeSeries, Sequence[TimeSeries]], *args, component_mask: Optional[np.array]=None, **kwargs) -> Union[TimeSeries, List[TimeSeries]]:
        if False:
            for i in range(10):
                print('nop')
        "Transforms a (sequence of) of series by calling the user-implemeneted `ts_transform` method.\n\n        In case a ``Sequence[TimeSeries]`` is passed as input data, this function takes care of\n        parallelising the transformation of multiple series in the sequence at the same time. Additionally,\n        if the `mask_components` attribute was set to `True` when instantiating `BaseDataTransformer`,\n        then any provided `component_mask`s will be automatically applied to each input `TimeSeries`;\n        please refer to 'Notes' for further details on component masking.\n\n        Any additionally specified `*args` and `**kwargs` are automatically passed to `ts_transform`.\n\n        Parameters\n        ----------\n        series\n            (sequence of) series to be transformed.\n        args\n            Additional positional arguments for each :func:`ts_transform()` method call\n        component_mask : Optional[np.ndarray] = None\n            Optionally, a 1-D boolean np.ndarray of length ``series.n_components`` that specifies which\n            components of the underlying `series` the transform should consider. If the `mask_components`\n            attribute was set to `True` when instantiating `BaseDataTransformer`, then the component mask\n            will be automatically applied to each `TimeSeries` input. Otherwise, `component_mask` will be\n            provided as an addition keyword argument to `ts_transform`. See 'Notes' for further details.\n        kwargs\n            Additional keyword arguments for each :func:`ts_transform()` method call\n\n        Returns\n        -------\n        Union[TimeSeries, List[TimeSeries]]\n            Transformed data.\n\n        Notes\n        -----\n        If the `mask_components` attribute was set to `True` when instantiating `BaseDataTransformer`,\n        then any provided `component_mask`s will be automatically applied to each `TimeSeries` input to\n        transform; `component_mask`s are simply boolean arrays of shape `(series.n_components,)` that\n        specify which components of each `series` should be transformed using `ts_transform` and which\n        components should not. If `component_mask[i]` is `True`, then the `i`th component of each\n        `series` will be transformed by `ts_transform`. Conversely, if `component_mask[i]` is `False`,\n        the `i`th component will be removed from each `series` before being passed to `ts_transform`;\n        after transforming this masked series, the untransformed `i`th component will be 'added back'\n        to the output. Note that automatic `component_mask`ing can only be performed if the `ts_transform`\n        does *not* change the number of timesteps in each series; if this were to happen, then the transformed\n        and untransformed components are unable to be concatenated back together along the component axis.\n\n        If `mask_components` was set to `False` when instantiating `BaseDataTransformer`, then any provided\n        `component_masks` will be passed as a keyword argument `ts_transform`; the user can then manually specify\n        how the `component_mask` should be applied to each series.\n        "
        desc = f'Transform ({self._name})'
        if isinstance(series, TimeSeries):
            input_series = [series]
            data = [series]
        else:
            input_series = series
            data = series
        if self._mask_components:
            data = [self.apply_component_mask(ts, component_mask, return_ts=True) for ts in data]
        else:
            kwargs['component_mask'] = component_mask
        input_iterator = _build_tqdm_iterator(zip(data, self._get_params(n_timeseries=len(data))), verbose=self._verbose, desc=desc, total=len(data))
        transformed_data = _parallel_apply(input_iterator, self.__class__.ts_transform, self._n_jobs, args, kwargs)
        if self._mask_components:
            unmasked = []
            for (ts, transformed_ts) in zip(input_series, transformed_data):
                unmasked.append(self.unapply_component_mask(ts, transformed_ts, component_mask))
            transformed_data = unmasked
        return transformed_data[0] if isinstance(series, TimeSeries) else transformed_data

    def _get_params(self, n_timeseries: int) -> Generator[Mapping[str, Any], None, None]:
        if False:
            print('Hello World!')
        '\n        Creates generator of dictionaries containing fixed parameter values\n        (i.e. attributes defined in the child-most class). Those fixed parameters\n        specified by `parallel_params` are given different values over each of the\n        parallel jobs. Called by `transform` and `inverse_transform`,\n        if `Transformer` does *not* inherit from `FittableTransformer`.\n        '
        self._check_fixed_params(n_timeseries)

        def params_generator(n_timeseries, fixed_params, parallel_params):
            if False:
                for i in range(10):
                    print('nop')
            fixed_params_copy = fixed_params.copy()
            for i in range(n_timeseries):
                for key in parallel_params:
                    fixed_params_copy[key] = fixed_params[key][i]
                if fixed_params_copy:
                    params = {'fixed': fixed_params_copy}
                else:
                    params = None
                yield params
            return None
        return params_generator(n_timeseries, self._fixed_params, self._parallel_params)

    def _check_fixed_params(self, n_timeseries: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Raises `ValueError` if `self._parallel_params` specifies a `key` in\n        `self._fixed_params` that should be distributed, but\n        `len(self._fixed_params[key])` does not equal `n_timeseries`.\n        '
        for key in self._parallel_params:
            raise_if(n_timeseries > len(self._fixed_params[key]), f'{n_timeseries} TimeSeries were provided but only {len(self._fixed_params[key])} {key} values were specified upon initialising {self.name}.')
        return None

    @staticmethod
    def apply_component_mask(series: TimeSeries, component_mask: Optional[np.ndarray]=None, return_ts=False) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Extracts components specified by `component_mask` from `series`\n\n        Parameters\n        ----------\n        series\n            input TimeSeries to be fed into transformer.\n        component_mask\n            Optionally, np.ndarray boolean mask of shape (n_components, 1) specifying which components to\n            extract from `series`. The `i`th component of `series` is kept only if `component_mask[i] = True`.\n            If not specified, no masking is performed.\n        return_ts\n            Optionally, specifies that a `TimeSeries` should be returned, rather than an `np.ndarray`.\n\n        Returns\n        -------\n        masked\n            `TimeSeries` (if `return_ts = True`) or `np.ndarray` (if `return_ts = False`) with only those components\n            specified by `component_mask` remaining.\n\n        '
        if component_mask is None:
            masked = series.copy() if return_ts else series.all_values()
        else:
            raise_if_not(isinstance(component_mask, np.ndarray) and component_mask.dtype == bool, f'`component_mask` must be a boolean `np.ndarray`, not a {type(component_mask)}.', logger)
            raise_if_not(series.width == len(component_mask), 'mismatch between number of components in `series` and length of `component_mask`', logger)
            masked = series.all_values(copy=False)[:, component_mask, :]
            if return_ts:
                coords = dict(series._xa.coords)
                coords['component'] = coords['component'][component_mask]
                new_xa = xr.DataArray(masked, dims=series._xa.dims, coords=coords, attrs=series._xa.attrs)
                masked = TimeSeries(new_xa)
        return masked

    @staticmethod
    def unapply_component_mask(series: TimeSeries, vals: Union[np.ndarray, TimeSeries], component_mask: Optional[np.ndarray]=None) -> Union[np.ndarray, TimeSeries]:
        if False:
            while True:
                i = 10
        "\n        Adds back components previously removed by `component_mask` in `apply_component_mask` method.\n\n        Parameters\n        ----------\n        series\n            input TimeSeries that was fed into transformer.\n        vals:\n            `np.ndarray` or `TimeSeries` to 'unmask'\n        component_mask\n            Optionally, np.ndarray boolean mask of shape (n_components, 1) specifying which components were extracted\n            from `series`. If given, insert `vals` back into the columns of the original array. If not specified,\n            nothing is 'unmasked'.\n\n        Returns\n        -------\n        unmasked\n            `TimeSeries` (if `vals` is a `TimeSeries`) or `np.ndarray` (if `vals` is an `np.ndarray`) with those\n            components previously removed by `component_mask` now 'added back'.\n        "
        if component_mask is None:
            unmasked = vals
        else:
            raise_if_not(isinstance(component_mask, np.ndarray) and component_mask.dtype == bool, 'If `component_mask` is given, must be a boolean np.ndarray`', logger)
            raise_if_not(series.width == len(component_mask), 'mismatch between number of components in `series` and length of `component_mask`', logger)
            unmasked = series.all_values()
            if isinstance(vals, TimeSeries):
                unmasked[:, component_mask, :] = vals.all_values()
                unmasked = series.slice_intersect(vals).with_values(unmasked)
            else:
                unmasked[:, component_mask, :] = vals
        return unmasked

    @staticmethod
    def stack_samples(vals: Union[np.ndarray, TimeSeries]) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        "\n        Creates an array of shape `(n_timesteps * n_samples, n_components)` from\n        either a `TimeSeries` or the `array_values` of a `TimeSeries`.\n\n        Each column of the returned array corresponds to a component (dimension)\n        of the series and is formed by concatenating all of the samples associated\n        with that component together. More specifically, the `i`th column is formed\n        by concatenating\n        `[component_i_sample_1, component_i_sample_2, ..., component_i_sample_n]`.\n\n        Stacking is useful when implementing a transformation that applies the exact same\n        change to every timestep in the timeseries. In such cases, the samples of each\n        component can be stacked together into a single column, and the transformation can then be\n        applied to each column, thereby 'vectorising' the transformation over all samples of that component;\n        the `unstack_samples` method can then be used to reshape the output. For transformations that depend\n        on the `time_index` or the temporal ordering of the observations, stacking should *not* be employed.\n\n        Parameters\n        ----------\n        vals\n            `Timeseries` or `np.ndarray` of shape `(n_timesteps, n_components, n_samples)` to be 'stacked'.\n\n        Returns\n        -------\n        stacked\n            `np.ndarray` of shape `(n_timesteps * n_samples, n_components)`, where the `i`th column is formed\n            by concatenating all of the samples of the `i`th component in `vals`.\n        "
        if isinstance(vals, TimeSeries):
            vals = vals.all_values()
        shape = vals.shape
        new_shape = (shape[0] * shape[2], shape[1])
        stacked = np.swapaxes(vals, 1, 2).reshape(new_shape)
        return stacked

    @staticmethod
    def unstack_samples(vals: np.ndarray, n_timesteps: Optional[int]=None, n_samples: Optional[int]=None, series: Optional[TimeSeries]=None) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        "\n        Reshapes the 2D array returned by `stack_samples` back into an array of shape\n        `(n_timesteps, n_components, n_samples)`; this 'undoes' the reshaping of\n        `stack_samples`. Either `n_components`, `n_samples`, or `series` must be specified.\n\n        Parameters\n        ----------\n        vals\n            `np.ndarray` of shape `(n_timesteps * n_samples, n_components)` to be 'unstacked'.\n        n_timesteps\n            Optionally, the number of timesteps in the array originally passed to `stack_samples`.\n            Does *not* need to be provided if `series` is specified.\n        n_samples\n            Optionally, the number of samples in the array originally passed to `stack_samples`.\n            Does *not* need to be provided if `series` is specified.\n        series\n            Optionally, the `TimeSeries` object used to create `vals`; `n_samples` is inferred\n            from this.\n\n        Returns\n        -------\n        unstacked\n            `np.ndarray` of shape `(n_timesteps, n_components, n_samples)`.\n\n        "
        if series is not None:
            n_samples = series.n_samples
        else:
            raise_if(all((x is None for x in [n_timesteps, n_samples])), 'Must specify either `n_timesteps`, `n_samples`, or `series`.')
        n_components = vals.shape[-1]
        if n_timesteps is not None:
            reshaped_vals = vals.reshape(n_timesteps, -1, n_components)
        else:
            reshaped_vals = vals.reshape(-1, n_samples, n_components)
        unstacked = np.swapaxes(reshaped_vals, 1, 2)
        return unstacked

    @property
    def name(self):
        if False:
            print('Hello World!')
        'Name of the data transformer.'
        return self._name

    def __str__(self):
        if False:
            return 10
        return self._name

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return self.__str__()