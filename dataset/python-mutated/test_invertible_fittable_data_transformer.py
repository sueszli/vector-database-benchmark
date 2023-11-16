from typing import Any, Mapping, Sequence, Union
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers.fittable_data_transformer import FittableDataTransformer
from darts.dataprocessing.transformers.invertible_data_transformer import InvertibleDataTransformer
from darts.utils.timeseries_generation import constant_timeseries

class TestLocalFittableInvertibleDataTransformer:
    """
    Tests that data transformers inheriting from both `FittableDataTransformer` and
    `InvertibleDataTransformer` classes behave correctly when `global_fit` attribute
    is `False`.
    """

    class DataTransformerMock(FittableDataTransformer, InvertibleDataTransformer):
        """
        Mock Fittable and Invertible data transformer that is locally fitted;
        used in test cases
        """

        def __init__(self, scale: float, translation: float, stack_samples: bool=False, mask_components: bool=True, parallel_params: Union[bool, Sequence[str]]=False):
            if False:
                while True:
                    i = 10
            "\n            Applies the (invertible) transform `transformed_series = scale * series + translation`.\n            When 'fitting' this transform, the `scale` and `translation` fixed parameters are returned.\n\n            Parameters\n            ----------\n            scale\n                Scale coefficient of transform.\n            translation\n                Translational constant of transform.\n            stack_samples\n                Whether to call `stack_samples` inside of `ts_transform`.\n            mask_components\n                Whether to automatically apply any provided `component_mask` key word arguments. See\n                `BaseDataTransformer` docstring for further details.\n            parallel_params\n                Specifies which parameters should vary between different parallel jobs, supposing that\n                multiple time series are given to `ts_transform`. See `BaseDataTransformer` docstring\n                for further details.\n\n            "
            self._scale = scale
            self._translation = translation
            self._stack_samples = stack_samples
            self._mask_components = mask_components
            super().__init__(name='DataTransformerMock', mask_components=mask_components, parallel_params=parallel_params)

        @staticmethod
        def ts_fit(series: TimeSeries, params: Mapping[str, Any], **kwargs):
            if False:
                return 10
            "\n            'Fits' transform by returning `scale` and `translation` fixed params.\n            "
            mask_components = params['fixed']['_mask_components']
            if mask_components:
                assert 'component_mask' not in kwargs
            assert 'fitted' not in params
            (scale, translation) = (params['fixed']['_scale'], params['fixed']['_translation'])
            return (scale, translation)

        @staticmethod
        def ts_transform(series: TimeSeries, params: Mapping[str, Any], **kwargs) -> TimeSeries:
            if False:
                return 10
            "\n            Implements the transform `scale * series + translation`.\n\n            If `component_mask` is in `kwargs`, this is manually applied and unapplied. If\n            `_stack_samples = True` in `params['fixed']`, then `stack_samples` and `unstack_samples`\n            all used when computing this transformation.\n\n            "
            stack_samples = params['fixed']['_stack_samples']
            mask_components = params['fixed']['_mask_components']
            (scale, translation) = params['fitted']
            if mask_components:
                assert 'component_mask' not in kwargs
            if not mask_components and 'component_mask' in kwargs:
                vals = TestLocalFittableInvertibleDataTransformer.DataTransformerMock.apply_component_mask(series, kwargs['component_mask'], return_ts=False)
            else:
                vals = series.all_values()
            if stack_samples:
                vals = TestLocalFittableInvertibleDataTransformer.DataTransformerMock.stack_samples(vals)
            vals = scale * vals + translation
            if stack_samples:
                vals = TestLocalFittableInvertibleDataTransformer.DataTransformerMock.unstack_samples(vals, series=series)
            if not mask_components and 'component_mask' in kwargs:
                vals = TestLocalFittableInvertibleDataTransformer.DataTransformerMock.unapply_component_mask(series, vals, kwargs['component_mask'])
            return series.with_values(vals)

        @staticmethod
        def ts_inverse_transform(series: TimeSeries, params: Mapping[str, Any], **kwargs) -> TimeSeries:
            if False:
                return 10
            "\n            Implements the inverse transform `(series - translation) / scale`.\n\n            If `component_mask` is in `kwargs`, this is manually applied and unapplied. If\n            `_stack_samples = True` in `params['fixed']`, then `stack_samples` and `unstack_samples`\n            all used when computing this transformation.\n\n            "
            stack_samples = params['fixed']['_stack_samples']
            mask_components = params['fixed']['_mask_components']
            (scale, translation) = params['fitted']
            if mask_components:
                assert 'component_mask' not in kwargs
            if not mask_components and 'component_mask' in kwargs:
                vals = TestLocalFittableInvertibleDataTransformer.DataTransformerMock.apply_component_mask(series, kwargs['component_mask'], return_ts=False)
            else:
                vals = series.all_values()
            if stack_samples:
                vals = TestLocalFittableInvertibleDataTransformer.DataTransformerMock.stack_samples(vals)
            vals = (vals - translation) / scale
            if stack_samples:
                vals = TestLocalFittableInvertibleDataTransformer.DataTransformerMock.unstack_samples(vals, series=series)
            if not mask_components and 'component_mask' in kwargs:
                vals = TestLocalFittableInvertibleDataTransformer.DataTransformerMock.unapply_component_mask(series, vals, kwargs['component_mask'])
            return series.with_values(vals)

    def test_input_transformed_single_series(self):
        if False:
            return 10
        '\n        Tests for correct (inverse) transformation of single series.\n        '
        test_input = constant_timeseries(value=1, length=10)
        mock = self.DataTransformerMock(scale=2, translation=10)
        transformed = mock.fit_transform(test_input)
        expected = constant_timeseries(value=12, length=10)
        assert transformed == expected
        assert mock.inverse_transform(transformed) == test_input

    def test_input_transformed_multiple_series(self):
        if False:
            return 10
        '\n        Tests for correct transformation of multiple series when\n        different param values are used for different parallel\n        jobs (i.e. test that `parallel_params` argument is treated\n        correctly). Also tests that transformer correctly handles\n        being provided with fewer input series than series used\n        to fit the transformer.\n        '
        test_input_1 = constant_timeseries(value=1, length=10)
        test_input_2 = constant_timeseries(value=2, length=11)
        test_input_3 = constant_timeseries(value=3, length=12)
        mock = self.DataTransformerMock(scale=2, translation=10, parallel_params=False)
        (transformed_1, transformed_2) = mock.fit_transform((test_input_1, test_input_2))
        assert transformed_1 == constant_timeseries(value=12, length=10)
        assert transformed_2 == constant_timeseries(value=14, length=11)
        (inv_1, inv_2) = mock.inverse_transform((transformed_1, transformed_2))
        assert inv_1 == test_input_1
        assert inv_2 == test_input_2
        transformed_1 = mock.transform(test_input_1)
        assert transformed_1 == constant_timeseries(value=12, length=10)
        inv_1 = mock.inverse_transform(transformed_1)
        assert inv_1 == test_input_1
        mock = self.DataTransformerMock(scale=(2, 3), translation=10, parallel_params=['_scale'])
        (transformed_1, transformed_2) = mock.fit_transform((test_input_1, test_input_2))
        assert transformed_1 == constant_timeseries(value=12, length=10)
        assert transformed_2 == constant_timeseries(value=16, length=11)
        (inv_1, inv_2) = mock.inverse_transform((transformed_1, transformed_2))
        assert inv_1 == test_input_1
        assert inv_2 == test_input_2
        mock = self.DataTransformerMock(scale=(2, 3), translation=(10, 11), stack_samples=(False, True), mask_components=(False, False), parallel_params=True)
        (transformed_1, transformed_2) = mock.fit_transform((test_input_1, test_input_2))
        assert transformed_1 == constant_timeseries(value=12, length=10)
        assert transformed_2 == constant_timeseries(value=17, length=11)
        (inv_1, inv_2) = mock.inverse_transform((transformed_1, transformed_2))
        assert inv_1 == test_input_1
        assert inv_2 == test_input_2
        mock = self.DataTransformerMock(scale=(2, 3, 4), translation=(10, 11, 12), stack_samples=(False, True, False), mask_components=(False, False, False), parallel_params=True)
        mock.fit([test_input_1, test_input_2, test_input_3])
        transformed_1 = mock.transform(test_input_1)
        assert transformed_1 == constant_timeseries(value=12, length=10)
        inv_1 = mock.inverse_transform(transformed_1)
        assert inv_1 == test_input_1
        (transformed_1, transformed_2) = mock.transform((test_input_1, test_input_2))
        assert transformed_1 == constant_timeseries(value=12, length=10)
        assert transformed_2 == constant_timeseries(value=17, length=11)
        (inv_1, inv_2) = mock.inverse_transform((transformed_1, transformed_2))
        assert inv_1 == test_input_1
        assert inv_2 == test_input_2

    def test_input_transformed_multiple_samples(self):
        if False:
            return 10
        '\n        Tests that `stack_samples` and `unstack_samples` correctly\n        implemented when considering multi-sample timeseries.\n        '
        test_input = constant_timeseries(value=1, length=10)
        test_input = test_input.concatenate(constant_timeseries(value=2, length=10), axis='sample')
        mock = self.DataTransformerMock(scale=2, translation=10, stack_samples=True)
        transformed = mock.fit_transform(test_input)
        expected = constant_timeseries(value=12, length=10)
        expected = expected.concatenate(constant_timeseries(value=14, length=10), axis='sample')
        assert transformed == expected
        inv = mock.inverse_transform(transformed)
        assert inv == test_input

    def test_input_transformed_masking(self):
        if False:
            print('Hello World!')
        '\n        Tests that automatic component masking is correctly implemented,\n        and that manual component masking is also handled correctly\n        through `kwargs` + `apply_component_mask`/`unapply_component_mask`\n        methods.\n        '
        test_input = TimeSeries.from_values(np.ones((4, 3, 5)))
        mask = np.array([True, False, True])
        scale = 2
        translation = 10
        expected = np.stack([12 * np.ones((4, 5)), np.ones((4, 5)), 12 * np.ones((4, 5))], axis=1)
        expected = TimeSeries.from_values(expected)
        mock = self.DataTransformerMock(scale=scale, translation=translation, mask_components=True)
        transformed = mock.fit_transform(test_input, component_mask=mask)
        assert transformed == expected
        inv = mock.inverse_transform(transformed, component_mask=mask)
        assert inv == test_input
        mock = self.DataTransformerMock(scale=2, translation=10, mask_components=False)
        transformed = mock.fit_transform(test_input, component_mask=mask)
        assert transformed == expected
        inv = mock.inverse_transform(transformed, component_mask=mask)
        assert inv == test_input

class TestGlobalFittableInvertibleDataTransformer:
    """
    Tests that data transformers inheriting from both `FittableDataTransformer` and
    `InvertibleDataTransformer` classes behave correctly when `global_fit` attribute
    is `True`.
    """

    class DataTransformerMock(FittableDataTransformer, InvertibleDataTransformer):
        """
        Mock Fittable and Invertible data transformer that is globally fitted;
        used in test cases
        """

        def __init__(self, global_fit: bool):
            if False:
                while True:
                    i = 10
            '\n            Subtracts off the time-averaged mean of each component in a `TimeSeries`.\n\n            If `global_fit` is `True`, then all of the `TimeSeries` provided to `fit` are\n            used to compute a single time-averaged mean that will be subtracted from\n            every `TimeSeries` subsequently provided to `transform`.\n\n            Conversely, if `global_fit` is `False`, then the time-averaged mean of each\n            `TimeSeries` pass to `fit` is individually computed, resulting in `n` means\n            being computed if `n` `TimeSeries` were passed to `fit`. If multiple `TimeSeries`\n            are subsequently passed to `transform`, the `i`th computed mean will be subtracted\n            from the `i`th provided `TimeSeries`.\n\n            Parameters\n            ----------\n            global_fit\n                Whether global fitting should be performed.\n            '
            super().__init__(name='DataTransformerMock', global_fit=global_fit)

        @staticmethod
        def ts_fit(series: Union[TimeSeries, Sequence[TimeSeries]], params: Mapping[str, Any], **kwargs):
            if False:
                return 10
            "\n            'Fits' transform by computing time-average of each sample and\n            component in `series`.\n\n            If `global_fit` is `True`, then `series` is a `Sequence[TimeSeries]` and the time-averaged mean\n            of each component over *all* of the `TimeSeries` is computed. If `global_fit` is `False`, then\n            `series` is a single `TimeSeries` and the time-averaged mean of the components of this single\n            `TimeSeries` are computed.\n            "
            if not isinstance(series, Sequence):
                series = [series]
            vals = np.concatenate([ts.all_values(copy=False) for ts in series], axis=0)
            return np.mean(vals, axis=0)

        @staticmethod
        def ts_transform(series: TimeSeries, params: Mapping[str, Any], **kwargs) -> TimeSeries:
            if False:
                while True:
                    i = 10
            '\n            Implements the transform `series - mean`.\n            '
            mean = params['fitted']
            vals = series.all_values()
            vals -= mean
            return series.from_values(vals)

        @staticmethod
        def ts_inverse_transform(series: TimeSeries, params: Mapping[str, Any], **kwargs) -> TimeSeries:
            if False:
                print('Hello World!')
            '\n            Implements the inverse transform `series + mean`.\n            '
            mean = params['fitted']
            vals = series.all_values()
            vals += mean
            return series.from_values(vals)

    def test_global_fitting(self):
        if False:
            return 10
        '\n        Tests that invertible time-averaged mean subtraction transformation\n        behaves correctly when `global_fit = False` and when `global_fit = True`.\n        '
        series_1 = TimeSeries.from_values(np.ones((3, 2, 1)))
        series_2 = TimeSeries.from_values(2 * np.ones((3, 2, 1)))
        transformer = self.DataTransformerMock(global_fit=False)
        (transformed_1, transformed_2) = transformer.fit_transform([series_1, series_2])
        assert transformed_1 == TimeSeries.from_values(np.zeros((3, 2, 1)))
        assert transformed_2 == TimeSeries.from_values(np.zeros((3, 2, 1)))
        (untransformed_1, untransformed_2) = transformer.inverse_transform([transformed_1, transformed_2])
        assert untransformed_1 == series_1
        assert untransformed_2 == series_2
        transformer = self.DataTransformerMock(global_fit=True)
        (transformed_1, transformed_2) = transformer.fit_transform([series_1, series_2])
        assert transformed_1 == TimeSeries.from_values(-0.5 * np.ones((3, 2, 1)))
        assert transformed_2 == TimeSeries.from_values(0.5 * np.ones((3, 2, 1)))
        (untransformed_1, untransformed_2) = transformer.inverse_transform([transformed_1, transformed_2])
        assert untransformed_1 == series_1
        assert untransformed_2 == series_2