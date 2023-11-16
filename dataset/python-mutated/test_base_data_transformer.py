from typing import Any, Mapping, Sequence, Union
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import BaseDataTransformer
from darts.utils.timeseries_generation import constant_timeseries

class TestBaseDataTransformer:

    class DataTransformerMock(BaseDataTransformer):

        def __init__(self, scale: float, translation: float, stack_samples: bool=False, mask_components: bool=True, parallel_params: Union[bool, Sequence[str]]=False):
            if False:
                print('Hello World!')
            '\n            Applies the transform `transformed_series = scale * series + translation`.\n\n            Parameters\n            ----------\n            scale\n                Scale coefficient of transform.\n            translation\n                Translational constant of transform.\n            stack_samples\n                Whether to call `stack_samples` inside of `ts_transform`.\n            mask_components\n                Whether to automatically apply any provided `component_mask` key word arguments. See\n                `BaseDataTransformer` docstring for further details.\n            parallel_params\n                Specifies which parameters should vary between different parallel jobs, supposing that\n                multiple time series are given to `ts_transform`. See `BaseDataTransformer` docstring\n                for further details.\n\n            '
            self._scale = scale
            self._translation = translation
            self._stack_samples = stack_samples
            self._mask_components = mask_components
            super().__init__(name='DataTransformerMock', mask_components=mask_components, parallel_params=parallel_params)

        @staticmethod
        def ts_transform(series: TimeSeries, params: Mapping[str, Any], **kwargs) -> TimeSeries:
            if False:
                while True:
                    i = 10
            "\n            Implements the transform `scale * series + translation`.\n\n            If `component_mask` is in `kwargs`, this is manually applied and unapplied. If\n            `_stack_samples = True` in `params['fixed']`, then `stack_samples` and `unstack_samples`\n            all used when computing this transformation.\n\n            "
            fixed_params = params['fixed']
            stack_samples = fixed_params['_stack_samples']
            mask_components = fixed_params['_mask_components']
            (scale, translation) = (fixed_params['_scale'], fixed_params['_translation'])
            if mask_components:
                assert 'component_mask' not in kwargs
            if not mask_components and 'component_mask' in kwargs:
                vals = TestBaseDataTransformer.DataTransformerMock.apply_component_mask(series, kwargs['component_mask'], return_ts=False)
            else:
                vals = series.all_values()
            if stack_samples:
                vals = TestBaseDataTransformer.DataTransformerMock.stack_samples(vals)
            vals = scale * vals + translation
            if stack_samples:
                vals = TestBaseDataTransformer.DataTransformerMock.unstack_samples(vals, series=series)
            if not mask_components and 'component_mask' in kwargs:
                vals = TestBaseDataTransformer.DataTransformerMock.unapply_component_mask(series, vals, kwargs['component_mask'])
            return series.with_values(vals)

    def test_input_transformed_single_series(self):
        if False:
            return 10
        '\n        Tests for correct transformation of single series.\n        '
        test_input = constant_timeseries(value=1, length=10)
        mock = self.DataTransformerMock(scale=2, translation=10)
        transformed = mock.transform(test_input)
        expected = constant_timeseries(value=12, length=10)
        assert transformed == expected

    def test_input_transformed_multiple_series(self):
        if False:
            while True:
                i = 10
        '\n        Tests for correct transformation of multiple series when\n        different param values are used for different parallel\n        jobs (i.e. test that `parallel_params` argument is treated\n        correctly). Also tests that transformer correctly handles\n        being provided with fewer input series than fixed parameter\n        value sets.\n        '
        test_input_1 = constant_timeseries(value=1, length=10)
        test_input_2 = constant_timeseries(value=2, length=11)
        mock = self.DataTransformerMock(scale=2, translation=10, parallel_params=False)
        (transformed_1, transformed_2) = mock.transform((test_input_1, test_input_2))
        assert transformed_1 == constant_timeseries(value=12, length=10)
        assert transformed_2 == constant_timeseries(value=14, length=11)
        mock = self.DataTransformerMock(scale=(2, 3), translation=10, parallel_params=['_scale'])
        (transformed_1, transformed_2) = mock.transform((test_input_1, test_input_2))
        assert transformed_1 == constant_timeseries(value=12, length=10)
        assert transformed_2 == constant_timeseries(value=16, length=11)
        transformed_1 = mock.transform(test_input_1)
        assert transformed_1 == constant_timeseries(value=12, length=10)
        mock = self.DataTransformerMock(scale=(2, 3), translation=(10, 11), stack_samples=(False, True), mask_components=(False, False), parallel_params=True)
        (transformed_1, transformed_2) = mock.transform((test_input_1, test_input_2))
        assert transformed_1 == constant_timeseries(value=12, length=10)
        assert transformed_2 == constant_timeseries(value=17, length=11)
        transformed_1 = mock.transform(test_input_1)
        assert transformed_1 == constant_timeseries(value=12, length=10)
        mock = self.DataTransformerMock(scale=(2, 3, 4), translation=(10, 11, 12), stack_samples=(False, True, False), mask_components=(False, False, False), parallel_params=True)
        transformed_1 = mock.transform(test_input_1)
        assert transformed_1 == constant_timeseries(value=12, length=10)
        (transformed_1, transformed_2) = mock.transform((test_input_1, test_input_2))
        assert transformed_1 == constant_timeseries(value=12, length=10)
        assert transformed_2 == constant_timeseries(value=17, length=11)

    def test_input_transformed_multiple_samples(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that `stack_samples` and `unstack_samples` correctly\n        implemented when considering multi-sample timeseries.\n        '
        test_input = constant_timeseries(value=1, length=10)
        test_input = test_input.concatenate(constant_timeseries(value=2, length=10), axis='sample')
        mock = self.DataTransformerMock(scale=2, translation=10, stack_samples=True)
        transformed = mock.transform(test_input)
        expected = constant_timeseries(value=12, length=10)
        expected = expected.concatenate(constant_timeseries(value=14, length=10), axis='sample')
        assert transformed == expected

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
        transformed = mock.transform(test_input, component_mask=mask)
        assert transformed == expected
        mock = self.DataTransformerMock(scale=2, translation=10, mask_components=False)
        transformed = mock.transform(test_input, component_mask=mask)
        assert transformed == expected