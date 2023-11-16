import pytest
from numpy.random import RandomState
from darts.logging import get_logger
logger = get_logger(__name__)
try:
    import torch
    from darts.utils.torch import random_method
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning('Torch not available. Torch utils will not be tested.')
    TORCH_AVAILABLE = False
if TORCH_AVAILABLE:

    class TorchModelMock:

        @random_method
        def __init__(self, some_params=None, **kwargs):
            if False:
                i = 10
                return i + 15
            self.model = torch.randn(5)

        @random_method
        def fit(self, some_params=None):
            if False:
                i = 10
                return i + 15
            self.fit_value = torch.randn(5)

    class TestRandomMethod:

        def test_it_raises_error_if_used_on_function(self):
            if False:
                i = 10
                return i + 15
            with pytest.raises(ValueError):

                @random_method
                def a_random_function():
                    if False:
                        return 10
                    pass

        def test_model_is_random_by_default(self):
            if False:
                print('Hello World!')
            model1 = TorchModelMock()
            model2 = TorchModelMock()
            assert not torch.equal(model1.model, model2.model)

        def test_model_is_random_when_None_random_state_specified(self):
            if False:
                for i in range(10):
                    print('nop')
            model1 = TorchModelMock(random_state=None)
            model2 = TorchModelMock(random_state=None)
            assert not torch.equal(model1.model, model2.model)

        def helper_test_reproducibility(self, model1, model2):
            if False:
                i = 10
                return i + 15
            assert torch.equal(model1.model, model2.model)
            model1.fit()
            model2.fit()
            assert torch.equal(model1.fit_value, model2.fit_value)

        def test_model_is_reproducible_when_seed_specified(self):
            if False:
                i = 10
                return i + 15
            model1 = TorchModelMock(random_state=42)
            model2 = TorchModelMock(random_state=42)
            self.helper_test_reproducibility(model1, model2)

        def test_model_is_reproducible_when_random_instance_specified(self):
            if False:
                while True:
                    i = 10
            model1 = TorchModelMock(random_state=RandomState(42))
            model2 = TorchModelMock(random_state=RandomState(42))
            self.helper_test_reproducibility(model1, model2)

        def test_model_is_different_for_different_seeds(self):
            if False:
                for i in range(10):
                    print('nop')
            model1 = TorchModelMock(random_state=42)
            model2 = TorchModelMock(random_state=43)
            assert not torch.equal(model1.model, model2.model)

        def test_model_is_different_for_different_random_instance(self):
            if False:
                for i in range(10):
                    print('nop')
            model1 = TorchModelMock(random_state=RandomState(42))
            model2 = TorchModelMock(random_state=RandomState(43))
            assert not torch.equal(model1.model, model2.model)

        def helper_test_successive_call_are_different(self, model):
            if False:
                return 10
            model.fit()
            assert not torch.equal(model.model, model.fit_value)
            old_fit_value = model.fit_value.clone()
            model.fit()
            assert not torch.equal(model.fit_value, old_fit_value)

        def test_successive_call_to_rng_are_different_when_seed_specified(self):
            if False:
                print('Hello World!')
            model = TorchModelMock(random_state=42)
            self.helper_test_successive_call_are_different(model)

        def test_successive_call_to_rng_are_different_when_random_instance_specified(self):
            if False:
                return 10
            model = TorchModelMock(random_state=RandomState(42))
            self.helper_test_successive_call_are_different(model)

        def test_no_side_effect_between_rng_with_seeds(self):
            if False:
                i = 10
                return i + 15
            model = TorchModelMock(random_state=42)
            model.fit()
            fit_value = model.fit_value.clone()
            model = TorchModelMock(random_state=42)
            model2 = TorchModelMock(random_state=42)
            model2.fit()
            model.fit()
            assert torch.equal(model.fit_value, fit_value)

        def test_no_side_effect_between_rng_with_random_instance(self):
            if False:
                print('Hello World!')
            model = TorchModelMock(random_state=RandomState(42))
            model.fit()
            fit_value = model.fit_value.clone()
            model = TorchModelMock(random_state=RandomState(42))
            model2 = TorchModelMock(random_state=RandomState(42))
            model2.fit()
            model.fit()
            assert torch.equal(model.fit_value, fit_value)