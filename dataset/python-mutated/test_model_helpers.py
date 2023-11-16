import pytest
from lightning.pytorch import LightningDataModule
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
from lightning.pytorch.utilities.model_helpers import _restricted_classmethod, is_overridden
from lightning_utilities import module_available

def test_is_overridden():
    if False:
        while True:
            i = 10
    assert not is_overridden('whatever', None)
    with pytest.raises(ValueError, match='Expected a parent'):
        is_overridden('whatever', object())
    model = BoringModel()
    assert not is_overridden('whatever', model)
    assert not is_overridden('whatever', model, parent=LightningDataModule)
    assert is_overridden('training_step', model)
    datamodule = BoringDataModule()
    assert is_overridden('train_dataloader', datamodule)

@pytest.mark.skipif(not module_available('lightning') or not module_available('pytorch_lightning'), reason='This test is ONLY relevant for the UNIFIED package')
def test_mixed_imports_unified():
    if False:
        for i in range(10):
            print('nop')
    from lightning.pytorch.utilities.compile import _maybe_unwrap_optimized as new_unwrap
    from lightning.pytorch.utilities.model_helpers import is_overridden as new_is_overridden
    from pytorch_lightning.callbacks import EarlyStopping as OldEarlyStopping
    from pytorch_lightning.demos.boring_classes import BoringModel as OldBoringModel
    model = OldBoringModel()
    with pytest.raises(TypeError, match='`pytorch_lightning` object \\(BoringModel\\) to a `lightning.pytorch`'):
        new_unwrap(model)
    with pytest.raises(TypeError, match='`pytorch_lightning` object \\(EarlyStopping\\) to a `lightning.pytorch`'):
        new_is_overridden('on_fit_start', OldEarlyStopping('foo'))

class RestrictedClass:

    @_restricted_classmethod
    def restricted_cmethod(cls):
        if False:
            i = 10
            return i + 15
        pass

    @classmethod
    def cmethod(cls):
        if False:
            while True:
                i = 10
        pass

def test_restricted_classmethod():
    if False:
        while True:
            i = 10
    with pytest.raises(TypeError, match='cannot be called on an instance'):
        RestrictedClass().restricted_cmethod()
    RestrictedClass.restricted_cmethod()