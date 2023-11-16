import pytest
import albumentations as A
from albumentations.imgaug.stubs import IAAStub
from tests.conftest import skipif_no_imgaug, skipif_imgaug

@skipif_no_imgaug
@pytest.mark.parametrize('aug_name', ['IAAEmboss', 'IAASuperpixels', 'IAASharpen', 'IAAAdditiveGaussianNoise', 'IAACropAndPad', 'IAAFliplr', 'IAAFlipud', 'IAAAffine', 'IAAPiecewiseAffine'])
def test_imgaug_augmentations_imported_when_imgaug_is_installed(aug_name):
    if False:
        print('Hello World!')
    aug_cls = getattr(A, aug_name)
    t = aug_cls()
    assert isinstance(t, A.BasicIAATransform)

@skipif_no_imgaug
def test_iaaperpective_augmentation_imported_when_imgaug_is_installed():
    if False:
        i = 10
        return i + 15
    from albumentations.imgaug.transforms import IAAPerspective
    t = IAAPerspective()
    assert isinstance(t, A.DualTransform)

@skipif_imgaug
@pytest.mark.parametrize('aug_name', ['IAAEmboss', 'IAASuperpixels', 'IAASharpen', 'IAAAdditiveGaussianNoise', 'IAACropAndPad', 'IAAFliplr', 'IAAFlipud', 'IAAAffine', 'IAAPiecewiseAffine', 'IAAPerspective'])
def test_imgaug_stubs_imported_when_imgaug_is_not_installed(aug_name):
    if False:
        i = 10
        return i + 15
    aug_cls = getattr(A, aug_name)
    assert issubclass(aug_cls, IAAStub)
    with pytest.raises(RuntimeError) as exc_info:
        aug_cls()
        message = f"You are trying to use a deprecated augmentation '{aug_name}' which depends on the imgaug library, but imgaug is not installed."
        assert message in str(exc_info.value)

@skipif_no_imgaug
def test_imports_from_imgaug_module_dont_raise_import_error():
    if False:
        return 10
    from albumentations.imgaug.transforms import IAAFlipud
    IAAFlipud()

@skipif_imgaug
def test_imports_from_imgaug_module_raise_import_error():
    if False:
        return 10
    with pytest.raises(ImportError) as exc_info:
        from albumentations.imgaug.transforms import IAAFlipud
        message = "You are trying to import an augmentation that depends on the imgaug library, but imgaug is not installed. To install a version of Albumentations that contains imgaug please run 'pip install -U albumentations[imgaug]'"
        assert message in str(exc_info.value)