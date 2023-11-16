from pathlib import Path
import shutil
import pytest
from pytest import approx
from matplotlib.testing.compare import compare_images
from matplotlib.testing.decorators import _image_directories

@pytest.mark.parametrize('im1, im2, tol, expect_rms', [('basn3p02.png', 'basn3p02-minorchange.png', 10, None), ('basn3p02.png', 'basn3p02-minorchange.png', 0, 6.50646), ('basn3p02.png', 'basn3p02-1px-offset.png', 0, 90.15611), ('basn3p02.png', 'basn3p02-half-1px-offset.png', 0, 63.75), ('basn3p02.png', 'basn3p02-scrambled.png', 0, 172.63582), ('all127.png', 'all128.png', 0, 1), ('all128.png', 'all127.png', 0, 1)])
def test_image_comparison_expect_rms(im1, im2, tol, expect_rms, tmp_path, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compare two images, expecting a particular RMS error.\n\n    im1 and im2 are filenames relative to the baseline_dir directory.\n\n    tol is the tolerance to pass to compare_images.\n\n    expect_rms is the expected RMS value, or None. If None, the test will\n    succeed if compare_images succeeds. Otherwise, the test will succeed if\n    compare_images fails and returns an RMS error almost equal to this value.\n    '
    monkeypatch.chdir(tmp_path)
    (baseline_dir, result_dir) = map(Path, _image_directories(lambda : 'dummy'))
    result_im2 = result_dir / im1
    shutil.copyfile(baseline_dir / im2, result_im2)
    results = compare_images(baseline_dir / im1, result_im2, tol=tol, in_decorator=True)
    if expect_rms is None:
        assert results is None
    else:
        assert results is not None
        assert results['rms'] == approx(expect_rms, abs=0.0001)