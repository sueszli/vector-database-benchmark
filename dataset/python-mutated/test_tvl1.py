import numpy as np
import pytest
from skimage._shared.utils import _supported_float_type
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp

def _sin_flow_gen(image0, max_motion=4.5, npics=5):
    if False:
        return 10
    'Generate a synthetic ground truth optical flow with a sinusoid as\n      first component.\n\n    Parameters\n    ----------\n    image0: ndarray\n        The base image to be warped.\n    max_motion: float\n        Maximum flow magnitude.\n    npics: int\n        Number of sinusoid pics.\n\n    Returns\n    -------\n    flow, image1 : ndarray\n        The synthetic ground truth optical flow with a sinusoid as\n        first component and the corresponding warped image.\n\n    '
    grid = np.meshgrid(*[np.arange(n) for n in image0.shape], indexing='ij')
    grid = np.stack(grid)
    gt_flow = np.zeros_like(grid, dtype=float)
    gt_flow[0, ...] = max_motion * np.sin(grid[0] / grid[0].max() * npics * np.pi)
    image1 = warp(image0, grid - gt_flow, mode='edge')
    return (gt_flow, image1)

@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_2d_motion(dtype):
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.default_rng(0)
    image0 = rng.normal(size=(256, 256))
    (gt_flow, image1) = _sin_flow_gen(image0)
    image1 = image1.astype(dtype, copy=False)
    float_dtype = _supported_float_type(dtype)
    flow = optical_flow_tvl1(image0, image1, attachment=5, dtype=float_dtype)
    assert flow.dtype == float_dtype
    assert abs(flow - gt_flow).mean() < 0.5
    if dtype != float_dtype:
        with pytest.raises(ValueError):
            optical_flow_tvl1(image0, image1, attachment=5, dtype=dtype)

def test_3d_motion():
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.default_rng(0)
    image0 = rng.normal(size=(100, 100, 100))
    (gt_flow, image1) = _sin_flow_gen(image0)
    flow = optical_flow_tvl1(image0, image1, attachment=10)
    assert abs(flow - gt_flow).mean() < 0.5

def test_no_motion_2d():
    if False:
        return 10
    rng = np.random.default_rng(0)
    img = rng.normal(size=(256, 256))
    flow = optical_flow_tvl1(img, img)
    assert np.all(flow == 0)

def test_no_motion_3d():
    if False:
        return 10
    rng = np.random.default_rng(0)
    img = rng.normal(size=(64, 64, 64))
    flow = optical_flow_tvl1(img, img)
    assert np.all(flow == 0)

def test_optical_flow_dtype():
    if False:
        return 10
    rng = np.random.default_rng(0)
    image0 = rng.normal(size=(256, 256))
    (gt_flow, image1) = _sin_flow_gen(image0)
    flow_f64 = optical_flow_tvl1(image0, image1, attachment=5, dtype=np.float64)
    assert flow_f64.dtype == np.float64
    flow_f32 = optical_flow_tvl1(image0, image1, attachment=5, dtype=np.float32)
    assert flow_f32.dtype == np.float32
    assert np.abs(flow_f64 - flow_f32).mean() < 0.001

def test_incompatible_shapes():
    if False:
        return 10
    rng = np.random.default_rng(0)
    I0 = rng.normal(size=(256, 256))
    I1 = rng.normal(size=(128, 256))
    with pytest.raises(ValueError):
        (u, v) = optical_flow_tvl1(I0, I1)

def test_wrong_dtype():
    if False:
        print('Hello World!')
    rng = np.random.default_rng(0)
    img = rng.normal(size=(256, 256))
    with pytest.raises(ValueError):
        (u, v) = optical_flow_tvl1(img, img, dtype=np.int64)