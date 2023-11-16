import numpy as np
import pytest
from skimage import io
from skimage._shared._warnings import expected_warnings
plt = pytest.importorskip('matplotlib.pyplot')

def setup():
    if False:
        return 10
    io.reset_plugins()
im8 = np.array([[0, 64], [128, 240]], np.uint8)
im16 = im8.astype(np.uint16) * 256
im64 = im8.astype(np.uint64)
imf = im8 / 255
im_lo = imf / 1000
im_hi = imf + 10
imshow_expected_warnings = ['tight_layout : falling back to Agg|\\A\\Z', 'tight_layout: falling back to Agg|\\A\\Z', 'np.asscalar|\\A\\Z', 'The figure layout has changed to tight|\\A\\Z']

def n_subplots(ax_im):
    if False:
        for i in range(10):
            print('nop')
    'Return the number of subplots in the figure containing an ``AxesImage``.\n\n    Parameters\n    ----------\n    ax_im : matplotlib.pyplot.AxesImage object\n        The input ``AxesImage``.\n\n    Returns\n    -------\n    n : int\n        The number of subplots in the corresponding figure.\n\n    Notes\n    -----\n    This function is intended to check whether a colorbar was drawn, in\n    which case two subplots are expected. For standard imshows, one\n    subplot is expected.\n    '
    return len(ax_im.get_figure().get_axes())

def test_uint8():
    if False:
        for i in range(10):
            print('nop')
    plt.figure()
    with expected_warnings(imshow_expected_warnings + ['CObject type is marked|\\A\\Z']):
        ax_im = io.imshow(im8)
    assert ax_im.cmap.name == 'gray'
    assert ax_im.get_clim() == (0, 255)
    assert n_subplots(ax_im) == 1
    assert ax_im.colorbar is None

def test_uint16():
    if False:
        return 10
    plt.figure()
    with expected_warnings(imshow_expected_warnings + ['CObject type is marked|\\A\\Z']):
        ax_im = io.imshow(im16)
    assert ax_im.cmap.name == 'gray'
    assert ax_im.get_clim() == (0, 65535)
    assert n_subplots(ax_im) == 1
    assert ax_im.colorbar is None

def test_float():
    if False:
        while True:
            i = 10
    plt.figure()
    with expected_warnings(imshow_expected_warnings + ['CObject type is marked|\\A\\Z']):
        ax_im = io.imshow(imf)
    assert ax_im.cmap.name == 'gray'
    assert ax_im.get_clim() == (0, 1)
    assert n_subplots(ax_im) == 1
    assert ax_im.colorbar is None

def test_low_data_range():
    if False:
        i = 10
        return i + 15
    with expected_warnings(imshow_expected_warnings + ['Low image data range|CObject type is marked']):
        ax_im = io.imshow(im_lo)
    assert ax_im.get_clim() == (im_lo.min(), im_lo.max())
    assert ax_im.colorbar is not None

def test_outside_standard_range():
    if False:
        i = 10
        return i + 15
    plt.figure()
    with expected_warnings(imshow_expected_warnings + ['out of standard range|CObject type is marked']):
        ax_im = io.imshow(im_hi)
    assert ax_im.get_clim() == (im_hi.min(), im_hi.max())
    assert n_subplots(ax_im) == 2
    assert ax_im.colorbar is not None

def test_nonstandard_type():
    if False:
        for i in range(10):
            print('nop')
    plt.figure()
    with expected_warnings(imshow_expected_warnings + ['Low image data range|CObject type is marked']):
        ax_im = io.imshow(im64)
    assert ax_im.get_clim() == (im64.min(), im64.max())
    assert n_subplots(ax_im) == 2
    assert ax_im.colorbar is not None

def test_signed_image():
    if False:
        print('Hello World!')
    plt.figure()
    im_signed = np.array([[-0.5, -0.2], [0.1, 0.4]])
    with expected_warnings(imshow_expected_warnings + ['CObject type is marked|\\A\\Z']):
        ax_im = io.imshow(im_signed)
    assert ax_im.get_clim() == (-0.5, 0.5)
    assert n_subplots(ax_im) == 2
    assert ax_im.colorbar is not None