"""Test utilities for comparing rendered arrays with expected array results."""
from typing import Optional, Any
import numpy as np
import pytest
try:
    from numpy.typing import ArrayLike, DtypeLike
except ImportError:
    ArrayLike = np.ndarray
    DtypeLike = Any

def compare_render(orig_data: ArrayLike, rendered_data: ArrayLike, previous_render: Optional[ArrayLike]=None, atol: Optional[float]=1.0):
    if False:
        for i in range(10):
            print('nop')
    'Compare an expected original array with the rendered result.\n\n    Parameters\n    ----------\n    orig_data\n        Expected output result array. This will be converted to an RGBA array\n        to be compared against the rendered data.\n    rendered_data\n        Actual rendered result as an RGBA 8-bit unsigned array.\n    previous_render\n        Previous instance of a render that the current render should not be\n        equal to.\n    atol\n        Absolute tolerance to be passed to\n        :func:`numpy.testing.assert_allclose`.\n\n    '
    predicted = make_rgba(orig_data)
    np.testing.assert_allclose(rendered_data.astype(float), predicted.astype(float), atol=atol)
    if previous_render is not None:
        pytest.raises(AssertionError, np.testing.assert_allclose, rendered_data, previous_render, atol=10)

def max_for_dtype(input_dtype: DtypeLike):
    if False:
        for i in range(10):
            print('nop')
    'Get maximum value an image array should have for a specific dtype.\n\n    This is max int for each integer type or 1.0 for floating point types.\n\n    '
    if np.issubdtype(input_dtype, np.integer):
        max_val = np.iinfo(input_dtype).max
    else:
        max_val = 1.0
    return max_val

def make_rgba(data_in: ArrayLike) -> ArrayLike:
    if False:
        for i in range(10):
            print('nop')
    'Convert any array to an RGBA array.\n\n    RGBA arrays have 3 dimensions where the last represents the channels. If\n    an Alpha channel needs to be added it will be made completely opaque.\n\n    Returns\n    -------\n    3D RGBA unsigned 8-bit array\n\n    '
    max_val = max_for_dtype(data_in.dtype)
    if data_in.ndim == 3 and data_in.shape[-1] == 1:
        data_in = data_in.squeeze()
    if data_in.ndim == 2:
        out = np.stack([data_in] * 4, axis=2)
        out[:, :, 3] = max_val
    elif data_in.shape[-1] == 3:
        out = np.concatenate((data_in, np.ones((*data_in.shape[:2], 1)) * max_val), axis=2)
    else:
        out = data_in
    return np.round(out.astype(np.float32) * 255 / max_val).astype(np.uint8)