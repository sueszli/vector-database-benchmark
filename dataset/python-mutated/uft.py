"""Function of unitary fourier transform (uft) and utilities

This module implements the unitary fourier transform, also known as
the ortho-normal transform. It is especially useful for convolution
[1], as it respects the Parseval equality. The value of the null
frequency is equal to

.. math::  \\frac{1}{\\sqrt{n}} \\sum_i x_i

so the Fourier transform has the same energy as the original image
(see ``image_quad_norm`` function). The transform is applied from the
last axis for performance (assuming a C-order array input).

References
----------
.. [1] B. R. Hunt "A matrix theory proof of the discrete convolution
       theorem", IEEE Trans. on Audio and Electroacoustics,
       vol. au-19, no. 4, pp. 285-288, dec. 1971

"""
import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type

def ufftn(inarray, dim=None):
    if False:
        for i in range(10):
            print('nop')
    'N-dimensional unitary Fourier transform.\n\n    Parameters\n    ----------\n    inarray : ndarray\n        The array to transform.\n    dim : int, optional\n        The last axis along which to compute the transform. All\n        axes by default.\n\n    Returns\n    -------\n    outarray : ndarray (same shape than inarray)\n        The unitary N-D Fourier transform of ``inarray``.\n\n    Examples\n    --------\n    >>> input = np.ones((3, 3, 3))\n    >>> output = ufftn(input)\n    >>> np.allclose(np.sum(input) / np.sqrt(input.size), output[0, 0, 0])\n    True\n    >>> output.shape\n    (3, 3, 3)\n    '
    if dim is None:
        dim = inarray.ndim
    outarray = fft.fftn(inarray, axes=range(-dim, 0), norm='ortho')
    return outarray

def uifftn(inarray, dim=None):
    if False:
        i = 10
        return i + 15
    'N-dimensional unitary inverse Fourier transform.\n\n    Parameters\n    ----------\n    inarray : ndarray\n        The array to transform.\n    dim : int, optional\n        The last axis along which to compute the transform. All\n        axes by default.\n\n    Returns\n    -------\n    outarray : ndarray\n        The unitary inverse nD Fourier transform of ``inarray``. Has the same shape as\n        ``inarray``.\n\n    Examples\n    --------\n    >>> input = np.ones((3, 3, 3))\n    >>> output = uifftn(input)\n    >>> np.allclose(np.sum(input) / np.sqrt(input.size), output[0, 0, 0])\n    True\n    >>> output.shape\n    (3, 3, 3)\n    '
    if dim is None:
        dim = inarray.ndim
    outarray = fft.ifftn(inarray, axes=range(-dim, 0), norm='ortho')
    return outarray

def urfftn(inarray, dim=None):
    if False:
        for i in range(10):
            print('nop')
    'N-dimensional real unitary Fourier transform.\n\n    This transform considers the Hermitian property of the transform on\n    real-valued input.\n\n    Parameters\n    ----------\n    inarray : ndarray, shape (M[, ...], P)\n        The array to transform.\n    dim : int, optional\n        The last axis along which to compute the transform. All\n        axes by default.\n\n    Returns\n    -------\n    outarray : ndarray, shape (M[, ...], P / 2 + 1)\n        The unitary N-D real Fourier transform of ``inarray``.\n\n    Notes\n    -----\n    The ``urfft`` functions assume an input array of real\n    values. Consequently, the output has a Hermitian property and\n    redundant values are not computed or returned.\n\n    Examples\n    --------\n    >>> input = np.ones((5, 5, 5))\n    >>> output = urfftn(input)\n    >>> np.allclose(np.sum(input) / np.sqrt(input.size), output[0, 0, 0])\n    True\n    >>> output.shape\n    (5, 5, 3)\n    '
    if dim is None:
        dim = inarray.ndim
    outarray = fft.rfftn(inarray, axes=range(-dim, 0), norm='ortho')
    return outarray

def uirfftn(inarray, dim=None, shape=None):
    if False:
        for i in range(10):
            print('nop')
    'N-dimensional inverse real unitary Fourier transform.\n\n    This transform considers the Hermitian property of the transform\n    from complex to real input.\n\n    Parameters\n    ----------\n    inarray : ndarray\n        The array to transform.\n    dim : int, optional\n        The last axis along which to compute the transform. All\n        axes by default.\n    shape : tuple of int, optional\n        The shape of the output. The shape of ``rfft`` is ambiguous in\n        case of odd-valued input shape. In this case, this parameter\n        should be provided. See ``np.fft.irfftn``.\n\n    Returns\n    -------\n    outarray : ndarray\n        The unitary N-D inverse real Fourier transform of ``inarray``.\n\n    Notes\n    -----\n    The ``uirfft`` function assumes that the output array is\n    real-valued. Consequently, the input is assumed to have a Hermitian\n    property and redundant values are implicit.\n\n    Examples\n    --------\n    >>> input = np.ones((5, 5, 5))\n    >>> output = uirfftn(urfftn(input), shape=input.shape)\n    >>> np.allclose(input, output)\n    True\n    >>> output.shape\n    (5, 5, 5)\n    '
    if dim is None:
        dim = inarray.ndim
    outarray = fft.irfftn(inarray, shape, axes=range(-dim, 0), norm='ortho')
    return outarray

def ufft2(inarray):
    if False:
        return 10
    '2-dimensional unitary Fourier transform.\n\n    Compute the Fourier transform on the last 2 axes.\n\n    Parameters\n    ----------\n    inarray : ndarray\n        The array to transform.\n\n    Returns\n    -------\n    outarray : ndarray (same shape as inarray)\n        The unitary 2-D Fourier transform of ``inarray``.\n\n    See Also\n    --------\n    uifft2, ufftn, urfftn\n\n    Examples\n    --------\n    >>> input = np.ones((10, 128, 128))\n    >>> output = ufft2(input)\n    >>> np.allclose(np.sum(input[1, ...]) / np.sqrt(input[1, ...].size),\n    ...             output[1, 0, 0])\n    True\n    >>> output.shape\n    (10, 128, 128)\n    '
    return ufftn(inarray, 2)

def uifft2(inarray):
    if False:
        while True:
            i = 10
    '2-dimensional inverse unitary Fourier transform.\n\n    Compute the inverse Fourier transform on the last 2 axes.\n\n    Parameters\n    ----------\n    inarray : ndarray\n        The array to transform.\n\n    Returns\n    -------\n    outarray : ndarray (same shape as inarray)\n        The unitary 2-D inverse Fourier transform of ``inarray``.\n\n    See Also\n    --------\n    uifft2, uifftn, uirfftn\n\n    Examples\n    --------\n    >>> input = np.ones((10, 128, 128))\n    >>> output = uifft2(input)\n    >>> np.allclose(np.sum(input[1, ...]) / np.sqrt(input[1, ...].size),\n    ...             output[0, 0, 0])\n    True\n    >>> output.shape\n    (10, 128, 128)\n    '
    return uifftn(inarray, 2)

def urfft2(inarray):
    if False:
        for i in range(10):
            print('nop')
    '2-dimensional real unitary Fourier transform\n\n    Compute the real Fourier transform on the last 2 axes. This\n    transform considers the Hermitian property of the transform from\n    complex to real-valued input.\n\n    Parameters\n    ----------\n    inarray : ndarray, shape (M[, ...], P)\n        The array to transform.\n\n    Returns\n    -------\n    outarray : ndarray, shape (M[, ...], 2 * (P - 1))\n        The unitary 2-D real Fourier transform of ``inarray``.\n\n    See Also\n    --------\n    ufft2, ufftn, urfftn\n\n    Examples\n    --------\n    >>> input = np.ones((10, 128, 128))\n    >>> output = urfft2(input)\n    >>> np.allclose(np.sum(input[1,...]) / np.sqrt(input[1,...].size),\n    ...             output[1, 0, 0])\n    True\n    >>> output.shape\n    (10, 128, 65)\n    '
    return urfftn(inarray, 2)

def uirfft2(inarray, shape=None):
    if False:
        i = 10
        return i + 15
    '2-dimensional inverse real unitary Fourier transform.\n\n    Compute the real inverse Fourier transform on the last 2 axes.\n    This transform considers the Hermitian property of the transform\n    from complex to real-valued input.\n\n    Parameters\n    ----------\n    inarray : ndarray, shape (M[, ...], P)\n        The array to transform.\n    shape : tuple of int, optional\n        The shape of the output. The shape of ``rfft`` is ambiguous in\n        case of odd-valued input shape. In this case, this parameter\n        should be provided. See ``np.fft.irfftn``.\n\n    Returns\n    -------\n    outarray : ndarray, shape (M[, ...], 2 * (P - 1))\n        The unitary 2-D inverse real Fourier transform of ``inarray``.\n\n    See Also\n    --------\n    urfft2, uifftn, uirfftn\n\n    Examples\n    --------\n    >>> input = np.ones((10, 128, 128))\n    >>> output = uirfftn(urfftn(input), shape=input.shape)\n    >>> np.allclose(input, output)\n    True\n    >>> output.shape\n    (10, 128, 128)\n    '
    return uirfftn(inarray, 2, shape=shape)

def image_quad_norm(inarray):
    if False:
        for i in range(10):
            print('nop')
    'Return the quadratic norm of images in Fourier space.\n\n    This function detects whether the input image satisfies the\n    Hermitian property.\n\n    Parameters\n    ----------\n    inarray : ndarray\n        Input image. The image data should reside in the final two\n        axes.\n\n    Returns\n    -------\n    norm : float\n        The quadratic norm of ``inarray``.\n\n    Examples\n    --------\n    >>> input = np.ones((5, 5))\n    >>> image_quad_norm(ufft2(input)) == np.sum(np.abs(input)**2)\n    True\n    >>> image_quad_norm(ufft2(input)) == image_quad_norm(urfft2(input))\n    True\n    '
    if inarray.shape[-1] != inarray.shape[-2]:
        return 2 * np.sum(np.sum(np.abs(inarray) ** 2, axis=-1), axis=-1) - np.sum(np.abs(inarray[..., 0]) ** 2, axis=-1)
    else:
        return np.sum(np.sum(np.abs(inarray) ** 2, axis=-1), axis=-1)

def ir2tf(imp_resp, shape, dim=None, is_real=True):
    if False:
        print('Hello World!')
    'Compute the transfer function of an impulse response (IR).\n\n    This function makes the necessary correct zero-padding, zero\n    convention, correct fft2, etc... to compute the transfer function\n    of IR. To use with unitary Fourier transform for the signal (ufftn\n    or equivalent).\n\n    Parameters\n    ----------\n    imp_resp : ndarray\n        The impulse responses.\n    shape : tuple of int\n        A tuple of integer corresponding to the target shape of the\n        transfer function.\n    dim : int, optional\n        The last axis along which to compute the transform. All\n        axes by default.\n    is_real : boolean, optional\n       If True (default), imp_resp is supposed real and the Hermitian property\n       is used with rfftn Fourier transform.\n\n    Returns\n    -------\n    y : complex ndarray\n       The transfer function of shape ``shape``.\n\n    See Also\n    --------\n    ufftn, uifftn, urfftn, uirfftn\n\n    Examples\n    --------\n    >>> np.all(np.array([[4, 0], [0, 0]]) == ir2tf(np.ones((2, 2)), (2, 2)))\n    True\n    >>> ir2tf(np.ones((2, 2)), (512, 512)).shape == (512, 257)\n    True\n    >>> ir2tf(np.ones((2, 2)), (512, 512), is_real=False).shape == (512, 512)\n    True\n\n    Notes\n    -----\n    The input array can be composed of multiple-dimensional IR with\n    an arbitrary number of IR. The individual IR must be accessed\n    through the first axes. The last ``dim`` axes contain the space\n    definition.\n    '
    if not dim:
        dim = imp_resp.ndim
    irpadded_dtype = _supported_float_type(imp_resp.dtype)
    irpadded = np.zeros(shape, dtype=irpadded_dtype)
    irpadded[tuple([slice(0, s) for s in imp_resp.shape])] = imp_resp
    for (axis, axis_size) in enumerate(imp_resp.shape):
        if axis >= imp_resp.ndim - dim:
            irpadded = np.roll(irpadded, shift=-int(np.floor(axis_size / 2)), axis=axis)
    func = fft.rfftn if is_real else fft.fftn
    out = func(irpadded, axes=range(-dim, 0))
    cplx_dtype = np.promote_types(irpadded_dtype, np.complex64)
    return out.astype(cplx_dtype, copy=False)

def laplacian(ndim, shape, is_real=True):
    if False:
        return 10
    'Return the transfer function of the Laplacian.\n\n    Laplacian is the second order difference, on row and column.\n\n    Parameters\n    ----------\n    ndim : int\n        The dimension of the Laplacian.\n    shape : tuple\n        The support on which to compute the transfer function.\n    is_real : boolean, optional\n       If True (default), imp_resp is assumed to be real-valued and\n       the Hermitian property is used with rfftn Fourier transform\n       to return the transfer function.\n\n    Returns\n    -------\n    tf : array_like, complex\n        The transfer function.\n    impr : array_like, real\n        The Laplacian.\n\n    Examples\n    --------\n    >>> tf, ir = laplacian(2, (32, 32))\n    >>> np.all(ir == np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))\n    True\n    >>> np.all(tf == ir2tf(ir, (32, 32)))\n    True\n    '
    impr = np.zeros([3] * ndim)
    for dim in range(ndim):
        idx = tuple([slice(1, 2)] * dim + [slice(None)] + [slice(1, 2)] * (ndim - dim - 1))
        impr[idx] = np.array([-1.0, 0.0, -1.0]).reshape([-1 if i == dim else 1 for i in range(ndim)])
    impr[(slice(1, 2),) * ndim] = 2.0 * ndim
    return (ir2tf(impr, shape, is_real=is_real), impr)