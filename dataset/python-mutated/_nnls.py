"""Non-negative least squares"""
import numpy as np
import scipy.optimize
from .utils import MAX_MEM_BLOCK
from typing import Any, Optional, Tuple, Sequence
__all__ = ['nnls']

def _nnls_obj(x: np.ndarray, shape: Sequence[int], A: np.ndarray, B: np.ndarray) -> Tuple[float, np.ndarray]:
    if False:
        i = 10
        return i + 15
    'Compute the objective and gradient for NNLS'
    x = x.reshape(shape)
    diff = np.einsum('mf,...ft->...mt', A, x, optimize=True) - B
    value = 1 / B.size * 0.5 * np.sum(diff ** 2)
    grad = 1 / B.size * np.einsum('mf,...mt->...ft', A, diff, optimize=True)
    return (value, grad.flatten())

def _nnls_lbfgs_block(A: np.ndarray, B: np.ndarray, x_init: Optional[np.ndarray]=None, **kwargs: Any) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    'Solve the constrained problem over a single block\n\n    Parameters\n    ----------\n    A : np.ndarray [shape=(m, d)]\n        The basis matrix\n    B : np.ndarray [shape=(m, N)]\n        The regression targets\n    x_init : np.ndarray [shape=(d, N)]\n        An initial guess\n    **kwargs\n        Additional keyword arguments to `scipy.optimize.fmin_l_bfgs_b`\n\n    Returns\n    -------\n    x : np.ndarray [shape=(d, N)]\n        Non-negative matrix such that Ax ~= B\n    '
    if x_init is None:
        x_init = np.einsum('fm,...mt->...ft', np.linalg.pinv(A), B, optimize=True)
        np.clip(x_init, 0, None, out=x_init)
    kwargs.setdefault('m', A.shape[1])
    bounds = [(0, None)] * x_init.size
    shape = x_init.shape
    x: np.ndarray
    (x, obj_value, diagnostics) = scipy.optimize.fmin_l_bfgs_b(_nnls_obj, x_init, args=(shape, A, B), bounds=bounds, **kwargs)
    return x.reshape(shape)

def nnls(A: np.ndarray, B: np.ndarray, **kwargs: Any) -> np.ndarray:
    if False:
        print('Hello World!')
    'Non-negative least squares.\n\n    Given two matrices A and B, find a non-negative matrix X\n    that minimizes the sum squared error::\n\n        err(X) = sum_i,j ((AX)[i,j] - B[i, j])^2\n\n    Parameters\n    ----------\n    A : np.ndarray [shape=(m, n)]\n        The basis matrix\n    B : np.ndarray [shape=(..., m, N)]\n        The target array.  Additional leading dimensions are supported.\n    **kwargs\n        Additional keyword arguments to `scipy.optimize.fmin_l_bfgs_b`\n\n    Returns\n    -------\n    X : np.ndarray [shape=(..., n, N), non-negative]\n        A minimizing solution to ``|AX - B|^2``\n\n    See Also\n    --------\n    scipy.optimize.nnls\n    scipy.optimize.fmin_l_bfgs_b\n\n    Examples\n    --------\n    Approximate a magnitude spectrum from its mel spectrogram\n\n    >>> y, sr = librosa.load(librosa.ex(\'trumpet\'), duration=3)\n    >>> S = np.abs(librosa.stft(y, n_fft=2048))\n    >>> M = librosa.feature.melspectrogram(S=S, sr=sr, power=1)\n    >>> mel_basis = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=M.shape[0])\n    >>> S_recover = librosa.util.nnls(mel_basis, M)\n\n    Plot the results\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)\n    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),\n    ...                          y_axis=\'log\', x_axis=\'time\', ax=ax[2])\n    >>> ax[2].set(title=\'Original spectrogram (1025 bins)\')\n    >>> ax[2].label_outer()\n    >>> librosa.display.specshow(librosa.amplitude_to_db(M, ref=np.max),\n    ...                          y_axis=\'mel\', x_axis=\'time\', ax=ax[0])\n    >>> ax[0].set(title=\'Mel spectrogram (128 bins)\')\n    >>> ax[0].label_outer()\n    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S_recover, ref=np.max(S)),\n    ...                          y_axis=\'log\', x_axis=\'time\', ax=ax[1])\n    >>> ax[1].set(title=\'Reconstructed spectrogram (1025 bins)\')\n    >>> ax[1].label_outer()\n    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")\n    '
    if B.ndim == 1:
        return scipy.optimize.nnls(A, B)[0]
    n_columns = int(MAX_MEM_BLOCK // (np.prod(B.shape[:-1]) * A.itemsize))
    n_columns = max(n_columns, 1)
    if B.shape[-1] <= n_columns:
        return _nnls_lbfgs_block(A, B, **kwargs).astype(A.dtype)
    x: np.ndarray
    x = np.einsum('fm,...mt->...ft', np.linalg.pinv(A), B, optimize=True)
    np.clip(x, 0, None, out=x)
    x_init = x
    for bl_s in range(0, x.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, B.shape[-1])
        x[..., bl_s:bl_t] = _nnls_lbfgs_block(A, B[..., bl_s:bl_t], x_init=x_init[..., bl_s:bl_t], **kwargs)
    return x