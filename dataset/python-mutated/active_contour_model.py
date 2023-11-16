import numpy as np
from scipy.interpolate import RectBivariateSpline
from .._shared.utils import _supported_float_type
from ..util import img_as_float
from ..filters import sobel

def active_contour(image, snake, alpha=0.01, beta=0.1, w_line=0, w_edge=1, gamma=0.01, max_px_move=1.0, max_num_iter=2500, convergence=0.1, *, boundary_condition='periodic'):
    if False:
        print('Hello World!')
    'Active contour model.\n\n    Active contours by fitting snakes to features of images. Supports single\n    and multichannel 2D images. Snakes can be periodic (for segmentation) or\n    have fixed and/or free ends.\n    The output snake has the same length as the input boundary.\n    As the number of points is constant, make sure that the initial snake\n    has enough points to capture the details of the final contour.\n\n    Parameters\n    ----------\n    image : (M, N) or (M, N, 3) ndarray\n        Input image.\n    snake : (K, 2) ndarray\n        Initial snake coordinates. For periodic boundary conditions, endpoints\n        must not be duplicated.\n    alpha : float, optional\n        Snake length shape parameter. Higher values makes snake contract\n        faster.\n    beta : float, optional\n        Snake smoothness shape parameter. Higher values makes snake smoother.\n    w_line : float, optional\n        Controls attraction to brightness. Use negative values to attract\n        toward dark regions.\n    w_edge : float, optional\n        Controls attraction to edges. Use negative values to repel snake from\n        edges.\n    gamma : float, optional\n        Explicit time stepping parameter.\n    max_px_move : float, optional\n        Maximum pixel distance to move per iteration.\n    max_num_iter : int, optional\n        Maximum iterations to optimize snake shape.\n    convergence : float, optional\n        Convergence criteria.\n    boundary_condition : string, optional\n        Boundary conditions for the contour. Can be one of \'periodic\',\n        \'free\', \'fixed\', \'free-fixed\', or \'fixed-free\'. \'periodic\' attaches\n        the two ends of the snake, \'fixed\' holds the end-points in place,\n        and \'free\' allows free movement of the ends. \'fixed\' and \'free\' can\n        be combined by parsing \'fixed-free\', \'free-fixed\'. Parsing\n        \'fixed-fixed\' or \'free-free\' yields same behaviour as \'fixed\' and\n        \'free\', respectively.\n\n    Returns\n    -------\n    snake : (K, 2) ndarray\n        Optimised snake, same shape as input parameter.\n\n    References\n    ----------\n    .. [1]  Kass, M.; Witkin, A.; Terzopoulos, D. "Snakes: Active contour\n            models". International Journal of Computer Vision 1 (4): 321\n            (1988). :DOI:`10.1007/BF00133570`\n\n    Examples\n    --------\n    >>> from skimage.draw import circle_perimeter\n    >>> from skimage.filters import gaussian\n\n    Create and smooth image:\n\n    >>> img = np.zeros((100, 100))\n    >>> rr, cc = circle_perimeter(35, 45, 25)\n    >>> img[rr, cc] = 1\n    >>> img = gaussian(img, 2, preserve_range=False)\n\n    Initialize spline:\n\n    >>> s = np.linspace(0, 2*np.pi, 100)\n    >>> init = 50 * np.array([np.sin(s), np.cos(s)]).T + 50\n\n    Fit spline to image:\n\n    >>> snake = active_contour(img, init, w_edge=0, w_line=1, coordinates=\'rc\')  # doctest: +SKIP\n    >>> dist = np.sqrt((45-snake[:, 0])**2 + (35-snake[:, 1])**2)  # doctest: +SKIP\n    >>> int(np.mean(dist))  # doctest: +SKIP\n    25\n\n    '
    max_num_iter = int(max_num_iter)
    if max_num_iter <= 0:
        raise ValueError('max_num_iter should be >0.')
    convergence_order = 10
    valid_bcs = ['periodic', 'free', 'fixed', 'free-fixed', 'fixed-free', 'fixed-fixed', 'free-free']
    if boundary_condition not in valid_bcs:
        raise ValueError('Invalid boundary condition.\n' + 'Should be one of: ' + ', '.join(valid_bcs) + '.')
    img = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    img = img.astype(float_dtype, copy=False)
    RGB = img.ndim == 3
    if w_edge != 0:
        if RGB:
            edge = [sobel(img[:, :, 0]), sobel(img[:, :, 1]), sobel(img[:, :, 2])]
        else:
            edge = [sobel(img)]
    else:
        edge = [0]
    if RGB:
        img = w_line * np.sum(img, axis=2) + w_edge * sum(edge)
    else:
        img = w_line * img + w_edge * edge[0]
    intp = RectBivariateSpline(np.arange(img.shape[1]), np.arange(img.shape[0]), img.T, kx=2, ky=2, s=0)
    snake_xy = snake[:, ::-1]
    x = snake_xy[:, 0].astype(float_dtype)
    y = snake_xy[:, 1].astype(float_dtype)
    n = len(x)
    xsave = np.empty((convergence_order, n), dtype=float_dtype)
    ysave = np.empty((convergence_order, n), dtype=float_dtype)
    eye_n = np.eye(n, dtype=float)
    a = np.roll(eye_n, -1, axis=0) + np.roll(eye_n, -1, axis=1) - 2 * eye_n
    b = np.roll(eye_n, -2, axis=0) + np.roll(eye_n, -2, axis=1) - 4 * np.roll(eye_n, -1, axis=0) - 4 * np.roll(eye_n, -1, axis=1) + 6 * eye_n
    A = -alpha * a + beta * b
    sfixed = False
    if boundary_condition.startswith('fixed'):
        A[0, :] = 0
        A[1, :] = 0
        A[1, :3] = [1, -2, 1]
        sfixed = True
    efixed = False
    if boundary_condition.endswith('fixed'):
        A[-1, :] = 0
        A[-2, :] = 0
        A[-2, -3:] = [1, -2, 1]
        efixed = True
    sfree = False
    if boundary_condition.startswith('free'):
        A[0, :] = 0
        A[0, :3] = [1, -2, 1]
        A[1, :] = 0
        A[1, :4] = [-1, 3, -3, 1]
        sfree = True
    efree = False
    if boundary_condition.endswith('free'):
        A[-1, :] = 0
        A[-1, -3:] = [1, -2, 1]
        A[-2, :] = 0
        A[-2, -4:] = [-1, 3, -3, 1]
        efree = True
    inv = np.linalg.inv(A + gamma * eye_n)
    inv = inv.astype(float_dtype, copy=False)
    for i in range(max_num_iter):
        fx = intp(x, y, dx=1, grid=False).astype(float_dtype, copy=False)
        fy = intp(x, y, dy=1, grid=False).astype(float_dtype, copy=False)
        if sfixed:
            fx[0] = 0
            fy[0] = 0
        if efixed:
            fx[-1] = 0
            fy[-1] = 0
        if sfree:
            fx[0] *= 2
            fy[0] *= 2
        if efree:
            fx[-1] *= 2
            fy[-1] *= 2
        xn = inv @ (gamma * x + fx)
        yn = inv @ (gamma * y + fy)
        dx = max_px_move * np.tanh(xn - x)
        dy = max_px_move * np.tanh(yn - y)
        if sfixed:
            dx[0] = 0
            dy[0] = 0
        if efixed:
            dx[-1] = 0
            dy[-1] = 0
        x += dx
        y += dy
        j = i % (convergence_order + 1)
        if j < convergence_order:
            xsave[j, :] = x
            ysave[j, :] = y
        else:
            dist = np.min(np.max(np.abs(xsave - x[None, :]) + np.abs(ysave - y[None, :]), 1))
            if dist < convergence:
                break
    return np.stack([y, x], axis=1)