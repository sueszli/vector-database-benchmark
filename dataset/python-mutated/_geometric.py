import math
import textwrap
from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial
from .._shared.utils import safe_as_int

def _affine_matrix_from_vector(v):
    if False:
        print('Hello World!')
    'Affine matrix from linearized (d, d + 1) matrix entries.'
    nparam = v.size
    d = (1 + np.sqrt(1 + 4 * nparam)) / 2 - 1
    dimensionality = int(np.round(d))
    if d != dimensionality:
        raise ValueError(f'Invalid number of elements for linearized matrix: {nparam}')
    matrix = np.eye(dimensionality + 1)
    matrix[:-1, :] = np.reshape(v, (dimensionality, dimensionality + 1))
    return matrix

def _center_and_normalize_points(points):
    if False:
        return 10
    'Center and normalize image points.\n\n    The points are transformed in a two-step procedure that is expressed\n    as a transformation matrix. The matrix of the resulting points is usually\n    better conditioned than the matrix of the original points.\n\n    Center the image points, such that the new coordinate system has its\n    origin at the centroid of the image points.\n\n    Normalize the image points, such that the mean distance from the points\n    to the origin of the coordinate system is sqrt(D).\n\n    If the points are all identical, the returned values will contain nan.\n\n    Parameters\n    ----------\n    points : (N, D) array\n        The coordinates of the image points.\n\n    Returns\n    -------\n    matrix : (D+1, D+1) array_like\n        The transformation matrix to obtain the new points.\n    new_points : (N, D) array\n        The transformed image points.\n\n    References\n    ----------\n    .. [1] Hartley, Richard I. "In defense of the eight-point algorithm."\n           Pattern Analysis and Machine Intelligence, IEEE Transactions on 19.6\n           (1997): 580-593.\n\n    '
    (n, d) = points.shape
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    rms = np.sqrt(np.sum(centered ** 2) / n)
    if rms == 0:
        return (np.full((d + 1, d + 1), np.nan), np.full_like(points, np.nan))
    norm_factor = np.sqrt(d) / rms
    part_matrix = norm_factor * np.concatenate((np.eye(d), -centroid[:, np.newaxis]), axis=1)
    matrix = np.concatenate((part_matrix, [[0] * d + [1]]), axis=0)
    points_h = np.vstack([points.T, np.ones(n)])
    new_points_h = (matrix @ points_h).T
    new_points = new_points_h[:, :d]
    new_points /= new_points_h[:, d:]
    return (matrix, new_points)

def _umeyama(src, dst, estimate_scale):
    if False:
        while True:
            i = 10
    'Estimate N-D similarity transformation with or without scaling.\n\n    Parameters\n    ----------\n    src : (M, N) array_like\n        Source coordinates.\n    dst : (M, N) array_like\n        Destination coordinates.\n    estimate_scale : bool\n        Whether to estimate scaling factor.\n\n    Returns\n    -------\n    T : (N + 1, N + 1)\n        The homogeneous similarity transformation matrix. The matrix contains\n        NaN values only if the problem is not well-conditioned.\n\n    References\n    ----------\n    .. [1] "Least-squares estimation of transformation parameters between two\n            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`\n\n    '
    src = np.asarray(src)
    dst = np.asarray(dst)
    num = src.shape[0]
    dim = src.shape[1]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = dst_demean.T @ src_demean / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.float64)
    (U, S, V) = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V
    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    return T

class _GeometricTransform(ABC):
    """Abstract base class for geometric transformations."""

    @abstractmethod
    def __call__(self, coords):
        if False:
            return 10
        'Apply forward transformation.\n\n        Parameters\n        ----------\n        coords : (N, 2) array_like\n            Source coordinates.\n\n        Returns\n        -------\n        coords : (N, 2) array\n            Destination coordinates.\n\n        '

    @property
    @abstractmethod
    def inverse(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a transform object representing the inverse.'

    def residuals(self, src, dst):
        if False:
            return 10
        'Determine residuals of transformed destination coordinates.\n\n        For each transformed source coordinate the Euclidean distance to the\n        respective destination coordinate is determined.\n\n        Parameters\n        ----------\n        src : (N, 2) array\n            Source coordinates.\n        dst : (N, 2) array\n            Destination coordinates.\n\n        Returns\n        -------\n        residuals : (N,) array\n            Residual for coordinate.\n\n        '
        return np.sqrt(np.sum((self(src) - dst) ** 2, axis=1))

class FundamentalMatrixTransform(_GeometricTransform):
    """Fundamental matrix transformation.

    The fundamental matrix relates corresponding points between a pair of
    uncalibrated images. The matrix transforms homogeneous image points in one
    image to epipolar lines in the other image.

    The fundamental matrix is only defined for a pair of moving images. In the
    case of pure rotation or planar scenes, the homography describes the
    geometric relation between two images (`ProjectiveTransform`). If the
    intrinsic calibration of the images is known, the essential matrix describes
    the metric relation between the two images (`EssentialMatrixTransform`).

    References
    ----------
    .. [1] Hartley, Richard, and Andrew Zisserman. Multiple view geometry in
           computer vision. Cambridge university press, 2003.

    Parameters
    ----------
    matrix : (3, 3) array_like, optional
        Fundamental matrix.

    Attributes
    ----------
    params : (3, 3) array
        Fundamental matrix.

    """

    def __init__(self, matrix=None, *, dimensionality=2):
        if False:
            print('Hello World!')
        if matrix is None:
            matrix = np.eye(dimensionality + 1)
        else:
            matrix = np.asarray(matrix)
            dimensionality = matrix.shape[0] - 1
            if matrix.shape != (dimensionality + 1, dimensionality + 1):
                raise ValueError('Invalid shape of transformation matrix')
        self.params = matrix
        if dimensionality != 2:
            raise NotImplementedError(f'{self.__class__} is only implemented for 2D coordinates (i.e. 3D transformation matrices).')

    def __call__(self, coords):
        if False:
            print('Hello World!')
        'Apply forward transformation.\n\n        Parameters\n        ----------\n        coords : (N, 2) array_like\n            Source coordinates.\n\n        Returns\n        -------\n        coords : (N, 3) array\n            Epipolar lines in the destination image.\n\n        '
        coords = np.asarray(coords)
        coords_homogeneous = np.column_stack([coords, np.ones(coords.shape[0])])
        return coords_homogeneous @ self.params.T

    @property
    def inverse(self):
        if False:
            while True:
                i = 10
        'Return a transform object representing the inverse.\n\n        See Hartley & Zisserman, Ch. 8: Epipolar Geometry and the Fundamental\n        Matrix, for an explanation of why F.T gives the inverse.\n        '
        return type(self)(matrix=self.params.T)

    def _setup_constraint_matrix(self, src, dst):
        if False:
            print('Hello World!')
        "Setup and solve the homogeneous epipolar constraint matrix::\n\n            dst' * F * src = 0.\n\n        Parameters\n        ----------\n        src : (N, 2) array_like\n            Source coordinates.\n        dst : (N, 2) array_like\n            Destination coordinates.\n\n        Returns\n        -------\n        F_normalized : (3, 3) array\n            The normalized solution to the homogeneous system. If the system\n            is not well-conditioned, this matrix contains NaNs.\n        src_matrix : (3, 3) array\n            The transformation matrix to obtain the normalized source\n            coordinates.\n        dst_matrix : (3, 3) array\n            The transformation matrix to obtain the normalized destination\n            coordinates.\n\n        "
        src = np.asarray(src)
        dst = np.asarray(dst)
        if src.shape != dst.shape:
            raise ValueError('src and dst shapes must be identical.')
        if src.shape[0] < 8:
            raise ValueError('src.shape[0] must be equal or larger than 8.')
        try:
            (src_matrix, src) = _center_and_normalize_points(src)
            (dst_matrix, dst) = _center_and_normalize_points(dst)
        except ZeroDivisionError:
            self.params = np.full((3, 3), np.nan)
            return 3 * [np.full((3, 3), np.nan)]
        A = np.ones((src.shape[0], 9))
        A[:, :2] = src
        A[:, :3] *= dst[:, 0, np.newaxis]
        A[:, 3:5] = src
        A[:, 3:6] *= dst[:, 1, np.newaxis]
        A[:, 6:8] = src
        (_, _, V) = np.linalg.svd(A)
        F_normalized = V[-1, :].reshape(3, 3)
        return (F_normalized, src_matrix, dst_matrix)

    def estimate(self, src, dst):
        if False:
            i = 10
            return i + 15
        'Estimate fundamental matrix using 8-point algorithm.\n\n        The 8-point algorithm requires at least 8 corresponding point pairs for\n        a well-conditioned solution, otherwise the over-determined solution is\n        estimated.\n\n        Parameters\n        ----------\n        src : (N, 2) array_like\n            Source coordinates.\n        dst : (N, 2) array_like\n            Destination coordinates.\n\n        Returns\n        -------\n        success : bool\n            True, if model estimation succeeds.\n\n        '
        (F_normalized, src_matrix, dst_matrix) = self._setup_constraint_matrix(src, dst)
        (U, S, V) = np.linalg.svd(F_normalized)
        S[2] = 0
        F = U @ np.diag(S) @ V
        self.params = dst_matrix.T @ F @ src_matrix
        return True

    def residuals(self, src, dst):
        if False:
            i = 10
            return i + 15
        'Compute the Sampson distance.\n\n        The Sampson distance is the first approximation to the geometric error.\n\n        Parameters\n        ----------\n        src : (N, 2) array\n            Source coordinates.\n        dst : (N, 2) array\n            Destination coordinates.\n\n        Returns\n        -------\n        residuals : (N,) array\n            Sampson distance.\n\n        '
        src_homogeneous = np.column_stack([src, np.ones(src.shape[0])])
        dst_homogeneous = np.column_stack([dst, np.ones(dst.shape[0])])
        F_src = self.params @ src_homogeneous.T
        Ft_dst = self.params.T @ dst_homogeneous.T
        dst_F_src = np.sum(dst_homogeneous * F_src.T, axis=1)
        return np.abs(dst_F_src) / np.sqrt(F_src[0] ** 2 + F_src[1] ** 2 + Ft_dst[0] ** 2 + Ft_dst[1] ** 2)

class EssentialMatrixTransform(FundamentalMatrixTransform):
    """Essential matrix transformation.

    The essential matrix relates corresponding points between a pair of
    calibrated images. The matrix transforms normalized, homogeneous image
    points in one image to epipolar lines in the other image.

    The essential matrix is only defined for a pair of moving images capturing a
    non-planar scene. In the case of pure rotation or planar scenes, the
    homography describes the geometric relation between two images
    (`ProjectiveTransform`). If the intrinsic calibration of the images is
    unknown, the fundamental matrix describes the projective relation between
    the two images (`FundamentalMatrixTransform`).

    References
    ----------
    .. [1] Hartley, Richard, and Andrew Zisserman. Multiple view geometry in
           computer vision. Cambridge university press, 2003.

    Parameters
    ----------
    rotation : (3, 3) array_like, optional
        Rotation matrix of the relative camera motion.
    translation : (3, 1) array_like, optional
        Translation vector of the relative camera motion. The vector must
        have unit length.
    matrix : (3, 3) array_like, optional
        Essential matrix.

    Attributes
    ----------
    params : (3, 3) array
        Essential matrix.

    Examples
    --------
    >>> import numpy as np
    >>> import skimage as ski
    >>>
    >>> tform_matrix = ski.transform.EssentialMatrixTransform(
    ...     rotation=np.eye(3), translation=np.array([0, 0, 1])
    ... )
    >>> tform_matrix.params
    array([[ 0., -1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  0.]])
    >>> src = np.array([[ 1.839035, 1.924743],
    ...                 [ 0.543582, 0.375221],
    ...                 [ 0.47324 , 0.142522],
    ...                 [ 0.96491 , 0.598376],
    ...                 [ 0.102388, 0.140092],
    ...                 [15.994343, 9.622164],
    ...                 [ 0.285901, 0.430055],
    ...                 [ 0.09115 , 0.254594]])
    >>> dst = np.array([[1.002114, 1.129644],
    ...                 [1.521742, 1.846002],
    ...                 [1.084332, 0.275134],
    ...                 [0.293328, 0.588992],
    ...                 [0.839509, 0.08729 ],
    ...                 [1.779735, 1.116857],
    ...                 [0.878616, 0.602447],
    ...                 [0.642616, 1.028681]])
    >>> tform_matrix.estimate(src, dst)
    True
    >>> tform_matrix.residuals(src, dst)
    array([0.4245518687, 0.0146044753, 0.1384703409, 0.1214095141,
           0.2775934609, 0.3245311807, 0.0021077555, 0.2651228318])

    """

    def __init__(self, rotation=None, translation=None, matrix=None, *, dimensionality=2):
        if False:
            while True:
                i = 10
        super().__init__(matrix=matrix, dimensionality=dimensionality)
        if rotation is not None:
            rotation = np.asarray(rotation)
            if translation is None:
                raise ValueError('Both rotation and translation required')
            translation = np.asarray(translation)
            if rotation.shape != (3, 3):
                raise ValueError('Invalid shape of rotation matrix')
            if abs(np.linalg.det(rotation) - 1) > 1e-06:
                raise ValueError('Rotation matrix must have unit determinant')
            if translation.size != 3:
                raise ValueError('Invalid shape of translation vector')
            if abs(np.linalg.norm(translation) - 1) > 1e-06:
                raise ValueError('Translation vector must have unit length')
            t_x = np.array([0, -translation[2], translation[1], translation[2], 0, -translation[0], -translation[1], translation[0], 0]).reshape(3, 3)
            self.params = t_x @ rotation
        elif matrix is not None:
            matrix = np.asarray(matrix)
            if matrix.shape != (3, 3):
                raise ValueError('Invalid shape of transformation matrix')
            self.params = matrix
        else:
            self.params = np.eye(3)

    def estimate(self, src, dst):
        if False:
            print('Hello World!')
        'Estimate essential matrix using 8-point algorithm.\n\n        The 8-point algorithm requires at least 8 corresponding point pairs for\n        a well-conditioned solution, otherwise the over-determined solution is\n        estimated.\n\n        Parameters\n        ----------\n        src : (N, 2) array_like\n            Source coordinates.\n        dst : (N, 2) array_like\n            Destination coordinates.\n\n        Returns\n        -------\n        success : bool\n            True, if model estimation succeeds.\n\n        '
        (E_normalized, src_matrix, dst_matrix) = self._setup_constraint_matrix(src, dst)
        (U, S, V) = np.linalg.svd(E_normalized)
        S[0] = (S[0] + S[1]) / 2.0
        S[1] = S[0]
        S[2] = 0
        E = U @ np.diag(S) @ V
        self.params = dst_matrix.T @ E @ src_matrix
        return True

class ProjectiveTransform(_GeometricTransform):
    """Projective transformation.

    Apply a projective transformation (homography) on coordinates.

    For each homogeneous coordinate :math:`\\mathbf{x} = [x, y, 1]^T`, its
    target position is calculated by multiplying with the given matrix,
    :math:`H`, to give :math:`H \\mathbf{x}`::

      [[a0 a1 a2]
       [b0 b1 b2]
       [c0 c1 1 ]].

    E.g., to rotate by theta degrees clockwise, the matrix should be::

      [[cos(theta) -sin(theta) 0]
       [sin(theta)  cos(theta) 0]
       [0            0         1]]

    or, to translate x by 10 and y by 20::

      [[1 0 10]
       [0 1 20]
       [0 0 1 ]].

    Parameters
    ----------
    matrix : (D+1, D+1) array_like, optional
        Homogeneous transformation matrix.
    dimensionality : int, optional
        The number of dimensions of the transform. This is ignored if
        ``matrix`` is not None.

    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.

    """

    def __init__(self, matrix=None, *, dimensionality=2):
        if False:
            print('Hello World!')
        if matrix is None:
            matrix = np.eye(dimensionality + 1)
        else:
            matrix = np.asarray(matrix)
            dimensionality = matrix.shape[0] - 1
            if matrix.shape != (dimensionality + 1, dimensionality + 1):
                raise ValueError('invalid shape of transformation matrix')
        self.params = matrix
        self._coeffs = range(matrix.size - 1)

    @property
    def _inv_matrix(self):
        if False:
            while True:
                i = 10
        return np.linalg.inv(self.params)

    def _apply_mat(self, coords, matrix):
        if False:
            while True:
                i = 10
        ndim = matrix.shape[0] - 1
        coords = np.array(coords, copy=False, ndmin=2)
        src = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
        dst = src @ matrix.T
        dst[dst[:, ndim] == 0, ndim] = np.finfo(float).eps
        dst[:, :ndim] /= dst[:, ndim:ndim + 1]
        return dst[:, :ndim]

    def __array__(self, dtype=None):
        if False:
            for i in range(10):
                print('nop')
        if dtype is None:
            return self.params
        else:
            return self.params.astype(dtype)

    def __call__(self, coords):
        if False:
            while True:
                i = 10
        'Apply forward transformation.\n\n        Parameters\n        ----------\n        coords : (N, D) array_like\n            Source coordinates.\n\n        Returns\n        -------\n        coords_out : (N, D) array\n            Destination coordinates.\n\n        '
        return self._apply_mat(coords, self.params)

    @property
    def inverse(self):
        if False:
            print('Hello World!')
        'Return a transform object representing the inverse.'
        return type(self)(matrix=self._inv_matrix)

    def estimate(self, src, dst, weights=None):
        if False:
            while True:
                i = 10
        'Estimate the transformation from a set of corresponding points.\n\n        You can determine the over-, well- and under-determined parameters\n        with the total least-squares method.\n\n        Number of source and destination coordinates must match.\n\n        The transformation is defined as::\n\n            X = (a0*x + a1*y + a2) / (c0*x + c1*y + 1)\n            Y = (b0*x + b1*y + b2) / (c0*x + c1*y + 1)\n\n        These equations can be transformed to the following form::\n\n            0 = a0*x + a1*y + a2 - c0*x*X - c1*y*X - X\n            0 = b0*x + b1*y + b2 - c0*x*Y - c1*y*Y - Y\n\n        which exist for each set of corresponding points, so we have a set of\n        N * 2 equations. The coefficients appear linearly so we can write\n        A x = 0, where::\n\n            A   = [[x y 1 0 0 0 -x*X -y*X -X]\n                   [0 0 0 x y 1 -x*Y -y*Y -Y]\n                    ...\n                    ...\n                  ]\n            x.T = [a0 a1 a2 b0 b1 b2 c0 c1 c3]\n\n        In case of total least-squares the solution of this homogeneous system\n        of equations is the right singular vector of A which corresponds to the\n        smallest singular value normed by the coefficient c3.\n\n        Weights can be applied to each pair of corresponding points to\n        indicate, particularly in an overdetermined system, if point pairs have\n        higher or lower confidence or uncertainties associated with them. From\n        the matrix treatment of least squares problems, these weight values are\n        normalised, square-rooted, then built into a diagonal matrix, by which\n        A is multiplied.\n\n        In case of the affine transformation the coefficients c0 and c1 are 0.\n        Thus the system of equations is::\n\n            A   = [[x y 1 0 0 0 -X]\n                   [0 0 0 x y 1 -Y]\n                    ...\n                    ...\n                  ]\n            x.T = [a0 a1 a2 b0 b1 b2 c3]\n\n        Parameters\n        ----------\n        src : (N, 2) array_like\n            Source coordinates.\n        dst : (N, 2) array_like\n            Destination coordinates.\n        weights : (N,) array_like, optional\n            Relative weight values for each pair of points.\n\n        Returns\n        -------\n        success : bool\n            True, if model estimation succeeds.\n\n        '
        src = np.asarray(src)
        dst = np.asarray(dst)
        (n, d) = src.shape
        (src_matrix, src) = _center_and_normalize_points(src)
        (dst_matrix, dst) = _center_and_normalize_points(dst)
        if not np.all(np.isfinite(src_matrix + dst_matrix)):
            self.params = np.full((d + 1, d + 1), np.nan)
            return False
        A = np.zeros((n * d, (d + 1) ** 2))
        for ddim in range(d):
            A[ddim * n:(ddim + 1) * n, ddim * (d + 1):ddim * (d + 1) + d] = src
            A[ddim * n:(ddim + 1) * n, ddim * (d + 1) + d] = 1
            A[ddim * n:(ddim + 1) * n, -d - 1:-1] = src
            A[ddim * n:(ddim + 1) * n, -1] = -1
            A[ddim * n:(ddim + 1) * n, -d - 1:] *= -dst[:, ddim:ddim + 1]
        A = A[:, list(self._coeffs) + [-1]]
        if weights is None:
            (_, _, V) = np.linalg.svd(A)
        else:
            weights = np.asarray(weights)
            W = np.diag(np.tile(np.sqrt(weights / np.max(weights)), d))
            (_, _, V) = np.linalg.svd(W @ A)
        if np.isclose(V[-1, -1], 0):
            self.params = np.full((d + 1, d + 1), np.nan)
            return False
        H = np.zeros((d + 1, d + 1))
        H.flat[list(self._coeffs) + [-1]] = -V[-1, :-1] / V[-1, -1]
        H[d, d] = 1
        H = np.linalg.inv(dst_matrix) @ H @ src_matrix
        H /= H[-1, -1]
        self.params = H
        return True

    def __add__(self, other):
        if False:
            print('Hello World!')
        'Combine this transformation with another.'
        if isinstance(other, ProjectiveTransform):
            if type(self) == type(other):
                tform = self.__class__
            else:
                tform = ProjectiveTransform
            return tform(other.params @ self.params)
        else:
            raise TypeError('Cannot combine transformations of differing types.')

    def __nice__(self):
        if False:
            i = 10
            return i + 15
        "common 'paramstr' used by __str__ and __repr__"
        npstring = np.array2string(self.params, separator=', ')
        paramstr = 'matrix=\n' + textwrap.indent(npstring, '    ')
        return paramstr

    def __repr__(self):
        if False:
            print('Hello World!')
        'Add standard repr formatting around a __nice__ string'
        paramstr = self.__nice__()
        classname = self.__class__.__name__
        classstr = classname
        return f'<{classstr}({paramstr}) at {hex(id(self))}>'

    def __str__(self):
        if False:
            i = 10
            return i + 15
        'Add standard str formatting around a __nice__ string'
        paramstr = self.__nice__()
        classname = self.__class__.__name__
        classstr = classname
        return f'<{classstr}({paramstr})>'

    @property
    def dimensionality(self):
        if False:
            i = 10
            return i + 15
        'The dimensionality of the transformation.'
        return self.params.shape[0] - 1

class AffineTransform(ProjectiveTransform):
    """Affine transformation.

    Has the following form::

        X = a0 * x + a1 * y + a2
          =   sx * x * [cos(rotation) + tan(shear_y) * sin(rotation)]
            - sy * y * [tan(shear_x) * cos(rotation) + sin(rotation)]
            + translation_x

        Y = b0 * x + b1 * y + b2
          =   sx * x * [sin(rotation) - tan(shear_y) * cos(rotation)]
            - sy * y * [tan(shear_x) * sin(rotation) - cos(rotation)]
            + translation_y

    where ``sx`` and ``sy`` are scale factors in the x and y directions.

    This is equivalent to applying the operations in the following order:

    1. Scale
    2. Shear
    3. Rotate
    4. Translate

    The homogeneous transformation matrix is::

        [[a0  a1  a2]
         [b0  b1  b2]
         [0   0    1]]

    In 2D, the transformation parameters can be given as the homogeneous
    transformation matrix, above, or as the implicit parameters, scale,
    rotation, shear, and translation in x (a2) and y (b2). For 3D and higher,
    only the matrix form is allowed.

    In narrower transforms, such as the Euclidean (only rotation and
    translation) or Similarity (rotation, translation, and a global scale
    factor) transforms, it is possible to specify 3D transforms using implicit
    parameters also.

    Parameters
    ----------
    matrix : (D+1, D+1) array_like, optional
        Homogeneous transformation matrix. If this matrix is provided, it is an
        error to provide any of scale, rotation, shear, or translation.
    scale : {s as float or (sx, sy) as array, list or tuple}, optional
        Scale factor(s). If a single value, it will be assigned to both
        sx and sy. Only available for 2D.

        .. versionadded:: 0.17
           Added support for supplying a single scalar value.
    rotation : float, optional
        Rotation angle, clockwise, as radians. Only available for 2D.
    shear : float or 2-tuple of float, optional
        The x and y shear angles, clockwise, by which these axes are
        rotated around the origin [2].
        If a single value is given, take that to be the x shear angle, with
        the y angle remaining 0. Only available in 2D.
    translation : (tx, ty) as array, list or tuple, optional
        Translation parameters. Only available for 2D.
    dimensionality : int, optional
        The dimensionality of the transform. This is not used if any other
        parameters are provided.

    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.

    Raises
    ------
    ValueError
        If both ``matrix`` and any of the other parameters are provided.

    Examples
    --------
    >>> import numpy as np
    >>> import skimage as ski
    >>> img = ski.data.astronaut()

    Define source and destination points:

    >>> src = np.array([[150, 150],
    ...                 [250, 100],
    ...                 [150, 200]])
    >>> dst = np.array([[200, 200],
    ...                 [300, 150],
    ...                 [150, 400]])

    Estimate the transformation matrix:

    >>> tform = ski.transform.AffineTransform()
    >>> tform.estimate(src, dst)
    True

    Apply the transformation:

    >>> warped = ski.transform.warp(img, inverse_map=tform.inverse)

    References
    ----------
    .. [1] Wikipedia, "Affine transformation",
           https://en.wikipedia.org/wiki/Affine_transformation#Image_transformation
    .. [2] Wikipedia, "Shear mapping",
           https://en.wikipedia.org/wiki/Shear_mapping
    """

    def __init__(self, matrix=None, scale=None, rotation=None, shear=None, translation=None, *, dimensionality=2):
        if False:
            i = 10
            return i + 15
        params = any((param is not None for param in (scale, rotation, shear, translation)))
        self._coeffs = range(dimensionality * (dimensionality + 1))
        if params and matrix is not None:
            raise ValueError('You cannot specify the transformation matrix and the implicit parameters at the same time.')
        if params and dimensionality > 2:
            raise ValueError('Parameter input is only supported in 2D.')
        elif matrix is not None:
            matrix = np.asarray(matrix)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError('Invalid shape of transformation matrix.')
            else:
                dimensionality = matrix.shape[0] - 1
                nparam = dimensionality * (dimensionality + 1)
            self._coeffs = range(nparam)
            self.params = matrix
        elif params:
            if scale is None:
                scale = (1, 1)
            if rotation is None:
                rotation = 0
            if shear is None:
                shear = 0
            if translation is None:
                translation = (0, 0)
            if np.isscalar(scale):
                sx = sy = scale
            else:
                (sx, sy) = scale
            if np.isscalar(shear):
                (shear_x, shear_y) = (shear, 0)
            else:
                (shear_x, shear_y) = shear
            a0 = sx * (math.cos(rotation) + math.tan(shear_y) * math.sin(rotation))
            a1 = -sy * (math.tan(shear_x) * math.cos(rotation) + math.sin(rotation))
            a2 = translation[0]
            b0 = sx * (math.sin(rotation) - math.tan(shear_y) * math.cos(rotation))
            b1 = -sy * (math.tan(shear_x) * math.sin(rotation) - math.cos(rotation))
            b2 = translation[1]
            self.params = np.array([[a0, a1, a2], [b0, b1, b2], [0, 0, 1]])
        else:
            self.params = np.eye(dimensionality + 1)

    @property
    def scale(self):
        if False:
            i = 10
            return i + 15
        if self.dimensionality != 2:
            return np.sqrt(np.sum(self.params ** 2, axis=0))[:self.dimensionality]
        else:
            ss = np.sum(self.params ** 2, axis=0)
            ss[1] = ss[1] / (math.tan(self.shear) ** 2 + 1)
            return np.sqrt(ss)[:self.dimensionality]

    @property
    def rotation(self):
        if False:
            print('Hello World!')
        if self.dimensionality != 2:
            raise NotImplementedError('The rotation property is only implemented for 2D transforms.')
        return math.atan2(self.params[1, 0], self.params[0, 0])

    @property
    def shear(self):
        if False:
            return 10
        if self.dimensionality != 2:
            raise NotImplementedError('The shear property is only implemented for 2D transforms.')
        beta = math.atan2(-self.params[0, 1], self.params[1, 1])
        return beta - self.rotation

    @property
    def translation(self):
        if False:
            return 10
        return self.params[0:self.dimensionality, self.dimensionality]

class PiecewiseAffineTransform(_GeometricTransform):
    """Piecewise affine transformation.

    Control points are used to define the mapping. The transform is based on
    a Delaunay triangulation of the points to form a mesh. Each triangle is
    used to find a local affine transform.

    Attributes
    ----------
    affines : list of AffineTransform objects
        Affine transformations for each triangle in the mesh.
    inverse_affines : list of AffineTransform objects
        Inverse affine transformations for each triangle in the mesh.

    """

    def __init__(self):
        if False:
            return 10
        self._tesselation = None
        self._inverse_tesselation = None
        self.affines = None
        self.inverse_affines = None

    def estimate(self, src, dst):
        if False:
            while True:
                i = 10
        'Estimate the transformation from a set of corresponding points.\n\n        Number of source and destination coordinates must match.\n\n        Parameters\n        ----------\n        src : (N, D) array_like\n            Source coordinates.\n        dst : (N, D) array_like\n            Destination coordinates.\n\n        Returns\n        -------\n        success : bool\n            True, if all pieces of the model are successfully estimated.\n\n        '
        src = np.asarray(src)
        dst = np.asarray(dst)
        ndim = src.shape[1]
        self._tesselation = spatial.Delaunay(src)
        success = True
        self.affines = []
        for tri in self._tesselation.simplices:
            affine = AffineTransform(dimensionality=ndim)
            success &= affine.estimate(src[tri, :], dst[tri, :])
            self.affines.append(affine)
        self._inverse_tesselation = spatial.Delaunay(dst)
        self.inverse_affines = []
        for tri in self._inverse_tesselation.simplices:
            affine = AffineTransform(dimensionality=ndim)
            success &= affine.estimate(dst[tri, :], src[tri, :])
            self.inverse_affines.append(affine)
        return success

    def __call__(self, coords):
        if False:
            i = 10
            return i + 15
        'Apply forward transformation.\n\n        Coordinates outside of the mesh will be set to `- 1`.\n\n        Parameters\n        ----------\n        coords : (N, D) array_like\n            Source coordinates.\n\n        Returns\n        -------\n        coords : (N, 2) array\n            Transformed coordinates.\n\n        '
        coords = np.asarray(coords)
        out = np.empty_like(coords, np.float64)
        simplex = self._tesselation.find_simplex(coords)
        out[simplex == -1, :] = -1
        for index in range(len(self._tesselation.simplices)):
            affine = self.affines[index]
            index_mask = simplex == index
            out[index_mask, :] = affine(coords[index_mask, :])
        return out

    @property
    def inverse(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a transform object representing the inverse.'
        tform = type(self)()
        tform._tesselation = self._inverse_tesselation
        tform._inverse_tesselation = self._tesselation
        tform.affines = self.inverse_affines
        tform.inverse_affines = self.affines
        return tform

def _euler_rotation(axis, angle):
    if False:
        return 10
    'Produce a single-axis Euler rotation matrix.\n\n    Parameters\n    ----------\n    axis : int in {0, 1, 2}\n        The axis of rotation.\n    angle : float\n        The angle of rotation in radians.\n\n    Returns\n    -------\n    Ri : array of float, shape (3, 3)\n        The rotation matrix along axis `axis`.\n    '
    i = axis
    s = (-1) ** i * np.sin(angle)
    c = np.cos(angle)
    R2 = np.array([[c, -s], [s, c]])
    Ri = np.eye(3)
    axes = sorted({0, 1, 2} - {axis})
    sl = slice(axes[0], axes[1] + 1, axes[1] - axes[0])
    Ri[sl, sl] = R2
    return Ri

def _euler_rotation_matrix(angles, axes=None):
    if False:
        while True:
            i = 10
    'Produce an Euler rotation matrix from the given angles.\n\n    The matrix will have dimension equal to the number of angles given.\n\n    Parameters\n    ----------\n    angles : array of float, shape (3,)\n        The transformation angles in radians.\n    axes : list of int\n        The axes about which to produce the rotation. Defaults to 0, 1, 2.\n\n    Returns\n    -------\n    R : array of float, shape (3, 3)\n        The Euler rotation matrix.\n    '
    if axes is None:
        axes = range(3)
    dim = len(angles)
    R = np.eye(dim)
    for (i, angle) in zip(axes, angles):
        R = R @ _euler_rotation(i, angle)
    return R

class EuclideanTransform(ProjectiveTransform):
    """Euclidean transformation, also known as a rigid transform.

    Has the following form::

        X = a0 * x - b0 * y + a1 =
          = x * cos(rotation) - y * sin(rotation) + a1

        Y = b0 * x + a0 * y + b1 =
          = x * sin(rotation) + y * cos(rotation) + b1

    where the homogeneous transformation matrix is::

        [[a0  b0  a1]
         [b0  a0  b1]
         [0   0    1]]

    The Euclidean transformation is a rigid transformation with rotation and
    translation parameters. The similarity transformation extends the Euclidean
    transformation with a single scaling factor.

    In 2D and 3D, the transformation parameters may be provided either via
    `matrix`, the homogeneous transformation matrix, above, or via the
    implicit parameters `rotation` and/or `translation` (where `a1` is the
    translation along `x`, `b1` along `y`, etc.). Beyond 3D, if the
    transformation is only a translation, you may use the implicit parameter
    `translation`; otherwise, you must use `matrix`.

    Parameters
    ----------
    matrix : (D+1, D+1) array_like, optional
        Homogeneous transformation matrix.
    rotation : float or sequence of float, optional
        Rotation angle, clockwise, as radians. If given as
        a vector, it is interpreted as Euler rotation angles [1]_. Only 2D
        (single rotation) and 3D (Euler rotations) values are supported. For
        higher dimensions, you must provide or estimate the transformation
        matrix.
    translation : (x, y[, z, ...]) sequence of float, length D, optional
        Translation parameters for each axis.
    dimensionality : int, optional
        The dimensionality of the transform.

    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    """

    def __init__(self, matrix=None, rotation=None, translation=None, *, dimensionality=2):
        if False:
            print('Hello World!')
        params_given = rotation is not None or translation is not None
        if params_given and matrix is not None:
            raise ValueError('You cannot specify the transformation matrix and the implicit parameters at the same time.')
        elif matrix is not None:
            matrix = np.asarray(matrix)
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError('Invalid shape of transformation matrix.')
            self.params = matrix
        elif params_given:
            if rotation is None:
                dimensionality = len(translation)
                if dimensionality == 2:
                    rotation = 0
                elif dimensionality == 3:
                    rotation = np.zeros(3)
                else:
                    raise ValueError(f'Parameters cannot be specified for dimension {dimensionality} transforms')
            elif not np.isscalar(rotation) and len(rotation) != 3:
                raise ValueError(f'Parameters cannot be specified for dimension {dimensionality} transforms')
            if translation is None:
                translation = (0,) * dimensionality
            if dimensionality == 2:
                self.params = np.array([[math.cos(rotation), -math.sin(rotation), 0], [math.sin(rotation), math.cos(rotation), 0], [0, 0, 1]])
            elif dimensionality == 3:
                self.params = np.eye(dimensionality + 1)
                self.params[:dimensionality, :dimensionality] = _euler_rotation_matrix(rotation)
            self.params[0:dimensionality, dimensionality] = translation
        else:
            self.params = np.eye(dimensionality + 1)

    def estimate(self, src, dst):
        if False:
            for i in range(10):
                print('nop')
        'Estimate the transformation from a set of corresponding points.\n\n        You can determine the over-, well- and under-determined parameters\n        with the total least-squares method.\n\n        Number of source and destination coordinates must match.\n\n        Parameters\n        ----------\n        src : (N, 2) array_like\n            Source coordinates.\n        dst : (N, 2) array_like\n            Destination coordinates.\n\n        Returns\n        -------\n        success : bool\n            True, if model estimation succeeds.\n\n        '
        self.params = _umeyama(src, dst, False)
        return not np.any(np.isnan(self.params))

    @property
    def rotation(self):
        if False:
            for i in range(10):
                print('nop')
        if self.dimensionality == 2:
            return math.atan2(self.params[1, 0], self.params[1, 1])
        elif self.dimensionality == 3:
            return self.params[:3, :3]
        else:
            raise NotImplementedError('Rotation only implemented for 2D and 3D transforms.')

    @property
    def translation(self):
        if False:
            return 10
        return self.params[0:self.dimensionality, self.dimensionality]

class SimilarityTransform(EuclideanTransform):
    """Similarity transformation.

    Has the following form in 2D::

        X = a0 * x - b0 * y + a1 =
          = s * x * cos(rotation) - s * y * sin(rotation) + a1

        Y = b0 * x + a0 * y + b1 =
          = s * x * sin(rotation) + s * y * cos(rotation) + b1

    where ``s`` is a scale factor and the homogeneous transformation matrix is::

        [[a0  -b0  a1]
         [b0  a0  b1]
         [0   0    1]]

    The similarity transformation extends the Euclidean transformation with a
    single scaling factor in addition to the rotation and translation
    parameters.

    Parameters
    ----------
    matrix : (dim+1, dim+1) array_like, optional
        Homogeneous transformation matrix.
    scale : float, optional
        Scale factor. Implemented only for 2D and 3D.
    rotation : float, optional
        Rotation angle, clockwise, as radians.
        Implemented only for 2D and 3D. For 3D, this is given in ZYX Euler
        angles.
    translation : (dim,) array_like, optional
        x, y[, z] translation parameters. Implemented only for 2D and 3D.

    Attributes
    ----------
    params : (dim+1, dim+1) array
        Homogeneous transformation matrix.

    """

    def __init__(self, matrix=None, scale=None, rotation=None, translation=None, *, dimensionality=2):
        if False:
            i = 10
            return i + 15
        self.params = None
        params = any((param is not None for param in (scale, rotation, translation)))
        if params and matrix is not None:
            raise ValueError('You cannot specify the transformation matrix and the implicit parameters at the same time.')
        elif matrix is not None:
            matrix = np.asarray(matrix)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError('Invalid shape of transformation matrix.')
            else:
                self.params = matrix
                dimensionality = matrix.shape[0] - 1
        if params:
            if dimensionality not in (2, 3):
                raise ValueError('Parameters only supported for 2D and 3D.')
            matrix = np.eye(dimensionality + 1, dtype=float)
            if scale is None:
                scale = 1
            if rotation is None:
                rotation = 0 if dimensionality == 2 else (0, 0, 0)
            if translation is None:
                translation = (0,) * dimensionality
            if dimensionality == 2:
                ax = (0, 1)
                (c, s) = (np.cos(rotation), np.sin(rotation))
                matrix[ax, ax] = c
                matrix[ax, ax[::-1]] = (-s, s)
            else:
                matrix[:3, :3] = _euler_rotation_matrix(rotation)
            matrix[:dimensionality, :dimensionality] *= scale
            matrix[:dimensionality, dimensionality] = translation
            self.params = matrix
        elif self.params is None:
            self.params = np.eye(dimensionality + 1)

    def estimate(self, src, dst):
        if False:
            i = 10
            return i + 15
        'Estimate the transformation from a set of corresponding points.\n\n        You can determine the over-, well- and under-determined parameters\n        with the total least-squares method.\n\n        Number of source and destination coordinates must match.\n\n        Parameters\n        ----------\n        src : (N, 2) array_like\n            Source coordinates.\n        dst : (N, 2) array_like\n            Destination coordinates.\n\n        Returns\n        -------\n        success : bool\n            True, if model estimation succeeds.\n\n        '
        self.params = _umeyama(src, dst, estimate_scale=True)
        return not np.any(np.isnan(self.params))

    @property
    def scale(self):
        if False:
            print('Hello World!')
        if self.dimensionality == 2:
            return np.sqrt(np.linalg.det(self.params))
        elif self.dimensionality == 3:
            return np.cbrt(np.linalg.det(self.params))
        else:
            raise NotImplementedError('Scale is only implemented for 2D and 3D.')

class PolynomialTransform(_GeometricTransform):
    """2D polynomial transformation.

    Has the following form::

        X = sum[j=0:order]( sum[i=0:j]( a_ji * x**(j - i) * y**i ))
        Y = sum[j=0:order]( sum[i=0:j]( b_ji * x**(j - i) * y**i ))

    Parameters
    ----------
    params : (2, N) array_like, optional
        Polynomial coefficients where `N * 2 = (order + 1) * (order + 2)`. So,
        a_ji is defined in `params[0, :]` and b_ji in `params[1, :]`.

    Attributes
    ----------
    params : (2, N) array
        Polynomial coefficients where `N * 2 = (order + 1) * (order + 2)`. So,
        a_ji is defined in `params[0, :]` and b_ji in `params[1, :]`.

    """

    def __init__(self, params=None, *, dimensionality=2):
        if False:
            return 10
        if dimensionality != 2:
            raise NotImplementedError('Polynomial transforms are only implemented for 2D.')
        if params is None:
            params = np.array([[0, 1, 0], [0, 0, 1]])
        else:
            params = np.asarray(params)
        if params.shape[0] != 2:
            raise ValueError('invalid shape of transformation parameters')
        self.params = params

    def estimate(self, src, dst, order=2, weights=None):
        if False:
            print('Hello World!')
        'Estimate the transformation from a set of corresponding points.\n\n        You can determine the over-, well- and under-determined parameters\n        with the total least-squares method.\n\n        Number of source and destination coordinates must match.\n\n        The transformation is defined as::\n\n            X = sum[j=0:order]( sum[i=0:j]( a_ji * x**(j - i) * y**i ))\n            Y = sum[j=0:order]( sum[i=0:j]( b_ji * x**(j - i) * y**i ))\n\n        These equations can be transformed to the following form::\n\n            0 = sum[j=0:order]( sum[i=0:j]( a_ji * x**(j - i) * y**i )) - X\n            0 = sum[j=0:order]( sum[i=0:j]( b_ji * x**(j - i) * y**i )) - Y\n\n        which exist for each set of corresponding points, so we have a set of\n        N * 2 equations. The coefficients appear linearly so we can write\n        A x = 0, where::\n\n            A   = [[1 x y x**2 x*y y**2 ... 0 ...             0 -X]\n                   [0 ...                 0 1 x y x**2 x*y y**2 -Y]\n                    ...\n                    ...\n                  ]\n            x.T = [a00 a10 a11 a20 a21 a22 ... ann\n                   b00 b10 b11 b20 b21 b22 ... bnn c3]\n\n        In case of total least-squares the solution of this homogeneous system\n        of equations is the right singular vector of A which corresponds to the\n        smallest singular value normed by the coefficient c3.\n\n        Weights can be applied to each pair of corresponding points to\n        indicate, particularly in an overdetermined system, if point pairs have\n        higher or lower confidence or uncertainties associated with them. From\n        the matrix treatment of least squares problems, these weight values are\n        normalised, square-rooted, then built into a diagonal matrix, by which\n        A is multiplied.\n\n        Parameters\n        ----------\n        src : (N, 2) array_like\n            Source coordinates.\n        dst : (N, 2) array_like\n            Destination coordinates.\n        order : int, optional\n            Polynomial order (number of coefficients is order + 1).\n        weights : (N,) array_like, optional\n            Relative weight values for each pair of points.\n\n        Returns\n        -------\n        success : bool\n            True, if model estimation succeeds.\n\n        '
        src = np.asarray(src)
        dst = np.asarray(dst)
        xs = src[:, 0]
        ys = src[:, 1]
        xd = dst[:, 0]
        yd = dst[:, 1]
        rows = src.shape[0]
        order = safe_as_int(order)
        u = (order + 1) * (order + 2)
        A = np.zeros((rows * 2, u + 1))
        pidx = 0
        for j in range(order + 1):
            for i in range(j + 1):
                A[:rows, pidx] = xs ** (j - i) * ys ** i
                A[rows:, pidx + u // 2] = xs ** (j - i) * ys ** i
                pidx += 1
        A[:rows, -1] = xd
        A[rows:, -1] = yd
        if weights is None:
            (_, _, V) = np.linalg.svd(A)
        else:
            weights = np.asarray(weights)
            W = np.diag(np.tile(np.sqrt(weights / np.max(weights)), 2))
            (_, _, V) = np.linalg.svd(W @ A)
        params = -V[-1, :-1] / V[-1, -1]
        self.params = params.reshape((2, u // 2))
        return True

    def __call__(self, coords):
        if False:
            print('Hello World!')
        'Apply forward transformation.\n\n        Parameters\n        ----------\n        coords : (N, 2) array_like\n            source coordinates\n\n        Returns\n        -------\n        coords : (N, 2) array\n            Transformed coordinates.\n\n        '
        coords = np.asarray(coords)
        x = coords[:, 0]
        y = coords[:, 1]
        u = len(self.params.ravel())
        order = int((-3 + math.sqrt(9 - 4 * (2 - u))) / 2)
        dst = np.zeros(coords.shape)
        pidx = 0
        for j in range(order + 1):
            for i in range(j + 1):
                dst[:, 0] += self.params[0, pidx] * x ** (j - i) * y ** i
                dst[:, 1] += self.params[1, pidx] * x ** (j - i) * y ** i
                pidx += 1
        return dst

    @property
    def inverse(self):
        if False:
            print('Hello World!')
        raise NotImplementedError('There is no explicit way to do the inverse polynomial transformation. Instead, estimate the inverse transformation parameters by exchanging source and destination coordinates,then apply the forward transformation.')
TRANSFORMS = {'euclidean': EuclideanTransform, 'similarity': SimilarityTransform, 'affine': AffineTransform, 'piecewise-affine': PiecewiseAffineTransform, 'projective': ProjectiveTransform, 'fundamental': FundamentalMatrixTransform, 'essential': EssentialMatrixTransform, 'polynomial': PolynomialTransform}

def estimate_transform(ttype, src, dst, *args, **kwargs):
    if False:
        print('Hello World!')
    "Estimate 2D geometric transformation parameters.\n\n    You can determine the over-, well- and under-determined parameters\n    with the total least-squares method.\n\n    Number of source and destination coordinates must match.\n\n    Parameters\n    ----------\n    ttype : {'euclidean', similarity', 'affine', 'piecewise-affine',              'projective', 'polynomial'}\n        Type of transform.\n    kwargs : array_like or int\n        Function parameters (src, dst, n, angle)::\n\n            NAME / TTYPE        FUNCTION PARAMETERS\n            'euclidean'         `src, `dst`\n            'similarity'        `src, `dst`\n            'affine'            `src, `dst`\n            'piecewise-affine'  `src, `dst`\n            'projective'        `src, `dst`\n            'polynomial'        `src, `dst`, `order` (polynomial order,\n                                                      default order is 2)\n\n        Also see examples below.\n\n    Returns\n    -------\n    tform : :class:`_GeometricTransform`\n        Transform object containing the transformation parameters and providing\n        access to forward and inverse transformation functions.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> import skimage as ski\n\n    >>> # estimate transformation parameters\n    >>> src = np.array([0, 0, 10, 10]).reshape((2, 2))\n    >>> dst = np.array([12, 14, 1, -20]).reshape((2, 2))\n\n    >>> tform = ski.transform.estimate_transform('similarity', src, dst)\n\n    >>> np.allclose(tform.inverse(tform(src)), src)\n    True\n\n    >>> # warp image using the estimated transformation\n    >>> image = ski.data.camera()\n\n    >>> ski.transform.warp(image, inverse_map=tform.inverse) # doctest: +SKIP\n\n    >>> # create transformation with explicit parameters\n    >>> tform2 = ski.transform.SimilarityTransform(scale=1.1, rotation=1,\n    ...     translation=(10, 20))\n\n    >>> # unite transformations, applied in order from left to right\n    >>> tform3 = tform + tform2\n    >>> np.allclose(tform3(src), tform2(tform(src)))\n    True\n\n    "
    ttype = ttype.lower()
    if ttype not in TRANSFORMS:
        raise ValueError(f"the transformation type '{ttype}' is not implemented")
    tform = TRANSFORMS[ttype](dimensionality=src.shape[1])
    tform.estimate(src, dst, *args, **kwargs)
    return tform

def matrix_transform(coords, matrix):
    if False:
        return 10
    'Apply 2D matrix transform.\n\n    Parameters\n    ----------\n    coords : (N, 2) array_like\n        x, y coordinates to transform\n    matrix : (3, 3) array_like\n        Homogeneous transformation matrix.\n\n    Returns\n    -------\n    coords : (N, 2) array\n        Transformed coordinates.\n\n    '
    return ProjectiveTransform(matrix)(coords)