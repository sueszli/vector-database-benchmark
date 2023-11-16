import copy
import math
import random
import threading
import time
import numpy as np
import PIL
import scipy
import scipy.ndimage as ndi
import skimage
from scipy import linalg
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from six.moves import range
from skimage import exposure, transform
from skimage.morphology import binary_dilation as _binary_dilation
from skimage.morphology import binary_erosion as _binary_erosion
from skimage.morphology import disk
from skimage.morphology import erosion as _erosion
import tensorlayer as tl
from tensorlayer.lazy_imports import LazyImport
cv2 = LazyImport('cv2')
__all__ = ['threading_data', 'affine_rotation_matrix', 'affine_horizontal_flip_matrix', 'affine_shift_matrix', 'affine_shear_matrix', 'affine_zoom_matrix', 'affine_respective_zoom_matrix', 'transform_matrix_offset_center', 'affine_transform', 'affine_transform_cv2', 'affine_transform_keypoints', 'projective_transform_by_points', 'rotation', 'rotation_multi', 'crop', 'crop_multi', 'flip_axis', 'flip_axis_multi', 'shift', 'shift_multi', 'shear', 'shear_multi', 'shear2', 'shear_multi2', 'swirl', 'swirl_multi', 'elastic_transform', 'elastic_transform_multi', 'zoom', 'respective_zoom', 'zoom_multi', 'brightness', 'brightness_multi', 'illumination', 'rgb_to_hsv', 'hsv_to_rgb', 'adjust_hue', 'imresize', 'pixel_value_scale', 'samplewise_norm', 'featurewise_norm', 'get_zca_whitening_principal_components_img', 'zca_whitening', 'channel_shift', 'channel_shift_multi', 'drop', 'array_to_img', 'find_contours', 'pt2map', 'binary_dilation', 'dilation', 'binary_erosion', 'erosion', 'obj_box_coords_rescale', 'obj_box_coord_rescale', 'obj_box_coord_scale_to_pixelunit', 'obj_box_coord_centroid_to_upleft_butright', 'obj_box_coord_upleft_butright_to_centroid', 'obj_box_coord_centroid_to_upleft', 'obj_box_coord_upleft_to_centroid', 'parse_darknet_ann_str_to_list', 'parse_darknet_ann_list_to_cls_box', 'obj_box_left_right_flip', 'obj_box_imresize', 'obj_box_crop', 'obj_box_shift', 'obj_box_zoom', 'pad_sequences', 'remove_pad_sequences', 'process_sequences', 'sequences_add_start_id', 'sequences_add_end_id', 'sequences_add_end_id_after_pad', 'sequences_get_mask', 'keypoint_random_crop', 'keypoint_resize_random_crop', 'keypoint_random_rotate', 'keypoint_random_flip', 'keypoint_random_resize', 'keypoint_random_resize_shortestedge']

def threading_data(data=None, fn=None, thread_count=None, **kwargs):
    if False:
        return 10
    "Process a batch of data by given function by threading.\n\n    Usually be used for data augmentation.\n\n    Parameters\n    -----------\n    data : numpy.array or others\n        The data to be processed.\n    thread_count : int\n        The number of threads to use.\n    fn : function\n        The function for data processing.\n    more args : the args for `fn`\n        Ssee Examples below.\n\n    Examples\n    --------\n    Process images.\n\n    >>> images, _, _, _ = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3))\n    >>> images = tl.prepro.threading_data(images[0:32], tl.prepro.zoom, zoom_range=[0.5, 1])\n\n    Customized image preprocessing function.\n\n    >>> def distort_img(x):\n    >>>     x = tl.prepro.flip_axis(x, axis=0, is_random=True)\n    >>>     x = tl.prepro.flip_axis(x, axis=1, is_random=True)\n    >>>     x = tl.prepro.crop(x, 100, 100, is_random=True)\n    >>>     return x\n    >>> images = tl.prepro.threading_data(images, distort_img)\n\n    Process images and masks together (Usually be used for image segmentation).\n\n    >>> X, Y --> [batch_size, row, col, 1]\n    >>> data = tl.prepro.threading_data([_ for _ in zip(X, Y)], tl.prepro.zoom_multi, zoom_range=[0.5, 1], is_random=True)\n    data --> [batch_size, 2, row, col, 1]\n    >>> X_, Y_ = data.transpose((1,0,2,3,4))\n    X_, Y_ --> [batch_size, row, col, 1]\n    >>> tl.vis.save_image(X_, 'images.png')\n    >>> tl.vis.save_image(Y_, 'masks.png')\n\n    Process images and masks together by using ``thread_count``.\n\n    >>> X, Y --> [batch_size, row, col, 1]\n    >>> data = tl.prepro.threading_data(X, tl.prepro.zoom_multi, 8, zoom_range=[0.5, 1], is_random=True)\n    data --> [batch_size, 2, row, col, 1]\n    >>> X_, Y_ = data.transpose((1,0,2,3,4))\n    X_, Y_ --> [batch_size, row, col, 1]\n    >>> tl.vis.save_image(X_, 'after.png')\n    >>> tl.vis.save_image(Y_, 'before.png')\n\n    Customized function for processing images and masks together.\n\n    >>> def distort_img(data):\n    >>>    x, y = data\n    >>>    x, y = tl.prepro.flip_axis_multi([x, y], axis=0, is_random=True)\n    >>>    x, y = tl.prepro.flip_axis_multi([x, y], axis=1, is_random=True)\n    >>>    x, y = tl.prepro.crop_multi([x, y], 100, 100, is_random=True)\n    >>>    return x, y\n\n    >>> X, Y --> [batch_size, row, col, channel]\n    >>> data = tl.prepro.threading_data([_ for _ in zip(X, Y)], distort_img)\n    >>> X_, Y_ = data.transpose((1,0,2,3,4))\n\n    Returns\n    -------\n    list or numpyarray\n        The processed results.\n\n    References\n    ----------\n    - `python queue <https://pymotw.com/2/Queue/index.html#module-Queue>`__\n    - `run with limited queue <http://effbot.org/librarybook/queue.htm>`__\n\n    "

    def apply_fn(results, i, data, kwargs):
        if False:
            while True:
                i = 10
        results[i] = fn(data, **kwargs)
    if thread_count is None:
        results = [None] * len(data)
        threads = []
        for (i, d) in enumerate(data):
            t = threading.Thread(name='threading_and_return', target=apply_fn, args=(results, i, d, kwargs))
            t.start()
            threads.append(t)
    else:
        divs = np.linspace(0, len(data), thread_count + 1)
        divs = np.round(divs).astype(int)
        results = [None] * thread_count
        threads = []
        for i in range(thread_count):
            t = threading.Thread(name='threading_and_return', target=apply_fn, args=(results, i, data[divs[i]:divs[i + 1]], kwargs))
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    if thread_count is None:
        try:
            return np.asarray(results)
        except Exception:
            return results
    else:
        return np.concatenate(results)

def affine_rotation_matrix(angle=(-20, 20)):
    if False:
        while True:
            i = 10
    'Create an affine transform matrix for image rotation.\n    NOTE: In OpenCV, x is width and y is height.\n\n    Parameters\n    -----------\n    angle : int/float or tuple of two int/float\n        Degree to rotate, usually -180 ~ 180.\n            - int/float, a fixed angle.\n            - tuple of 2 floats/ints, randomly sample a value as the angle between these 2 values.\n\n    Returns\n    -------\n    numpy.array\n        An affine transform matrix.\n\n    '
    if isinstance(angle, tuple):
        theta = np.pi / 180 * np.random.uniform(angle[0], angle[1])
    else:
        theta = np.pi / 180 * angle
    rotation_matrix = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return rotation_matrix

def affine_horizontal_flip_matrix(prob=0.5):
    if False:
        while True:
            i = 10
    'Create an affine transformation matrix for image horizontal flipping.\n    NOTE: In OpenCV, x is width and y is height.\n\n    Parameters\n    ----------\n    prob : float\n        Probability to flip the image. 1.0 means always flip.\n\n    Returns\n    -------\n    numpy.array\n        An affine transform matrix.\n\n    '
    factor = np.random.uniform(0, 1)
    if prob >= factor:
        filp_matrix = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        return filp_matrix
    else:
        filp_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        return filp_matrix

def affine_vertical_flip_matrix(prob=0.5):
    if False:
        for i in range(10):
            print('nop')
    'Create an affine transformation for image vertical flipping.\n    NOTE: In OpenCV, x is width and y is height.\n\n    Parameters\n    ----------\n    prob : float\n        Probability to flip the image. 1.0 means always flip.\n\n    Returns\n    -------\n    numpy.array\n        An affine transform matrix.\n\n    '
    factor = np.random.uniform(0, 1)
    if prob >= factor:
        filp_matrix = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        return filp_matrix
    else:
        filp_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        return filp_matrix

def affine_shift_matrix(wrg=(-0.1, 0.1), hrg=(-0.1, 0.1), w=200, h=200):
    if False:
        i = 10
        return i + 15
    'Create an affine transform matrix for image shifting.\n    NOTE: In OpenCV, x is width and y is height.\n\n    Parameters\n    -----------\n    wrg : float or tuple of floats\n        Range to shift on width axis, -1 ~ 1.\n            - float, a fixed distance.\n            - tuple of 2 floats, randomly sample a value as the distance between these 2 values.\n    hrg : float or tuple of floats\n        Range to shift on height axis, -1 ~ 1.\n            - float, a fixed distance.\n            - tuple of 2 floats, randomly sample a value as the distance between these 2 values.\n    w, h : int\n        The width and height of the image.\n\n    Returns\n    -------\n    numpy.array\n        An affine transform matrix.\n\n    '
    if isinstance(wrg, tuple):
        tx = np.random.uniform(wrg[0], wrg[1]) * w
    else:
        tx = wrg * w
    if isinstance(hrg, tuple):
        ty = np.random.uniform(hrg[0], hrg[1]) * h
    else:
        ty = hrg * h
    shift_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    return shift_matrix

def affine_shear_matrix(x_shear=(-0.1, 0.1), y_shear=(-0.1, 0.1)):
    if False:
        print('Hello World!')
    'Create affine transform matrix for image shearing.\n    NOTE: In OpenCV, x is width and y is height.\n\n    Parameters\n    -----------\n    shear : tuple of two floats\n        Percentage of shears for width and height directions.\n\n    Returns\n    -------\n    numpy.array\n        An affine transform matrix.\n\n    '
    if isinstance(x_shear, tuple):
        x_shear = np.random.uniform(x_shear[0], x_shear[1])
    if isinstance(y_shear, tuple):
        y_shear = np.random.uniform(y_shear[0], y_shear[1])
    shear_matrix = np.array([[1, x_shear, 0], [y_shear, 1, 0], [0, 0, 1]])
    return shear_matrix

def affine_zoom_matrix(zoom_range=(0.8, 1.1)):
    if False:
        while True:
            i = 10
    "Create an affine transform matrix for zooming/scaling an image's height and width.\n    OpenCV format, x is width.\n\n    Parameters\n    -----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    zoom_range : float or tuple of 2 floats\n        The zooming/scaling ratio, greater than 1 means larger.\n            - float, a fixed ratio.\n            - tuple of 2 floats, randomly sample a value as the ratio between these 2 values.\n\n    Returns\n    -------\n    numpy.array\n        An affine transform matrix.\n\n    "
    if isinstance(zoom_range, (float, int)):
        scale = zoom_range
    elif isinstance(zoom_range, tuple):
        scale = np.random.uniform(zoom_range[0], zoom_range[1])
    else:
        raise Exception('zoom_range: float or tuple of 2 floats')
    zoom_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    return zoom_matrix

def affine_respective_zoom_matrix(w_range=0.8, h_range=1.1):
    if False:
        return 10
    'Get affine transform matrix for zooming/scaling that height and width are changed independently.\n    OpenCV format, x is width.\n\n    Parameters\n    -----------\n    w_range : float or tuple of 2 floats\n        The zooming/scaling ratio of width, greater than 1 means larger.\n            - float, a fixed ratio.\n            - tuple of 2 floats, randomly sample a value as the ratio between 2 values.\n    h_range : float or tuple of 2 floats\n        The zooming/scaling ratio of height, greater than 1 means larger.\n            - float, a fixed ratio.\n            - tuple of 2 floats, randomly sample a value as the ratio between 2 values.\n\n    Returns\n    -------\n    numpy.array\n        An affine transform matrix.\n\n    '
    if isinstance(h_range, (float, int)):
        zy = h_range
    elif isinstance(h_range, tuple):
        zy = np.random.uniform(h_range[0], h_range[1])
    else:
        raise Exception('h_range: float or tuple of 2 floats')
    if isinstance(w_range, (float, int)):
        zx = w_range
    elif isinstance(w_range, tuple):
        zx = np.random.uniform(w_range[0], w_range[1])
    else:
        raise Exception('w_range: float or tuple of 2 floats')
    zoom_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
    return zoom_matrix

def transform_matrix_offset_center(matrix, y, x):
    if False:
        print('Hello World!')
    'Convert the matrix from Cartesian coordinates (the origin in the middle of image) to Image coordinates (the origin on the top-left of image).\n\n    Parameters\n    ----------\n    matrix : numpy.array\n        Transform matrix.\n    x and y : 2 int\n        Size of image.\n\n    Returns\n    -------\n    numpy.array\n        The transform matrix.\n\n    Examples\n    --------\n    - See ``tl.prepro.rotation``, ``tl.prepro.shear``, ``tl.prepro.zoom``.\n    '
    o_x = (x - 1) / 2.0
    o_y = (y - 1) / 2.0
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def affine_transform(x, transform_matrix, channel_index=2, fill_mode='nearest', cval=0.0, order=1):
    if False:
        print('Hello World!')
    "Return transformed images by given an affine matrix in Scipy format (x is height).\n\n    Parameters\n    ----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    transform_matrix : numpy.array\n        Transform matrix (offset center), can be generated by ``transform_matrix_offset_center``\n    channel_index : int\n        Index of channel, default 2.\n    fill_mode : str\n        Method to fill missing pixel, default `nearest`, more options `constant`, `reflect` or `wrap`, see `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`__\n    cval : float\n        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0\n    order : int\n        The order of interpolation. The order has to be in the range 0-5:\n            - 0 Nearest-neighbor\n            - 1 Bi-linear (default)\n            - 2 Bi-quadratic\n            - 3 Bi-cubic\n            - 4 Bi-quartic\n            - 5 Bi-quintic\n            - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`__\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    Examples\n    --------\n    >>> M_shear = tl.prepro.affine_shear_matrix(intensity=0.2, is_random=False)\n    >>> M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=0.8)\n    >>> M_combined = M_shear.dot(M_zoom)\n    >>> transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, h, w)\n    >>> result = tl.prepro.affine_transform(image, transform_matrix)\n\n    "
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix, final_offset, order=order, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x
apply_transform = affine_transform

def affine_transform_cv2(x, transform_matrix, flags=None, border_mode='constant'):
    if False:
        while True:
            i = 10
    'Return transformed images by given an affine matrix in OpenCV format (x is width). (Powered by OpenCV2, faster than ``tl.prepro.affine_transform``)\n\n    Parameters\n    ----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    transform_matrix : numpy.array\n        A transform matrix, OpenCV format.\n    border_mode : str\n        - `constant`, pad the image with a constant value (i.e. black or 0)\n        - `replicate`, the row or column at the very edge of the original is replicated to the extra border.\n\n    Examples\n    --------\n    >>> M_shear = tl.prepro.affine_shear_matrix(intensity=0.2, is_random=False)\n    >>> M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=0.8)\n    >>> M_combined = M_shear.dot(M_zoom)\n    >>> result = tl.prepro.affine_transform_cv2(image, M_combined)\n    '
    (rows, cols) = (x.shape[0], x.shape[1])
    if flags is None:
        flags = cv2.INTER_AREA
    if border_mode is 'constant':
        border_mode = cv2.BORDER_CONSTANT
    elif border_mode is 'replicate':
        border_mode = cv2.BORDER_REPLICATE
    else:
        raise Exception('unsupport border_mode, check cv.BORDER_ for more details.')
    return cv2.warpAffine(x, transform_matrix[0:2, :], (cols, rows), flags=flags, borderMode=border_mode)

def affine_transform_keypoints(coords_list, transform_matrix):
    if False:
        while True:
            i = 10
    'Transform keypoint coordinates according to a given affine transform matrix.\n    OpenCV format, x is width.\n\n    Note that, for pose estimation task, flipping requires maintaining the left and right body information.\n    We should not flip the left and right body, so please use ``tl.prepro.keypoint_random_flip``.\n\n    Parameters\n    -----------\n    coords_list : list of list of tuple/list\n        The coordinates\n        e.g., the keypoint coordinates of every person in an image.\n    transform_matrix : numpy.array\n        Transform matrix, OpenCV format.\n\n    Examples\n    ---------\n    >>> # 1. get all affine transform matrices\n    >>> M_rotate = tl.prepro.affine_rotation_matrix(angle=20)\n    >>> M_flip = tl.prepro.affine_horizontal_flip_matrix(prob=1)\n    >>> # 2. combine all affine transform matrices to one matrix\n    >>> M_combined = dot(M_flip).dot(M_rotate)\n    >>> # 3. transfrom the matrix from Cartesian coordinate (the origin in the middle of image)\n    >>> # to Image coordinate (the origin on the top-left of image)\n    >>> transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=w, y=h)\n    >>> # 4. then we can transfrom the image once for all transformations\n    >>> result = tl.prepro.affine_transform_cv2(image, transform_matrix)  # 76 times faster\n    >>> # 5. transform keypoint coordinates\n    >>> coords = [[(50, 100), (100, 100), (100, 50), (200, 200)], [(250, 50), (200, 50), (200, 100)]]\n    >>> coords_result = tl.prepro.affine_transform_keypoints(coords, transform_matrix)\n    '
    coords_result_list = []
    for coords in coords_list:
        coords = np.asarray(coords)
        coords = coords.transpose([1, 0])
        coords = np.insert(coords, 2, 1, axis=0)
        coords_result = np.matmul(transform_matrix, coords)
        coords_result = coords_result[0:2, :].transpose([1, 0])
        coords_result_list.append(coords_result)
    return coords_result_list

def projective_transform_by_points(x, src, dst, map_args=None, output_shape=None, order=1, mode='constant', cval=0.0, clip=True, preserve_range=False):
    if False:
        print('Hello World!')
    'Projective transform by given coordinates, usually 4 coordinates.\n\n    see `scikit-image <http://scikit-image.org/docs/dev/auto_examples/applications/plot_geometric.html>`__.\n\n    Parameters\n    -----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    src : list or numpy\n        The original coordinates, usually 4 coordinates of (width, height).\n    dst : list or numpy\n        The coordinates after transformation, the number of coordinates is the same with src.\n    map_args : dictionary or None\n        Keyword arguments passed to inverse map.\n    output_shape : tuple of 2 int\n        Shape of the output image generated. By default the shape of the input image is preserved. Note that, even for multi-band images, only rows and columns need to be specified.\n    order : int\n        The order of interpolation. The order has to be in the range 0-5:\n            - 0 Nearest-neighbor\n            - 1 Bi-linear (default)\n            - 2 Bi-quadratic\n            - 3 Bi-cubic\n            - 4 Bi-quartic\n            - 5 Bi-quintic\n    mode : str\n        One of `constant` (default), `edge`, `symmetric`, `reflect` or `wrap`.\n        Points outside the boundaries of the input are filled according to the given mode. Modes match the behaviour of numpy.pad.\n    cval : float\n        Used in conjunction with mode `constant`, the value outside the image boundaries.\n    clip : boolean\n        Whether to clip the output to the range of values of the input image. This is enabled by default, since higher order interpolation may produce values outside the given input range.\n    preserve_range : boolean\n        Whether to keep the original range of values. Otherwise, the input image is converted according to the conventions of img_as_float.\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    Examples\n    --------\n    Assume X is an image from CIFAR-10, i.e. shape == (32, 32, 3)\n\n    >>> src = [[0,0],[0,32],[32,0],[32,32]]     # [w, h]\n    >>> dst = [[10,10],[0,32],[32,0],[32,32]]\n    >>> x = tl.prepro.projective_transform_by_points(X, src, dst)\n\n    References\n    -----------\n    - `scikit-image : geometric transformations <http://scikit-image.org/docs/dev/auto_examples/applications/plot_geometric.html>`__\n    - `scikit-image : examples <http://scikit-image.org/docs/dev/auto_examples/index.html>`__\n\n    '
    if map_args is None:
        map_args = {}
    if isinstance(src, list):
        src = np.array(src)
    if isinstance(dst, list):
        dst = np.array(dst)
    if np.max(x) > 1:
        x = x / 255
    m = transform.ProjectiveTransform()
    m.estimate(dst, src)
    warped = transform.warp(x, m, map_args=map_args, output_shape=output_shape, order=order, mode=mode, cval=cval, clip=clip, preserve_range=preserve_range)
    return warped

def rotation(x, rg=20, is_random=False, row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0.0, order=1):
    if False:
        while True:
            i = 10
    "Rotate an image randomly or non-randomly.\n\n    Parameters\n    -----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    rg : int or float\n        Degree to rotate, usually 0 ~ 180.\n    is_random : boolean\n        If True, randomly rotate. Default is False\n    row_index col_index and channel_index : int\n        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).\n    fill_mode : str\n        Method to fill missing pixel, default `nearest`, more options `constant`, `reflect` or `wrap`, see `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`__\n    cval : float\n        Value used for points outside the boundaries of the input if mode=`constant`. Default is 0.0\n    order : int\n        The order of interpolation. The order has to be in the range 0-5. See ``tl.prepro.affine_transform`` and `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`__\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    Examples\n    ---------\n    >>> x --> [row, col, 1]\n    >>> x = tl.prepro.rotation(x, rg=40, is_random=False)\n    >>> tl.vis.save_image(x, 'im.png')\n\n    "
    if is_random:
        theta = np.pi / 180 * np.random.uniform(-rg, rg)
    else:
        theta = np.pi / 180 * rg
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    (h, w) = (x.shape[row_index], x.shape[col_index])
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = affine_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x

def rotation_multi(x, rg=20, is_random=False, row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0.0, order=1):
    if False:
        i = 10
        return i + 15
    'Rotate multiple images with the same arguments, randomly or non-randomly.\n    Usually be used for image segmentation which x=[X, Y], X and Y should be matched.\n\n    Parameters\n    -----------\n    x : list of numpy.array\n        List of images with dimension of [n_images, row, col, channel] (default).\n    others : args\n        See ``tl.prepro.rotation``.\n\n    Returns\n    -------\n    numpy.array\n        A list of processed images.\n\n    Examples\n    --------\n    >>> x, y --> [row, col, 1]  greyscale\n    >>> x, y = tl.prepro.rotation_multi([x, y], rg=90, is_random=False)\n\n    '
    if is_random:
        theta = np.pi / 180 * np.random.uniform(-rg, rg)
    else:
        theta = np.pi / 180 * rg
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    (h, w) = (x[0].shape[row_index], x[0].shape[col_index])
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    results = []
    for data in x:
        results.append(affine_transform(data, transform_matrix, channel_index, fill_mode, cval, order))
    return np.asarray(results)

def crop(x, wrg, hrg, is_random=False, row_index=0, col_index=1):
    if False:
        i = 10
        return i + 15
    'Randomly or centrally crop an image.\n\n    Parameters\n    ----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    wrg : int\n        Size of width.\n    hrg : int\n        Size of height.\n    is_random : boolean,\n        If True, randomly crop, else central crop. Default is False.\n    row_index: int\n        index of row.\n    col_index: int\n        index of column.\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    '
    (h, w) = (x.shape[row_index], x.shape[col_index])
    if h < hrg or w < wrg:
        raise AssertionError('The size of cropping should smaller than or equal to the original image')
    if is_random:
        h_offset = int(np.random.uniform(0, h - hrg))
        w_offset = int(np.random.uniform(0, w - wrg))
        return x[h_offset:hrg + h_offset, w_offset:wrg + w_offset]
    else:
        h_offset = int(np.floor((h - hrg) / 2.0))
        w_offset = int(np.floor((w - wrg) / 2.0))
        h_end = h_offset + hrg
        w_end = w_offset + wrg
        return x[h_offset:h_end, w_offset:w_end]

def crop_multi(x, wrg, hrg, is_random=False, row_index=0, col_index=1):
    if False:
        i = 10
        return i + 15
    'Randomly or centrally crop multiple images.\n\n    Parameters\n    ----------\n    x : list of numpy.array\n        List of images with dimension of [n_images, row, col, channel] (default).\n    others : args\n        See ``tl.prepro.crop``.\n\n    Returns\n    -------\n    numpy.array\n        A list of processed images.\n\n    '
    (h, w) = (x[0].shape[row_index], x[0].shape[col_index])
    if h < hrg or w < wrg:
        raise AssertionError('The size of cropping should smaller than or equal to the original image')
    if is_random:
        h_offset = int(np.random.uniform(0, h - hrg))
        w_offset = int(np.random.uniform(0, w - wrg))
        results = []
        for data in x:
            results.append(data[h_offset:hrg + h_offset, w_offset:wrg + w_offset])
        return np.asarray(results)
    else:
        h_offset = int(np.floor((h - hrg) / 2.0))
        w_offset = int(np.floor((w - wrg) / 2.0))
        results = []
        for data in x:
            results.append(data[h_offset:h - h_offset, w_offset:w - w_offset])
        return np.asarray(results)

def flip_axis(x, axis=1, is_random=False):
    if False:
        while True:
            i = 10
    'Flip the axis of an image, such as flip left and right, up and down, randomly or non-randomly,\n\n    Parameters\n    ----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    axis : int\n        Which axis to flip.\n            - 0, flip up and down\n            - 1, flip left and right\n            - 2, flip channel\n    is_random : boolean\n        If True, randomly flip. Default is False.\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    '
    if is_random:
        factor = np.random.uniform(-1, 1)
        if factor > 0:
            x = np.asarray(x).swapaxes(axis, 0)
            x = x[::-1, ...]
            x = x.swapaxes(0, axis)
            return x
        else:
            return x
    else:
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x

def flip_axis_multi(x, axis, is_random=False):
    if False:
        return 10
    'Flip the axises of multiple images together, such as flip left and right, up and down, randomly or non-randomly,\n\n    Parameters\n    -----------\n    x : list of numpy.array\n        List of images with dimension of [n_images, row, col, channel] (default).\n    others : args\n        See ``tl.prepro.flip_axis``.\n\n    Returns\n    -------\n    numpy.array\n        A list of processed images.\n\n    '
    if is_random:
        factor = np.random.uniform(-1, 1)
        if factor > 0:
            results = []
            for data in x:
                data = np.asarray(data).swapaxes(axis, 0)
                data = data[::-1, ...]
                data = data.swapaxes(0, axis)
                results.append(data)
            return np.asarray(results)
        else:
            return np.asarray(x)
    else:
        results = []
        for data in x:
            data = np.asarray(data).swapaxes(axis, 0)
            data = data[::-1, ...]
            data = data.swapaxes(0, axis)
            results.append(data)
        return np.asarray(results)

def shift(x, wrg=0.1, hrg=0.1, is_random=False, row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0.0, order=1):
    if False:
        i = 10
        return i + 15
    "Shift an image randomly or non-randomly.\n\n    Parameters\n    -----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    wrg : float\n        Percentage of shift in axis x, usually -0.25 ~ 0.25.\n    hrg : float\n        Percentage of shift in axis y, usually -0.25 ~ 0.25.\n    is_random : boolean\n        If True, randomly shift. Default is False.\n    row_index col_index and channel_index : int\n        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).\n    fill_mode : str\n        Method to fill missing pixel, default `nearest`, more options `constant`, `reflect` or `wrap`, see `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`__\n    cval : float\n        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0.\n    order : int\n        The order of interpolation. The order has to be in the range 0-5. See ``tl.prepro.affine_transform`` and `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`__\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    "
    (h, w) = (x.shape[row_index], x.shape[col_index])
    if is_random:
        tx = np.random.uniform(-hrg, hrg) * h
        ty = np.random.uniform(-wrg, wrg) * w
    else:
        (tx, ty) = (hrg * h, wrg * w)
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    transform_matrix = translation_matrix
    x = affine_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x

def shift_multi(x, wrg=0.1, hrg=0.1, is_random=False, row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0.0, order=1):
    if False:
        for i in range(10):
            print('nop')
    'Shift images with the same arguments, randomly or non-randomly.\n    Usually be used for image segmentation which x=[X, Y], X and Y should be matched.\n\n    Parameters\n    -----------\n    x : list of numpy.array\n        List of images with dimension of [n_images, row, col, channel] (default).\n    others : args\n        See ``tl.prepro.shift``.\n\n    Returns\n    -------\n    numpy.array\n        A list of processed images.\n\n    '
    (h, w) = (x[0].shape[row_index], x[0].shape[col_index])
    if is_random:
        tx = np.random.uniform(-hrg, hrg) * h
        ty = np.random.uniform(-wrg, wrg) * w
    else:
        (tx, ty) = (hrg * h, wrg * w)
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    transform_matrix = translation_matrix
    results = []
    for data in x:
        results.append(affine_transform(data, transform_matrix, channel_index, fill_mode, cval, order))
    return np.asarray(results)

def shear(x, intensity=0.1, is_random=False, row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0.0, order=1):
    if False:
        print('Hello World!')
    "Shear an image randomly or non-randomly.\n\n    Parameters\n    -----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    intensity : float\n        Percentage of shear, usually -0.5 ~ 0.5 (is_random==True), 0 ~ 0.5 (is_random==False),\n        you can have a quick try by shear(X, 1).\n    is_random : boolean\n        If True, randomly shear. Default is False.\n    row_index col_index and channel_index : int\n        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).\n    fill_mode : str\n        Method to fill missing pixel, default `nearest`, more options `constant`, `reflect` or `wrap`, see and `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`__\n    cval : float\n        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0.\n    order : int\n        The order of interpolation. The order has to be in the range 0-5. See ``tl.prepro.affine_transform`` and `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`__\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    References\n    -----------\n    - `Affine transformation <https://uk.mathworks.com/discovery/affine-transformation.html>`__\n\n    "
    if is_random:
        shear = np.random.uniform(-intensity, intensity)
    else:
        shear = intensity
    shear_matrix = np.array([[1, -np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    (h, w) = (x.shape[row_index], x.shape[col_index])
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = affine_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x

def shear_multi(x, intensity=0.1, is_random=False, row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0.0, order=1):
    if False:
        i = 10
        return i + 15
    'Shear images with the same arguments, randomly or non-randomly.\n    Usually be used for image segmentation which x=[X, Y], X and Y should be matched.\n\n    Parameters\n    -----------\n    x : list of numpy.array\n        List of images with dimension of [n_images, row, col, channel] (default).\n    others : args\n        See ``tl.prepro.shear``.\n\n    Returns\n    -------\n    numpy.array\n        A list of processed images.\n\n    '
    if is_random:
        shear = np.random.uniform(-intensity, intensity)
    else:
        shear = intensity
    shear_matrix = np.array([[1, -np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    (h, w) = (x[0].shape[row_index], x[0].shape[col_index])
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    results = []
    for data in x:
        results.append(affine_transform(data, transform_matrix, channel_index, fill_mode, cval, order))
    return np.asarray(results)

def shear2(x, shear=(0.1, 0.1), is_random=False, row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0.0, order=1):
    if False:
        for i in range(10):
            print('nop')
    "Shear an image randomly or non-randomly.\n\n    Parameters\n    -----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    shear : tuple of two floats\n        Percentage of shear for height and width direction (0, 1).\n    is_random : boolean\n        If True, randomly shear. Default is False.\n    row_index col_index and channel_index : int\n        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).\n    fill_mode : str\n        Method to fill missing pixel, default `nearest`, more options `constant`, `reflect` or `wrap`, see `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`__\n    cval : float\n        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0.\n    order : int\n        The order of interpolation. The order has to be in the range 0-5. See ``tl.prepro.affine_transform`` and `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`__\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    References\n    -----------\n    - `Affine transformation <https://uk.mathworks.com/discovery/affine-transformation.html>`__\n\n    "
    if len(shear) != 2:
        raise AssertionError('shear should be tuple of 2 floats, or you want to use tl.prepro.shear rather than tl.prepro.shear2 ?')
    if isinstance(shear, tuple):
        shear = list(shear)
    if is_random:
        shear[0] = np.random.uniform(-shear[0], shear[0])
        shear[1] = np.random.uniform(-shear[1], shear[1])
    shear_matrix = np.array([[1, shear[0], 0], [shear[1], 1, 0], [0, 0, 1]])
    (h, w) = (x.shape[row_index], x.shape[col_index])
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = affine_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x

def shear_multi2(x, shear=(0.1, 0.1), is_random=False, row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0.0, order=1):
    if False:
        i = 10
        return i + 15
    'Shear images with the same arguments, randomly or non-randomly.\n    Usually be used for image segmentation which x=[X, Y], X and Y should be matched.\n\n    Parameters\n    -----------\n    x : list of numpy.array\n        List of images with dimension of [n_images, row, col, channel] (default).\n    others : args\n        See ``tl.prepro.shear2``.\n\n    Returns\n    -------\n    numpy.array\n        A list of processed images.\n\n    '
    if len(shear) != 2:
        raise AssertionError('shear should be tuple of 2 floats, or you want to use tl.prepro.shear_multi rather than tl.prepro.shear_multi2 ?')
    if isinstance(shear, tuple):
        shear = list(shear)
    if is_random:
        shear[0] = np.random.uniform(-shear[0], shear[0])
        shear[1] = np.random.uniform(-shear[1], shear[1])
    shear_matrix = np.array([[1, shear[0], 0], [shear[1], 1, 0], [0, 0, 1]])
    (h, w) = (x[0].shape[row_index], x[0].shape[col_index])
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    results = []
    for data in x:
        results.append(affine_transform(data, transform_matrix, channel_index, fill_mode, cval, order))
    return np.asarray(results)

def swirl(x, center=None, strength=1, radius=100, rotation=0, output_shape=None, order=1, mode='constant', cval=0, clip=True, preserve_range=False, is_random=False):
    if False:
        i = 10
        return i + 15
    'Swirl an image randomly or non-randomly, see `scikit-image swirl API <http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.swirl>`__\n    and `example <http://scikit-image.org/docs/dev/auto_examples/plot_swirl.html>`__.\n\n    Parameters\n    -----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    center : tuple or 2 int or None\n        Center coordinate of transformation (optional).\n    strength : float\n        The amount of swirling applied.\n    radius : float\n        The extent of the swirl in pixels. The effect dies out rapidly beyond radius.\n    rotation : float\n        Additional rotation applied to the image, usually [0, 360], relates to center.\n    output_shape : tuple of 2 int or None\n        Shape of the output image generated (height, width). By default the shape of the input image is preserved.\n    order : int, optional\n        The order of the spline interpolation, default is 1. The order has to be in the range 0-5. See skimage.transform.warp for detail.\n    mode : str\n        One of `constant` (default), `edge`, `symmetric` `reflect` and `wrap`.\n        Points outside the boundaries of the input are filled according to the given mode, with `constant` used as the default. Modes match the behaviour of numpy.pad.\n    cval : float\n        Used in conjunction with mode `constant`, the value outside the image boundaries.\n    clip : boolean\n        Whether to clip the output to the range of values of the input image. This is enabled by default, since higher order interpolation may produce values outside the given input range.\n    preserve_range : boolean\n        Whether to keep the original range of values. Otherwise, the input image is converted according to the conventions of img_as_float.\n    is_random : boolean,\n        If True, random swirl. Default is False.\n            - random center = [(0 ~ x.shape[0]), (0 ~ x.shape[1])]\n            - random strength = [0, strength]\n            - random radius = [1e-10, radius]\n            - random rotation = [-rotation, rotation]\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    Examples\n    ---------\n    >>> x --> [row, col, 1] greyscale\n    >>> x = tl.prepro.swirl(x, strength=4, radius=100)\n\n    '
    if radius == 0:
        raise AssertionError('Invalid radius value')
    rotation = np.pi / 180 * rotation
    if is_random:
        center_h = int(np.random.uniform(0, x.shape[0]))
        center_w = int(np.random.uniform(0, x.shape[1]))
        center = (center_h, center_w)
        strength = np.random.uniform(0, strength)
        radius = np.random.uniform(1e-10, radius)
        rotation = np.random.uniform(-rotation, rotation)
    max_v = np.max(x)
    if max_v > 1:
        x = x / max_v
    swirled = skimage.transform.swirl(x, center=center, strength=strength, radius=radius, rotation=rotation, output_shape=output_shape, order=order, mode=mode, cval=cval, clip=clip, preserve_range=preserve_range)
    if max_v > 1:
        swirled = swirled * max_v
    return swirled

def swirl_multi(x, center=None, strength=1, radius=100, rotation=0, output_shape=None, order=1, mode='constant', cval=0, clip=True, preserve_range=False, is_random=False):
    if False:
        return 10
    'Swirl multiple images with the same arguments, randomly or non-randomly.\n    Usually be used for image segmentation which x=[X, Y], X and Y should be matched.\n\n    Parameters\n    -----------\n    x : list of numpy.array\n        List of images with dimension of [n_images, row, col, channel] (default).\n    others : args\n        See ``tl.prepro.swirl``.\n\n    Returns\n    -------\n    numpy.array\n        A list of processed images.\n\n    '
    if radius == 0:
        raise AssertionError('Invalid radius value')
    rotation = np.pi / 180 * rotation
    if is_random:
        center_h = int(np.random.uniform(0, x[0].shape[0]))
        center_w = int(np.random.uniform(0, x[0].shape[1]))
        center = (center_h, center_w)
        strength = np.random.uniform(0, strength)
        radius = np.random.uniform(1e-10, radius)
        rotation = np.random.uniform(-rotation, rotation)
    results = []
    for data in x:
        max_v = np.max(data)
        if max_v > 1:
            data = data / max_v
        swirled = skimage.transform.swirl(data, center=center, strength=strength, radius=radius, rotation=rotation, output_shape=output_shape, order=order, mode=mode, cval=cval, clip=clip, preserve_range=preserve_range)
        if max_v > 1:
            swirled = swirled * max_v
        results.append(swirled)
    return np.asarray(results)

def elastic_transform(x, alpha, sigma, mode='constant', cval=0, is_random=False):
    if False:
        print('Hello World!')
    'Elastic transformation for image as described in `[Simard2003] <http://deeplearning.cs.cmu.edu/pdfs/Simard.pdf>`__.\n\n    Parameters\n    -----------\n    x : numpy.array\n        A greyscale image.\n    alpha : float\n        Alpha value for elastic transformation.\n    sigma : float or sequence of float\n        The smaller the sigma, the more transformation. Standard deviation for Gaussian kernel. The standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number, in which case it is equal for all axes.\n    mode : str\n        See `scipy.ndimage.filters.gaussian_filter <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.gaussian_filter.html>`__. Default is `constant`.\n    cval : float,\n        Used in conjunction with `mode` of `constant`, the value outside the image boundaries.\n    is_random : boolean\n        Default is False.\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    Examples\n    ---------\n    >>> x = tl.prepro.elastic_transform(x, alpha=x.shape[1]*3, sigma=x.shape[1]*0.07)\n\n    References\n    ------------\n    - `Github <https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a>`__.\n    - `Kaggle <https://www.kaggle.com/pscion/ultrasound-nerve-segmentation/elastic-transform-for-data-augmentation-0878921a>`__\n\n    '
    if is_random is False:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(int(time.time()))
    is_3d = False
    if len(x.shape) == 3 and x.shape[-1] == 1:
        x = x[:, :, 0]
        is_3d = True
    elif len(x.shape) == 3 and x.shape[-1] != 1:
        raise Exception('Only support greyscale image')
    if len(x.shape) != 2:
        raise AssertionError('input should be grey-scale image')
    shape = x.shape
    dx = gaussian_filter(random_state.rand(*shape) * 2 - 1, sigma, mode=mode, cval=cval) * alpha
    dy = gaussian_filter(random_state.rand(*shape) * 2 - 1, sigma, mode=mode, cval=cval) * alpha
    (x_, y_) = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = (np.reshape(x_ + dx, (-1, 1)), np.reshape(y_ + dy, (-1, 1)))
    if is_3d:
        return map_coordinates(x, indices, order=1).reshape((shape[0], shape[1], 1))
    else:
        return map_coordinates(x, indices, order=1).reshape(shape)

def elastic_transform_multi(x, alpha, sigma, mode='constant', cval=0, is_random=False):
    if False:
        print('Hello World!')
    'Elastic transformation for images as described in `[Simard2003] <http://deeplearning.cs.cmu.edu/pdfs/Simard.pdf>`__.\n\n    Parameters\n    -----------\n    x : list of numpy.array\n        List of greyscale images.\n    others : args\n        See ``tl.prepro.elastic_transform``.\n\n    Returns\n    -------\n    numpy.array\n        A list of processed images.\n\n    '
    if is_random is False:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(int(time.time()))
    shape = x[0].shape
    if len(shape) == 3:
        shape = (shape[0], shape[1])
    new_shape = random_state.rand(*shape)
    results = []
    for data in x:
        is_3d = False
        if len(data.shape) == 3 and data.shape[-1] == 1:
            data = data[:, :, 0]
            is_3d = True
        elif len(data.shape) == 3 and data.shape[-1] != 1:
            raise Exception('Only support greyscale image')
        if len(data.shape) != 2:
            raise AssertionError('input should be grey-scale image')
        dx = gaussian_filter(new_shape * 2 - 1, sigma, mode=mode, cval=cval) * alpha
        dy = gaussian_filter(new_shape * 2 - 1, sigma, mode=mode, cval=cval) * alpha
        (x_, y_) = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = (np.reshape(x_ + dx, (-1, 1)), np.reshape(y_ + dy, (-1, 1)))
        if is_3d:
            results.append(map_coordinates(data, indices, order=1).reshape((shape[0], shape[1], 1)))
        else:
            results.append(map_coordinates(data, indices, order=1).reshape(shape))
    return np.asarray(results)

def zoom(x, zoom_range=(0.9, 1.1), flags=None, border_mode='constant'):
    if False:
        i = 10
        return i + 15
    'Zooming/Scaling a single image that height and width are changed together.\n\n    Parameters\n    -----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    zoom_range : float or tuple of 2 floats\n        The zooming/scaling ratio, greater than 1 means larger.\n            - float, a fixed ratio.\n            - tuple of 2 floats, randomly sample a value as the ratio between 2 values.\n    border_mode : str\n        - `constant`, pad the image with a constant value (i.e. black or 0)\n        - `replicate`, the row or column at the very edge of the original is replicated to the extra border.\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    '
    zoom_matrix = affine_zoom_matrix(zoom_range=zoom_range)
    (h, w) = (x.shape[0], x.shape[1])
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = affine_transform_cv2(x, transform_matrix, flags=flags, border_mode=border_mode)
    return x

def respective_zoom(x, h_range=(0.9, 1.1), w_range=(0.9, 1.1), flags=None, border_mode='constant'):
    if False:
        print('Hello World!')
    'Zooming/Scaling a single image that height and width are changed independently.\n\n    Parameters\n    -----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    h_range : float or tuple of 2 floats\n        The zooming/scaling ratio of height, greater than 1 means larger.\n            - float, a fixed ratio.\n            - tuple of 2 floats, randomly sample a value as the ratio between 2 values.\n    w_range : float or tuple of 2 floats\n        The zooming/scaling ratio of width, greater than 1 means larger.\n            - float, a fixed ratio.\n            - tuple of 2 floats, randomly sample a value as the ratio between 2 values.\n    border_mode : str\n        - `constant`, pad the image with a constant value (i.e. black or 0)\n        - `replicate`, the row or column at the very edge of the original is replicated to the extra border.\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    '
    zoom_matrix = affine_respective_zoom_matrix(h_range=h_range, w_range=w_range)
    (h, w) = (x.shape[0], x.shape[1])
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = affine_transform_cv2(x, transform_matrix, flags=flags, border_mode=border_mode)
    return x

def zoom_multi(x, zoom_range=(0.9, 1.1), flags=None, border_mode='constant'):
    if False:
        while True:
            i = 10
    'Zoom in and out of images with the same arguments, randomly or non-randomly.\n    Usually be used for image segmentation which x=[X, Y], X and Y should be matched.\n\n    Parameters\n    -----------\n    x : list of numpy.array\n        List of images with dimension of [n_images, row, col, channel] (default).\n    others : args\n        See ``tl.prepro.zoom``.\n\n    Returns\n    -------\n    numpy.array\n        A list of processed images.\n\n    '
    zoom_matrix = affine_zoom_matrix(zoom_range=zoom_range)
    results = []
    for img in x:
        (h, w) = (x.shape[0], x.shape[1])
        transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
        results.append(affine_transform_cv2(x, transform_matrix, flags=flags, border_mode=border_mode))
    return np.asarray(results)

def brightness(x, gamma=1, gain=1, is_random=False):
    if False:
        i = 10
        return i + 15
    'Change the brightness of a single image, randomly or non-randomly.\n\n    Parameters\n    -----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    gamma : float\n        Non negative real number. Default value is 1.\n            - Small than 1 means brighter.\n            - If `is_random` is True, gamma in a range of (1-gamma, 1+gamma).\n    gain : float\n        The constant multiplier. Default value is 1.\n    is_random : boolean\n        If True, randomly change brightness. Default is False.\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    References\n    -----------\n    - `skimage.exposure.adjust_gamma <http://scikit-image.org/docs/dev/api/skimage.exposure.html>`__\n    - `chinese blog <http://www.cnblogs.com/denny402/p/5124402.html>`__\n\n    '
    if is_random:
        gamma = np.random.uniform(1 - gamma, 1 + gamma)
    x = exposure.adjust_gamma(x, gamma, gain)
    return x

def brightness_multi(x, gamma=1, gain=1, is_random=False):
    if False:
        return 10
    'Change the brightness of multiply images, randomly or non-randomly.\n    Usually be used for image segmentation which x=[X, Y], X and Y should be matched.\n\n    Parameters\n    -----------\n    x : list of numpyarray\n        List of images with dimension of [n_images, row, col, channel] (default).\n    others : args\n        See ``tl.prepro.brightness``.\n\n    Returns\n    -------\n    numpy.array\n        A list of processed images.\n\n    '
    if is_random:
        gamma = np.random.uniform(1 - gamma, 1 + gamma)
    results = []
    for data in x:
        results.append(exposure.adjust_gamma(data, gamma, gain))
    return np.asarray(results)

def illumination(x, gamma=1.0, contrast=1.0, saturation=1.0, is_random=False):
    if False:
        while True:
            i = 10
    'Perform illumination augmentation for a single image, randomly or non-randomly.\n\n    Parameters\n    -----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    gamma : float\n        Change brightness (the same with ``tl.prepro.brightness``)\n            - if is_random=False, one float number, small than one means brighter, greater than one means darker.\n            - if is_random=True, tuple of two float numbers, (min, max).\n    contrast : float\n        Change contrast.\n            - if is_random=False, one float number, small than one means blur.\n            - if is_random=True, tuple of two float numbers, (min, max).\n    saturation : float\n        Change saturation.\n            - if is_random=False, one float number, small than one means unsaturation.\n            - if is_random=True, tuple of two float numbers, (min, max).\n    is_random : boolean\n        If True, randomly change illumination. Default is False.\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    Examples\n    ---------\n    Random\n\n    >>> x = tl.prepro.illumination(x, gamma=(0.5, 5.0), contrast=(0.3, 1.0), saturation=(0.7, 1.0), is_random=True)\n\n    Non-random\n\n    >>> x = tl.prepro.illumination(x, 0.5, 0.6, 0.8, is_random=False)\n\n    '
    if is_random:
        if not len(gamma) == len(contrast) == len(saturation) == 2:
            raise AssertionError('if is_random = True, the arguments are (min, max)')
        illum_settings = np.random.randint(0, 3)
        if illum_settings == 0:
            gamma = np.random.uniform(gamma[0], 1.0)
        elif illum_settings == 1:
            gamma = np.random.uniform(1.0, gamma[1])
        else:
            gamma = 1
        im_ = brightness(x, gamma=gamma, gain=1, is_random=False)
        image = PIL.Image.fromarray(im_)
        contrast_adjust = PIL.ImageEnhance.Contrast(image)
        image = contrast_adjust.enhance(np.random.uniform(contrast[0], contrast[1]))
        saturation_adjust = PIL.ImageEnhance.Color(image)
        image = saturation_adjust.enhance(np.random.uniform(saturation[0], saturation[1]))
        im_ = np.array(image)
    else:
        im_ = brightness(x, gamma=gamma, gain=1, is_random=False)
        image = PIL.Image.fromarray(im_)
        contrast_adjust = PIL.ImageEnhance.Contrast(image)
        image = contrast_adjust.enhance(contrast)
        saturation_adjust = PIL.ImageEnhance.Color(image)
        image = saturation_adjust.enhance(saturation)
        im_ = np.array(image)
    return np.asarray(im_)

def rgb_to_hsv(rgb):
    if False:
        while True:
            i = 10
    'Input RGB image [0~255] return HSV image [0~1].\n\n    Parameters\n    ------------\n    rgb : numpy.array\n        An image with values between 0 and 255.\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    '
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    hsv[..., 3:] = rgb[..., 3:]
    (r, g, b) = (rgb[..., 0], rgb[..., 1], rgb[..., 2])
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = hsv[..., 0] / 6.0 % 1.0
    return hsv

def hsv_to_rgb(hsv):
    if False:
        i = 10
        return i + 15
    'Input HSV image [0~1] return RGB image [0~255].\n\n    Parameters\n    -------------\n    hsv : numpy.array\n        An image with values between 0.0 and 1.0\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n    '
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    (h, s, v) = (hsv[..., 0], hsv[..., 1], hsv[..., 2])
    i = (h * 6.0).astype('uint8')
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')

def adjust_hue(im, hout=0.66, is_offset=True, is_clip=True, is_random=False):
    if False:
        while True:
            i = 10
    'Adjust hue of an RGB image.\n\n    This is a convenience method that converts an RGB image to float representation, converts it to HSV, add an offset to the hue channel, converts back to RGB and then back to the original data type.\n    For TF, see `tf.image.adjust_hue <https://www.tensorflow.org/api_docs/python/tf/image/adjust_hue>`__.and `tf.image.random_hue <https://www.tensorflow.org/api_docs/python/tf/image/random_hue>`__.\n\n    Parameters\n    -----------\n    im : numpy.array\n        An image with values between 0 and 255.\n    hout : float\n        The scale value for adjusting hue.\n            - If is_offset is False, set all hue values to this value. 0 is red; 0.33 is green; 0.66 is blue.\n            - If is_offset is True, add this value as the offset to the hue channel.\n    is_offset : boolean\n        Whether `hout` is added on HSV as offset or not. Default is True.\n    is_clip : boolean\n        If HSV value smaller than 0, set to 0. Default is True.\n    is_random : boolean\n        If True, randomly change hue. Default is False.\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    Examples\n    ---------\n    Random, add a random value between -0.2 and 0.2 as the offset to every hue values.\n\n    >>> im_hue = tl.prepro.adjust_hue(image, hout=0.2, is_offset=True, is_random=False)\n\n    Non-random, make all hue to green.\n\n    >>> im_green = tl.prepro.adjust_hue(image, hout=0.66, is_offset=False, is_random=False)\n\n    References\n    -----------\n    - `tf.image.random_hue <https://www.tensorflow.org/api_docs/python/tf/image/random_hue>`__.\n    - `tf.image.adjust_hue <https://www.tensorflow.org/api_docs/python/tf/image/adjust_hue>`__.\n    - `StackOverflow: Changing image hue with python PIL <https://stackoverflow.com/questions/7274221/changing-image-hue-with-python-pil>`__.\n\n    '
    hsv = rgb_to_hsv(im)
    if is_random:
        hout = np.random.uniform(-hout, hout)
    if is_offset:
        hsv[..., 0] += hout
    else:
        hsv[..., 0] = hout
    if is_clip:
        hsv[..., 0] = np.clip(hsv[..., 0], 0, np.inf)
    rgb = hsv_to_rgb(hsv)
    return rgb

def imresize(x, size=None, interp='bicubic', mode=None):
    if False:
        return 10
    'Resize an image by given output size and method.\n\n    Warning, this function will rescale the value to [0, 255].\n\n    Parameters\n    -----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    size : list of 2 int or None\n        For height and width.\n    interp : str\n        Interpolation method for re-sizing (`nearest`, `lanczos`, `bilinear`, `bicubic` (default) or `cubic`).\n    mode : str\n        The PIL image mode (`P`, `L`, etc.) to convert image before resizing.\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    References\n    ------------\n    - `scipy.misc.imresize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imresize.html>`__\n\n    '
    if size is None:
        size = [100, 100]
    if x.shape[-1] == 1:
        x = scipy.misc.imresize(x[:, :, 0], size, interp=interp, mode=mode)
        return x[:, :, np.newaxis]
    else:
        return scipy.misc.imresize(x, size, interp=interp, mode=mode)

def pixel_value_scale(im, val=0.9, clip=None, is_random=False):
    if False:
        return 10
    'Scales each value in the pixels of the image.\n\n    Parameters\n    -----------\n    im : numpy.array\n        An image.\n    val : float\n        The scale value for changing pixel value.\n            - If is_random=False, multiply this value with all pixels.\n            - If is_random=True, multiply a value between [1-val, 1+val] with all pixels.\n    clip : tuple of 2 numbers\n        The minimum and maximum value.\n    is_random : boolean\n        If True, see ``val``.\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    Examples\n    ----------\n    Random\n\n    >>> im = pixel_value_scale(im, 0.1, [0, 255], is_random=True)\n\n    Non-random\n\n    >>> im = pixel_value_scale(im, 0.9, [0, 255], is_random=False)\n\n    '
    clip = clip if clip is not None else (-np.inf, np.inf)
    if is_random:
        scale = 1 + np.random.uniform(-val, val)
        im = im * scale
    else:
        im = im * val
    if len(clip) == 2:
        im = np.clip(im, clip[0], clip[1])
    else:
        raise Exception('clip : tuple of 2 numbers')
    return im

def samplewise_norm(x, rescale=None, samplewise_center=False, samplewise_std_normalization=False, channel_index=2, epsilon=1e-07):
    if False:
        i = 10
        return i + 15
    'Normalize an image by rescale, samplewise centering and samplewise centering in order.\n\n    Parameters\n    -----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    rescale : float\n        Rescaling factor. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (before applying any other transformation)\n    samplewise_center : boolean\n        If True, set each sample mean to 0.\n    samplewise_std_normalization : boolean\n        If True, divide each input by its std.\n    epsilon : float\n        A small position value for dividing standard deviation.\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    Examples\n    --------\n    >>> x = samplewise_norm(x, samplewise_center=True, samplewise_std_normalization=True)\n    >>> print(x.shape, np.mean(x), np.std(x))\n    (160, 176, 1), 0.0, 1.0\n\n    Notes\n    ------\n    When samplewise_center and samplewise_std_normalization are True.\n    - For greyscale image, every pixels are subtracted and divided by the mean and std of whole image.\n    - For RGB image, every pixels are subtracted and divided by the mean and std of this pixel i.e. the mean and std of a pixel is 0 and 1.\n\n    '
    if rescale:
        x *= rescale
    if x.shape[channel_index] == 1:
        if samplewise_center:
            x = x - np.mean(x)
        if samplewise_std_normalization:
            x = x / np.std(x)
        return x
    elif x.shape[channel_index] == 3:
        if samplewise_center:
            x = x - np.mean(x, axis=channel_index, keepdims=True)
        if samplewise_std_normalization:
            x = x / (np.std(x, axis=channel_index, keepdims=True) + epsilon)
        return x
    else:
        raise Exception('Unsupported channels %d' % x.shape[channel_index])

def featurewise_norm(x, mean=None, std=None, epsilon=1e-07):
    if False:
        for i in range(10):
            print('nop')
    'Normalize every pixels by the same given mean and std, which are usually\n    compute from all examples.\n\n    Parameters\n    -----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    mean : float\n        Value for subtraction.\n    std : float\n        Value for division.\n    epsilon : float\n        A small position value for dividing standard deviation.\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    '
    if mean:
        x = x - mean
    if std:
        x = x / (std + epsilon)
    return x

def get_zca_whitening_principal_components_img(X):
    if False:
        while True:
            i = 10
    'Return the ZCA whitening principal components matrix.\n\n    Parameters\n    -----------\n    x : numpy.array\n        Batch of images with dimension of [n_example, row, col, channel] (default).\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    '
    flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
    tl.logging.info('zca : computing sigma ..')
    sigma = np.dot(flatX.T, flatX) / flatX.shape[0]
    tl.logging.info('zca : computing U, S and V ..')
    (U, S, _) = linalg.svd(sigma)
    tl.logging.info('zca : computing principal components ..')
    principal_components = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(S + 1e-06))), U.T)
    return principal_components

def zca_whitening(x, principal_components):
    if False:
        return 10
    'Apply ZCA whitening on an image by given principal components matrix.\n\n    Parameters\n    -----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    principal_components : matrix\n        Matrix from ``get_zca_whitening_principal_components_img``.\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    '
    flatx = np.reshape(x, x.size)
    whitex = np.dot(flatx, principal_components)
    x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
    return x

def channel_shift(x, intensity, is_random=False, channel_index=2):
    if False:
        return 10
    'Shift the channels of an image, randomly or non-randomly, see `numpy.rollaxis <https://docs.scipy.org/doc/numpy/reference/generated/numpy.rollaxis.html>`__.\n\n    Parameters\n    -----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    intensity : float\n        Intensity of shifting.\n    is_random : boolean\n        If True, randomly shift. Default is False.\n    channel_index : int\n        Index of channel. Default is 2.\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    '
    if is_random:
        factor = np.random.uniform(-intensity, intensity)
    else:
        factor = intensity
    x = np.rollaxis(x, channel_index, 0)
    (min_x, max_x) = (np.min(x), np.max(x))
    channel_images = [np.clip(x_channel + factor, min_x, max_x) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x

def channel_shift_multi(x, intensity, is_random=False, channel_index=2):
    if False:
        i = 10
        return i + 15
    'Shift the channels of images with the same arguments, randomly or non-randomly, see `numpy.rollaxis <https://docs.scipy.org/doc/numpy/reference/generated/numpy.rollaxis.html>`__.\n    Usually be used for image segmentation which x=[X, Y], X and Y should be matched.\n\n    Parameters\n    -----------\n    x : list of numpy.array\n        List of images with dimension of [n_images, row, col, channel] (default).\n    others : args\n        See ``tl.prepro.channel_shift``.\n\n    Returns\n    -------\n    numpy.array\n        A list of processed images.\n\n    '
    if is_random:
        factor = np.random.uniform(-intensity, intensity)
    else:
        factor = intensity
    results = []
    for data in x:
        data = np.rollaxis(data, channel_index, 0)
        (min_x, max_x) = (np.min(data), np.max(data))
        channel_images = [np.clip(x_channel + factor, min_x, max_x) for x_channel in x]
        data = np.stack(channel_images, axis=0)
        data = np.rollaxis(x, 0, channel_index + 1)
        results.append(data)
    return np.asarray(results)

def drop(x, keep=0.5):
    if False:
        print('Hello World!')
    'Randomly set some pixels to zero by a given keeping probability.\n\n    Parameters\n    -----------\n    x : numpy.array\n        An image with dimension of [row, col, channel] or [row, col].\n    keep : float\n        The keeping probability (0, 1), the lower more values will be set to zero.\n\n    Returns\n    -------\n    numpy.array\n        A processed image.\n\n    '
    if len(x.shape) == 3:
        if x.shape[-1] == 3:
            img_size = x.shape
            mask = np.random.binomial(n=1, p=keep, size=x.shape[:-1])
            for i in range(3):
                x[:, :, i] = np.multiply(x[:, :, i], mask)
        elif x.shape[-1] == 1:
            img_size = x.shape
            x = np.multiply(x, np.random.binomial(n=1, p=keep, size=img_size))
        else:
            raise Exception('Unsupported shape {}'.format(x.shape))
    elif len(x.shape) == 2 or 1:
        img_size = x.shape
        x = np.multiply(x, np.random.binomial(n=1, p=keep, size=img_size))
    else:
        raise Exception('Unsupported shape {}'.format(x.shape))
    return x

def array_to_img(x, dim_ordering=(0, 1, 2), scale=True):
    if False:
        print('Hello World!')
    'Converts a numpy array to PIL image object (uint8 format).\n\n    Parameters\n    ----------\n    x : numpy.array\n        An image with dimension of 3 and channels of 1 or 3.\n    dim_ordering : tuple of 3 int\n        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).\n    scale : boolean\n        If True, converts image to [0, 255] from any range of value like [-1, 2]. Default is True.\n\n    Returns\n    -------\n    PIL.image\n        An image.\n\n    References\n    -----------\n    `PIL Image.fromarray <http://pillow.readthedocs.io/en/3.1.x/reference/Image.html?highlight=fromarray>`__\n\n    '
    x = x.transpose(dim_ordering)
    if scale:
        x += max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x = x / x_max
        x *= 255
    if x.shape[2] == 3:
        return PIL.Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        return PIL.Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise Exception('Unsupported channel number: ', x.shape[2])

def find_contours(x, level=0.8, fully_connected='low', positive_orientation='low'):
    if False:
        print('Hello World!')
    'Find iso-valued contours in a 2D array for a given level value, returns list of (n, 2)-ndarrays\n    see `skimage.measure.find_contours <http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.find_contours>`__.\n\n    Parameters\n    ------------\n    x : 2D ndarray of double.\n        Input data in which to find contours.\n    level : float\n        Value along which to find contours in the array.\n    fully_connected : str\n        Either `low` or `high`. Indicates whether array elements below the given level value are to be considered fully-connected (and hence elements above the value will only be face connected), or vice-versa. (See notes below for details.)\n    positive_orientation : str\n        Either `low` or `high`. Indicates whether the output contours will produce positively-oriented polygons around islands of low- or high-valued elements. If `low` then contours will wind counter-clockwise around elements below the iso-value. Alternately, this means that low-valued elements are always on the left of the contour.\n\n    Returns\n    --------\n    list of (n,2)-ndarrays\n        Each contour is an ndarray of shape (n, 2), consisting of n (row, column) coordinates along the contour.\n\n    '
    return skimage.measure.find_contours(x, level, fully_connected=fully_connected, positive_orientation=positive_orientation)

def pt2map(list_points=None, size=(100, 100), val=1):
    if False:
        for i in range(10):
            print('nop')
    'Inputs a list of points, return a 2D image.\n\n    Parameters\n    --------------\n    list_points : list of 2 int\n        [[x, y], [x, y]..] for point coordinates.\n    size : tuple of 2 int\n        (w, h) for output size.\n    val : float or int\n        For the contour value.\n\n    Returns\n    -------\n    numpy.array\n        An image.\n\n    '
    if list_points is None:
        raise Exception('list_points : list of 2 int')
    i_m = np.zeros(size)
    if len(list_points) == 0:
        return i_m
    for xx in list_points:
        for x in xx:
            i_m[int(np.round(x[0]))][int(np.round(x[1]))] = val
    return i_m

def binary_dilation(x, radius=3):
    if False:
        for i in range(10):
            print('nop')
    'Return fast binary morphological dilation of an image.\n    see `skimage.morphology.binary_dilation <http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.binary_dilation>`__.\n\n    Parameters\n    -----------\n    x : 2D array\n        A binary image.\n    radius : int\n        For the radius of mask.\n\n    Returns\n    -------\n    numpy.array\n        A processed binary image.\n\n    '
    mask = disk(radius)
    x = _binary_dilation(x, selem=mask)
    return x

def dilation(x, radius=3):
    if False:
        while True:
            i = 10
    'Return greyscale morphological dilation of an image,\n    see `skimage.morphology.dilation <http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.dilation>`__.\n\n    Parameters\n    -----------\n    x : 2D array\n        An greyscale image.\n    radius : int\n        For the radius of mask.\n\n    Returns\n    -------\n    numpy.array\n        A processed greyscale image.\n\n    '
    mask = disk(radius)
    x = dilation(x, selem=mask)
    return x

def binary_erosion(x, radius=3):
    if False:
        print('Hello World!')
    'Return binary morphological erosion of an image,\n    see `skimage.morphology.binary_erosion <http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.binary_erosion>`__.\n\n    Parameters\n    -----------\n    x : 2D array\n        A binary image.\n    radius : int\n        For the radius of mask.\n\n    Returns\n    -------\n    numpy.array\n        A processed binary image.\n\n    '
    mask = disk(radius)
    x = _binary_erosion(x, selem=mask)
    return x

def erosion(x, radius=3):
    if False:
        print('Hello World!')
    'Return greyscale morphological erosion of an image,\n    see `skimage.morphology.erosion <http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.erosion>`__.\n\n    Parameters\n    -----------\n    x : 2D array\n        A greyscale image.\n    radius : int\n        For the radius of mask.\n\n    Returns\n    -------\n    numpy.array\n        A processed greyscale image.\n\n    '
    mask = disk(radius)
    x = _erosion(x, selem=mask)
    return x

def obj_box_coords_rescale(coords=None, shape=None):
    if False:
        return 10
    'Scale down a list of coordinates from pixel unit to the ratio of image size i.e. in the range of [0, 1].\n\n    Parameters\n    ------------\n    coords : list of list of 4 ints or None\n        For coordinates of more than one images .e.g.[[x, y, w, h], [x, y, w, h], ...].\n    shape : list of 2 int or None\n        【height, width].\n\n    Returns\n    -------\n    list of list of 4 numbers\n        A list of new bounding boxes.\n\n\n    Examples\n    ---------\n    >>> coords = obj_box_coords_rescale(coords=[[30, 40, 50, 50], [10, 10, 20, 20]], shape=[100, 100])\n    >>> print(coords)\n      [[0.3, 0.4, 0.5, 0.5], [0.1, 0.1, 0.2, 0.2]]\n    >>> coords = obj_box_coords_rescale(coords=[[30, 40, 50, 50]], shape=[50, 100])\n    >>> print(coords)\n      [[0.3, 0.8, 0.5, 1.0]]\n    >>> coords = obj_box_coords_rescale(coords=[[30, 40, 50, 50]], shape=[100, 200])\n    >>> print(coords)\n      [[0.15, 0.4, 0.25, 0.5]]\n\n    Returns\n    -------\n    list of 4 numbers\n        New coordinates.\n\n    '
    if coords is None:
        coords = []
    if shape is None:
        shape = [100, 200]
    (imh, imw) = (shape[0], shape[1])
    imh = imh * 1.0
    imw = imw * 1.0
    coords_new = list()
    for coord in coords:
        if len(coord) != 4:
            raise AssertionError('coordinate should be 4 values : [x, y, w, h]')
        x = coord[0] / imw
        y = coord[1] / imh
        w = coord[2] / imw
        h = coord[3] / imh
        coords_new.append([x, y, w, h])
    return coords_new

def obj_box_coord_rescale(coord=None, shape=None):
    if False:
        print('Hello World!')
    'Scale down one coordinates from pixel unit to the ratio of image size i.e. in the range of [0, 1].\n    It is the reverse process of ``obj_box_coord_scale_to_pixelunit``.\n\n    Parameters\n    ------------\n    coords : list of 4 int or None\n        One coordinates of one image e.g. [x, y, w, h].\n    shape : list of 2 int or None\n        For [height, width].\n\n    Returns\n    -------\n    list of 4 numbers\n        New bounding box.\n\n    Examples\n    ---------\n    >>> coord = tl.prepro.obj_box_coord_rescale(coord=[30, 40, 50, 50], shape=[100, 100])\n      [0.3, 0.4, 0.5, 0.5]\n\n    '
    if coord is None:
        coord = []
    if shape is None:
        shape = [100, 200]
    return obj_box_coords_rescale(coords=[coord], shape=shape)[0]

def obj_box_coord_scale_to_pixelunit(coord, shape=None):
    if False:
        for i in range(10):
            print('nop')
    'Convert one coordinate [x, y, w (or x2), h (or y2)] in ratio format to image coordinate format.\n    It is the reverse process of ``obj_box_coord_rescale``.\n\n    Parameters\n    -----------\n    coord : list of 4 float\n        One coordinate of one image [x, y, w (or x2), h (or y2)] in ratio format, i.e value range [0~1].\n    shape : tuple of 2 or None\n        For [height, width].\n\n    Returns\n    -------\n    list of 4 numbers\n        New bounding box.\n\n    Examples\n    ---------\n    >>> x, y, x2, y2 = tl.prepro.obj_box_coord_scale_to_pixelunit([0.2, 0.3, 0.5, 0.7], shape=(100, 200, 3))\n      [40, 30, 100, 70]\n\n    '
    if shape is None:
        shape = [100, 100]
    (imh, imw) = shape[0:2]
    x = int(coord[0] * imw)
    x2 = int(coord[2] * imw)
    y = int(coord[1] * imh)
    y2 = int(coord[3] * imh)
    return [x, y, x2, y2]

def obj_box_coord_centroid_to_upleft_butright(coord, to_int=False):
    if False:
        for i in range(10):
            print('nop')
    'Convert one coordinate [x_center, y_center, w, h] to [x1, y1, x2, y2] in up-left and botton-right format.\n\n    Parameters\n    ------------\n    coord : list of 4 int/float\n        One coordinate.\n    to_int : boolean\n        Whether to convert output as integer.\n\n    Returns\n    -------\n    list of 4 numbers\n        New bounding box.\n\n    Examples\n    ---------\n    >>> coord = obj_box_coord_centroid_to_upleft_butright([30, 40, 20, 20])\n      [20, 30, 40, 50]\n\n    '
    if len(coord) != 4:
        raise AssertionError('coordinate should be 4 values : [x, y, w, h]')
    (x_center, y_center, w, h) = coord
    x = x_center - w / 2.0
    y = y_center - h / 2.0
    x2 = x + w
    y2 = y + h
    if to_int:
        return [int(x), int(y), int(x2), int(y2)]
    else:
        return [x, y, x2, y2]

def obj_box_coord_upleft_butright_to_centroid(coord):
    if False:
        i = 10
        return i + 15
    'Convert one coordinate [x1, y1, x2, y2] to [x_center, y_center, w, h].\n    It is the reverse process of ``obj_box_coord_centroid_to_upleft_butright``.\n\n    Parameters\n    ------------\n    coord : list of 4 int/float\n        One coordinate.\n\n    Returns\n    -------\n    list of 4 numbers\n        New bounding box.\n\n    '
    if len(coord) != 4:
        raise AssertionError('coordinate should be 4 values : [x1, y1, x2, y2]')
    (x1, y1, x2, y2) = coord
    w = x2 - x1
    h = y2 - y1
    x_c = x1 + w / 2.0
    y_c = y1 + h / 2.0
    return [x_c, y_c, w, h]

def obj_box_coord_centroid_to_upleft(coord):
    if False:
        print('Hello World!')
    'Convert one coordinate [x_center, y_center, w, h] to [x, y, w, h].\n    It is the reverse process of ``obj_box_coord_upleft_to_centroid``.\n\n    Parameters\n    ------------\n    coord : list of 4 int/float\n        One coordinate.\n\n    Returns\n    -------\n    list of 4 numbers\n        New bounding box.\n\n    '
    if len(coord) != 4:
        raise AssertionError('coordinate should be 4 values : [x, y, w, h]')
    (x_center, y_center, w, h) = coord
    x = x_center - w / 2.0
    y = y_center - h / 2.0
    return [x, y, w, h]

def obj_box_coord_upleft_to_centroid(coord):
    if False:
        for i in range(10):
            print('nop')
    'Convert one coordinate [x, y, w, h] to [x_center, y_center, w, h].\n    It is the reverse process of ``obj_box_coord_centroid_to_upleft``.\n\n    Parameters\n    ------------\n    coord : list of 4 int/float\n        One coordinate.\n\n    Returns\n    -------\n    list of 4 numbers\n        New bounding box.\n\n    '
    if len(coord) != 4:
        raise AssertionError('coordinate should be 4 values : [x, y, w, h]')
    (x, y, w, h) = coord
    x_center = x + w / 2.0
    y_center = y + h / 2.0
    return [x_center, y_center, w, h]

def parse_darknet_ann_str_to_list(annotations):
    if False:
        for i in range(10):
            print('nop')
    'Input string format of class, x, y, w, h, return list of list format.\n\n    Parameters\n    -----------\n    annotations : str\n        The annotations in darkent format "class, x, y, w, h ...." seperated by "\\\\n".\n\n    Returns\n    -------\n    list of list of 4 numbers\n        List of bounding box.\n\n    '
    annotations = annotations.split('\n')
    ann = []
    for a in annotations:
        a = a.split()
        if len(a) == 5:
            for (i, _v) in enumerate(a):
                if i == 0:
                    a[i] = int(a[i])
                else:
                    a[i] = float(a[i])
            ann.append(a)
    return ann

def parse_darknet_ann_list_to_cls_box(annotations):
    if False:
        while True:
            i = 10
    'Parse darknet annotation format into two lists for class and bounding box.\n\n    Input list of [[class, x, y, w, h], ...], return two list of [class ...] and [[x, y, w, h], ...].\n\n    Parameters\n    ------------\n    annotations : list of list\n        A list of class and bounding boxes of images e.g. [[class, x, y, w, h], ...]\n\n    Returns\n    -------\n    list of int\n        List of class labels.\n\n    list of list of 4 numbers\n        List of bounding box.\n\n    '
    class_list = []
    bbox_list = []
    for ann in annotations:
        class_list.append(ann[0])
        bbox_list.append(ann[1:])
    return (class_list, bbox_list)

def obj_box_horizontal_flip(im, coords=None, is_rescale=False, is_center=False, is_random=False):
    if False:
        i = 10
        return i + 15
    'Left-right flip the image and coordinates for object detection.\n\n    Parameters\n    ----------\n    im : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    coords : list of list of 4 int/float or None\n        Coordinates [[x, y, w, h], [x, y, w, h], ...].\n    is_rescale : boolean\n        Set to True, if the input coordinates are rescaled to [0, 1]. Default is False.\n    is_center : boolean\n        Set to True, if the x and y of coordinates are the centroid (i.e. darknet format). Default is False.\n    is_random : boolean\n        If True, randomly flip. Default is False.\n\n    Returns\n    -------\n    numpy.array\n        A processed image\n    list of list of 4 numbers\n        A list of new bounding boxes.\n\n    Examples\n    --------\n    >>> im = np.zeros([80, 100])    # as an image with shape width=100, height=80\n    >>> im, coords = obj_box_left_right_flip(im, coords=[[0.2, 0.4, 0.3, 0.3], [0.1, 0.5, 0.2, 0.3]], is_rescale=True, is_center=True, is_random=False)\n    >>> print(coords)\n      [[0.8, 0.4, 0.3, 0.3], [0.9, 0.5, 0.2, 0.3]]\n    >>> im, coords = obj_box_left_right_flip(im, coords=[[0.2, 0.4, 0.3, 0.3]], is_rescale=True, is_center=False, is_random=False)\n    >>> print(coords)\n      [[0.5, 0.4, 0.3, 0.3]]\n    >>> im, coords = obj_box_left_right_flip(im, coords=[[20, 40, 30, 30]], is_rescale=False, is_center=True, is_random=False)\n    >>> print(coords)\n      [[80, 40, 30, 30]]\n    >>> im, coords = obj_box_left_right_flip(im, coords=[[20, 40, 30, 30]], is_rescale=False, is_center=False, is_random=False)\n    >>> print(coords)\n      [[50, 40, 30, 30]]\n\n    '
    if coords is None:
        coords = []

    def _flip(im, coords):
        if False:
            i = 10
            return i + 15
        im = flip_axis(im, axis=1, is_random=False)
        coords_new = list()
        for coord in coords:
            if len(coord) != 4:
                raise AssertionError('coordinate should be 4 values : [x, y, w, h]')
            if is_rescale:
                if is_center:
                    x = 1.0 - coord[0]
                else:
                    x = 1.0 - coord[0] - coord[2]
            elif is_center:
                x = im.shape[1] - coord[0]
            else:
                x = im.shape[1] - coord[0] - coord[2]
            coords_new.append([x, coord[1], coord[2], coord[3]])
        return (im, coords_new)
    if is_random:
        factor = np.random.uniform(-1, 1)
        if factor > 0:
            return _flip(im, coords)
        else:
            return (im, coords)
    else:
        return _flip(im, coords)
obj_box_left_right_flip = obj_box_horizontal_flip

def obj_box_imresize(im, coords=None, size=None, interp='bicubic', mode=None, is_rescale=False):
    if False:
        while True:
            i = 10
    'Resize an image, and compute the new bounding box coordinates.\n\n    Parameters\n    -------------\n    im : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    coords : list of list of 4 int/float or None\n        Coordinates [[x, y, w, h], [x, y, w, h], ...]\n    size interp and mode : args\n        See ``tl.prepro.imresize``.\n    is_rescale : boolean\n        Set to True, if the input coordinates are rescaled to [0, 1], then return the original coordinates. Default is False.\n\n    Returns\n    -------\n    numpy.array\n        A processed image\n    list of list of 4 numbers\n        A list of new bounding boxes.\n\n    Examples\n    --------\n    >>> im = np.zeros([80, 100, 3])    # as an image with shape width=100, height=80\n    >>> _, coords = obj_box_imresize(im, coords=[[20, 40, 30, 30], [10, 20, 20, 20]], size=[160, 200], is_rescale=False)\n    >>> print(coords)\n      [[40, 80, 60, 60], [20, 40, 40, 40]]\n    >>> _, coords = obj_box_imresize(im, coords=[[20, 40, 30, 30]], size=[40, 100], is_rescale=False)\n    >>> print(coords)\n      [[20, 20, 30, 15]]\n    >>> _, coords = obj_box_imresize(im, coords=[[20, 40, 30, 30]], size=[60, 150], is_rescale=False)\n    >>> print(coords)\n      [[30, 30, 45, 22]]\n    >>> im2, coords = obj_box_imresize(im, coords=[[0.2, 0.4, 0.3, 0.3]], size=[160, 200], is_rescale=True)\n    >>> print(coords, im2.shape)\n      [[0.2, 0.4, 0.3, 0.3]] (160, 200, 3)\n\n    '
    if coords is None:
        coords = []
    if size is None:
        size = [100, 100]
    (imh, imw) = im.shape[0:2]
    imh = imh * 1.0
    imw = imw * 1.0
    im = imresize(im, size=size, interp=interp, mode=mode)
    if is_rescale is False:
        coords_new = list()
        for coord in coords:
            if len(coord) != 4:
                raise AssertionError('coordinate should be 4 values : [x, y, w, h]')
            x = int(coord[0] * (size[1] / imw))
            y = int(coord[1] * (size[0] / imh))
            w = int(coord[2] * (size[1] / imw))
            h = int(coord[3] * (size[0] / imh))
            coords_new.append([x, y, w, h])
        return (im, coords_new)
    else:
        return (im, coords)

def obj_box_crop(im, classes=None, coords=None, wrg=100, hrg=100, is_rescale=False, is_center=False, is_random=False, thresh_wh=0.02, thresh_wh2=12.0):
    if False:
        return 10
    'Randomly or centrally crop an image, and compute the new bounding box coordinates.\n    Objects outside the cropped image will be removed.\n\n    Parameters\n    -----------\n    im : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    classes : list of int or None\n        Class IDs.\n    coords : list of list of 4 int/float or None\n        Coordinates [[x, y, w, h], [x, y, w, h], ...]\n    wrg hrg and is_random : args\n        See ``tl.prepro.crop``.\n    is_rescale : boolean\n        Set to True, if the input coordinates are rescaled to [0, 1]. Default is False.\n    is_center : boolean, default False\n        Set to True, if the x and y of coordinates are the centroid (i.e. darknet format). Default is False.\n    thresh_wh : float\n        Threshold, remove the box if its ratio of width(height) to image size less than the threshold.\n    thresh_wh2 : float\n        Threshold, remove the box if its ratio of width to height or vice verse higher than the threshold.\n\n    Returns\n    -------\n    numpy.array\n        A processed image\n    list of int\n        A list of classes\n    list of list of 4 numbers\n        A list of new bounding boxes.\n\n    '
    if classes is None:
        classes = []
    if coords is None:
        coords = []
    (h, w) = (im.shape[0], im.shape[1])
    if h <= hrg or w <= wrg:
        raise AssertionError('The size of cropping should smaller than the original image')
    if is_random:
        h_offset = int(np.random.uniform(0, h - hrg) - 1)
        w_offset = int(np.random.uniform(0, w - wrg) - 1)
        h_end = hrg + h_offset
        w_end = wrg + w_offset
        im_new = im[h_offset:h_end, w_offset:w_end]
    else:
        h_offset = int(np.floor((h - hrg) / 2.0))
        w_offset = int(np.floor((w - wrg) / 2.0))
        h_end = h_offset + hrg
        w_end = w_offset + wrg
        im_new = im[h_offset:h_end, w_offset:w_end]

    def _get_coord(coord):
        if False:
            i = 10
            return i + 15
        'Input pixel-unit [x, y, w, h] format, then make sure [x, y] it is the up-left coordinates,\n        before getting the new coordinates.\n        Boxes outsides the cropped image will be removed.\n\n        '
        if is_center:
            coord = obj_box_coord_centroid_to_upleft(coord)
        x = coord[0] - w_offset
        y = coord[1] - h_offset
        w = coord[2]
        h = coord[3]
        if x < 0:
            if x + w <= 0:
                return None
            w = w + x
            x = 0
        elif x > im_new.shape[1]:
            return None
        if y < 0:
            if y + h <= 0:
                return None
            h = h + y
            y = 0
        elif y > im_new.shape[0]:
            return None
        if x is not None and x + w > im_new.shape[1]:
            w = im_new.shape[1] - x
        if y is not None and y + h > im_new.shape[0]:
            h = im_new.shape[0] - y
        if w / (h + 1.0) > thresh_wh2 or h / (w + 1.0) > thresh_wh2:
            return None
        if w / (im_new.shape[1] * 1.0) < thresh_wh or h / (im_new.shape[0] * 1.0) < thresh_wh:
            return None
        coord = [x, y, w, h]
        if is_center:
            coord = obj_box_coord_upleft_to_centroid(coord)
        return coord
    coords_new = list()
    classes_new = list()
    for (i, _) in enumerate(coords):
        coord = coords[i]
        if len(coord) != 4:
            raise AssertionError('coordinate should be 4 values : [x, y, w, h]')
        if is_rescale:
            coord = obj_box_coord_scale_to_pixelunit(coord, im.shape)
            coord = _get_coord(coord)
            if coord is not None:
                coord = obj_box_coord_rescale(coord, im_new.shape)
                coords_new.append(coord)
                classes_new.append(classes[i])
        else:
            coord = _get_coord(coord)
            if coord is not None:
                coords_new.append(coord)
                classes_new.append(classes[i])
    return (im_new, classes_new, coords_new)

def obj_box_shift(im, classes=None, coords=None, wrg=0.1, hrg=0.1, row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0.0, order=1, is_rescale=False, is_center=False, is_random=False, thresh_wh=0.02, thresh_wh2=12.0):
    if False:
        while True:
            i = 10
    'Shift an image randomly or non-randomly, and compute the new bounding box coordinates.\n    Objects outside the cropped image will be removed.\n\n    Parameters\n    -----------\n    im : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    classes : list of int or None\n        Class IDs.\n    coords : list of list of 4 int/float or None\n        Coordinates [[x, y, w, h], [x, y, w, h], ...]\n    wrg, hrg row_index col_index channel_index is_random fill_mode cval and order : see ``tl.prepro.shift``.\n    is_rescale : boolean\n        Set to True, if the input coordinates are rescaled to [0, 1]. Default is False.\n    is_center : boolean\n        Set to True, if the x and y of coordinates are the centroid (i.e. darknet format). Default is False.\n    thresh_wh : float\n        Threshold, remove the box if its ratio of width(height) to image size less than the threshold.\n    thresh_wh2 : float\n        Threshold, remove the box if its ratio of width to height or vice verse higher than the threshold.\n\n\n    Returns\n    -------\n    numpy.array\n        A processed image\n    list of int\n        A list of classes\n    list of list of 4 numbers\n        A list of new bounding boxes.\n\n    '
    if classes is None:
        classes = []
    if coords is None:
        coords = []
    (imh, imw) = (im.shape[row_index], im.shape[col_index])
    if hrg >= 1.0 and hrg <= 0.0 and (wrg >= 1.0) and (wrg <= 0.0):
        raise AssertionError('shift range should be (0, 1)')
    if is_random:
        tx = np.random.uniform(-hrg, hrg) * imh
        ty = np.random.uniform(-wrg, wrg) * imw
    else:
        (tx, ty) = (hrg * imh, wrg * imw)
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    transform_matrix = translation_matrix
    im_new = affine_transform(im, transform_matrix, channel_index, fill_mode, cval, order)

    def _get_coord(coord):
        if False:
            while True:
                i = 10
        'Input pixel-unit [x, y, w, h] format, then make sure [x, y] it is the up-left coordinates,\n        before getting the new coordinates.\n        Boxes outsides the cropped image will be removed.\n\n        '
        if is_center:
            coord = obj_box_coord_centroid_to_upleft(coord)
        x = coord[0] - ty
        y = coord[1] - tx
        w = coord[2]
        h = coord[3]
        if x < 0:
            if x + w <= 0:
                return None
            w = w + x
            x = 0
        elif x > im_new.shape[1]:
            return None
        if y < 0:
            if y + h <= 0:
                return None
            h = h + y
            y = 0
        elif y > im_new.shape[0]:
            return None
        if x is not None and x + w > im_new.shape[1]:
            w = im_new.shape[1] - x
        if y is not None and y + h > im_new.shape[0]:
            h = im_new.shape[0] - y
        if w / (h + 1.0) > thresh_wh2 or h / (w + 1.0) > thresh_wh2:
            return None
        if w / (im_new.shape[1] * 1.0) < thresh_wh or h / (im_new.shape[0] * 1.0) < thresh_wh:
            return None
        coord = [x, y, w, h]
        if is_center:
            coord = obj_box_coord_upleft_to_centroid(coord)
        return coord
    coords_new = list()
    classes_new = list()
    for (i, _) in enumerate(coords):
        coord = coords[i]
        if len(coord) != 4:
            raise AssertionError('coordinate should be 4 values : [x, y, w, h]')
        if is_rescale:
            coord = obj_box_coord_scale_to_pixelunit(coord, im.shape)
            coord = _get_coord(coord)
            if coord is not None:
                coord = obj_box_coord_rescale(coord, im_new.shape)
                coords_new.append(coord)
                classes_new.append(classes[i])
        else:
            coord = _get_coord(coord)
            if coord is not None:
                coords_new.append(coord)
                classes_new.append(classes[i])
    return (im_new, classes_new, coords_new)

def obj_box_zoom(im, classes=None, coords=None, zoom_range=(0.9, 1.1), row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0.0, order=1, is_rescale=False, is_center=False, is_random=False, thresh_wh=0.02, thresh_wh2=12.0):
    if False:
        i = 10
        return i + 15
    'Zoom in and out of a single image, randomly or non-randomly, and compute the new bounding box coordinates.\n    Objects outside the cropped image will be removed.\n\n    Parameters\n    -----------\n    im : numpy.array\n        An image with dimension of [row, col, channel] (default).\n    classes : list of int or None\n        Class IDs.\n    coords : list of list of 4 int/float or None\n        Coordinates [[x, y, w, h], [x, y, w, h], ...].\n    zoom_range row_index col_index channel_index is_random fill_mode cval and order : see ``tl.prepro.zoom``.\n    is_rescale : boolean\n        Set to True, if the input coordinates are rescaled to [0, 1]. Default is False.\n    is_center : boolean\n        Set to True, if the x and y of coordinates are the centroid. (i.e. darknet format). Default is False.\n    thresh_wh : float\n        Threshold, remove the box if its ratio of width(height) to image size less than the threshold.\n    thresh_wh2 : float\n        Threshold, remove the box if its ratio of width to height or vice verse higher than the threshold.\n\n    Returns\n    -------\n    numpy.array\n        A processed image\n    list of int\n        A list of classes\n    list of list of 4 numbers\n        A list of new bounding boxes.\n\n    '
    if classes is None:
        classes = []
    if coords is None:
        coords = []
    if len(zoom_range) != 2:
        raise Exception('zoom_range should be a tuple or list of two floats. Received arg: ', zoom_range)
    if is_random:
        if zoom_range[0] == 1 and zoom_range[1] == 1:
            (zx, zy) = (1, 1)
            tl.logging.info(' random_zoom : not zoom in/out')
        else:
            (zx, zy) = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    else:
        (zx, zy) = zoom_range
    zoom_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
    (h, w) = (im.shape[row_index], im.shape[col_index])
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    im_new = affine_transform(im, transform_matrix, channel_index, fill_mode, cval, order)

    def _get_coord(coord):
        if False:
            print('Hello World!')
        'Input pixel-unit [x, y, w, h] format, then make sure [x, y] it is the up-left coordinates,\n        before getting the new coordinates.\n        Boxes outsides the cropped image will be removed.\n\n        '
        if is_center:
            coord = obj_box_coord_centroid_to_upleft(coord)
        x = (coord[0] - im.shape[1] / 2) / zy + im.shape[1] / 2
        y = (coord[1] - im.shape[0] / 2) / zx + im.shape[0] / 2
        w = coord[2] / zy
        h = coord[3] / zx
        if x < 0:
            if x + w <= 0:
                return None
            w = w + x
            x = 0
        elif x > im_new.shape[1]:
            return None
        if y < 0:
            if y + h <= 0:
                return None
            h = h + y
            y = 0
        elif y > im_new.shape[0]:
            return None
        if x is not None and x + w > im_new.shape[1]:
            w = im_new.shape[1] - x
        if y is not None and y + h > im_new.shape[0]:
            h = im_new.shape[0] - y
        if w / (h + 1.0) > thresh_wh2 or h / (w + 1.0) > thresh_wh2:
            return None
        if w / (im_new.shape[1] * 1.0) < thresh_wh or h / (im_new.shape[0] * 1.0) < thresh_wh:
            return None
        coord = [x, y, w, h]
        if is_center:
            coord = obj_box_coord_upleft_to_centroid(coord)
        return coord
    coords_new = list()
    classes_new = list()
    for (i, _) in enumerate(coords):
        coord = coords[i]
        if len(coord) != 4:
            raise AssertionError('coordinate should be 4 values : [x, y, w, h]')
        if is_rescale:
            coord = obj_box_coord_scale_to_pixelunit(coord, im.shape)
            coord = _get_coord(coord)
            if coord is not None:
                coord = obj_box_coord_rescale(coord, im_new.shape)
                coords_new.append(coord)
                classes_new.append(classes[i])
        else:
            coord = _get_coord(coord)
            if coord is not None:
                coords_new.append(coord)
                classes_new.append(classes[i])
    return (im_new, classes_new, coords_new)

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post', truncating='pre', value=0.0):
    if False:
        return 10
    "Pads each sequence to the same length:\n    the length of the longest sequence.\n    If maxlen is provided, any sequence longer\n    than maxlen is truncated to maxlen.\n    Truncation happens off either the beginning (default) or\n    the end of the sequence.\n    Supports post-padding and pre-padding (default).\n\n    Parameters\n    ----------\n    sequences : list of list of int\n        All sequences where each row is a sequence.\n    maxlen : int\n        Maximum length.\n    dtype : numpy.dtype or str\n        Data type to cast the resulting sequence.\n    padding : str\n        Either 'pre' or 'post', pad either before or after each sequence.\n    truncating : str\n        Either 'pre' or 'post', remove values from sequences larger than maxlen either in the beginning or in the end of the sequence\n    value : float\n        Value to pad the sequences to the desired value.\n\n    Returns\n    ----------\n    x : numpy.array\n        With dimensions (number_of_sequences, maxlen)\n\n    Examples\n    ----------\n    >>> sequences = [[1,1,1,1,1],[2,2,2],[3,3]]\n    >>> sequences = pad_sequences(sequences, maxlen=None, dtype='int32',\n    ...                  padding='post', truncating='pre', value=0.)\n    [[1 1 1 1 1]\n     [2 2 2 0 0]\n     [3 3 0 0 0]]\n\n    "
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for (idx, s) in enumerate(sequences):
        if len(s) == 0:
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' % (trunc.shape[1:], idx, sample_shape))
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x.tolist()

def remove_pad_sequences(sequences, pad_id=0):
    if False:
        for i in range(10):
            print('nop')
    'Remove padding.\n\n    Parameters\n    -----------\n    sequences : list of list of int\n        All sequences where each row is a sequence.\n    pad_id : int\n        The pad ID.\n\n    Returns\n    ----------\n    list of list of int\n        The processed sequences.\n\n    Examples\n    ----------\n    >>> sequences = [[2,3,4,0,0], [5,1,2,3,4,0,0,0], [4,5,0,2,4,0,0,0]]\n    >>> print(remove_pad_sequences(sequences, pad_id=0))\n    [[2, 3, 4], [5, 1, 2, 3, 4], [4, 5, 0, 2, 4]]\n\n    '
    sequences_out = copy.deepcopy(sequences)
    for (i, _) in enumerate(sequences):
        for j in range(1, len(sequences[i])):
            if sequences[i][-j] != pad_id:
                sequences_out[i] = sequences_out[i][0:-j + 1]
                break
    return sequences_out

def process_sequences(sequences, end_id=0, pad_val=0, is_shorten=True, remain_end_id=False):
    if False:
        for i in range(10):
            print('nop')
    'Set all tokens(ids) after END token to the padding value, and then shorten (option) it to the maximum sequence length in this batch.\n\n    Parameters\n    -----------\n    sequences : list of list of int\n        All sequences where each row is a sequence.\n    end_id : int\n        The special token for END.\n    pad_val : int\n        Replace the `end_id` and the IDs after `end_id` to this value.\n    is_shorten : boolean\n        Shorten the sequences. Default is True.\n    remain_end_id : boolean\n        Keep an `end_id` in the end. Default is False.\n\n    Returns\n    ----------\n    list of list of int\n        The processed sequences.\n\n    Examples\n    ---------\n    >>> sentences_ids = [[4, 3, 5, 3, 2, 2, 2, 2],  <-- end_id is 2\n    ...                  [5, 3, 9, 4, 9, 2, 2, 3]]  <-- end_id is 2\n    >>> sentences_ids = precess_sequences(sentences_ids, end_id=vocab.end_id, pad_val=0, is_shorten=True)\n    [[4, 3, 5, 3, 0], [5, 3, 9, 4, 9]]\n\n    '
    max_length = 0
    for (_, seq) in enumerate(sequences):
        is_end = False
        for (i_w, n) in enumerate(seq):
            if n == end_id and is_end == False:
                is_end = True
                if max_length < i_w:
                    max_length = i_w
                if remain_end_id is False:
                    seq[i_w] = pad_val
            elif is_end == True:
                seq[i_w] = pad_val
    if remain_end_id is True:
        max_length += 1
    if is_shorten:
        for (i, seq) in enumerate(sequences):
            sequences[i] = seq[:max_length]
    return sequences

def sequences_add_start_id(sequences, start_id=0, remove_last=False):
    if False:
        for i in range(10):
            print('nop')
    'Add special start token(id) in the beginning of each sequence.\n\n    Parameters\n    ------------\n    sequences : list of list of int\n        All sequences where each row is a sequence.\n    start_id : int\n        The start ID.\n    remove_last : boolean\n        Remove the last value of each sequences. Usually be used for removing the end ID.\n\n    Returns\n    ----------\n    list of list of int\n        The processed sequences.\n\n    Examples\n    ---------\n    >>> sentences_ids = [[4,3,5,3,2,2,2,2], [5,3,9,4,9,2,2,3]]\n    >>> sentences_ids = sequences_add_start_id(sentences_ids, start_id=2)\n    [[2, 4, 3, 5, 3, 2, 2, 2, 2], [2, 5, 3, 9, 4, 9, 2, 2, 3]]\n    >>> sentences_ids = sequences_add_start_id(sentences_ids, start_id=2, remove_last=True)\n    [[2, 4, 3, 5, 3, 2, 2, 2], [2, 5, 3, 9, 4, 9, 2, 2]]\n\n    For Seq2seq\n\n    >>> input = [a, b, c]\n    >>> target = [x, y, z]\n    >>> decode_seq = [start_id, a, b] <-- sequences_add_start_id(input, start_id, True)\n\n    '
    sequences_out = [[] for _ in range(len(sequences))]
    for (i, _) in enumerate(sequences):
        if remove_last:
            sequences_out[i] = [start_id] + sequences[i][:-1]
        else:
            sequences_out[i] = [start_id] + sequences[i]
    return sequences_out

def sequences_add_end_id(sequences, end_id=888):
    if False:
        for i in range(10):
            print('nop')
    'Add special end token(id) in the end of each sequence.\n\n    Parameters\n    -----------\n    sequences : list of list of int\n        All sequences where each row is a sequence.\n    end_id : int\n        The end ID.\n\n    Returns\n    ----------\n    list of list of int\n        The processed sequences.\n\n    Examples\n    ---------\n    >>> sequences = [[1,2,3],[4,5,6,7]]\n    >>> print(sequences_add_end_id(sequences, end_id=999))\n    [[1, 2, 3, 999], [4, 5, 6, 999]]\n\n    '
    sequences_out = [[] for _ in range(len(sequences))]
    for (i, _) in enumerate(sequences):
        sequences_out[i] = sequences[i] + [end_id]
    return sequences_out

def sequences_add_end_id_after_pad(sequences, end_id=888, pad_id=0):
    if False:
        while True:
            i = 10
    'Add special end token(id) in the end of each sequence.\n\n    Parameters\n    -----------\n    sequences : list of list of int\n        All sequences where each row is a sequence.\n    end_id : int\n        The end ID.\n    pad_id : int\n        The pad ID.\n\n    Returns\n    ----------\n    list of list of int\n        The processed sequences.\n\n    Examples\n    ---------\n    >>> sequences = [[1,2,0,0], [1,2,3,0], [1,2,3,4]]\n    >>> print(sequences_add_end_id_after_pad(sequences, end_id=99, pad_id=0))\n    [[1, 2, 99, 0], [1, 2, 3, 99], [1, 2, 3, 4]]\n\n    '
    sequences_out = copy.deepcopy(sequences)
    for (i, v) in enumerate(sequences):
        for (j, _v2) in enumerate(v):
            if sequences[i][j] == pad_id:
                sequences_out[i][j] = end_id
                break
    return sequences_out

def sequences_get_mask(sequences, pad_val=0):
    if False:
        i = 10
        return i + 15
    'Return mask for sequences.\n\n    Parameters\n    -----------\n    sequences : list of list of int\n        All sequences where each row is a sequence.\n    pad_val : int\n        The pad value.\n\n    Returns\n    ----------\n    list of list of int\n        The mask.\n\n    Examples\n    ---------\n    >>> sentences_ids = [[4, 0, 5, 3, 0, 0],\n    ...                  [5, 3, 9, 4, 9, 0]]\n    >>> mask = sequences_get_mask(sentences_ids, pad_val=0)\n    [[1 1 1 1 0 0]\n     [1 1 1 1 1 0]]\n\n    '
    mask = np.ones_like(sequences)
    for (i, seq) in enumerate(sequences):
        for i_w in reversed(range(len(seq))):
            if seq[i_w] == pad_val:
                mask[i, i_w] = 0
            else:
                break
    return mask

def keypoint_random_crop(image, annos, mask=None, size=(368, 368)):
    if False:
        print('Hello World!')
    'Randomly crop an image and corresponding keypoints without influence scales, given by ``keypoint_random_resize_shortestedge``.\n\n    Parameters\n    -----------\n    image : 3 channel image\n        The given image for augmentation.\n    annos : list of list of floats\n        The keypoints annotation of people.\n    mask : single channel image or None\n        The mask if available.\n    size : tuple of int\n        The size of returned image.\n\n    Returns\n    ----------\n    preprocessed image, annotation, mask\n\n    '
    _target_height = size[0]
    _target_width = size[1]
    target_size = (_target_width, _target_height)
    if len(np.shape(image)) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    (height, width, _) = np.shape(image)
    for _ in range(50):
        x = random.randrange(0, width - target_size[0]) if width > target_size[0] else 0
        y = random.randrange(0, height - target_size[1]) if height > target_size[1] else 0
        for joint in annos:
            if x <= joint[0][0] < x + target_size[0] and y <= joint[0][1] < y + target_size[1]:
                break

    def pose_crop(image, annos, mask, x, y, w, h):
        if False:
            for i in range(10):
                print('nop')
        target_size = (w, h)
        img = image
        resized = img[y:y + target_size[1], x:x + target_size[0], :]
        resized_mask = mask[y:y + target_size[1], x:x + target_size[0]]
        adjust_joint_list = []
        for joint in annos:
            adjust_joint = []
            for point in joint:
                if point[0] < -10 or point[1] < -10:
                    adjust_joint.append((-1000, -1000))
                    continue
                (new_x, new_y) = (point[0] - x, point[1] - y)
                if new_x > w - 1 or new_y > h - 1:
                    adjust_joint.append((-1000, -1000))
                    continue
                adjust_joint.append((new_x, new_y))
            adjust_joint_list.append(adjust_joint)
        return (resized, adjust_joint_list, resized_mask)
    return pose_crop(image, annos, mask, x, y, target_size[0], target_size[1])

def keypoint_resize_random_crop(image, annos, mask=None, size=(368, 368)):
    if False:
        for i in range(10):
            print('nop')
    'Reszie the image to make either its width or height equals to the given sizes.\n    Then randomly crop image without influence scales.\n    Resize the image match with the minimum size before cropping, this API will change the zoom scale of object.\n\n    Parameters\n    -----------\n    image : 3 channel image\n        The given image for augmentation.\n    annos : list of list of floats\n        The keypoints annotation of people.\n    mask : single channel image or None\n        The mask if available.\n    size : tuple of int\n        The size (height, width) of returned image.\n\n    Returns\n    ----------\n    preprocessed image, annos, mask\n\n    '
    if len(np.shape(image)) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    def resize_image(image, annos, mask, target_width, target_height):
        if False:
            for i in range(10):
                print('nop')
        'Reszie image\n\n        Parameters\n        -----------\n        image : 3 channel image\n            The given image.\n        annos : list of list of floats\n            Keypoints of people\n        mask : single channel image or None\n            The mask if available.\n        target_width : int\n            Expected width of returned image.\n        target_height : int\n            Expected height of returned image.\n\n        Returns\n        ----------\n        preprocessed input image, annos, mask\n\n        '
        (y, x, _) = np.shape(image)
        ratio_y = target_height / y
        ratio_x = target_width / x
        new_joints = []
        for people in annos:
            new_keypoints = []
            for keypoints in people:
                if keypoints[0] < 0 or keypoints[1] < 0:
                    new_keypoints.append((-1000, -1000))
                    continue
                pts = (int(keypoints[0] * ratio_x + 0.5), int(keypoints[1] * ratio_y + 0.5))
                if pts[0] > target_width - 1 or pts[1] > target_height - 1:
                    new_keypoints.append((-1000, -1000))
                    continue
                new_keypoints.append(pts)
            new_joints.append(new_keypoints)
        annos = new_joints
        new_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        if mask is not None:
            new_mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_AREA)
            return (new_image, annos, new_mask)
        else:
            return (new_image, annos, None)
    _target_height = size[0]
    _target_width = size[1]
    if len(np.shape(image)) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    (input_height, input_width, _) = np.shape(image)
    vertical_ratio = _target_height / input_height
    horizontal_ratio = _target_width / input_width
    rescale_ratio = max(vertical_ratio, horizontal_ratio)
    (image, annos, mask) = resize_image(image, annos, mask, round(input_width * rescale_ratio), round(input_height * rescale_ratio))
    (new_height, new_width, _) = np.shape(image)
    if new_height > _target_height:
        crop_range_y = np.random.randint(0, new_height - _target_height)
        image = image[crop_range_y:crop_range_y + _target_height, :, :]
        if mask is not None:
            mask = mask[crop_range_y:crop_range_y + _target_height, :]
        new_joints = []
        for people in annos:
            new_keypoints = []
            for keypoints in people:
                if keypoints[1] >= crop_range_y and keypoints[1] <= crop_range_y + _target_height - 1:
                    pts = (int(keypoints[0]), int(keypoints[1] - crop_range_y))
                else:
                    pts = (-1000, -1000)
                new_keypoints.append(pts)
            new_joints.append(new_keypoints)
        annos = new_joints
    elif new_width > _target_width:
        crop_range_x = np.random.randint(0, new_width - _target_width)
        image = image[:, crop_range_x:crop_range_x + _target_width, :]
        if mask is not None:
            mask = mask[:, crop_range_x:crop_range_x + _target_width]
        new_joints = []
        for people in annos:
            new_keypoints = []
            for keypoints in people:
                if keypoints[0] >= crop_range_x and keypoints[0] <= crop_range_x + _target_width - 1:
                    pts = (int(keypoints[0] - crop_range_x), int(keypoints[1]))
                else:
                    pts = (-1000, -1000)
                new_keypoints.append(pts)
            new_joints.append(new_keypoints)
        annos = new_joints
    if mask is not None:
        return (image, annos, mask)
    else:
        return (image, annos, None)

def keypoint_random_rotate(image, annos, mask=None, rg=15.0):
    if False:
        while True:
            i = 10
    'Rotate an image and corresponding keypoints.\n\n    Parameters\n    -----------\n    image : 3 channel image\n        The given image for augmentation.\n    annos : list of list of floats\n        The keypoints annotation of people.\n    mask : single channel image or None\n        The mask if available.\n    rg : int or float\n        Degree to rotate, usually 0 ~ 180.\n\n    Returns\n    ----------\n    preprocessed image, annos, mask\n\n    '

    def _rotate_coord(shape, newxy, point, angle):
        if False:
            i = 10
            return i + 15
        angle = -1 * angle / 180.0 * math.pi
        (ox, oy) = shape
        (px, py) = point
        ox /= 2
        oy /= 2
        qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        (new_x, new_y) = newxy
        qx += ox - new_x
        qy += oy - new_y
        return (int(qx + 0.5), int(qy + 0.5))

    def _largest_rotated_rect(w, h, angle):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get largest rectangle after rotation.\n        http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders\n        '
        angle = angle / 180.0 * math.pi
        if w <= 0 or h <= 0:
            return (0, 0)
        width_is_longer = w >= h
        (side_long, side_short) = (w, h) if width_is_longer else (h, w)
        (sin_a, cos_a) = (abs(math.sin(angle)), abs(math.cos(angle)))
        if side_short <= 2.0 * sin_a * cos_a * side_long:
            x = 0.5 * side_short
            (wr, hr) = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
        else:
            cos_2a = cos_a * cos_a - sin_a * sin_a
            (wr, hr) = ((w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a)
        return (int(np.round(wr)), int(np.round(hr)))
    img_shape = np.shape(image)
    height = img_shape[0]
    width = img_shape[1]
    deg = np.random.uniform(-rg, rg)
    img = image
    center = (img.shape[1] * 0.5, img.shape[0] * 0.5)
    rot_m = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), deg, 1)
    ret = cv2.warpAffine(img, rot_m, img.shape[1::-1], flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
    if img.ndim == 3 and ret.ndim == 2:
        ret = ret[:, :, np.newaxis]
    (neww, newh) = _largest_rotated_rect(ret.shape[1], ret.shape[0], deg)
    neww = min(neww, ret.shape[1])
    newh = min(newh, ret.shape[0])
    newx = int(center[0] - neww * 0.5)
    newy = int(center[1] - newh * 0.5)
    img = ret[newy:newy + newh, newx:newx + neww]
    adjust_joint_list = []
    for joint in annos:
        adjust_joint = []
        for point in joint:
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            (x, y) = _rotate_coord((width, height), (newx, newy), point, deg)
            if x > neww - 1 or y > newh - 1:
                adjust_joint.append((-1000, -1000))
                continue
            if x < 0 or y < 0:
                adjust_joint.append((-1000, -1000))
                continue
            adjust_joint.append((x, y))
        adjust_joint_list.append(adjust_joint)
    joint_list = adjust_joint_list
    if mask is not None:
        msk = mask
        center = (msk.shape[1] * 0.5, msk.shape[0] * 0.5)
        rot_m = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), deg, 1)
        ret = cv2.warpAffine(msk, rot_m, msk.shape[1::-1], flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
        if msk.ndim == 3 and msk.ndim == 2:
            ret = ret[:, :, np.newaxis]
        (neww, newh) = _largest_rotated_rect(ret.shape[1], ret.shape[0], deg)
        neww = min(neww, ret.shape[1])
        newh = min(newh, ret.shape[0])
        newx = int(center[0] - neww * 0.5)
        newy = int(center[1] - newh * 0.5)
        msk = ret[newy:newy + newh, newx:newx + neww]
        return (img, joint_list, msk)
    else:
        return (img, joint_list, None)

def keypoint_random_flip(image, annos, mask=None, prob=0.5, flip_list=(0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16, 18)):
    if False:
        print('Hello World!')
    "Flip an image and corresponding keypoints.\n\n    Parameters\n    -----------\n    image : 3 channel image\n        The given image for augmentation.\n    annos : list of list of floats\n        The keypoints annotation of people.\n    mask : single channel image or None\n        The mask if available.\n    prob : float, 0 to 1\n        The probability to flip the image, if 1, always flip the image.\n    flip_list : tuple of int\n        Denotes how the keypoints number be changed after flipping which is required for pose estimation task.\n        The left and right body should be maintained rather than switch.\n        (Default COCO format).\n        Set to an empty tuple if you don't need to maintain left and right information.\n\n    Returns\n    ----------\n    preprocessed image, annos, mask\n\n    "
    _prob = np.random.uniform(0, 1.0)
    if _prob < prob:
        return (image, annos, mask)
    (_, width, _) = np.shape(image)
    image = cv2.flip(image, 1)
    mask = cv2.flip(mask, 1)
    new_joints = []
    for people in annos:
        new_keypoints = []
        for k in flip_list:
            point = people[k]
            if point[0] < 0 or point[1] < 0:
                new_keypoints.append((-1000, -1000))
                continue
            if point[0] > image.shape[1] - 1 or point[1] > image.shape[0] - 1:
                new_keypoints.append((-1000, -1000))
                continue
            if width - point[0] > image.shape[1] - 1:
                new_keypoints.append((-1000, -1000))
                continue
            new_keypoints.append((width - point[0], point[1]))
        new_joints.append(new_keypoints)
    annos = new_joints
    return (image, annos, mask)

def keypoint_random_resize(image, annos, mask=None, zoom_range=(0.8, 1.2)):
    if False:
        print('Hello World!')
    'Randomly resize an image and corresponding keypoints.\n    The height and width of image will be changed independently, so the scale will be changed.\n\n    Parameters\n    -----------\n    image : 3 channel image\n        The given image for augmentation.\n    annos : list of list of floats\n        The keypoints annotation of people.\n    mask : single channel image or None\n        The mask if available.\n    zoom_range : tuple of two floats\n        The minimum and maximum factor to zoom in or out, e.g (0.5, 1) means zoom out 1~2 times.\n\n    Returns\n    ----------\n    preprocessed image, annos, mask\n\n    '
    height = image.shape[0]
    width = image.shape[1]
    (_min, _max) = zoom_range
    scalew = np.random.uniform(_min, _max)
    scaleh = np.random.uniform(_min, _max)
    neww = int(width * scalew)
    newh = int(height * scaleh)
    dst = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)
    if mask is not None:
        mask = cv2.resize(mask, (neww, newh), interpolation=cv2.INTER_AREA)
    adjust_joint_list = []
    for joint in annos:
        adjust_joint = []
        for point in joint:
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            adjust_joint.append((int(point[0] * scalew + 0.5), int(point[1] * scaleh + 0.5)))
        adjust_joint_list.append(adjust_joint)
    if mask is not None:
        return (dst, adjust_joint_list, mask)
    else:
        return (dst, adjust_joint_list, None)

def keypoint_random_resize_shortestedge(image, annos, mask=None, min_size=(368, 368), zoom_range=(0.8, 1.2), pad_val=(0, 0, np.random.uniform(0.0, 1.0))):
    if False:
        for i in range(10):
            print('nop')
    'Randomly resize an image and corresponding keypoints based on shorter edgeself.\n    If the resized image is smaller than `min_size`, uses padding to make shape matchs `min_size`.\n    The height and width of image will be changed together, the scale would not be changed.\n\n    Parameters\n    -----------\n    image : 3 channel image\n        The given image for augmentation.\n    annos : list of list of floats\n        The keypoints annotation of people.\n    mask : single channel image or None\n        The mask if available.\n    min_size : tuple of two int\n        The minimum size of height and width.\n    zoom_range : tuple of two floats\n        The minimum and maximum factor to zoom in or out, e.g (0.5, 1) means zoom out 1~2 times.\n    pad_val : int/float, or tuple of int or random function\n        The three padding values for RGB channels respectively.\n\n    Returns\n    ----------\n    preprocessed image, annos, mask\n\n    '
    _target_height = min_size[0]
    _target_width = min_size[1]
    if len(np.shape(image)) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    (height, width, _) = np.shape(image)
    ratio_w = _target_width / width
    ratio_h = _target_height / height
    ratio = min(ratio_w, ratio_h)
    target_size = int(min(width * ratio + 0.5, height * ratio + 0.5))
    random_target = np.random.uniform(zoom_range[0], zoom_range[1])
    target_size = int(target_size * random_target)

    def pose_resize_shortestedge(image, annos, mask, target_size):
        if False:
            i = 10
            return i + 15
        ' '
        (height, width, _) = np.shape(image)
        scale = target_size / min(height, width)
        if height < width:
            (newh, neww) = (target_size, int(scale * width + 0.5))
        else:
            (newh, neww) = (int(scale * height + 0.5), target_size)
        dst = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (neww, newh), interpolation=cv2.INTER_AREA)
        pw = ph = 0
        if neww < _target_width or newh < _target_height:
            pw = max(0, (_target_width - neww) // 2)
            ph = max(0, (_target_height - newh) // 2)
            mw = (_target_width - neww) % 2
            mh = (_target_height - newh) % 2
            dst = cv2.copyMakeBorder(dst, ph, ph + mh, pw, pw + mw, cv2.BORDER_CONSTANT, value=pad_val)
            if mask is not None:
                mask = cv2.copyMakeBorder(mask, ph, ph + mh, pw, pw + mw, cv2.BORDER_CONSTANT, value=1)
        adjust_joint_list = []
        for joint in annos:
            adjust_joint = []
            for point in joint:
                if point[0] < -100 or point[1] < -100:
                    adjust_joint.append((-1000, -1000))
                    continue
                adjust_joint.append((int(point[0] * scale + 0.5) + pw, int(point[1] * scale + 0.5) + ph))
            adjust_joint_list.append(adjust_joint)
        if mask is not None:
            return (dst, adjust_joint_list, mask)
        else:
            return (dst, adjust_joint_list, None)
    return pose_resize_shortestedge(image, annos, mask, target_size)