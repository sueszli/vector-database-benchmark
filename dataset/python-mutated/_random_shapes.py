import math
import numpy as np
from .draw import polygon as draw_polygon, disk as draw_disk, ellipse as draw_ellipse
from .._shared.utils import deprecate_kwarg, warn

def _generate_rectangle_mask(point, image, shape, random):
    if False:
        while True:
            i = 10
    'Generate a mask for a filled rectangle shape.\n\n    The height and width of the rectangle are generated randomly.\n\n    Parameters\n    ----------\n    point : tuple\n        The row and column of the top left corner of the rectangle.\n    image : tuple\n        The height, width and depth of the image into which the shape\n        is placed.\n    shape : tuple\n        The minimum and maximum size of the shape to fit.\n    random : `numpy.random.Generator`\n\n        The random state to use for random sampling.\n\n    Raises\n    ------\n    ArithmeticError\n        When a shape cannot be fit into the image with the given starting\n        coordinates. This usually means the image dimensions are too small or\n        shape dimensions too large.\n\n    Returns\n    -------\n    label : tuple\n        A (category, ((r0, r1), (c0, c1))) tuple specifying the category and\n        bounding box coordinates of the shape.\n    indices : 2-D array\n        A mask of indices that the shape fills.\n\n    '
    available_width = min(image[1] - point[1], shape[1]) - shape[0]
    available_height = min(image[0] - point[0], shape[1]) - shape[0]
    r = shape[0] + random.integers(max(1, available_height)) - 1
    c = shape[0] + random.integers(max(1, available_width)) - 1
    rectangle = draw_polygon([point[0], point[0] + r, point[0] + r, point[0]], [point[1], point[1], point[1] + c, point[1] + c])
    label = ('rectangle', ((point[0], point[0] + r + 1), (point[1], point[1] + c + 1)))
    return (rectangle, label)

def _generate_circle_mask(point, image, shape, random):
    if False:
        while True:
            i = 10
    'Generate a mask for a filled circle shape.\n\n    The radius of the circle is generated randomly.\n\n    Parameters\n    ----------\n    point : tuple\n        The row and column of the top left corner of the rectangle.\n    image : tuple\n        The height, width and depth of the image into which the shape is placed.\n    shape : tuple\n        The minimum and maximum size and color of the shape to fit.\n    random : `numpy.random.Generator`\n        The random state to use for random sampling.\n\n    Raises\n    ------\n    ArithmeticError\n        When a shape cannot be fit into the image with the given starting\n        coordinates. This usually means the image dimensions are too small or\n        shape dimensions too large.\n\n    Returns\n    -------\n    label : tuple\n        A (category, ((r0, r1), (c0, c1))) tuple specifying the category and\n        bounding box coordinates of the shape.\n    indices : 2-D array\n        A mask of indices that the shape fills.\n    '
    if shape[0] == 1 or shape[1] == 1:
        raise ValueError('size must be > 1 for circles')
    min_radius = shape[0] // 2.0
    max_radius = shape[1] // 2.0
    left = point[1]
    right = image[1] - point[1]
    top = point[0]
    bottom = image[0] - point[0]
    available_radius = min(left, right, top, bottom, max_radius) - min_radius
    if available_radius < 0:
        raise ArithmeticError('cannot fit shape to image')
    radius = int(min_radius + random.integers(max(1, available_radius)))
    disk = draw_disk((point[0], point[1]), radius)
    label = ('circle', ((point[0] - radius + 1, point[0] + radius), (point[1] - radius + 1, point[1] + radius)))
    return (disk, label)

def _generate_triangle_mask(point, image, shape, random):
    if False:
        while True:
            i = 10
    'Generate a mask for a filled equilateral triangle shape.\n\n    The length of the sides of the triangle is generated randomly.\n\n    Parameters\n    ----------\n    point : tuple\n        The row and column of the top left corner of a up-pointing triangle.\n    image : tuple\n        The height, width and depth of the image into which the shape\n        is placed.\n    shape : tuple\n        The minimum and maximum size and color of the shape to fit.\n    random : `numpy.random.Generator`\n        The random state to use for random sampling.\n\n    Raises\n    ------\n    ArithmeticError\n        When a shape cannot be fit into the image with the given starting\n        coordinates. This usually means the image dimensions are too small or\n        shape dimensions too large.\n\n    Returns\n    -------\n    label : tuple\n        A (category, ((r0, r1), (c0, c1))) tuple specifying the category and\n        bounding box coordinates of the shape.\n    indices : 2-D array\n        A mask of indices that the shape fills.\n\n    '
    if shape[0] == 1 or shape[1] == 1:
        raise ValueError('dimension must be > 1 for triangles')
    available_side = min(image[1] - point[1], point[0], shape[1]) - shape[0]
    side = shape[0] + random.integers(max(1, available_side)) - 1
    triangle_height = int(np.ceil(np.sqrt(3 / 4.0) * side))
    triangle = draw_polygon([point[0], point[0] - triangle_height, point[0]], [point[1], point[1] + side // 2, point[1] + side])
    label = ('triangle', ((point[0] - triangle_height, point[0] + 1), (point[1], point[1] + side + 1)))
    return (triangle, label)

def _generate_ellipse_mask(point, image, shape, random):
    if False:
        for i in range(10):
            print('nop')
    'Generate a mask for a filled ellipse shape.\n\n    The rotation, major and minor semi-axes of the ellipse are generated\n    randomly.\n\n    Parameters\n    ----------\n    point : tuple\n        The row and column of the top left corner of the rectangle.\n    image : tuple\n        The height, width and depth of the image into which the shape is\n        placed.\n    shape : tuple\n        The minimum and maximum size and color of the shape to fit.\n    random : `numpy.random.Generator`\n        The random state to use for random sampling.\n\n    Raises\n    ------\n    ArithmeticError\n        When a shape cannot be fit into the image with the given starting\n        coordinates. This usually means the image dimensions are too small or\n        shape dimensions too large.\n\n    Returns\n    -------\n    label : tuple\n        A (category, ((r0, r1), (c0, c1))) tuple specifying the category and\n        bounding box coordinates of the shape.\n    indices : 2-D array\n        A mask of indices that the shape fills.\n    '
    if shape[0] == 1 or shape[1] == 1:
        raise ValueError('size must be > 1 for ellipses')
    min_radius = shape[0] / 2.0
    max_radius = shape[1] / 2.0
    left = point[1]
    right = image[1] - point[1]
    top = point[0]
    bottom = image[0] - point[0]
    available_radius = min(left, right, top, bottom, max_radius)
    if available_radius < min_radius:
        raise ArithmeticError('cannot fit shape to image')
    r_radius = random.uniform(min_radius, available_radius + 1)
    c_radius = random.uniform(min_radius, available_radius + 1)
    rotation = random.uniform(-np.pi, np.pi)
    ellipse = draw_ellipse(point[0], point[1], r_radius, c_radius, shape=image[:2], rotation=rotation)
    max_radius = math.ceil(max(r_radius, c_radius))
    min_x = np.min(ellipse[0])
    max_x = np.max(ellipse[0]) + 1
    min_y = np.min(ellipse[1])
    max_y = np.max(ellipse[1]) + 1
    label = ('ellipse', ((min_x, max_x), (min_y, max_y)))
    return (ellipse, label)
SHAPE_GENERATORS = dict(rectangle=_generate_rectangle_mask, circle=_generate_circle_mask, triangle=_generate_triangle_mask, ellipse=_generate_ellipse_mask)
SHAPE_CHOICES = list(SHAPE_GENERATORS.values())

def _generate_random_colors(num_colors, num_channels, intensity_range, random):
    if False:
        for i in range(10):
            print('nop')
    'Generate an array of random colors.\n\n    Parameters\n    ----------\n    num_colors : int\n        Number of colors to generate.\n    num_channels : int\n        Number of elements representing color.\n    intensity_range : {tuple of tuples of ints, tuple of ints}, optional\n        The range of values to sample pixel values from. For grayscale images\n        the format is (min, max). For multichannel - ((min, max),) if the\n        ranges are equal across the channels, and\n        ((min_0, max_0), ... (min_N, max_N)) if they differ.\n    random : `numpy.random.Generator`\n        The random state to use for random sampling.\n\n    Raises\n    ------\n    ValueError\n        When the `intensity_range` is not in the interval (0, 255).\n\n    Returns\n    -------\n    colors : array\n        An array of shape (num_colors, num_channels), where the values for\n        each channel are drawn from the corresponding `intensity_range`.\n\n    '
    if num_channels == 1:
        intensity_range = (intensity_range,)
    elif len(intensity_range) == 1:
        intensity_range = intensity_range * num_channels
    colors = [random.integers(r[0], r[1] + 1, size=num_colors) for r in intensity_range]
    return np.transpose(colors)

@deprecate_kwarg({'random_seed': 'rng'}, deprecated_version='0.21', removed_version='0.23')
def random_shapes(image_shape, max_shapes, min_shapes=1, min_size=2, max_size=None, num_channels=3, shape=None, intensity_range=None, allow_overlap=False, num_trials=100, rng=None, *, channel_axis=-1):
    if False:
        for i in range(10):
            print('nop')
    "Generate an image with random shapes, labeled with bounding boxes.\n\n    The image is populated with random shapes with random sizes, random\n    locations, and random colors, with or without overlap.\n\n    Shapes have random (row, col) starting coordinates and random sizes bounded\n    by `min_size` and `max_size`. It can occur that a randomly generated shape\n    will not fit the image at all. In that case, the algorithm will try again\n    with new starting coordinates a certain number of times. However, it also\n    means that some shapes may be skipped altogether. In that case, this\n    function will generate fewer shapes than requested.\n\n    Parameters\n    ----------\n    image_shape : tuple\n        The number of rows and columns of the image to generate.\n    max_shapes : int\n        The maximum number of shapes to (attempt to) fit into the shape.\n    min_shapes : int, optional\n        The minimum number of shapes to (attempt to) fit into the shape.\n    min_size : int, optional\n        The minimum dimension of each shape to fit into the image.\n    max_size : int, optional\n        The maximum dimension of each shape to fit into the image.\n    num_channels : int, optional\n        Number of channels in the generated image. If 1, generate monochrome\n        images, else color images with multiple channels. Ignored if\n        ``multichannel`` is set to False.\n    shape : {rectangle, circle, triangle, ellipse, None} str, optional\n        The name of the shape to generate or `None` to pick random ones.\n    intensity_range : {tuple of tuples of uint8, tuple of uint8}, optional\n        The range of values to sample pixel values from. For grayscale\n        images the format is (min, max). For multichannel - ((min, max),)\n        if the ranges are equal across the channels, and\n        ((min_0, max_0), ... (min_N, max_N)) if they differ. As the\n        function supports generation of uint8 arrays only, the maximum\n        range is (0, 255). If None, set to (0, 254) for each channel\n        reserving color of intensity = 255 for background.\n    allow_overlap : bool, optional\n        If `True`, allow shapes to overlap.\n    num_trials : int, optional\n        How often to attempt to fit a shape into the image before skipping it.\n    rng : {`numpy.random.Generator`, int}, optional\n        Pseudo-random number generator.\n        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).\n        If `rng` is an int, it is used to seed the generator.\n    channel_axis : int or None, optional\n        If None, the image is assumed to be a grayscale (single channel) image.\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    image : uint8 array\n        An image with the fitted shapes.\n    labels : list\n        A list of labels, one per shape in the image. Each label is a\n        (category, ((r0, r1), (c0, c1))) tuple specifying the category and\n        bounding box coordinates of the shape.\n\n    Examples\n    --------\n    >>> import skimage.draw\n    >>> image, labels = skimage.draw.random_shapes((32, 32), max_shapes=3)\n    >>> image # doctest: +SKIP\n    array([\n       [[255, 255, 255],\n        [255, 255, 255],\n        [255, 255, 255],\n        ...,\n        [255, 255, 255],\n        [255, 255, 255],\n        [255, 255, 255]]], dtype=uint8)\n    >>> labels # doctest: +SKIP\n    [('circle', ((22, 18), (25, 21))),\n     ('triangle', ((5, 6), (13, 13)))]\n    "
    if min_size > image_shape[0] or min_size > image_shape[1]:
        raise ValueError('Minimum dimension must be less than ncols and nrows')
    max_size = max_size or max(image_shape[0], image_shape[1])
    if channel_axis is None:
        num_channels = 1
    if intensity_range is None:
        intensity_range = (0, 254) if num_channels == 1 else ((0, 254),)
    else:
        tmp = (intensity_range,) if num_channels == 1 else intensity_range
        for intensity_pair in tmp:
            for intensity in intensity_pair:
                if not 0 <= intensity <= 255:
                    msg = 'Intensity range must lie within (0, 255) interval'
                    raise ValueError(msg)
    rng = np.random.default_rng(rng)
    user_shape = shape
    image_shape = (image_shape[0], image_shape[1], num_channels)
    image = np.full(image_shape, 255, dtype=np.uint8)
    filled = np.zeros(image_shape, dtype=bool)
    labels = []
    num_shapes = rng.integers(min_shapes, max_shapes + 1)
    colors = _generate_random_colors(num_shapes, num_channels, intensity_range, rng)
    shape = (min_size, max_size)
    for shape_idx in range(num_shapes):
        if user_shape is None:
            shape_generator = rng.choice(SHAPE_CHOICES)
        else:
            shape_generator = SHAPE_GENERATORS[user_shape]
        for _ in range(num_trials):
            column = rng.integers(max(1, image_shape[1] - min_size))
            row = rng.integers(max(1, image_shape[0] - min_size))
            point = (row, column)
            try:
                (indices, label) = shape_generator(point, image_shape, shape, rng)
            except ArithmeticError:
                indices = []
                continue
            if allow_overlap or not filled[indices].any():
                image[indices] = colors[shape_idx]
                filled[indices] = True
                labels.append(label)
                break
        else:
            warn('Could not fit any shapes to image, consider reducing the minimum dimension')
    if channel_axis is None:
        image = np.squeeze(image, axis=2)
    else:
        image = np.moveaxis(image, -1, channel_axis)
    return (image, labels)