"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.

"""
import collections
import functools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
import tensorflow.compat.v2 as tf
from official.vision.detection.utils.object_detection import shape_utils
_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = ['AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque', 'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan', 'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange', 'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet', 'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite', 'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod', 'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki', 'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue', 'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey', 'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue', 'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime', 'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid', 'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen', 'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin', 'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed', 'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed', 'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple', 'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown', 'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue', 'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow', 'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White', 'WhiteSmoke', 'Yellow', 'YellowGreen']

def save_image_array_as_png(image, output_path):
    if False:
        while True:
            i = 10
    'Saves an image (represented as a numpy array) to PNG.\n\n  Args:\n    image: a numpy array with shape [height, width, 3].\n    output_path: path to which image should be written.\n  '
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    with tf.io.gfile.GFile(output_path, 'w') as fid:
        image_pil.save(fid, 'PNG')

def encode_image_array_as_png_str(image):
    if False:
        return 10
    'Encodes a numpy array into a PNG string.\n\n  Args:\n    image: a numpy array with shape [height, width, 3].\n\n  Returns:\n    PNG encoded image string.\n  '
    image_pil = Image.fromarray(np.uint8(image))
    output = six.BytesIO()
    image_pil.save(output, format='PNG')
    png_string = output.getvalue()
    output.close()
    return png_string

def draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax, color='red', thickness=4, display_str_list=(), use_normalized_coordinates=True):
    if False:
        while True:
            i = 10
    'Adds a bounding box to an image (numpy array).\n\n  Bounding box coordinates can be specified in either absolute (pixel) or\n  normalized coordinates by setting the use_normalized_coordinates argument.\n\n  Args:\n    image: a numpy array with shape [height, width, 3].\n    ymin: ymin of bounding box.\n    xmin: xmin of bounding box.\n    ymax: ymax of bounding box.\n    xmax: xmax of bounding box.\n    color: color to draw bounding box. Default is red.\n    thickness: line thickness. Default value is 4.\n    display_str_list: list of strings to display in box\n                      (each to be shown on its own line).\n    use_normalized_coordinates: If True (default), treat coordinates\n      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat\n      coordinates as absolute.\n  '
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, thickness, display_str_list, use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color='red', thickness=4, display_str_list=(), use_normalized_coordinates=True):
    if False:
        while True:
            i = 10
    "Adds a bounding box to an image.\n\n  Bounding box coordinates can be specified in either absolute (pixel) or\n  normalized coordinates by setting the use_normalized_coordinates argument.\n\n  Each string in display_str_list is displayed on a separate line above the\n  bounding box in black text on a rectangle filled with the input 'color'.\n  If the top of the bounding box extends to the edge of the image, the strings\n  are displayed below the bounding box.\n\n  Args:\n    image: a PIL.Image object.\n    ymin: ymin of bounding box.\n    xmin: xmin of bounding box.\n    ymax: ymax of bounding box.\n    xmax: xmax of bounding box.\n    color: color to draw bounding box. Default is red.\n    thickness: line thickness. Default value is 4.\n    display_str_list: list of strings to display in box\n                      (each to be shown on its own line).\n    use_normalized_coordinates: If True (default), treat coordinates\n      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat\n      coordinates as absolute.\n  "
    draw = ImageDraw.Draw(image)
    (im_width, im_height) = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    for display_str in display_str_list[::-1]:
        (text_width, text_height) = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin), display_str, fill='black', font=font)
        text_bottom -= text_height - 2 * margin

def draw_bounding_boxes_on_image_array(image, boxes, color='red', thickness=4, display_str_list_list=()):
    if False:
        print('Hello World!')
    'Draws bounding boxes on image (numpy array).\n\n  Args:\n    image: a numpy array object.\n    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).\n           The coordinates are in normalized format between [0, 1].\n    color: color to draw bounding box. Default is red.\n    thickness: line thickness. Default value is 4.\n    display_str_list_list: list of list of strings.\n                           a list of strings for each bounding box.\n                           The reason to pass a list of strings for a\n                           bounding box is that it might contain\n                           multiple labels.\n\n  Raises:\n    ValueError: if boxes is not a [N, 4] array\n  '
    image_pil = Image.fromarray(image)
    draw_bounding_boxes_on_image(image_pil, boxes, color, thickness, display_str_list_list)
    np.copyto(image, np.array(image_pil))

def draw_bounding_boxes_on_image(image, boxes, color='red', thickness=4, display_str_list_list=()):
    if False:
        print('Hello World!')
    'Draws bounding boxes on image.\n\n  Args:\n    image: a PIL.Image object.\n    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).\n           The coordinates are in normalized format between [0, 1].\n    color: color to draw bounding box. Default is red.\n    thickness: line thickness. Default value is 4.\n    display_str_list_list: list of list of strings.\n                           a list of strings for each bounding box.\n                           The reason to pass a list of strings for a\n                           bounding box is that it might contain\n                           multiple labels.\n\n  Raises:\n    ValueError: if boxes is not a [N, 4] array\n  '
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        display_str_list = ()
        if display_str_list_list:
            display_str_list = display_str_list_list[i]
        draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], color, thickness, display_str_list)

def _visualize_boxes(image, boxes, classes, scores, category_index, **kwargs):
    if False:
        print('Hello World!')
    return visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index=category_index, **kwargs)

def _visualize_boxes_and_masks(image, boxes, classes, scores, masks, category_index, **kwargs):
    if False:
        i = 10
        return i + 15
    return visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index=category_index, instance_masks=masks, **kwargs)

def _visualize_boxes_and_keypoints(image, boxes, classes, scores, keypoints, category_index, **kwargs):
    if False:
        return 10
    return visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index=category_index, keypoints=keypoints, **kwargs)

def _visualize_boxes_and_masks_and_keypoints(image, boxes, classes, scores, masks, keypoints, category_index, **kwargs):
    if False:
        while True:
            i = 10
    return visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index=category_index, instance_masks=masks, keypoints=keypoints, **kwargs)

def _resize_original_image(image, image_shape):
    if False:
        while True:
            i = 10
    image = tf.expand_dims(image, 0)
    image = tf.image.resize(image, image_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return tf.cast(tf.squeeze(image, 0), tf.uint8)

def draw_bounding_boxes_on_image_tensors(images, boxes, classes, scores, category_index, original_image_spatial_shape=None, true_image_shape=None, instance_masks=None, keypoints=None, max_boxes_to_draw=20, min_score_thresh=0.2, use_normalized_coordinates=True):
    if False:
        for i in range(10):
            print('nop')
    "Draws bounding boxes, masks, and keypoints on batch of image tensors.\n\n  Args:\n    images: A 4D uint8 image tensor of shape [N, H, W, C]. If C > 3, additional\n      channels will be ignored. If C = 1, then we convert the images to RGB\n      images.\n    boxes: [N, max_detections, 4] float32 tensor of detection boxes.\n    classes: [N, max_detections] int tensor of detection classes. Note that\n      classes are 1-indexed.\n    scores: [N, max_detections] float32 tensor of detection scores.\n    category_index: a dict that maps integer ids to category dicts. e.g.\n      {1: {1: 'dog'}, 2: {2: 'cat'}, ...}\n    original_image_spatial_shape: [N, 2] tensor containing the spatial size of\n      the original image.\n    true_image_shape: [N, 3] tensor containing the spatial size of unpadded\n      original_image.\n    instance_masks: A 4D uint8 tensor of shape [N, max_detection, H, W] with\n      instance masks.\n    keypoints: A 4D float32 tensor of shape [N, max_detection, num_keypoints, 2]\n      with keypoints.\n    max_boxes_to_draw: Maximum number of boxes to draw on an image. Default 20.\n    min_score_thresh: Minimum score threshold for visualization. Default 0.2.\n    use_normalized_coordinates: Whether to assume boxes and kepoints are in\n      normalized coordinates (as opposed to absolute coordiantes).\n      Default is True.\n\n  Returns:\n    4D image tensor of type uint8, with boxes drawn on top.\n  "
    if images.shape[3] > 3:
        images = images[:, :, :, 0:3]
    elif images.shape[3] == 1:
        images = tf.image.grayscale_to_rgb(images)
    visualization_keyword_args = {'use_normalized_coordinates': use_normalized_coordinates, 'max_boxes_to_draw': max_boxes_to_draw, 'min_score_thresh': min_score_thresh, 'agnostic_mode': False, 'line_thickness': 4}
    if true_image_shape is None:
        true_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 3])
    else:
        true_shapes = true_image_shape
    if original_image_spatial_shape is None:
        original_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 2])
    else:
        original_shapes = original_image_spatial_shape
    if instance_masks is not None and keypoints is None:
        visualize_boxes_fn = functools.partial(_visualize_boxes_and_masks, category_index=category_index, **visualization_keyword_args)
        elems = [true_shapes, original_shapes, images, boxes, classes, scores, instance_masks]
    elif instance_masks is None and keypoints is not None:
        visualize_boxes_fn = functools.partial(_visualize_boxes_and_keypoints, category_index=category_index, **visualization_keyword_args)
        elems = [true_shapes, original_shapes, images, boxes, classes, scores, keypoints]
    elif instance_masks is not None and keypoints is not None:
        visualize_boxes_fn = functools.partial(_visualize_boxes_and_masks_and_keypoints, category_index=category_index, **visualization_keyword_args)
        elems = [true_shapes, original_shapes, images, boxes, classes, scores, instance_masks, keypoints]
    else:
        visualize_boxes_fn = functools.partial(_visualize_boxes, category_index=category_index, **visualization_keyword_args)
        elems = [true_shapes, original_shapes, images, boxes, classes, scores]

    def draw_boxes(image_and_detections):
        if False:
            print('Hello World!')
        'Draws boxes on image.'
        true_shape = image_and_detections[0]
        original_shape = image_and_detections[1]
        if true_image_shape is not None:
            image = shape_utils.pad_or_clip_nd(image_and_detections[2], [true_shape[0], true_shape[1], 3])
        if original_image_spatial_shape is not None:
            image_and_detections[2] = _resize_original_image(image, original_shape)
        image_with_boxes = tf.compat.v1.py_func(visualize_boxes_fn, image_and_detections[2:], tf.uint8)
        return image_with_boxes
    images = tf.map_fn(draw_boxes, elems, dtype=tf.uint8, back_prop=False)
    return images

def draw_keypoints_on_image_array(image, keypoints, color='red', radius=2, use_normalized_coordinates=True):
    if False:
        print('Hello World!')
    'Draws keypoints on an image (numpy array).\n\n  Args:\n    image: a numpy array with shape [height, width, 3].\n    keypoints: a numpy array with shape [num_keypoints, 2].\n    color: color to draw the keypoints with. Default is red.\n    radius: keypoint radius. Default value is 2.\n    use_normalized_coordinates: if True (default), treat keypoint values as\n      relative to the image.  Otherwise treat them as absolute.\n  '
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_keypoints_on_image(image_pil, keypoints, color, radius, use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))

def draw_keypoints_on_image(image, keypoints, color='red', radius=2, use_normalized_coordinates=True):
    if False:
        print('Hello World!')
    'Draws keypoints on an image.\n\n  Args:\n    image: a PIL.Image object.\n    keypoints: a numpy array with shape [num_keypoints, 2].\n    color: color to draw the keypoints with. Default is red.\n    radius: keypoint radius. Default value is 2.\n    use_normalized_coordinates: if True (default), treat keypoint values as\n      relative to the image.  Otherwise treat them as absolute.\n  '
    draw = ImageDraw.Draw(image)
    (im_width, im_height) = image.size
    keypoints_x = [k[1] for k in keypoints]
    keypoints_y = [k[0] for k in keypoints]
    if use_normalized_coordinates:
        keypoints_x = tuple([im_width * x for x in keypoints_x])
        keypoints_y = tuple([im_height * y for y in keypoints_y])
    for (keypoint_x, keypoint_y) in zip(keypoints_x, keypoints_y):
        draw.ellipse([(keypoint_x - radius, keypoint_y - radius), (keypoint_x + radius, keypoint_y + radius)], outline=color, fill=color)

def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
    if False:
        i = 10
        return i + 15
    'Draws mask on an image.\n\n  Args:\n    image: uint8 numpy array with shape (img_height, img_height, 3)\n    mask: a uint8 numpy array of shape (img_height, img_height) with\n      values between either 0 or 1.\n    color: color to draw the keypoints with. Default is red.\n    alpha: transparency value between 0 and 1. (default: 0.4)\n\n  Raises:\n    ValueError: On incorrect data type for image or masks.\n  '
    if image.dtype != np.uint8:
        raise ValueError('`image` not of type np.uint8')
    if mask.dtype != np.uint8:
        raise ValueError('`mask` not of type np.uint8')
    if np.any(np.logical_and(mask != 1, mask != 0)):
        raise ValueError('`mask` elements should be in [0, 1]')
    if image.shape[:2] != mask.shape:
        raise ValueError('The image has spatial dimensions %s but the mask has dimensions %s' % (image.shape[:2], mask.shape))
    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(image)
    solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
    pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert('L')
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    np.copyto(image, np.array(pil_image.convert('RGB')))

def visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index, instance_masks=None, instance_boundaries=None, keypoints=None, use_normalized_coordinates=False, max_boxes_to_draw=20, min_score_thresh=0.5, agnostic_mode=False, line_thickness=4, groundtruth_box_visualization_color='black', skip_scores=False, skip_labels=False):
    if False:
        return 10
    'Overlay labeled boxes on an image with formatted scores and label names.\n\n  This function groups boxes that correspond to the same location\n  and creates a display string for each detection and overlays these\n  on the image. Note that this function modifies the image in place, and returns\n  that same image.\n\n  Args:\n    image: uint8 numpy array with shape (img_height, img_width, 3)\n    boxes: a numpy array of shape [N, 4]\n    classes: a numpy array of shape [N]. Note that class indices are 1-based,\n      and match the keys in the label map.\n    scores: a numpy array of shape [N] or None.  If scores=None, then\n      this function assumes that the boxes to be plotted are groundtruth\n      boxes and plot all boxes as black with no classes or scores.\n    category_index: a dict containing category dictionaries (each holding\n      category index `id` and category name `name`) keyed by category indices.\n    instance_masks: a numpy array of shape [N, image_height, image_width] with\n      values ranging between 0 and 1, can be None.\n    instance_boundaries: a numpy array of shape [N, image_height, image_width]\n      with values ranging between 0 and 1, can be None.\n    keypoints: a numpy array of shape [N, num_keypoints, 2], can\n      be None\n    use_normalized_coordinates: whether boxes is to be interpreted as\n      normalized coordinates or not.\n    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw\n      all boxes.\n    min_score_thresh: minimum score threshold for a box to be visualized\n    agnostic_mode: boolean (default: False) controlling whether to evaluate in\n      class-agnostic mode or not.  This mode will display scores but ignore\n      classes.\n    line_thickness: integer (default: 4) controlling line width of the boxes.\n    groundtruth_box_visualization_color: box color for visualizing groundtruth\n      boxes\n    skip_scores: whether to skip score when drawing a single detection\n    skip_labels: whether to skip label when drawing a single detection\n\n  Returns:\n    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.\n  '
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_instance_boundaries_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ''
                if not skip_labels:
                    if not agnostic_mode:
                        if classes[i] in category_index.keys():
                            class_name = category_index[classes[i]]['name']
                        else:
                            class_name = 'N/A'
                        display_str = str(class_name)
                if not skip_scores:
                    if not display_str:
                        display_str = '{}%'.format(int(100 * scores[i]))
                    else:
                        display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                else:
                    box_to_color_map[box] = STANDARD_COLORS[classes[i] % len(STANDARD_COLORS)]
    for (box, color) in box_to_color_map.items():
        (ymin, xmin, ymax, xmax) = box
        if instance_masks is not None:
            draw_mask_on_image_array(image, box_to_instance_masks_map[box], color=color)
        if instance_boundaries is not None:
            draw_mask_on_image_array(image, box_to_instance_boundaries_map[box], color='red', alpha=1.0)
        draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax, color=color, thickness=line_thickness, display_str_list=box_to_display_str_map[box], use_normalized_coordinates=use_normalized_coordinates)
        if keypoints is not None:
            draw_keypoints_on_image_array(image, box_to_keypoints_map[box], color=color, radius=line_thickness / 2, use_normalized_coordinates=use_normalized_coordinates)
    return image

def add_cdf_image_summary(values, name):
    if False:
        while True:
            i = 10
    'Adds a tf.summary.image for a CDF plot of the values.\n\n  Normalizes `values` such that they sum to 1, plots the cumulative distribution\n  function and creates a tf image summary.\n\n  Args:\n    values: a 1-D float32 tensor containing the values.\n    name: name for the image summary.\n  '

    def cdf_plot(values):
        if False:
            for i in range(10):
                print('nop')
        'Numpy function to plot CDF.'
        normalized_values = values / np.sum(values)
        sorted_values = np.sort(normalized_values)
        cumulative_values = np.cumsum(sorted_values)
        fraction_of_examples = np.arange(cumulative_values.size, dtype=np.float32) / cumulative_values.size
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot('111')
        ax.plot(fraction_of_examples, cumulative_values)
        ax.set_ylabel('cumulative normalized values')
        ax.set_xlabel('fraction of examples')
        fig.canvas.draw()
        (width, height) = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(1, int(height), int(width), 3)
        return image
    cdf_plot = tf.compat.v1.py_func(cdf_plot, [values], tf.uint8)
    tf.compat.v1.summary.image(name, cdf_plot)

def add_hist_image_summary(values, bins, name):
    if False:
        i = 10
        return i + 15
    'Adds a tf.summary.image for a histogram plot of the values.\n\n  Plots the histogram of values and creates a tf image summary.\n\n  Args:\n    values: a 1-D float32 tensor containing the values.\n    bins: bin edges which will be directly passed to np.histogram.\n    name: name for the image summary.\n  '

    def hist_plot(values, bins):
        if False:
            i = 10
            return i + 15
        'Numpy function to plot hist.'
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot('111')
        (y, x) = np.histogram(values, bins=bins)
        ax.plot(x[:-1], y)
        ax.set_ylabel('count')
        ax.set_xlabel('value')
        fig.canvas.draw()
        (width, height) = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(1, int(height), int(width), 3)
        return image
    hist_plot = tf.compat.v1.py_func(hist_plot, [values, bins], tf.uint8)
    tf.compat.v1.summary.image(name, hist_plot)