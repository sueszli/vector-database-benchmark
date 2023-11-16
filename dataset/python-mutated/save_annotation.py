"""Saves an annotation as one png image.

This script saves an annotation as one png image, and has the option to add
colormap to the png image for better visualization.
"""
import numpy as np
import PIL.Image as img
import tensorflow as tf
from deeplab.utils import get_dataset_colormap

def save_annotation(label, save_dir, filename, add_colormap=True, normalize_to_unit_values=False, scale_values=False, colormap_type=get_dataset_colormap.get_pascal_name()):
    if False:
        return 10
    'Saves the given label to image on disk.\n\n  Args:\n    label: The numpy array to be saved. The data will be converted\n      to uint8 and saved as png image.\n    save_dir: String, the directory to which the results will be saved.\n    filename: String, the image filename.\n    add_colormap: Boolean, add color map to the label or not.\n    normalize_to_unit_values: Boolean, normalize the input values to [0, 1].\n    scale_values: Boolean, scale the input values to [0, 255] for visualization.\n    colormap_type: String, colormap type for visualization.\n  '
    if add_colormap:
        colored_label = get_dataset_colormap.label_to_color_image(label, colormap_type)
    else:
        colored_label = label
        if normalize_to_unit_values:
            min_value = np.amin(colored_label)
            max_value = np.amax(colored_label)
            range_value = max_value - min_value
            if range_value != 0:
                colored_label = (colored_label - min_value) / range_value
        if scale_values:
            colored_label = 255.0 * colored_label
    pil_image = img.fromarray(colored_label.astype(dtype=np.uint8))
    with tf.gfile.Open('%s/%s.png' % (save_dir, filename), mode='w') as f:
        pil_image.save(f, 'PNG')