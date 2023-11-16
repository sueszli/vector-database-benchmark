"""Functions to make unit testing easier."""
import StringIO
import numpy as np
from PIL import Image as PILImage
import tensorflow as tf

def create_random_image(image_format, shape):
    if False:
        for i in range(10):
            print('nop')
    'Creates an image with random values.\n\n  Args:\n    image_format: An image format (PNG or JPEG).\n    shape: A tuple with image shape (including channels).\n\n  Returns:\n    A tuple (<numpy ndarray>, <a string with encoded image>)\n  '
    image = np.random.randint(low=0, high=255, size=shape, dtype='uint8')
    io = StringIO.StringIO()
    image_pil = PILImage.fromarray(image)
    image_pil.save(io, image_format, subsampling=0, quality=100)
    return (image, io.getvalue())

def create_serialized_example(name_to_values):
    if False:
        while True:
            i = 10
    'Creates a tf.Example proto using a dictionary.\n\n  It automatically detects type of values and define a corresponding feature.\n\n  Args:\n    name_to_values: A dictionary.\n\n  Returns:\n    tf.Example proto.\n  '
    example = tf.train.Example()
    for (name, values) in name_to_values.items():
        feature = example.features.feature[name]
        if isinstance(values[0], str):
            add = feature.bytes_list.value.extend
        elif isinstance(values[0], float):
            add = feature.float32_list.value.extend
        elif isinstance(values[0], int):
            add = feature.int64_list.value.extend
        else:
            raise AssertionError('Unsupported type: %s' % type(values[0]))
        add(values)
    return example.SerializeToString()