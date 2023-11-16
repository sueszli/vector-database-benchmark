"""This tool converts an image file into a CSV data array.

Loads JPEG or PNG input files, resizes them, optionally converts to grayscale,
and writes out as comma-separated variables, one image per row. Designed to
help create test inputs that can be shared between Python and on-device test
cases to investigate accuracy issues.

"""
import sys
from absl import app
from absl import flags
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
FLAGS = flags.FLAGS
flags.DEFINE_multi_string('image_file_names', None, 'List of paths to the input images.')
flags.DEFINE_integer('width', 96, 'Width to scale images to.')
flags.DEFINE_integer('height', 96, 'Height to scale images to.')
flags.DEFINE_boolean('want_grayscale', False, 'Whether to convert the image to monochrome.')

def get_image(width, height, want_grayscale, filepath):
    if False:
        i = 10
        return i + 15
    'Returns an image loaded into an np.ndarray with dims [height, width, (3 or 1)].\n\n  Args:\n    width: Width to rescale the image to.\n    height: Height to rescale the image to.\n    want_grayscale: Whether the result should be converted to grayscale.\n    filepath: Path of the image file..\n\n  Returns:\n    np.ndarray of shape (height, width, channels) where channels is 1 if\n      want_grayscale is true, otherwise 3.\n  '
    with ops.Graph().as_default():
        with session.Session():
            file_data = io_ops.read_file(filepath)
            channels = 1 if want_grayscale else 3
            image_tensor = image_ops.decode_image(file_data, channels=channels).eval()
            resized_tensor = image_ops.resize_images_v2(image_tensor, (height, width)).eval()
    return resized_tensor

def array_to_int_csv(array_data):
    if False:
        while True:
            i = 10
    'Converts all elements in a numerical array to a comma-separated string.\n\n  Args:\n    array_data: Numerical array to convert.\n\n  Returns:\n    String containing array values as integers, separated by commas.\n  '
    flattened_array = array_data.flatten()
    array_as_strings = [item.astype(int).astype(str) for item in flattened_array]
    return ','.join(array_as_strings)

def main(_):
    if False:
        i = 10
        return i + 15
    for image_file_name in FLAGS.image_file_names:
        try:
            image_data = get_image(FLAGS.width, FLAGS.height, FLAGS.want_grayscale, image_file_name)
            print(array_to_int_csv(image_data))
        except NotFoundError:
            sys.stderr.write('Image file not found at {0}\n'.format(image_file_name))
            sys.exit(1)
if __name__ == '__main__':
    app.run(main)