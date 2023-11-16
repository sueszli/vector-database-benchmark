"""Configuration to read FSNS dataset https://goo.gl/3Ldm8v."""
import os
import re
import tensorflow as tf
from tensorflow.contrib import slim
import logging
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(__file__), 'data/fsns')
DEFAULT_CONFIG = {'name': 'FSNS', 'splits': {'train': {'size': 1044868, 'pattern': 'train/train*'}, 'test': {'size': 20404, 'pattern': 'test/test*'}, 'validation': {'size': 16150, 'pattern': 'validation/validation*'}}, 'charset_filename': 'charset_size=134.txt', 'image_shape': (150, 600, 3), 'num_of_views': 4, 'max_sequence_length': 37, 'null_code': 133, 'items_to_descriptions': {'image': 'A [150 x 600 x 3] color image.', 'label': 'Characters codes.', 'text': 'A unicode string.', 'length': 'A length of the encoded text.', 'num_of_views': 'A number of different views stored within the image.'}}

def read_charset(filename, null_character=u'░'):
    if False:
        return 10
    "Reads a charset definition from a tab separated text file.\n\n  charset file has to have format compatible with the FSNS dataset.\n\n  Args:\n    filename: a path to the charset file.\n    null_character: a unicode character used to replace '<null>' character. the\n      default value is a light shade block '░'.\n\n  Returns:\n    a dictionary with keys equal to character codes and values - unicode\n    characters.\n  "
    pattern = re.compile('(\\d+)\\t(.+)')
    charset = {}
    with tf.gfile.GFile(filename) as f:
        for (i, line) in enumerate(f):
            m = pattern.match(line)
            if m is None:
                logging.warning('incorrect charset file. line #%d: %s', i, line)
                continue
            code = int(m.group(1))
            char = m.group(2).decode('utf-8')
            if char == '<nul>':
                char = null_character
            charset[code] = char
    return charset

class _NumOfViewsHandler(slim.tfexample_decoder.ItemHandler):
    """Convenience handler to determine number of views stored in an image."""

    def __init__(self, width_key, original_width_key, num_of_views):
        if False:
            while True:
                i = 10
        super(_NumOfViewsHandler, self).__init__([width_key, original_width_key])
        self._width_key = width_key
        self._original_width_key = original_width_key
        self._num_of_views = num_of_views

    def tensors_to_item(self, keys_to_tensors):
        if False:
            while True:
                i = 10
        return tf.to_int64(self._num_of_views * keys_to_tensors[self._original_width_key] / keys_to_tensors[self._width_key])

def get_split(split_name, dataset_dir=None, config=None):
    if False:
        print('Hello World!')
    'Returns a dataset tuple for FSNS dataset.\n\n  Args:\n    split_name: A train/test split name.\n    dataset_dir: The base directory of the dataset sources, by default it uses\n      a predefined CNS path (see DEFAULT_DATASET_DIR).\n    config: A dictionary with dataset configuration. If None - will use the\n      DEFAULT_CONFIG.\n\n  Returns:\n    A `Dataset` namedtuple.\n\n  Raises:\n    ValueError: if `split_name` is not a valid train/test split.\n  '
    if not dataset_dir:
        dataset_dir = DEFAULT_DATASET_DIR
    if not config:
        config = DEFAULT_CONFIG
    if split_name not in config['splits']:
        raise ValueError('split name %s was not recognized.' % split_name)
    logging.info('Using %s dataset split_name=%s dataset_dir=%s', config['name'], split_name, dataset_dir)
    zero = tf.zeros([1], dtype=tf.int64)
    keys_to_features = {'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''), 'image/format': tf.FixedLenFeature((), tf.string, default_value='png'), 'image/width': tf.FixedLenFeature([1], tf.int64, default_value=zero), 'image/orig_width': tf.FixedLenFeature([1], tf.int64, default_value=zero), 'image/class': tf.FixedLenFeature([config['max_sequence_length']], tf.int64), 'image/unpadded_class': tf.VarLenFeature(tf.int64), 'image/text': tf.FixedLenFeature([1], tf.string, default_value='')}
    items_to_handlers = {'image': slim.tfexample_decoder.Image(shape=config['image_shape'], image_key='image/encoded', format_key='image/format'), 'label': slim.tfexample_decoder.Tensor(tensor_key='image/class'), 'text': slim.tfexample_decoder.Tensor(tensor_key='image/text'), 'num_of_views': _NumOfViewsHandler(width_key='image/width', original_width_key='image/orig_width', num_of_views=config['num_of_views'])}
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    charset_file = os.path.join(dataset_dir, config['charset_filename'])
    charset = read_charset(charset_file)
    file_pattern = os.path.join(dataset_dir, config['splits'][split_name]['pattern'])
    return slim.dataset.Dataset(data_sources=file_pattern, reader=tf.TFRecordReader, decoder=decoder, num_samples=config['splits'][split_name]['size'], items_to_descriptions=config['items_to_descriptions'], charset=charset, num_char_classes=len(charset), num_of_views=config['num_of_views'], max_sequence_length=config['max_sequence_length'], null_code=config['null_code'])