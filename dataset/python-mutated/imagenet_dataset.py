import math
import os
import random
from typing import TYPE_CHECKING, Iterable, List, Mapping, Union, Tuple
from bigdl.dllib.utils.log4Error import invalidInputError
if TYPE_CHECKING:
    from numpy import ndarray
    import tensorflow as tf
    from tensorflow.core.example.example_pb2 import Example
    from tensorflow.core.example.feature_pb2 import Feature
TRAINING_SHARDS = 1024
VALIDATION_SHARDS = 128
TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'validation'
VALIDATION_LABELS = 'synset_labels.txt'

def convert_imagenet_to_tf_records(raw_data_dir: str, output_dir: str) -> None:
    if False:
        while True:
            i = 10
    'Converts the Imagenet dataset into TF-Record dumps.'
    import tensorflow as tf
    random.seed(0)

    def make_shuffle_idx(n):
        if False:
            print('Hello World!')
        order = list(range(n))
        random.shuffle(order)
        return order
    training_files = tf.gfile.Glob(os.path.join(raw_data_dir, TRAINING_DIRECTORY, '*', '*.JPEG'))
    training_synsets = [os.path.basename(os.path.dirname(f)) for f in training_files]
    training_synsets = list(map(lambda x: bytes(x, 'utf-8'), training_synsets))
    training_shuffle_idx = make_shuffle_idx(len(training_files))
    training_files = [training_files[i] for i in training_shuffle_idx]
    training_synsets = [training_synsets[i] for i in training_shuffle_idx]
    validation_files = sorted(tf.gfile.Glob(os.path.join(raw_data_dir, VALIDATION_DIRECTORY, '*.JPEG')))
    validation_synsets = tf.gfile.FastGFile(os.path.join(raw_data_dir, VALIDATION_LABELS), 'rb').read().splitlines()
    labels = {v: k + 1 for (k, v) in enumerate(sorted(set(validation_synsets + training_synsets)))}
    print('Processing the training data.')
    _process_dataset(training_files, training_synsets, labels, os.path.join(output_dir, TRAINING_DIRECTORY), TRAINING_DIRECTORY, TRAINING_SHARDS)
    print('Processing the validation data.')
    _process_dataset(validation_files, validation_synsets, labels, os.path.join(output_dir, VALIDATION_DIRECTORY), VALIDATION_DIRECTORY, VALIDATION_SHARDS)

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self) -> None:
        if False:
            return 10
        import tensorflow as tf
        self._sess = tf.Session()
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data: bytes) -> 'tf.Tensor':
        if False:
            print('Hello World!')
        'Converts a PNG compressed image to a JPEG Tensor.'
        return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data: bytes) -> 'tf.Tensor':
        if False:
            i = 10
            return i + 15
        'Converts a CMYK image to RGB Tensor.'
        return self._sess.run(self._cmyk_to_rgb, feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data: bytes) -> 'tf.Tensor':
        if False:
            print('Hello World!')
        'Decodes a JPEG image.'
        image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        invalidInputError(len(image.shape) == 3, 'expect image dim is 3')
        invalidInputError(image.shape[2] == 3, 'expect image.shape[2] == 3')
        return image

def _process_dataset(filenames: List[str], synsets: List[str], labels: Mapping[str, int], output_directory: str, prefix: str, num_shards: int) -> List[str]:
    if False:
        while True:
            i = 10
    'Processes and saves list of images as TFRecords.\n\n    Args:\n    filenames: iterable of strings; each string is a path to an image file.\n    synsets: iterable of strings; each string is a unique WordNet ID.\n    labels: map of string to integer; id for all synset labels.\n    output_directory: path where output files should be created.\n    prefix: string; prefix for each file.\n    num_shards: number of chunks to split the filenames into.\n\n    Returns:\n    files: list of tf-record filepaths created from processing the dataset.\n\n    '
    _check_or_create_dir(output_directory)
    chunksize = int(math.ceil(len(filenames) / num_shards))
    coder = ImageCoder()
    files = []
    for shard in range(num_shards):
        chunk_files = filenames[shard * chunksize:(shard + 1) * chunksize]
        chunk_synsets = synsets[shard * chunksize:(shard + 1) * chunksize]
        output_file = os.path.join(output_directory, '%s-%.5d-of-%.5d' % (prefix, shard, num_shards))
        _process_image_files_batch(coder, output_file, chunk_files, chunk_synsets, labels)
        print('Finished writing file: %s', output_file)
        files.append(output_file)
    return files

def _process_image(filename: str, coder: ImageCoder) -> Tuple[str, int, int]:
    if False:
        i = 10
        return i + 15
    "Processes a single image file.\n\n    Args:\n    filename: string, path to an image file e.g., '/path/to/example.JPG'.\n    coder: instance of ImageCoder to provide TensorFlow image coding utils.\n\n    Returns:\n    image_buffer: string, JPEG encoding of RGB image.\n    height: integer, image height in pixels.\n    width: integer, image width in pixels.\n\n    "
    import tensorflow as tf
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()
    if _is_png(filename):
        print('Converting PNG to JPEG for %s', filename)
        image_data = coder.png_to_jpeg(image_data)
    elif _is_cmyk(filename):
        print('Converting CMYK to RGB for %s', filename)
        image_data = coder.cmyk_to_rgb(image_data)
    image = coder.decode_jpeg(image_data)
    invalidInputError(len(image.shape) == 3, 'expect image dim is 3')
    height = image.shape[0]
    width = image.shape[1]
    invalidInputError(image.shape[2] == 3, 'expect image.shape[2] is 3')
    return (image_data, height, width)

def _process_image_files_batch(coder: ImageCoder, output_file: str, filenames: List[str], synsets: List[str], labels: Mapping[str, int]) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Processes and saves a list of images as TFRecords.\n\n    Args:\n    coder: instance of ImageCoder to provide TensorFlow image coding utils.\n    output_file: string, unique identifier specifying the data set.\n    filenames: list of strings; each string is a path to an image file.\n    synsets: list of strings; each string is a unique WordNet ID.\n    labels: map of string to integer; id for all synset labels.\n\n    '
    import tensorflow as tf
    writer = tf.python_io.TFRecordWriter(output_file)
    for (filename, synset) in zip(filenames, synsets):
        (image_buffer, height, width) = _process_image(filename, coder)
        label = labels[synset]
        example = _convert_to_example(filename, image_buffer, label, synset, height, width)
        writer.write(example.SerializeToString())
    writer.close()

def _check_or_create_dir(directory: str) -> None:
    if False:
        while True:
            i = 10
    import tensorflow as tf
    'Checks if directory exists otherwise creates it.'
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)

def _int64_feature(value: Union[int, Iterable[int]]) -> 'Feature':
    if False:
        return 10
    'Inserts int64 features into Example proto.'
    import tensorflow as tf
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value: Union[bytes, str]) -> 'Feature':
    if False:
        for i in range(10):
            print('nop')
    'Inserts bytes features into Example proto.'
    import tensorflow as tf
    if isinstance(value, str):
        value = bytes(value, 'utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(filename: str, image_buffer: str, label: int, synset: str, height: int, width: int) -> 'Example':
    if False:
        i = 10
        return i + 15
    "\n    Builds an Example proto for an ImageNet example.\n\n    Args:\n    filename: string, path to an image file, e.g., '/path/to/example.JPG'\n    image_buffer: string, JPEG encoding of RGB image\n    label: integer, identifier for the ground truth for the network\n    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'\n    height: integer, image height in pixels\n    width: integer, image width in pixels\n    Returns:\n    Example proto\n\n    "
    import tensorflow as tf
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={'image/height': _int64_feature(height), 'image/width': _int64_feature(width), 'image/colorspace': _bytes_feature(colorspace), 'image/channels': _int64_feature(channels), 'image/class/label': _int64_feature(label), 'image/class/synset': _bytes_feature(synset), 'image/format': _bytes_feature(image_format), 'image/filename': _bytes_feature(os.path.basename(filename)), 'image/encoded': _bytes_feature(image_buffer)}))
    return example

def _is_png(filename: str) -> bool:
    if False:
        while True:
            i = 10
    '\n    Determines if a file contains a PNG format image.\n\n    Args:\n    filename: string, path of the image file.\n\n    Returns:\n    boolean indicating if the image is a PNG.\n\n    '
    return 'n02105855_2933.JPEG' in filename

def _is_cmyk(filename: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Determines if file contains a CMYK JPEG format image.\n\n    Args:\n    filename: string, path of the image file.\n\n    Returns:\n    boolean indicating if the image is a JPEG encoded with CMYK color space.\n\n    '
    blacklist = set(['n01739381_1309.JPEG', 'n02077923_14822.JPEG', 'n02447366_23489.JPEG', 'n02492035_15739.JPEG', 'n02747177_10752.JPEG', 'n03018349_4028.JPEG', 'n03062245_4620.JPEG', 'n03347037_9675.JPEG', 'n03467068_12171.JPEG', 'n03529860_11437.JPEG', 'n03544143_17228.JPEG', 'n03633091_5218.JPEG', 'n03710637_5125.JPEG', 'n03961711_5286.JPEG', 'n04033995_2932.JPEG', 'n04258138_17003.JPEG', 'n04264628_27969.JPEG', 'n04336792_7448.JPEG', 'n04371774_5854.JPEG', 'n04596742_4225.JPEG', 'n07583066_647.JPEG', 'n13037406_4650.JPEG'])
    return os.path.basename(filename) in blacklist