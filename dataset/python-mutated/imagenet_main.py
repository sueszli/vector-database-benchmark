"""Runs a ResNet model on the ImageNet dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import app as absl_app
from absl import flags
import tensorflow as tf
from official.r1.resnet import imagenet_preprocessing
from official.r1.resnet import resnet_model
from official.r1.resnet import resnet_run_loop
from official.utils.flags import core as flags_core
from official.utils.logs import logger
DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 1001
NUM_IMAGES = {'train': 1281167, 'validation': 50000}
_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 10000
DATASET_NAME = 'ImageNet'

def get_filenames(is_training, data_dir):
    if False:
        return 10
    'Return filenames for dataset.'
    if is_training:
        return [os.path.join(data_dir, 'train-%05d-of-01024' % i) for i in range(_NUM_TRAIN_FILES)]
    else:
        return [os.path.join(data_dir, 'validation-%05d-of-00128' % i) for i in range(128)]

def _parse_example_proto(example_serialized):
    if False:
        return 10
    "Parses an Example proto containing a training example of an image.\n\n  The output of the build_image_data.py image preprocessing script is a dataset\n  containing serialized Example protocol buffers. Each Example proto contains\n  the following fields (values are included as examples):\n\n    image/height: 462\n    image/width: 581\n    image/colorspace: 'RGB'\n    image/channels: 3\n    image/class/label: 615\n    image/class/synset: 'n03623198'\n    image/class/text: 'knee pad'\n    image/object/bbox/xmin: 0.1\n    image/object/bbox/xmax: 0.9\n    image/object/bbox/ymin: 0.2\n    image/object/bbox/ymax: 0.6\n    image/object/bbox/label: 615\n    image/format: 'JPEG'\n    image/filename: 'ILSVRC2012_val_00041207.JPEG'\n    image/encoded: <JPEG encoded string>\n\n  Args:\n    example_serialized: scalar Tensor tf.string containing a serialized\n      Example protocol buffer.\n\n  Returns:\n    image_buffer: Tensor tf.string containing the contents of a JPEG file.\n    label: Tensor tf.int32 containing the label.\n    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]\n      where each coordinate is [0, 1) and the coordinates are arranged as\n      [ymin, xmin, ymax, xmax].\n  "
    feature_map = {'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''), 'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1), 'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string, default_value='')}
    sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
    feature_map.update({k: sparse_float32 for k in ['image/object/bbox/xmin', 'image/object/bbox/ymin', 'image/object/bbox/xmax', 'image/object/bbox/ymax']})
    features = tf.io.parse_single_example(serialized=example_serialized, features=feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)
    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(a=bbox, perm=[0, 2, 1])
    return (features['image/encoded'], label, bbox)

def parse_record(raw_record, is_training, dtype):
    if False:
        for i in range(10):
            print('nop')
    'Parses a record containing a training example of an image.\n\n  The input record is parsed into a label and image, and the image is passed\n  through preprocessing steps (cropping, flipping, and so on).\n\n  Args:\n    raw_record: scalar Tensor tf.string containing a serialized\n      Example protocol buffer.\n    is_training: A boolean denoting whether the input is for training.\n    dtype: data type to use for images/features.\n\n  Returns:\n    Tuple with processed image tensor and one-hot-encoded label tensor.\n  '
    (image_buffer, label, bbox) = _parse_example_proto(raw_record)
    image = imagenet_preprocessing.preprocess_image(image_buffer=image_buffer, bbox=bbox, output_height=DEFAULT_IMAGE_SIZE, output_width=DEFAULT_IMAGE_SIZE, num_channels=NUM_CHANNELS, is_training=is_training)
    image = tf.cast(image, dtype)
    return (image, label)

def input_fn(is_training, data_dir, batch_size, num_epochs=1, dtype=tf.float32, datasets_num_private_threads=None, parse_record_fn=parse_record, input_context=None, drop_remainder=False, tf_data_experimental_slack=False):
    if False:
        print('Hello World!')
    "Input function which provides batches for train or eval.\n\n  Args:\n    is_training: A boolean denoting whether the input is for training.\n    data_dir: The directory containing the input data.\n    batch_size: The number of samples per batch.\n    num_epochs: The number of epochs to repeat the dataset.\n    dtype: Data type to use for images/features\n    datasets_num_private_threads: Number of private threads for tf.data.\n    parse_record_fn: Function to use for parsing the records.\n    input_context: A `tf.distribute.InputContext` object passed in by\n      `tf.distribute.Strategy`.\n    drop_remainder: A boolean indicates whether to drop the remainder of the\n      batches. If True, the batch dimension will be static.\n    tf_data_experimental_slack: Whether to enable tf.data's\n      `experimental_slack` option.\n\n  Returns:\n    A dataset that can be used for iteration.\n  "
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if input_context:
        tf.compat.v1.logging.info('Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (input_context.input_pipeline_id, input_context.num_input_pipelines))
        dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    if is_training:
        dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=10, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return resnet_run_loop.process_record_dataset(dataset=dataset, is_training=is_training, batch_size=batch_size, shuffle_buffer=_SHUFFLE_BUFFER, parse_record_fn=parse_record_fn, num_epochs=num_epochs, dtype=dtype, datasets_num_private_threads=datasets_num_private_threads, drop_remainder=drop_remainder, tf_data_experimental_slack=tf_data_experimental_slack)

def get_synth_input_fn(dtype):
    if False:
        return 10
    return resnet_run_loop.get_synth_input_fn(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, NUM_CHANNELS, NUM_CLASSES, dtype=dtype)

class ImagenetModel(resnet_model.Model):
    """Model class with appropriate defaults for Imagenet data."""

    def __init__(self, resnet_size, data_format=None, num_classes=NUM_CLASSES, resnet_version=resnet_model.DEFAULT_VERSION, dtype=resnet_model.DEFAULT_DTYPE):
        if False:
            for i in range(10):
                print('nop')
        "These are the parameters that work for Imagenet data.\n\n    Args:\n      resnet_size: The number of convolutional layers needed in the model.\n      data_format: Either 'channels_first' or 'channels_last', specifying which\n        data format to use when setting up the model.\n      num_classes: The number of output classes needed from the model. This\n        enables users to extend the same model to their own datasets.\n      resnet_version: Integer representing which version of the ResNet network\n        to use. See README for details. Valid values: [1, 2]\n      dtype: The TensorFlow dtype to use for calculations.\n    "
        if resnet_size < 50:
            bottleneck = False
        else:
            bottleneck = True
        super(ImagenetModel, self).__init__(resnet_size=resnet_size, bottleneck=bottleneck, num_classes=num_classes, num_filters=64, kernel_size=7, conv_stride=2, first_pool_size=3, first_pool_stride=2, block_sizes=_get_block_sizes(resnet_size), block_strides=[1, 2, 2, 2], resnet_version=resnet_version, data_format=data_format, dtype=dtype)

def _get_block_sizes(resnet_size):
    if False:
        return 10
    'Retrieve the size of each block_layer in the ResNet model.\n\n  The number of block layers used for the Resnet model varies according\n  to the size of the model. This helper grabs the layer set we want, throwing\n  an error if a non-standard size has been selected.\n\n  Args:\n    resnet_size: The number of convolutional layers needed in the model.\n\n  Returns:\n    A list of block sizes to use in building the model.\n\n  Raises:\n    KeyError: if invalid resnet_size is received.\n  '
    choices = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
    try:
        return choices[resnet_size]
    except KeyError:
        err = 'Could not find layers for selected Resnet size.\nSize received: {}; sizes allowed: {}.'.format(resnet_size, choices.keys())
        raise ValueError(err)

def imagenet_model_fn(features, labels, mode, params):
    if False:
        while True:
            i = 10
    'Our model_fn for ResNet to be used with our Estimator.'
    if params['fine_tune']:
        warmup = False
        base_lr = 0.1
    else:
        warmup = True
        base_lr = 0.128
    learning_rate_fn = resnet_run_loop.learning_rate_with_decay(batch_size=params['batch_size'] * params.get('num_workers', 1), batch_denom=256, num_images=NUM_IMAGES['train'], boundary_epochs=[30, 60, 80, 90], decay_rates=[1, 0.1, 0.01, 0.001, 0.0001], warmup=warmup, base_lr=base_lr)
    return resnet_run_loop.resnet_model_fn(features=features, labels=labels, mode=mode, model_class=ImagenetModel, resnet_size=params['resnet_size'], weight_decay=flags.FLAGS.weight_decay, learning_rate_fn=learning_rate_fn, momentum=0.9, data_format=params['data_format'], resnet_version=params['resnet_version'], loss_scale=params['loss_scale'], loss_filter_fn=None, dtype=params['dtype'], fine_tune=params['fine_tune'], label_smoothing=flags.FLAGS.label_smoothing)

def define_imagenet_flags():
    if False:
        for i in range(10):
            print('nop')
    resnet_run_loop.define_resnet_flags(resnet_size_choices=['18', '34', '50', '101', '152', '200'], dynamic_loss_scale=True, fp16_implementation=True)
    flags.adopt_module_key_flags(resnet_run_loop)
    flags_core.set_defaults(train_epochs=90)

def run_imagenet(flags_obj):
    if False:
        return 10
    'Run ResNet ImageNet training and eval loop.\n\n  Args:\n    flags_obj: An object containing parsed flag values.\n\n  Returns:\n    Dict of results of the run.  Contains the keys `eval_results` and\n      `train_hooks`. `eval_results` contains accuracy (top_1) and\n      accuracy_top_5. `train_hooks` is a list the instances of hooks used during\n      training.\n  '
    input_function = flags_obj.use_synthetic_data and get_synth_input_fn(flags_core.get_tf_dtype(flags_obj)) or input_fn
    result = resnet_run_loop.resnet_main(flags_obj, imagenet_model_fn, input_function, DATASET_NAME, shape=[DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, NUM_CHANNELS])
    return result

def main(_):
    if False:
        i = 10
        return i + 15
    with logger.benchmark_context(flags.FLAGS):
        run_imagenet(flags.FLAGS)
if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    define_imagenet_flags()
    absl_app.run(main)