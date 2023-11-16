"""Simple transfer learning with image modules from TensorFlow Hub.

WARNING: This code is deprecated in favor of
https://github.com/tensorflow/hub/tree/master/tensorflow_hub/tools/make_image_classifier

This example shows how to train an image classifier based on any
TensorFlow Hub module that computes image feature vectors. By default,
it uses the feature vectors computed by Inception V3 trained on ImageNet.
For more options, search https://tfhub.dev for image feature vector modules.

The top layer receives as input a 2048-dimensional vector (assuming
Inception V3) for each image. We train a softmax layer on top of this
representation. If the softmax layer contains N labels, this corresponds
to learning N + 2048*N model parameters for the biases and weights.

Here's an example, which assumes you have a folder containing class-named
subfolders, each full of images for each label. The example folder flower_photos
should have a structure like this:

~/flower_photos/daisy/photo1.jpg
~/flower_photos/daisy/photo2.jpg
...
~/flower_photos/rose/anotherphoto77.jpg
...
~/flower_photos/sunflower/somepicture.jpg

The subfolder names are important, since they define what label is applied to
each image, but the filenames themselves don't matter. (For a working example,
download http://download.tensorflow.org/example_images/flower_photos.tgz
and run  tar xzf flower_photos.tgz  to unpack it.)

Once your images are prepared, and you have pip-installed tensorflow-hub and
a sufficiently recent version of tensorflow, you can run the training with a
command like this:

```bash
python retrain.py --image_dir ~/flower_photos
```

You can replace the image_dir argument with any folder containing subfolders of
images. The label for each image is taken from the name of the subfolder it's
in.

This produces a new model file that can be loaded and run by any TensorFlow
program, for example the tensorflow/examples/label_image sample code.

By default this script will use the highly accurate, but comparatively large and
slow Inception V3 model architecture. It's recommended that you start with this
to validate that you have gathered good training data, but if you want to deploy
on resource-limited platforms, you can try the `--tfhub_module` flag with a
Mobilenet model. For more information on Mobilenet, see
https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html

For example:

Run floating-point version of Mobilenet:

```bash
python retrain.py --image_dir ~/flower_photos \\
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/3
```

Run Mobilenet, instrumented for quantization:

```bash
python retrain.py --image_dir ~/flower_photos/ \\
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/quantops/feature_vector/3
```

These instrumented models can be converted to fully quantized mobile models via
TensorFlow Lite.

There are different Mobilenet models to choose from, with a variety of file
size and latency options.
  - The first number can be '100', '075', '050', or '025' to control the number
    of neurons (activations of hidden layers); the number of weights (and hence
    to some extent the file size and speed) shrinks with the square of that
    fraction.
  - The second number is the input image size. You can choose '224', '192',
    '160', or '128', with smaller sizes giving faster speeds.

To use with TensorBoard:

By default, this script will log summaries to /tmp/retrain_logs directory

Visualize the summaries with this command:

tensorboard --logdir /tmp/retrain_logs

To use with Tensorflow Serving, run this tool with --saved_model_dir set
to some increasingly numbered export location under the model base path, e.g.:

```bash
python retrain.py (... other args as before ...) \\
    --saved_model_dir=/tmp/saved_models/$(date +%s)/
tensorflow_model_server --port=9000 --model_name=my_image_classifier \\
    --model_base_path=/tmp/saved_models/
```
"""
from absl import logging
import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib import quantize as contrib_quantize
FLAGS = None
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxVarsPerChannel')

def create_image_lists(image_dir, testing_percentage, validation_percentage):
    if False:
        return 10
    'Builds a list of training images from the file system.\n\n  Analyzes the sub folders in the image directory, splits them into stable\n  training, testing, and validation sets, and returns a data structure\n  describing the lists of images for each label and their paths.\n\n  Args:\n    image_dir: String path to a folder containing subfolders of images.\n    testing_percentage: Integer percentage of the images to reserve for tests.\n    validation_percentage: Integer percentage of images reserved for validation.\n\n  Returns:\n    An OrderedDict containing an entry for each label subfolder, with images\n    split into training, testing, and validation sets within each label.\n    The order of items defines the class indices.\n  '
    if not tf.gfile.Exists(image_dir):
        logging.error("Image directory '" + image_dir + "' not found.")
        return None
    result = collections.OrderedDict()
    sub_dirs = sorted((x[0] for x in tf.gfile.Walk(image_dir)))
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = sorted(set((os.path.normcase(ext) for ext in ['JPEG', 'JPG', 'jpeg', 'jpg', 'png'])))
        file_list = []
        dir_name = os.path.basename(sub_dir[:-1] if sub_dir.endswith('/') else sub_dir)
        if dir_name == image_dir:
            continue
        logging.info("Looking for images in '%s'", dir_name)
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))
        if not file_list:
            logging.warning('No files found')
            continue
        if len(file_list) < 20:
            logging.warning('WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            logging.warning('WARNING: Folder %s has more than %s images. Some images will never be selected.', dir_name, MAX_NUM_IMAGES_PER_CLASS)
        label_name = re.sub('[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            hash_name = re.sub('_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = int(hash_name_hashed, 16) % (MAX_NUM_IMAGES_PER_CLASS + 1) * (100.0 / MAX_NUM_IMAGES_PER_CLASS)
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < testing_percentage + validation_percentage:
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {'dir': dir_name, 'training': training_images, 'testing': testing_images, 'validation': validation_images}
    return result

def get_image_path(image_lists, label_name, index, image_dir, category):
    if False:
        while True:
            i = 10
    'Returns a path to an image for a label at the given index.\n\n  Args:\n    image_lists: OrderedDict of training images for each label.\n    label_name: Label string we want to get an image for.\n    index: Int offset of the image we want. This will be moduloed by the\n    available number of images for the label, so it can be arbitrarily large.\n    image_dir: Root folder string of the subfolders containing the training\n    images.\n    category: Name string of set to pull images from - training, testing, or\n    validation.\n\n  Returns:\n    File system path string to an image that meets the requested parameters.\n\n  '
    if label_name not in image_lists:
        logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        logging.fatal('Label %s has no images in the category %s.', label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path

def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category, module_name):
    if False:
        i = 10
        return i + 15
    'Returns a path to a bottleneck file for a label at the given index.\n\n  Args:\n    image_lists: OrderedDict of training images for each label.\n    label_name: Label string we want to get an image for.\n    index: Integer offset of the image we want. This will be moduloed by the\n    available number of images for the label, so it can be arbitrarily large.\n    bottleneck_dir: Folder string holding cached files of bottleneck values.\n    category: Name string of set to pull images from - training, testing, or\n    validation.\n    module_name: The name of the image module being used.\n\n  Returns:\n    File system path string to an image that meets the requested parameters.\n  '
    module_name = module_name.replace('://', '~').replace('/', '~').replace(':', '~').replace('\\', '~')
    return get_image_path(image_lists, label_name, index, bottleneck_dir, category) + '_' + module_name + '.txt'

def create_module_graph(module_spec):
    if False:
        i = 10
        return i + 15
    'Creates a graph and loads Hub Module into it.\n\n  Args:\n    module_spec: the hub.ModuleSpec for the image module being used.\n\n  Returns:\n    graph: the tf.Graph that was created.\n    bottleneck_tensor: the bottleneck values output by the module.\n    resized_input_tensor: the input images, resized as expected by the module.\n    wants_quantization: a boolean, whether the module has been instrumented\n      with fake quantization ops.\n  '
    (height, width) = hub.get_expected_image_size(module_spec)
    with tf.Graph().as_default() as graph:
        resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
        m = hub.Module(module_spec)
        bottleneck_tensor = m(resized_input_tensor)
        wants_quantization = any((node.op in FAKE_QUANT_OPS for node in graph.as_graph_def().node))
    return (graph, bottleneck_tensor, resized_input_tensor, wants_quantization)

def run_bottleneck_on_image(sess, image_data, image_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor):
    if False:
        i = 10
        return i + 15
    "Runs inference on an image to extract the 'bottleneck' summary layer.\n\n  Args:\n    sess: Current active TensorFlow Session.\n    image_data: String of raw JPEG data.\n    image_data_tensor: Input data layer in the graph.\n    decoded_image_tensor: Output of initial image resizing and preprocessing.\n    resized_input_tensor: The input node of the recognition graph.\n    bottleneck_tensor: Layer before the final softmax.\n\n  Returns:\n    Numpy array of bottleneck values.\n  "
    resized_input_values = sess.run(decoded_image_tensor, {image_data_tensor: image_data})
    bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

def ensure_dir_exists(dir_name):
    if False:
        print('Hello World!')
    'Makes sure the folder exists on disk.\n\n  Args:\n    dir_name: Path string to the folder we want to create.\n  '
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor):
    if False:
        return 10
    'Create a single bottleneck file.'
    logging.debug('Creating bottleneck at %s', bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index, image_dir, category)
    if not tf.gfile.Exists(image_path):
        logging.fatal('File does not exist %s', image_path)
    image_data = tf.gfile.GFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path, str(e)))
    bottleneck_string = ','.join((str(x) for x in bottleneck_values))
    with tf.gfile.GFile(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)

def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category, bottleneck_dir, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, module_name):
    if False:
        for i in range(10):
            print('nop')
    'Retrieves or calculates bottleneck values for an image.\n\n  If a cached version of the bottleneck data exists on-disk, return that,\n  otherwise calculate the data and save it to disk for future use.\n\n  Args:\n    sess: The current active TensorFlow Session.\n    image_lists: OrderedDict of training images for each label.\n    label_name: Label string we want to get an image for.\n    index: Integer offset of the image we want. This will be modulo-ed by the\n    available number of images for the label, so it can be arbitrarily large.\n    image_dir: Root folder string of the subfolders containing the training\n    images.\n    category: Name string of which set to pull images from - training, testing,\n    or validation.\n    bottleneck_dir: Folder string holding cached files of bottleneck values.\n    jpeg_data_tensor: The tensor to feed loaded jpeg data into.\n    decoded_image_tensor: The output of decoding and resizing the image.\n    resized_input_tensor: The input node of the recognition graph.\n    bottleneck_tensor: The output tensor for the bottleneck values.\n    module_name: The name of the image module being used.\n\n  Returns:\n    Numpy array of values produced by the bottleneck layer for the image.\n  '
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category, module_name)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor)
    with tf.gfile.GFile(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        logging.warning('Invalid float found, recreating bottleneck')
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor)
        with tf.gfile.GFile(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values

def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, module_name):
    if False:
        i = 10
        return i + 15
    "Ensures all the training, testing, and validation bottlenecks are cached.\n\n  Because we're likely to read the same image multiple times (if there are no\n  distortions applied during training) it can speed things up a lot if we\n  calculate the bottleneck layer values once for each image during\n  preprocessing, and then just read those cached values repeatedly during\n  training. Here we go through all the images we've found, calculate those\n  values, and save them off.\n\n  Args:\n    sess: The current active TensorFlow Session.\n    image_lists: OrderedDict of training images for each label.\n    image_dir: Root folder string of the subfolders containing the training\n    images.\n    bottleneck_dir: Folder string holding cached files of bottleneck values.\n    jpeg_data_tensor: Input tensor for jpeg data from file.\n    decoded_image_tensor: The output of decoding and resizing the image.\n    resized_input_tensor: The input node of the recognition graph.\n    bottleneck_tensor: The penultimate output layer of the graph.\n    module_name: The name of the image module being used.\n\n  Returns:\n    Nothing.\n  "
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for (label_name, label_lists) in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for (index, unused_base_name) in enumerate(category_list):
                get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category, bottleneck_dir, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, module_name)
                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    logging.info('%s bottleneck files created.', how_many_bottlenecks)

def get_random_cached_bottlenecks(sess, image_lists, how_many, category, bottleneck_dir, image_dir, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, module_name):
    if False:
        return 10
    'Retrieves bottleneck values for cached images.\n\n  If no distortions are being applied, this function can retrieve the cached\n  bottleneck values directly from disk for images. It picks a random set of\n  images from the specified category.\n\n  Args:\n    sess: Current TensorFlow Session.\n    image_lists: OrderedDict of training images for each label.\n    how_many: If positive, a random sample of this size will be chosen.\n    If negative, all bottlenecks will be retrieved.\n    category: Name string of which set to pull from - training, testing, or\n    validation.\n    bottleneck_dir: Folder string holding cached files of bottleneck values.\n    image_dir: Root folder string of the subfolders containing the training\n    images.\n    jpeg_data_tensor: The layer to feed jpeg image data into.\n    decoded_image_tensor: The output of decoding and resizing the image.\n    resized_input_tensor: The input node of the recognition graph.\n    bottleneck_tensor: The bottleneck output layer of the CNN graph.\n    module_name: The name of the image module being used.\n\n  Returns:\n    List of bottleneck arrays, their corresponding ground truths, and the\n    relevant filenames.\n  '
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, image_dir, category, bottleneck_dir, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, module_name)
            bottlenecks.append(bottleneck)
            ground_truths.append(label_index)
            filenames.append(image_name)
    else:
        for (label_index, label_name) in enumerate(image_lists.keys()):
            for (image_index, image_name) in enumerate(image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)
                bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, image_dir, category, bottleneck_dir, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, module_name)
                bottlenecks.append(bottleneck)
                ground_truths.append(label_index)
                filenames.append(image_name)
    return (bottlenecks, ground_truths, filenames)

def get_random_distorted_bottlenecks(sess, image_lists, how_many, category, image_dir, input_jpeg_tensor, distorted_image, resized_input_tensor, bottleneck_tensor):
    if False:
        while True:
            i = 10
    "Retrieves bottleneck values for training images, after distortions.\n\n  If we're training with distortions like crops, scales, or flips, we have to\n  recalculate the full model for every image, and so we can't use cached\n  bottleneck values. Instead we find random images for the requested category,\n  run them through the distortion graph, and then the full graph to get the\n  bottleneck results for each.\n\n  Args:\n    sess: Current TensorFlow Session.\n    image_lists: OrderedDict of training images for each label.\n    how_many: The integer number of bottleneck values to return.\n    category: Name string of which set of images to fetch - training, testing,\n    or validation.\n    image_dir: Root folder string of the subfolders containing the training\n    images.\n    input_jpeg_tensor: The input layer we feed the image data to.\n    distorted_image: The output node of the distortion graph.\n    resized_input_tensor: The input node of the recognition graph.\n    bottleneck_tensor: The bottleneck output layer of the CNN graph.\n\n  Returns:\n    List of bottleneck arrays and their corresponding ground truths.\n  "
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name, image_index, image_dir, category)
        if not tf.gfile.Exists(image_path):
            logging.fatal('File does not exist %s', image_path)
        jpeg_data = tf.gfile.GFile(image_path, 'rb').read()
        distorted_image_data = sess.run(distorted_image, {input_jpeg_tensor: jpeg_data})
        bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: distorted_image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        bottlenecks.append(bottleneck_values)
        ground_truths.append(label_index)
    return (bottlenecks, ground_truths)

def should_distort_images(flip_left_right, random_crop, random_scale, random_brightness):
    if False:
        while True:
            i = 10
    'Whether any distortions are enabled, from the input flags.\n\n  Args:\n    flip_left_right: Boolean whether to randomly mirror images horizontally.\n    random_crop: Integer percentage setting the total margin used around the\n    crop box.\n    random_scale: Integer percentage of how much to vary the scale by.\n    random_brightness: Integer range to randomly multiply the pixel values by.\n\n  Returns:\n    Boolean value indicating whether any distortions should be applied.\n  '
    return flip_left_right or random_crop != 0 or random_scale != 0 or (random_brightness != 0)

def add_input_distortions(flip_left_right, random_crop, random_scale, random_brightness, module_spec):
    if False:
        for i in range(10):
            print('nop')
    "Creates the operations to apply the specified distortions.\n\n  During training it can help to improve the results if we run the images\n  through simple distortions like crops, scales, and flips. These reflect the\n  kind of variations we expect in the real world, and so can help train the\n  model to cope with natural data more effectively. Here we take the supplied\n  parameters and construct a network of operations to apply them to an image.\n\n  Cropping\n  ~~~~~~~~\n\n  Cropping is done by placing a bounding box at a random position in the full\n  image. The cropping parameter controls the size of that box relative to the\n  input image. If it's zero, then the box is the same size as the input and no\n  cropping is performed. If the value is 50%, then the crop box will be half the\n  width and height of the input. In a diagram it looks like this:\n\n  <       width         >\n  +---------------------+\n  |                     |\n  |   width - crop%     |\n  |    <      >         |\n  |    +------+         |\n  |    |      |         |\n  |    |      |         |\n  |    |      |         |\n  |    +------+         |\n  |                     |\n  |                     |\n  +---------------------+\n\n  Scaling\n  ~~~~~~~\n\n  Scaling is a lot like cropping, except that the bounding box is always\n  centered and its size varies randomly within the given range. For example if\n  the scale percentage is zero, then the bounding box is the same size as the\n  input and no scaling is applied. If it's 50%, then the bounding box will be in\n  a random range between half the width and height and full size.\n\n  Args:\n    flip_left_right: Boolean whether to randomly mirror images horizontally.\n    random_crop: Integer percentage setting the total margin used around the\n    crop box.\n    random_scale: Integer percentage of how much to vary the scale by.\n    random_brightness: Integer range to randomly multiply the pixel values by.\n    graph.\n    module_spec: The hub.ModuleSpec for the image module being used.\n\n  Returns:\n    The jpeg input layer and the distorted result tensor.\n  "
    (input_height, input_width) = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)
    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image, tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + random_crop / 100.0
    resize_scale = 1.0 + random_scale / 100.0
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(shape=[], minval=1.0, maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, input_width)
    precrop_height = tf.multiply(scale_value, input_height)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d, precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, axis=[0])
    cropped_image = tf.random_crop(precropped_image_3d, [input_height, input_width, input_depth])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - random_brightness / 100.0
    brightness_max = 1.0 + random_brightness / 100.0
    brightness_value = tf.random_uniform(shape=[], minval=brightness_min, maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
    return (jpeg_data, distort_result)

def variable_summaries(var):
    if False:
        for i in range(10):
            print('nop')
    'Attach a lot of summaries to a Tensor (for TensorBoard visualization).'
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def add_final_retrain_ops(class_count, final_tensor_name, bottleneck_tensor, quantize_layer, is_training):
    if False:
        while True:
            i = 10
    "Adds a new softmax and fully-connected layer for training and eval.\n\n  We need to retrain the top layer to identify our new classes, so this function\n  adds the right operations to the graph, along with some variables to hold the\n  weights, and then sets up all the gradients for the backward pass.\n\n  The set up for the softmax and fully-connected layers is based on:\n  https://www.tensorflow.org/tutorials/mnist/beginners/index.html\n\n  Args:\n    class_count: Integer of how many categories of things we're trying to\n        recognize.\n    final_tensor_name: Name string for the new final node that produces results.\n    bottleneck_tensor: The output of the main CNN graph.\n    quantize_layer: Boolean, specifying whether the newly added layer should be\n        instrumented for quantization with TF-Lite.\n    is_training: Boolean, specifying whether the newly add layer is for training\n        or eval.\n\n  Returns:\n    The tensors for the training and cross entropy results, and tensors for the\n    bottleneck input and ground truth input.\n  "
    (batch_size, bottleneck_tensor_size) = bottleneck_tensor.get_shape().as_list()
    assert batch_size is None, 'We want to work with arbitrary batch size.'
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(bottleneck_tensor, shape=[batch_size, bottleneck_tensor_size], name='BottleneckInputPlaceholder')
        ground_truth_input = tf.placeholder(tf.int64, [batch_size], name='GroundTruthInput')
    layer_name = 'final_retrain_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal([bottleneck_tensor_size, class_count], stddev=0.001)
            layer_weights = tf.Variable(initial_value, name='final_weights')
            variable_summaries(layer_weights)
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)
    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    if quantize_layer:
        if is_training:
            contrib_quantize.create_training_graph()
        else:
            contrib_quantize.create_eval_graph()
    tf.summary.histogram('activations', final_tensor)
    if not is_training:
        return (None, None, bottleneck_input, ground_truth_input, final_tensor)
    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels=ground_truth_input, logits=logits)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)
    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)

def add_evaluation_step(result_tensor, ground_truth_tensor):
    if False:
        return 10
    'Inserts the operations we need to evaluate the accuracy of our results.\n\n  Args:\n    result_tensor: The new final node that produces results.\n    ground_truth_tensor: The node we feed ground truth data\n    into.\n\n  Returns:\n    Tuple of (evaluation step, prediction).\n  '
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, ground_truth_tensor)
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return (evaluation_step, prediction)

def run_final_eval(train_session, module_spec, class_count, image_lists, jpeg_data_tensor, decoded_image_tensor, resized_image_tensor, bottleneck_tensor):
    if False:
        return 10
    'Runs a final evaluation on an eval graph using the test data set.\n\n  Args:\n    train_session: Session for the train graph with the tensors below.\n    module_spec: The hub.ModuleSpec for the image module being used.\n    class_count: Number of classes\n    image_lists: OrderedDict of training images for each label.\n    jpeg_data_tensor: The layer to feed jpeg image data into.\n    decoded_image_tensor: The output of decoding and resizing the image.\n    resized_image_tensor: The input node of the recognition graph.\n    bottleneck_tensor: The bottleneck output layer of the CNN graph.\n  '
    (test_bottlenecks, test_ground_truth, test_filenames) = get_random_cached_bottlenecks(train_session, image_lists, FLAGS.test_batch_size, 'testing', FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor, decoded_image_tensor, resized_image_tensor, bottleneck_tensor, FLAGS.tfhub_module)
    (eval_session, _, bottleneck_input, ground_truth_input, evaluation_step, prediction) = build_eval_session(module_spec, class_count)
    (test_accuracy, predictions) = eval_session.run([evaluation_step, prediction], feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
    logging.info('Final test accuracy = %.1f%% (N=%d)', test_accuracy * 100, len(test_bottlenecks))
    if FLAGS.print_misclassified_test_images:
        logging.info('=== MISCLASSIFIED TEST IMAGES ===')
        for (i, test_filename) in enumerate(test_filenames):
            if predictions[i] != test_ground_truth[i]:
                logging.info('%70s  %s', test_filename, list(image_lists.keys())[predictions[i]])

def build_eval_session(module_spec, class_count):
    if False:
        for i in range(10):
            print('nop')
    'Builds an restored eval session without train operations for exporting.\n\n  Args:\n    module_spec: The hub.ModuleSpec for the image module being used.\n    class_count: Number of classes\n\n  Returns:\n    Eval session containing the restored eval graph.\n    The bottleneck input, ground truth, eval step, and prediction tensors.\n  '
    (eval_graph, bottleneck_tensor, resized_input_tensor, wants_quantization) = create_module_graph(module_spec)
    eval_sess = tf.Session(graph=eval_graph)
    with eval_graph.as_default():
        (_, _, bottleneck_input, ground_truth_input, final_tensor) = add_final_retrain_ops(class_count, FLAGS.final_tensor_name, bottleneck_tensor, wants_quantization, is_training=False)
        tf.train.Saver().restore(eval_sess, FLAGS.checkpoint_path)
        (evaluation_step, prediction) = add_evaluation_step(final_tensor, ground_truth_input)
    return (eval_sess, resized_input_tensor, bottleneck_input, ground_truth_input, evaluation_step, prediction)

def save_graph_to_file(graph_file_name, module_spec, class_count):
    if False:
        return 10
    'Saves an graph to file, creating a valid quantized one if necessary.'
    (sess, _, _, _, _, _) = build_eval_session(module_spec, class_count)
    graph = sess.graph
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
    with tf.gfile.GFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())

def prepare_file_system():
    if False:
        return 10
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    if FLAGS.intermediate_store_frequency > 0:
        ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
    return

def add_jpeg_decoding(module_spec):
    if False:
        i = 10
        return i + 15
    'Adds operations that perform JPEG decoding and resizing to the graph..\n\n  Args:\n    module_spec: The hub.ModuleSpec for the image module being used.\n\n  Returns:\n    Tensors for the node to feed JPEG data into, and the output of the\n      preprocessing steps.\n  '
    (input_height, input_width) = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image, tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
    return (jpeg_data, resized_image)

def export_model(module_spec, class_count, saved_model_dir):
    if False:
        i = 10
        return i + 15
    'Exports model for serving.\n\n  Args:\n    module_spec: The hub.ModuleSpec for the image module being used.\n    class_count: The number of classes.\n    saved_model_dir: Directory in which to save exported model and variables.\n  '
    (sess, in_image, _, _, _, _) = build_eval_session(module_spec, class_count)
    with sess.graph.as_default() as graph:
        tf.saved_model.simple_save(sess, saved_model_dir, inputs={'image': in_image}, outputs={'prediction': graph.get_tensor_by_name('final_result:0')}, legacy_init_op=tf.group(tf.tables_initializer(), name='legacy_init_op'))

def logging_level_verbosity(logging_verbosity):
    if False:
        while True:
            i = 10
    "Converts logging_level into TensorFlow logging verbosity value.\n\n  Args:\n    logging_verbosity: String value representing logging level: 'DEBUG', 'INFO',\n    'WARN', 'ERROR', 'FATAL'\n  "
    name_to_level = {'FATAL': logging.FATAL, 'ERROR': logging.ERROR, 'WARN': logging.WARN, 'INFO': logging.INFO, 'DEBUG': logging.DEBUG}
    try:
        return name_to_level[logging_verbosity]
    except Exception as e:
        raise RuntimeError('Not supported logs verbosity (%s). Use one of %s.' % (str(e), list(name_to_level)))

def main(_):
    if False:
        return 10
    logging_verbosity = logging_level_verbosity(FLAGS.logging_verbosity)
    logging.set_verbosity(logging_verbosity)
    logging.error('WARNING: This tool is deprecated in favor of https://github.com/tensorflow/hub/tree/master/tensorflow_hub/tools/make_image_classifier')
    if not FLAGS.image_dir:
        logging.error('Must set flag --image_dir.')
        return -1
    prepare_file_system()
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage, FLAGS.validation_percentage)
    class_count = len(image_lists.keys())
    if class_count == 0:
        logging.error('No valid folders of images found at %s', FLAGS.image_dir)
        return -1
    if class_count == 1:
        logging.error('Only one valid folder of images found at %s  - multiple classes are needed for classification.', FLAGS.image_dir)
        return -1
    do_distort_images = should_distort_images(FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale, FLAGS.random_brightness)
    module_spec = hub.load_module_spec(FLAGS.tfhub_module)
    (graph, bottleneck_tensor, resized_image_tensor, wants_quantization) = create_module_graph(module_spec)
    with graph.as_default():
        (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) = add_final_retrain_ops(class_count, FLAGS.final_tensor_name, bottleneck_tensor, wants_quantization, is_training=True)
    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        (jpeg_data_tensor, decoded_image_tensor) = add_jpeg_decoding(module_spec)
        if do_distort_images:
            (distorted_jpeg_data_tensor, distorted_image_tensor) = add_input_distortions(FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale, FLAGS.random_brightness, module_spec)
        else:
            cache_bottlenecks(sess, image_lists, FLAGS.image_dir, FLAGS.bottleneck_dir, jpeg_data_tensor, decoded_image_tensor, resized_image_tensor, bottleneck_tensor, FLAGS.tfhub_module)
        (evaluation_step, _) = add_evaluation_step(final_tensor, ground_truth_input)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')
        train_saver = tf.train.Saver()
        for i in range(FLAGS.how_many_training_steps):
            if do_distort_images:
                (train_bottlenecks, train_ground_truth) = get_random_distorted_bottlenecks(sess, image_lists, FLAGS.train_batch_size, 'training', FLAGS.image_dir, distorted_jpeg_data_tensor, distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
            else:
                (train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(sess, image_lists, FLAGS.train_batch_size, 'training', FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor, decoded_image_tensor, resized_image_tensor, bottleneck_tensor, FLAGS.tfhub_module)
            (train_summary, _) = sess.run([merged, train_step], feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)
            is_last_step = i + 1 == FLAGS.how_many_training_steps
            if i % FLAGS.eval_step_interval == 0 or is_last_step:
                (train_accuracy, cross_entropy_value) = sess.run([evaluation_step, cross_entropy], feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
                logging.info('%s: Step %d: Train accuracy = %.1f%%', datetime.now(), i, train_accuracy * 100)
                logging.info('%s: Step %d: Cross entropy = %f', datetime.now(), i, cross_entropy_value)
                (validation_bottlenecks, validation_ground_truth, _) = get_random_cached_bottlenecks(sess, image_lists, FLAGS.validation_batch_size, 'validation', FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor, decoded_image_tensor, resized_image_tensor, bottleneck_tensor, FLAGS.tfhub_module)
                (validation_summary, validation_accuracy) = sess.run([merged, evaluation_step], feed_dict={bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                validation_writer.add_summary(validation_summary, i)
                logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)', datetime.now(), i, validation_accuracy * 100, len(validation_bottlenecks))
            intermediate_frequency = FLAGS.intermediate_store_frequency
            if intermediate_frequency > 0 and i % intermediate_frequency == 0 and (i > 0):
                train_saver.save(sess, FLAGS.checkpoint_path)
                intermediate_file_name = FLAGS.intermediate_output_graphs_dir + 'intermediate_' + str(i) + '.pb'
                logging.info('Save intermediate result to : %s', intermediate_file_name)
                save_graph_to_file(intermediate_file_name, module_spec, class_count)
        train_saver.save(sess, FLAGS.checkpoint_path)
        run_final_eval(sess, module_spec, class_count, image_lists, jpeg_data_tensor, decoded_image_tensor, resized_image_tensor, bottleneck_tensor)
        logging.info('Save final result to : %s', FLAGS.output_graph)
        if wants_quantization:
            logging.info('The model is instrumented for quantization with TF-Lite')
        save_graph_to_file(FLAGS.output_graph, module_spec, class_count)
        with tf.gfile.GFile(FLAGS.output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')
        if FLAGS.saved_model_dir:
            export_model(module_spec, class_count, FLAGS.saved_model_dir)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='', help='Path to folders of labeled images.')
    parser.add_argument('--output_graph', type=str, default='/tmp/output_graph.pb', help='Where to save the trained graph.')
    parser.add_argument('--intermediate_output_graphs_dir', type=str, default='/tmp/intermediate_graph/', help='Where to save the intermediate graphs.')
    parser.add_argument('--intermediate_store_frequency', type=int, default=0, help='         How many steps to store intermediate graph. If "0" then will not\n         store.      ')
    parser.add_argument('--output_labels', type=str, default='/tmp/output_labels.txt', help="Where to save the trained graph's labels.")
    parser.add_argument('--summaries_dir', type=str, default='/tmp/retrain_logs', help='Where to save summary logs for TensorBoard.')
    parser.add_argument('--how_many_training_steps', type=int, default=4000, help='How many training steps to run before ending.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='How large a learning rate to use when training.')
    parser.add_argument('--testing_percentage', type=int, default=10, help='What percentage of images to use as a test set.')
    parser.add_argument('--validation_percentage', type=int, default=10, help='What percentage of images to use as a validation set.')
    parser.add_argument('--eval_step_interval', type=int, default=10, help='How often to evaluate the training results.')
    parser.add_argument('--train_batch_size', type=int, default=100, help='How many images to train on at a time.')
    parser.add_argument('--test_batch_size', type=int, default=-1, help='      How many images to test on. This test set is only used once, to evaluate\n      the final accuracy of the model after training completes.\n      A value of -1 causes the entire test set to be used, which leads to more\n      stable results across runs.      ')
    parser.add_argument('--validation_batch_size', type=int, default=100, help='      How many images to use in an evaluation batch. This validation set is\n      used much more often than the test set, and is an early indicator of how\n      accurate the model is during training.\n      A value of -1 causes the entire validation set to be used, which leads to\n      more stable results across training iterations, but may be slower on large\n      training sets.      ')
    parser.add_argument('--print_misclassified_test_images', default=False, help='      Whether to print out a list of all misclassified test images.      ', action='store_true')
    parser.add_argument('--bottleneck_dir', type=str, default='/tmp/bottleneck', help='Path to cache bottleneck layer values as files.')
    parser.add_argument('--final_tensor_name', type=str, default='final_result', help='      The name of the output classification layer in the retrained graph.      ')
    parser.add_argument('--flip_left_right', default=False, help='      Whether to randomly flip half of the training images horizontally.      ', action='store_true')
    parser.add_argument('--random_crop', type=int, default=0, help='      A percentage determining how much of a margin to randomly crop off the\n      training images.      ')
    parser.add_argument('--random_scale', type=int, default=0, help='      A percentage determining how much to randomly scale up the size of the\n      training images by.      ')
    parser.add_argument('--random_brightness', type=int, default=0, help='      A percentage determining how much to randomly multiply the training image\n      input pixels up or down by.      ')
    parser.add_argument('--tfhub_module', type=str, default='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/3', help='      Which TensorFlow Hub module to use. For more options,\n      search https://tfhub.dev for image feature vector modules.      ')
    parser.add_argument('--saved_model_dir', type=str, default='', help='Where to save the exported graph.')
    parser.add_argument('--logging_verbosity', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'], help='How much logging output should be produced.')
    parser.add_argument('--checkpoint_path', type=str, default='/tmp/_retrain_checkpoint', help='Where to save checkpoint files.')
    (FLAGS, unparsed) = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)