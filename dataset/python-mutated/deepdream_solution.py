"""DeepDream.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import zipfile
import sys
sys.path.extend(['', '/Users/shlens/Desktop/Neural-Art/homebrew/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python27.zip', '/Users/shlens/Desktop/Neural-Art/homebrew/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7', '/Users/shlens/Desktop/Neural-Art/homebrew/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-darwin', '/Users/shlens/Desktop/Neural-Art/homebrew/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-mac', '/Users/shlens/Desktop/Neural-Art/homebrew/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-mac/lib-scriptpackages', '/Users/shlens/Desktop/Neural-Art/homebrew/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-tk', '/Users/shlens/Desktop/Neural-Art/homebrew/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-old', '/Users/shlens/Desktop/Neural-Art/homebrew/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-dynload', '/Users/shlens/Desktop/Neural-Art/homebrew/lib/python2.7/site-packages', '/Users/shlens/Desktop/Neural-Art/homebrew/lib/python2.7/site-packages/gtk-2.0', '/Users/shlens/Desktop/Neural-Art/homebrew/lib/python2.7/site-packages/gtk-2.0'])
import numpy as np
import PIL.Image
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/tmp/inception/', 'Directory for storing Inception network.')
tf.app.flags.DEFINE_string('jpeg_file', 'output.jpg', 'Where to save the resulting JPEG.')

def get_layer(layer):
    if False:
        while True:
            i = 10
    'Helper for getting layer output Tensor in model Graph.\n\n  Args:\n   layer: string, layer name\n\n  Returns:\n    Tensor for that layer.\n  '
    graph = tf.get_default_graph()
    return graph.get_tensor_by_name('import/%s:0' % layer)

def maybe_download(data_dir):
    if False:
        print('Hello World!')
    'Maybe download pretrained Inception network.\n\n  Args:\n    data_dir: string, path to data\n  '
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    basename = 'inception5h.zip'
    local_file = tf.contrib.learn.python.learn.datasets.base.maybe_download(basename, data_dir, url)
    print('Extracting', local_file)
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall(FLAGS.data_dir)
    zip_ref.close()

def normalize_image(image):
    if False:
        print('Hello World!')
    'Stretch the range and prepare the image for saving as a JPEG.\n\n  Args:\n    image: numpy array\n\n  Returns:\n    numpy array of image in uint8\n  '
    image = np.clip(image, 0, 1)
    image = np.uint8(image * 255)
    return image

def save_jpeg(jpeg_file, image):
    if False:
        print('Hello World!')
    pil_image = PIL.Image.fromarray(image)
    pil_image.save(jpeg_file)
    print('Saved to file: ', jpeg_file)

def main(unused_argv):
    if False:
        for i in range(10):
            print('nop')
    maybe_download(FLAGS.data_dir)
    model_fn = os.path.join(FLAGS.data_dir, 'tensorflow_inception_graph.pb')
    with tf.gfile.FastGFile(model_fn, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default():
        input_image = tf.placeholder(np.float32, name='input')
        pixel_mean = 117.0
        input_preprocessed = tf.expand_dims(input_image - pixel_mean, 0)
        tf.import_graph_def(graph_def, {'input': input_preprocessed})
        graph = tf.get_default_graph()
        layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
        feature_nums = [int(graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]
        print('Number of layers', len(layers))
        print('Number of features:', sum(feature_nums))
        layer = 'mixed4d_3x3_bottleneck_pre_relu'
        channel = 139
        layer_channel = get_layer(layer)[:, :, :, channel]
        print('layer %s, channel %d: %s' % (layer, channel, layer_channel))
        score = tf.reduce_mean(layer_channel)
        input_gradient = tf.gradients(score, input_image)[0]
        noise_image = np.random.uniform(size=(224, 224, 3)) + 100.0
        image = noise_image.copy()
        step_scale = 1.0
        num_iter = 20
        with tf.Session() as sess:
            for i in xrange(num_iter):
                (image_gradient, score_value) = sess.run([input_gradient, score], {input_image: image})
                image_gradient /= image_gradient.std() + 1e-08
                image += image_gradient * step_scale
                print('At step = %d, score = %.3f' % (i, score_value))
    stddev = 0.1
    image = (image - image.mean()) / max(image.std(), 0.0001) * stddev + 0.5
    image = normalize_image(image)
    save_jpeg(FLAGS.jpeg_file, image)
if __name__ == '__main__':
    tf.app.run()