"""A script to export TF-Hub SavedModel."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import app
from absl import flags
import tensorflow as tf
from official.vision.image_classification import imagenet_preprocessing
from official.vision.image_classification import resnet_model
FLAGS = flags.FLAGS
flags.DEFINE_string('model_path', None, 'File path to TF model checkpoint or H5 file.')
flags.DEFINE_string('export_path', None, 'TF-Hub SavedModel destination path to export.')

def export_tfhub(model_path, hub_destination):
    if False:
        return 10
    'Restores a tf.keras.Model and saves for TF-Hub.'
    model = resnet_model.resnet50(num_classes=imagenet_preprocessing.NUM_CLASSES, rescale_inputs=True)
    model.load_weights(model_path)
    model.save(os.path.join(hub_destination, 'classification'), include_optimizer=False)
    image_input = model.get_layer(index=0).get_output_at(0)
    feature_vector_output = model.get_layer(name='reduce_mean').get_output_at(0)
    hub_model = tf.keras.Model(image_input, feature_vector_output)
    hub_model.save(os.path.join(hub_destination, 'feature-vector'), include_optimizer=False)

def main(argv):
    if False:
        while True:
            i = 10
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    assert tf.version.VERSION.startswith('2.')
    export_tfhub(FLAGS.model_path, FLAGS.export_path)
if __name__ == '__main__':
    app.run(main)