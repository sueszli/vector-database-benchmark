"""Base wrapper class for performing inference with an image-to-text model.

Subclasses must implement the following methods:

  build_model():
    Builds the model for inference and returns the model object.

  feed_image():
    Takes an encoded image and returns the initial model state, where "state"
    is a numpy array whose specifics are defined by the subclass, e.g.
    concatenated LSTM state. It's assumed that feed_image() will be called
    precisely once at the start of inference for each image. Subclasses may
    compute and/or save per-image internal context in this method.

  inference_step():
    Takes a batch of inputs and states at a single time-step. Returns the
    softmax output corresponding to the inputs, and the new states of the batch.
    Optionally also returns metadata about the current inference step, e.g. a
    serialized numpy array containing activations from a particular model layer.

Client usage:
  1. Build the model inference graph via build_graph_from_config() or
     build_graph_from_proto().
  2. Call the resulting restore_fn to load the model checkpoint.
  3. For each image in a batch of images:
     a) Call feed_image() once to get the initial state.
     b) For each step of caption generation, call inference_step().
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import tensorflow as tf

class InferenceWrapperBase(object):
    """Base wrapper class for performing inference with an image-to-text model."""

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def build_model(self, model_config):
        if False:
            for i in range(10):
                print('nop')
        'Builds the model for inference.\n\n    Args:\n      model_config: Object containing configuration for building the model.\n\n    Returns:\n      model: The model object.\n    '
        tf.logging.fatal('Please implement build_model in subclass')

    def _create_restore_fn(self, checkpoint_path, saver):
        if False:
            return 10
        'Creates a function that restores a model from checkpoint.\n\n    Args:\n      checkpoint_path: Checkpoint file or a directory containing a checkpoint\n        file.\n      saver: Saver for restoring variables from the checkpoint file.\n\n    Returns:\n      restore_fn: A function such that restore_fn(sess) loads model variables\n        from the checkpoint file.\n\n    Raises:\n      ValueError: If checkpoint_path does not refer to a checkpoint file or a\n        directory containing a checkpoint file.\n    '
        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            if not checkpoint_path:
                raise ValueError('No checkpoint file found in: %s' % checkpoint_path)

        def _restore_fn(sess):
            if False:
                while True:
                    i = 10
            tf.logging.info('Loading model from checkpoint: %s', checkpoint_path)
            saver.restore(sess, checkpoint_path)
            tf.logging.info('Successfully loaded checkpoint: %s', os.path.basename(checkpoint_path))
        return _restore_fn

    def build_graph_from_config(self, model_config, checkpoint_path):
        if False:
            print('Hello World!')
        'Builds the inference graph from a configuration object.\n\n    Args:\n      model_config: Object containing configuration for building the model.\n      checkpoint_path: Checkpoint file or a directory containing a checkpoint\n        file.\n\n    Returns:\n      restore_fn: A function such that restore_fn(sess) loads model variables\n        from the checkpoint file.\n    '
        tf.logging.info('Building model.')
        self.build_model(model_config)
        saver = tf.train.Saver()
        return self._create_restore_fn(checkpoint_path, saver)

    def build_graph_from_proto(self, graph_def_file, saver_def_file, checkpoint_path):
        if False:
            i = 10
            return i + 15
        'Builds the inference graph from serialized GraphDef and SaverDef protos.\n\n    Args:\n      graph_def_file: File containing a serialized GraphDef proto.\n      saver_def_file: File containing a serialized SaverDef proto.\n      checkpoint_path: Checkpoint file or a directory containing a checkpoint\n        file.\n\n    Returns:\n      restore_fn: A function such that restore_fn(sess) loads model variables\n        from the checkpoint file.\n    '
        tf.logging.info('Loading GraphDef from file: %s', graph_def_file)
        graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(graph_def_file, 'rb') as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.logging.info('Loading SaverDef from file: %s', saver_def_file)
        saver_def = tf.train.SaverDef()
        with tf.gfile.FastGFile(saver_def_file, 'rb') as f:
            saver_def.ParseFromString(f.read())
        saver = tf.train.Saver(saver_def=saver_def)
        return self._create_restore_fn(checkpoint_path, saver)

    def feed_image(self, sess, encoded_image):
        if False:
            print('Hello World!')
        'Feeds an image and returns the initial model state.\n\n    See comments at the top of file.\n\n    Args:\n      sess: TensorFlow Session object.\n      encoded_image: An encoded image string.\n\n    Returns:\n      state: A numpy array of shape [1, state_size].\n    '
        tf.logging.fatal('Please implement feed_image in subclass')

    def inference_step(self, sess, input_feed, state_feed):
        if False:
            print('Hello World!')
        'Runs one step of inference.\n\n    Args:\n      sess: TensorFlow Session object.\n      input_feed: A numpy array of shape [batch_size].\n      state_feed: A numpy array of shape [batch_size, state_size].\n\n    Returns:\n      softmax_output: A numpy array of shape [batch_size, vocab_size].\n      new_state: A numpy array of shape [batch_size, state_size].\n      metadata: Optional. If not None, a string containing metadata about the\n        current inference step (e.g. serialized numpy array containing\n        activations from a particular model layer.).\n    '
        tf.logging.fatal('Please implement inference_step in subclass')