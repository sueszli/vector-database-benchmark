"""Base estimator defining TCN training, test, and inference."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABCMeta
from abc import abstractmethod
import os
import numpy as np
import numpy as np
import data_providers
import preprocessing
from utils import util
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.python.training import session_run_hook
tf.app.flags.DEFINE_integer('tf_random_seed', 0, 'Random seed.')
FLAGS = tf.app.flags.FLAGS

class InitFromPretrainedCheckpointHook(session_run_hook.SessionRunHook):
    """Hook that can init graph from a pretrained checkpoint."""

    def __init__(self, pretrained_checkpoint_dir):
        if False:
            while True:
                i = 10
        'Initializes a `InitFromPretrainedCheckpointHook`.\n\n    Args:\n      pretrained_checkpoint_dir: The dir of pretrained checkpoint.\n\n    Raises:\n      ValueError: If pretrained_checkpoint_dir is invalid.\n    '
        if pretrained_checkpoint_dir is None:
            raise ValueError('pretrained_checkpoint_dir must be specified.')
        self._pretrained_checkpoint_dir = pretrained_checkpoint_dir

    def begin(self):
        if False:
            while True:
                i = 10
        checkpoint_reader = tf.contrib.framework.load_checkpoint(self._pretrained_checkpoint_dir)
        variable_shape_map = checkpoint_reader.get_variable_to_shape_map()
        exclude_scopes = 'logits/,final_layer/,aux_'
        exclusions = ['global_step']
        if exclude_scopes:
            exclusions.extend([scope.strip() for scope in exclude_scopes.split(',')])
        variable_to_restore = tf.contrib.framework.get_model_variables()
        filtered_variables_to_restore = {}
        for v in variable_to_restore:
            for exclusion in exclusions:
                if v.name.startswith(exclusion):
                    break
            else:
                var_name = v.name.split(':')[0]
                filtered_variables_to_restore[var_name] = v
        final_variables_to_restore = {}
        for (var_name, var_tensor) in filtered_variables_to_restore.iteritems():
            if var_name not in variable_shape_map:
                var_name = os.path.join(var_name, 'ExponentialMovingAverage')
                if var_name not in variable_shape_map:
                    tf.logging.info('Skip init [%s] because it is not in ckpt.', var_name)
                    continue
            if not var_tensor.get_shape().is_compatible_with(variable_shape_map[var_name]):
                tf.logging.info('Skip init [%s] from [%s] in ckpt because shape dismatch: %s vs %s', var_tensor.name, var_name, var_tensor.get_shape(), variable_shape_map[var_name])
                continue
            tf.logging.info('Init %s from %s in ckpt' % (var_tensor, var_name))
            final_variables_to_restore[var_name] = var_tensor
        self._init_fn = tf.contrib.framework.assign_from_checkpoint_fn(self._pretrained_checkpoint_dir, final_variables_to_restore)

    def after_create_session(self, session, coord):
        if False:
            return 10
        tf.logging.info('Restoring InceptionV3 weights.')
        self._init_fn(session)
        tf.logging.info('Done restoring InceptionV3 weights.')

class BaseEstimator(object):
    """Abstract TCN base estimator class."""
    __metaclass__ = ABCMeta

    def __init__(self, config, logdir):
        if False:
            return 10
        'Constructor.\n\n    Args:\n      config: A Luatable-like T object holding training config.\n      logdir: String, a directory where checkpoints and summaries are written.\n    '
        self._config = config
        self._logdir = logdir

    @abstractmethod
    def construct_input_fn(self, records, is_training):
        if False:
            return 10
        "Builds an estimator input_fn.\n\n    The input_fn is used to pass feature and target data to the train,\n    evaluate, and predict methods of the Estimator.\n\n    Method to be overridden by implementations.\n\n    Args:\n      records: A list of Strings, paths to TFRecords with image data.\n      is_training: Boolean, whether or not we're training.\n\n    Returns:\n      Function, that has signature of ()->(dict of features, target).\n        features is a dict mapping feature names to `Tensors`\n        containing the corresponding feature data (typically, just a single\n        key/value pair 'raw_data' -> image `Tensor` for TCN.\n        labels is a 1-D int32 `Tensor` holding labels.\n    "
        pass

    def preprocess_data(self, images, is_training):
        if False:
            i = 10
            return i + 15
        "Preprocesses raw images for either training or inference.\n\n    Args:\n      images: A 4-D float32 `Tensor` holding images to preprocess.\n      is_training: Boolean, whether or not we're in training.\n\n    Returns:\n      data_preprocessed: data after the preprocessor.\n    "
        config = self._config
        height = config.data.height
        width = config.data.width
        min_scale = config.data.augmentation.minscale
        max_scale = config.data.augmentation.maxscale
        p_scale_up = config.data.augmentation.proportion_scaled_up
        aug_color = config.data.augmentation.color
        fast_mode = config.data.augmentation.fast_mode
        crop_strategy = config.data.preprocessing.eval_cropping
        preprocessed_images = preprocessing.preprocess_images(images, is_training, height, width, min_scale, max_scale, p_scale_up, aug_color=aug_color, fast_mode=fast_mode, crop_strategy=crop_strategy)
        return preprocessed_images

    @abstractmethod
    def forward(self, images, is_training, reuse=False):
        if False:
            i = 10
            return i + 15
        "Defines the forward pass that converts batch images to embeddings.\n\n    Method to be overridden by implementations.\n\n    Args:\n      images: A 4-D float32 `Tensor` holding images to be embedded.\n      is_training: Boolean, whether or not we're in training mode.\n      reuse: Boolean, whether or not to reuse embedder.\n    Returns:\n      embeddings: A 2-D float32 `Tensor` holding embedded images.\n    "
        pass

    @abstractmethod
    def define_loss(self, embeddings, labels, is_training):
        if False:
            print('Hello World!')
        "Defines the loss function on the embedding vectors.\n\n    Method to be overridden by implementations.\n\n    Args:\n      embeddings: A 2-D float32 `Tensor` holding embedded images.\n      labels: A 1-D int32 `Tensor` holding problem labels.\n      is_training: Boolean, whether or not we're in training mode.\n\n    Returns:\n      loss: tf.float32 scalar.\n    "
        pass

    @abstractmethod
    def define_eval_metric_ops(self):
        if False:
            return 10
        'Defines the dictionary of eval metric tensors.\n\n    Method to be overridden by implementations.\n\n    Returns:\n      eval_metric_ops:  A dict of name/value pairs specifying the\n        metrics that will be calculated when the model runs in EVAL mode.\n    '
        pass

    def get_train_op(self, loss):
        if False:
            for i in range(10):
                print('nop')
        "Creates a training op.\n\n    Args:\n      loss: A float32 `Tensor` representing the total training loss.\n    Returns:\n      train_op: A slim.learning.create_train_op train_op.\n    Raises:\n      ValueError: If specified optimizer isn't supported.\n    "
        assert self.variables_to_train
        decay_steps = self._config.learning.decay_steps
        decay_factor = self._config.learning.decay_factor
        learning_rate = float(self._config.learning.learning_rate)
        global_step = slim.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_factor, staircase=True)
        opt_type = self._config.learning.optimizer
        if opt_type == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate)
        elif opt_type == 'momentum':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
        elif opt_type == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9, epsilon=1.0, decay=0.9)
        else:
            raise ValueError('Unsupported optimizer %s' % opt_type)
        if self._config.use_tpu:
            opt = tpu_optimizer.CrossShardOptimizer(opt)
        train_op = slim.learning.create_train_op(loss, optimizer=opt, variables_to_train=self.variables_to_train, update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        return train_op

    def _get_model_fn(self):
        if False:
            print('Hello World!')
        'Defines behavior for training, evaluation, and inference (prediction).\n\n    Returns:\n      `model_fn` for `Estimator`.\n    '

        def model_fn(features, labels, mode, params):
            if False:
                print('Hello World!')
            'Build the model based on features, labels, and mode.\n\n      Args:\n        features: Dict, strings to `Tensor` input data, returned by the\n          input_fn.\n        labels: The labels Tensor returned by the input_fn.\n        mode: A string indicating the mode. This will be either\n          tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.PREDICT,\n          or tf.estimator.ModeKeys.EVAL.\n        params: A dict holding training parameters, passed in during TPU\n          training.\n\n      Returns:\n        A tf.estimator.EstimatorSpec specifying train/test/inference behavior.\n      '
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            batch_preprocessed = features['batch_preprocessed']
            batch_encoded = self.forward(batch_preprocessed, is_training)
            initializer_fn = None
            if mode == tf.estimator.ModeKeys.TRAIN:
                initializer_fn = self.pretrained_init_fn
            total_loss = None
            if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
                loss = self.define_loss(batch_encoded, labels, is_training)
                tf.losses.add_loss(loss)
                total_loss = tf.losses.get_total_loss()
            train_op = None
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = self.get_train_op(total_loss)
            predictions_dict = None
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions_dict = {'embeddings': batch_encoded}
                for (k, v) in features.iteritems():
                    predictions_dict[k] = v
            eval_metric_ops = None
            if mode == tf.estimator.ModeKeys.EVAL:
                eval_metric_ops = self.define_eval_metric_ops()
            num_checkpoint_to_keep = self._config.logging.checkpoint.num_to_keep
            saver = tf.train.Saver(max_to_keep=num_checkpoint_to_keep)
            if is_training and self._config.use_tpu:
                return tpu_estimator.TPUEstimatorSpec(mode, loss=total_loss, eval_metrics=None, train_op=train_op, predictions=predictions_dict)
            else:
                scaffold = tf.train.Scaffold(init_fn=initializer_fn, saver=saver, summary_op=None)
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict, loss=total_loss, train_op=train_op, eval_metric_ops=eval_metric_ops, scaffold=scaffold)
        return model_fn

    def train(self):
        if False:
            print('Hello World!')
        'Runs training.'
        config = self._config
        training_dir = config.data.training
        training_records = util.GetFilesRecursively(training_dir)
        self._batch_size = config.data.batch_size
        train_input_fn = self.construct_input_fn(training_records, is_training=True)
        estimator = self._build_estimator(is_training=True)
        train_hooks = None
        if config.use_tpu:
            train_hooks = []
            if tf.train.latest_checkpoint(self._logdir) is None:
                train_hooks.append(InitFromPretrainedCheckpointHook(config[config.embedder_strategy].pretrained_checkpoint))
        estimator.train(input_fn=train_input_fn, hooks=train_hooks, steps=config.learning.max_step)

    def _build_estimator(self, is_training):
        if False:
            print('Hello World!')
        "Returns an Estimator object.\n\n    Args:\n      is_training: Boolean, whether or not we're in training mode.\n\n    Returns:\n      A tf.estimator.Estimator.\n    "
        config = self._config
        save_checkpoints_steps = config.logging.checkpoint.save_checkpoints_steps
        keep_checkpoint_max = self._config.logging.checkpoint.num_to_keep
        if is_training and config.use_tpu:
            iterations = config.tpu.iterations
            num_shards = config.tpu.num_shards
            run_config = tpu_config.RunConfig(save_checkpoints_secs=None, save_checkpoints_steps=save_checkpoints_steps, keep_checkpoint_max=keep_checkpoint_max, master=FLAGS.master, evaluation_master=FLAGS.master, model_dir=self._logdir, tpu_config=tpu_config.TPUConfig(iterations_per_loop=iterations, num_shards=num_shards, per_host_input_for_training=num_shards <= 8), tf_random_seed=FLAGS.tf_random_seed)
            batch_size = config.data.batch_size
            return tpu_estimator.TPUEstimator(model_fn=self._get_model_fn(), config=run_config, use_tpu=True, train_batch_size=batch_size, eval_batch_size=batch_size)
        else:
            run_config = tf.estimator.RunConfig().replace(model_dir=self._logdir, save_checkpoints_steps=save_checkpoints_steps, keep_checkpoint_max=keep_checkpoint_max, tf_random_seed=FLAGS.tf_random_seed)
            return tf.estimator.Estimator(model_fn=self._get_model_fn(), config=run_config)

    def evaluate(self):
        if False:
            while True:
                i = 10
        'Runs `Estimator` validation.\n    '
        config = self._config
        validation_dir = config.data.validation
        validation_records = util.GetFilesRecursively(validation_dir)
        self._batch_size = config.data.batch_size
        validation_input_fn = self.construct_input_fn(validation_records, False)
        estimator = self._build_estimator(is_training=False)
        eval_batch_size = config.data.batch_size
        num_eval_samples = config.val.num_eval_samples
        num_eval_batches = int(num_eval_samples / eval_batch_size)
        estimator.evaluate(input_fn=validation_input_fn, steps=num_eval_batches)

    def inference(self, inference_input, checkpoint_path, batch_size=None, **kwargs):
        if False:
            print('Hello World!')
        "Defines 3 of modes of inference.\n\n    Inputs:\n    * Mode 1: Input is an input_fn.\n    * Mode 2: Input is a TFRecord (or list of TFRecords).\n    * Mode 3: Input is a numpy array holding an image (or array of images).\n\n    Outputs:\n    * Mode 1: this returns an iterator over embeddings and additional\n      metadata. See\n      https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#predict\n      for details.\n    * Mode 2: Returns an iterator over tuples of\n      (embeddings, raw_image_strings, sequence_name), where embeddings is a\n      2-D float32 numpy array holding [sequence_size, embedding_size] image\n      embeddings, raw_image_strings is a 1-D string numpy array holding\n      [sequence_size] jpeg-encoded image strings, and sequence_name is a\n      string holding the name of the embedded sequence.\n    * Mode 3: Returns a tuple of (embeddings, raw_image_strings), where\n      embeddings is a 2-D float32 numpy array holding\n      [batch_size, embedding_size] image embeddings, raw_image_strings is a\n      1-D string numpy array holding [batch_size] jpeg-encoded image strings.\n\n    Args:\n      inference_input: This can be a tf.Estimator input_fn, a TFRecord path,\n        a list of TFRecord paths, a numpy image, or an array of numpy images.\n      checkpoint_path: String, path to the checkpoint to restore for inference.\n      batch_size: Int, the size of the batch to use for inference.\n      **kwargs: Additional keyword arguments, depending on the mode.\n        See _input_fn_inference, _tfrecord_inference, and _np_inference.\n    Returns:\n      inference_output: Inference output depending on mode, see above for\n        details.\n    Raises:\n      ValueError: If inference_input isn't a tf.Estimator input_fn,\n        a TFRecord path, a list of TFRecord paths, or a numpy array,\n    "
        if callable(inference_input):
            return self._input_fn_inference(input_fn=inference_input, checkpoint_path=checkpoint_path, **kwargs)
        elif util.is_tfrecord_input(inference_input):
            return self._tfrecord_inference(records=inference_input, checkpoint_path=checkpoint_path, batch_size=batch_size, **kwargs)
        elif util.is_np_array(inference_input):
            return self._np_inference(np_images=inference_input, checkpoint_path=checkpoint_path, **kwargs)
        else:
            raise ValueError('inference input must be a tf.Estimator input_fn, a TFRecord path,a list of TFRecord paths, or a numpy array. Got: %s' % str(type(inference_input)))

    def _input_fn_inference(self, input_fn, checkpoint_path, predict_keys=None):
        if False:
            print('Hello World!')
        'Mode 1: tf.Estimator inference.\n\n    Args:\n      input_fn: Function, that has signature of ()->(dict of features, None).\n        This is a function called by the estimator to get input tensors (stored\n        in the features dict) to do inference over.\n      checkpoint_path: String, path to a specific checkpoint to restore.\n      predict_keys: List of strings, the keys of the `Tensors` in the features\n        dict (returned by the input_fn) to evaluate during inference.\n    Returns:\n      predictions: An Iterator, yielding evaluated values of `Tensors`\n        specified in `predict_keys`.\n    '
        estimator = self._build_estimator(is_training=False)
        predictions = estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path, predict_keys=predict_keys)
        return predictions

    def _tfrecord_inference(self, records, checkpoint_path, batch_size, num_sequences=-1, reuse=False):
        if False:
            return 10
        'Mode 2: TFRecord inference.\n\n    Args:\n      records: List of strings, paths to TFRecords.\n      checkpoint_path: String, path to a specific checkpoint to restore.\n      batch_size: Int, size of inference batch.\n      num_sequences: Int, number of sequences to embed. If -1,\n        embed everything.\n      reuse: Boolean, whether or not to reuse embedder weights.\n    Yields:\n      (embeddings, raw_image_strings, sequence_name):\n        embeddings is a 2-D float32 numpy array holding\n        [sequence_size, embedding_size] image embeddings.\n        raw_image_strings is a 1-D string numpy array holding\n        [sequence_size] jpeg-encoded image strings.\n        sequence_name is a string holding the name of the embedded sequence.\n    '
        tf.reset_default_graph()
        if not isinstance(records, list):
            records = list(records)
        num_views = self._config.data.num_views
        (views, task, seq_len) = data_providers.full_sequence_provider(records, num_views)
        tensor_dict = {'raw_image_strings': views, 'task': task, 'seq_len': seq_len}
        image_str_placeholder = tf.placeholder(tf.string, shape=[None])
        decoded = preprocessing.decode_images(image_str_placeholder)
        decoded.set_shape([batch_size, None, None, 3])
        preprocessed = self.preprocess_data(decoded, is_training=False)
        embeddings = self.forward(preprocessed, is_training=False, reuse=reuse)
        tf.train.get_or_create_global_step()
        saver = tf.train.Saver(tf.all_variables())
        with tf.train.MonitoredSession() as sess:
            saver.restore(sess, checkpoint_path)
            cnt = 0
            try:
                while cnt < num_sequences if num_sequences != -1 else True:
                    np_data = sess.run(tensor_dict)
                    np_raw_images = np_data['raw_image_strings']
                    np_seq_len = np_data['seq_len']
                    np_task = np_data['task']
                    embedding_size = self._config.embedding_size
                    view_embeddings = [np.zeros((0, embedding_size)) for _ in range(num_views)]
                    for view_index in range(num_views):
                        view_raw = np_raw_images[view_index]
                        t = 0
                        while t < np_seq_len:
                            embeddings_np = sess.run(embeddings, feed_dict={image_str_placeholder: view_raw[t:t + batch_size]})
                            view_embeddings[view_index] = np.append(view_embeddings[view_index], embeddings_np, axis=0)
                            tf.logging.info('Embedded %d images for task %s' % (t, np_task))
                            t += batch_size
                    view_raw_images = np_data['raw_image_strings']
                    yield (view_embeddings, view_raw_images, np_task)
                    cnt += 1
            except tf.errors.OutOfRangeError:
                tf.logging.info('Done embedding entire dataset.')

    def _np_inference(self, np_images, checkpoint_path):
        if False:
            while True:
                i = 10
        'Mode 3: Call this repeatedly to do inference over numpy images.\n\n    This mode is for when we we want to do real-time inference over\n    some stream of images (represented as numpy arrays).\n\n    Args:\n      np_images: A float32 numpy array holding images to embed.\n      checkpoint_path: String, path to a specific checkpoint to restore.\n    Returns:\n      (embeddings, raw_image_strings):\n        embeddings is a 2-D float32 numpy array holding\n        [inferred batch_size, embedding_size] image embeddings.\n        raw_image_strings is a 1-D string numpy array holding\n        [inferred batch_size] jpeg-encoded image strings.\n    '
        if isinstance(np_images, list):
            np_images = np.asarray(np_images)
        if len(np_images.shape) == 3:
            np_images = np.expand_dims(np_images, axis=0)
        assert np.min(np_images) >= 0.0
        if (np.min(np_images), np.max(np_images)) == (0, 255):
            np_images = np_images.astype(np.float32) / 255.0
            assert (np.min(np_images), np.max(np_images)) == (0.0, 1.0)
        if not hasattr(self, '_np_inf_tensor_dict'):
            self._setup_np_inference(np_images, checkpoint_path)
        np_tensor_dict = self._sess.run(self._np_inf_tensor_dict, feed_dict={self._image_placeholder: np_images})
        return (np_tensor_dict['embeddings'], np_tensor_dict['raw_image_strings'])

    def _setup_np_inference(self, np_images, checkpoint_path):
        if False:
            while True:
                i = 10
        'Sets up and restores inference graph, creates and caches a Session.'
        tf.logging.info('Restoring model weights.')
        (_, height, width, _) = np.shape(np_images)
        image_placeholder = tf.placeholder(tf.float32, shape=(None, height, width, 3))
        preprocessed = self.preprocess_data(image_placeholder, is_training=False)
        im_strings = preprocessing.unscale_jpeg_encode(preprocessed)
        embeddings = self.forward(preprocessed, is_training=False)
        tf.train.get_or_create_global_step()
        saver = tf.train.Saver(tf.all_variables())
        self._image_placeholder = image_placeholder
        self._batch_encoded = embeddings
        self._np_inf_tensor_dict = {'embeddings': embeddings, 'raw_image_strings': im_strings}
        self._sess = tf.Session()
        saver.restore(self._sess, checkpoint_path)