"""Callbacks: utilities called at certain points during model training."""
import os
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.summary import summary as tf_summary
from tensorflow.python.training import saver

class TensorBoard(callbacks.TensorBoard):
    """Enable visualizations for TensorBoard.

  TensorBoard is a visualization tool provided with TensorFlow.

  This callback logs events for TensorBoard, including:
  * Metrics summary plots
  * Training graph visualization
  * Activation histograms
  * Sampled profiling

  If you have installed TensorFlow with pip, you should be able
  to launch TensorBoard from the command line:

  ```sh
  tensorboard --logdir=path_to_your_logs
  ```

  You can find more information about TensorBoard
  [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

  Args:
      log_dir: the path of the directory where to save the log files to be
        parsed by TensorBoard.
      histogram_freq: frequency (in epochs) at which to compute activation and
        weight histograms for the layers of the model. If set to 0, histograms
        won't be computed. Validation data (or split) must be specified for
        histogram visualizations.
      write_graph: whether to visualize the graph in TensorBoard. The log file
        can become quite large when write_graph is set to True.
      write_grads: whether to visualize gradient histograms in TensorBoard.
        `histogram_freq` must be greater than 0.
      batch_size: size of batch of inputs to feed to the network for histograms
        computation.
      write_images: whether to write model weights to visualize as image in
        TensorBoard.
      embeddings_freq: frequency (in epochs) at which selected embedding layers
        will be saved. If set to 0, embeddings won't be computed. Data to be
        visualized in TensorBoard's Embedding tab must be passed as
        `embeddings_data`.
      embeddings_layer_names: a list of names of layers to keep eye on. If None
        or empty list all the embedding layer will be watched.
      embeddings_metadata: a dictionary which maps layer name to a file name in
        which metadata for this embedding layer is saved.
          [Here are details](
            https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
      embeddings_data: data to be embedded at layers specified in
        `embeddings_layer_names`. Numpy array (if the model has a single input)
        or list of Numpy arrays (if the model has multiple inputs). Learn more
        about embeddings [in this guide](
          https://www.tensorflow.org/programmers_guide/embedding).
      update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
        writes the losses and metrics to TensorBoard after each batch. The same
        applies for `'epoch'`. If using an integer, let's say `1000`, the
        callback will write the metrics and losses to TensorBoard every 1000
        samples. Note that writing too frequently to TensorBoard can slow down
        your training.
      profile_batch: Profile the batch to sample compute characteristics. By
        default, it will profile the second batch. Set profile_batch=0 to
        disable profiling.

  Raises:
      ValueError: If histogram_freq is set and no validation data is provided.

  @compatibility(eager)
  Using the `TensorBoard` callback will work when eager execution is enabled,
  with the restriction that outputting histogram summaries of weights and
  gradients is not supported. Consequently, `histogram_freq` will be ignored.
  @end_compatibility
  """

    def __init__(self, log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch', profile_batch=2):
        if False:
            return 10
        callbacks.Callback.__init__(self)
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        if self.histogram_freq and context.executing_eagerly():
            logging.warning(UserWarning('Weight and gradient histograms not supported for eagerexecution, setting `histogram_freq` to `0`.'))
            self.histogram_freq = 0
        self.merged = None
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.write_images = write_images
        self.batch_size = batch_size
        self._current_batch = 0
        self._total_batches_seen = 0
        self._total_val_batches_seen = 0
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata
        self.embeddings_data = embeddings_data
        if update_freq == 'batch':
            self.update_freq = 1
        else:
            self.update_freq = update_freq
        self._samples_seen = 0
        self._samples_seen_at_last_write = 0
        self._profile_batch = profile_batch
        self._profiler_started = False
        self._chief_worker_only = True

    def _init_writer(self, model):
        if False:
            for i in range(10):
                print('nop')
        'Sets file writer.'
        if context.executing_eagerly():
            self.writer = summary_ops_v2.create_file_writer_v2(self.log_dir)
            if not model.run_eagerly and self.write_graph:
                with self.writer.as_default():
                    summary_ops_v2.graph(K.get_graph())
        elif self.write_graph:
            self.writer = tf_summary.FileWriter(self.log_dir, K.get_graph())
        else:
            self.writer = tf_summary.FileWriter(self.log_dir)

    def _make_histogram_ops(self, model):
        if False:
            for i in range(10):
                print('nop')
        'Defines histogram ops when histogram_freq > 0.'
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:
                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    tf_summary.histogram(mapped_weight_name, weight)
                    if self.write_images:
                        w_img = array_ops.squeeze(weight)
                        shape = K.int_shape(w_img)
                        if len(shape) == 2:
                            if shape[0] > shape[1]:
                                w_img = array_ops.transpose(w_img)
                                shape = K.int_shape(w_img)
                            w_img = array_ops.reshape(w_img, [1, shape[0], shape[1], 1])
                        elif len(shape) == 3:
                            if K.image_data_format() == 'channels_last':
                                w_img = array_ops.transpose(w_img, perm=[2, 0, 1])
                                shape = K.int_shape(w_img)
                            w_img = array_ops.reshape(w_img, [shape[0], shape[1], shape[2], 1])
                        elif len(shape) == 1:
                            w_img = array_ops.reshape(w_img, [1, shape[0], 1, 1])
                        else:
                            continue
                        shape = K.int_shape(w_img)
                        assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                        tf_summary.image(mapped_weight_name, w_img)
                if self.write_grads:
                    for weight in layer.trainable_weights:
                        mapped_weight_name = weight.name.replace(':', '_')
                        grads = model.optimizer.get_gradients(model.total_loss, weight)

                        def is_indexed_slices(grad):
                            if False:
                                while True:
                                    i = 10
                            return type(grad).__name__ == 'IndexedSlices'
                        grads = [grad.values if is_indexed_slices(grad) else grad for grad in grads]
                        tf_summary.histogram('{}_grad'.format(mapped_weight_name), grads)
                if hasattr(layer, 'output'):
                    if isinstance(layer.output, list):
                        for (i, output) in enumerate(layer.output):
                            tf_summary.histogram('{}_out_{}'.format(layer.name, i), output)
                    else:
                        tf_summary.histogram('{}_out'.format(layer.name), layer.output)

    def set_model(self, model):
        if False:
            while True:
                i = 10
        'Sets Keras model and creates summary ops.'
        self.model = model
        self._init_writer(model)
        if not context.executing_eagerly():
            self._make_histogram_ops(model)
            self.merged = tf_summary.merge_all()
        if self.embeddings_freq and self.embeddings_data is not None:
            from tensorflow.python.keras.engine import training_utils_v1
            self.embeddings_data = training_utils_v1.standardize_input_data(self.embeddings_data, model.input_names)
            embeddings_layer_names = self.embeddings_layer_names
            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in self.model.layers if type(layer).__name__ == 'Embedding']
            self.assign_embeddings = []
            embeddings_vars = {}
            self.batch_id = batch_id = array_ops.placeholder(dtypes.int32)
            self.step = step = array_ops.placeholder(dtypes.int32)
            for layer in self.model.layers:
                if layer.name in embeddings_layer_names:
                    embedding_input = self.model.get_layer(layer.name).output
                    embedding_size = np.prod(embedding_input.shape[1:])
                    embedding_input = array_ops.reshape(embedding_input, (step, int(embedding_size)))
                    shape = (self.embeddings_data[0].shape[0], int(embedding_size))
                    embedding = variables.Variable(array_ops.zeros(shape), name=layer.name + '_embedding')
                    embeddings_vars[layer.name] = embedding
                    batch = state_ops.assign(embedding[batch_id:batch_id + step], embedding_input)
                    self.assign_embeddings.append(batch)
            self.saver = saver.Saver(list(embeddings_vars.values()))
            if isinstance(self.embeddings_metadata, str):
                embeddings_metadata = {layer_name: self.embeddings_metadata for layer_name in embeddings_vars.keys()}
            else:
                embeddings_metadata = self.embeddings_metadata
            try:
                from tensorboard.plugins import projector
            except ImportError:
                raise ImportError('Failed to import TensorBoard. Please make sure that TensorBoard integration is complete."')
            config = projector.ProjectorConfig()
            for (layer_name, tensor) in embeddings_vars.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name
                if embeddings_metadata is not None and layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]
            projector.visualize_embeddings(self.writer, config)

    def _fetch_callback(self, summary):
        if False:
            while True:
                i = 10
        self.writer.add_summary(summary, self._total_val_batches_seen)
        self._total_val_batches_seen += 1

    def _write_custom_summaries(self, step, logs=None):
        if False:
            return 10
        'Writes metrics out as custom scalar summaries.\n\n    Args:\n        step: the global step to use for TensorBoard.\n        logs: dict. Keys are scalar summary names, values are\n            NumPy scalars.\n\n    '
        logs = logs or {}
        if context.executing_eagerly():
            with self.writer.as_default(), summary_ops_v2.record_if(True):
                for (name, value) in logs.items():
                    if isinstance(value, np.ndarray):
                        value = value.item()
                    summary_ops_v2.scalar(name, value, step=step)
        else:
            for (name, value) in logs.items():
                if isinstance(value, np.ndarray):
                    value = value.item()
                summary = tf_summary.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value
                summary_value.tag = name
                self.writer.add_summary(summary, step)
        self.writer.flush()

    def on_train_batch_begin(self, batch, logs=None):
        if False:
            return 10
        if self._total_batches_seen == self._profile_batch - 1:
            self._start_profiler()

    def on_train_batch_end(self, batch, logs=None):
        if False:
            i = 10
            return i + 15
        return self.on_batch_end(batch, logs)

    def on_test_begin(self, logs=None):
        if False:
            for i in range(10):
                print('nop')
        pass

    def on_test_end(self, logs=None):
        if False:
            for i in range(10):
                print('nop')
        pass

    def on_batch_end(self, batch, logs=None):
        if False:
            while True:
                i = 10
        'Writes scalar summaries for metrics on every training batch.\n\n    Performs profiling if current batch is in profiler_batches.\n    '
        logs = logs or {}
        self._samples_seen += logs.get('size', 1)
        samples_seen_since = self._samples_seen - self._samples_seen_at_last_write
        if self.update_freq != 'epoch' and samples_seen_since >= self.update_freq:
            batch_logs = {'batch_' + k: v for (k, v) in logs.items() if k not in ['batch', 'size', 'num_steps']}
            self._write_custom_summaries(self._total_batches_seen, batch_logs)
            self._samples_seen_at_last_write = self._samples_seen
        self._total_batches_seen += 1
        self._stop_profiler()

    def on_train_begin(self, logs=None):
        if False:
            return 10
        pass

    def on_epoch_begin(self, epoch, logs=None):
        if False:
            print('Hello World!')
        'Add histogram op to Model eval_function callbacks, reset batch count.'
        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self.model._make_test_function()
            if self.merged not in self.model.test_function.fetches:
                self.model.test_function.fetches.append(self.merged)
                self.model.test_function.fetch_callbacks[self.merged] = self._fetch_callback

    def on_epoch_end(self, epoch, logs=None):
        if False:
            return 10
        'Checks if summary ops should run next epoch, logs scalar summaries.'
        logs = {'epoch_' + k: v for (k, v) in logs.items() if k not in ['batch', 'size', 'num_steps']}
        if self.update_freq == 'epoch':
            step = epoch
        else:
            step = self._samples_seen
        self._write_custom_summaries(step, logs)
        if self.histogram_freq:
            if self.merged in self.model.test_function.fetches:
                self.model.test_function.fetches.remove(self.merged)
            if self.merged in self.model.test_function.fetch_callbacks:
                self.model.test_function.fetch_callbacks.pop(self.merged)
        if self.embeddings_data is None and self.embeddings_freq:
            raise ValueError('To visualize embeddings, embeddings_data must be provided.')
        if self.embeddings_freq and self.embeddings_data is not None:
            if epoch % self.embeddings_freq == 0:
                embeddings_data = self.embeddings_data
                n_samples = embeddings_data[0].shape[0]
                i = 0
                sess = K.get_session()
                while i < n_samples:
                    step = min(self.batch_size, n_samples - i)
                    batch = slice(i, i + step)
                    if isinstance(self.model.input, list):
                        feed_dict = {model_input: embeddings_data[idx][batch] for (idx, model_input) in enumerate(self.model.input)}
                    else:
                        feed_dict = {self.model.input: embeddings_data[0][batch]}
                    feed_dict.update({self.batch_id: i, self.step: step})
                    if not isinstance(K.learning_phase(), int):
                        feed_dict[K.learning_phase()] = False
                    sess.run(self.assign_embeddings, feed_dict=feed_dict)
                    self.saver.save(sess, os.path.join(self.log_dir, 'keras_embedding.ckpt'), epoch)
                    i += self.batch_size

    def on_train_end(self, logs=None):
        if False:
            i = 10
            return i + 15
        self._stop_profiler()
        self.writer.close()

    def _start_profiler(self):
        if False:
            print('Hello World!')
        'Starts the profiler if currently inactive.'
        if self._profiler_started:
            return
        try:
            profiler.start(logdir=self.log_dir)
            self._profiler_started = True
        except errors.AlreadyExistsError as e:
            logging.error('Failed to start profiler: %s', e.message)

    def _stop_profiler(self):
        if False:
            print('Hello World!')
        'Stops the profiler if currently active.'
        if not self._profiler_started:
            return
        try:
            profiler.stop()
        except errors.UnavailableError as e:
            logging.error('Failed to stop profiler: %s', e.message)
        finally:
            self._profiler_started = False