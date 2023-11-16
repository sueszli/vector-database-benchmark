"""Custom training loop for running TensorFlow 2.0 models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, List, Text, Callable, Union, Iterator, Any
from official.modeling.hyperparams import params_dict
from official.utils.misc import tpu_lib
FLAGS = flags.FLAGS

def strategy_flags_dict():
    if False:
        return 10
    'Returns TPU related flags in a dictionary.'
    return {'tpu': FLAGS.tpu, 'worker_hosts': FLAGS.worker_hosts, 'task_index': FLAGS.task_index}

def hparam_flags_dict():
    if False:
        i = 10
        return i + 15
    'Returns model params related flags in a dictionary.'
    return {'data_dir': FLAGS.data_dir, 'model_dir': FLAGS.model_dir, 'train_batch_size': FLAGS.train_batch_size, 'eval_batch_size': FLAGS.eval_batch_size, 'precision': FLAGS.precision, 'config_file': FLAGS.config_file, 'params_override': FLAGS.params_override}

def _save_checkpoint(checkpoint, model_dir, checkpoint_prefix):
    if False:
        return 10
    'Saves model to model_dir with provided checkpoint prefix.'
    checkpoint_path = os.path.join(model_dir, checkpoint_prefix)
    saved_path = checkpoint.save(checkpoint_path)
    logging.info('Saving model as TF checkpoint: %s', saved_path)

def _steps_to_run(current_step, total_steps, steps_per_loop):
    if False:
        return 10
    'Calculates steps to run on device.'
    if steps_per_loop <= 0:
        raise ValueError('steps_per_loop should be positive integer.')
    return min(total_steps - current_step, steps_per_loop)

def _no_metric():
    if False:
        i = 10
        return i + 15
    return None

class SummaryWriter(object):
    """Simple SummaryWriter for writing dictionary of metrics.

  Attributes:
    _writer: The tf.SummaryWriter.
  """

    def __init__(self, model_dir: Text, name: Text):
        if False:
            print('Hello World!')
        'Inits SummaryWriter with paths.\n\n    Arguments:\n      model_dir: the model folder path.\n      name: the summary subfolder name.\n    '
        self._writer = tf.summary.create_file_writer(os.path.join(model_dir, name))

    def __call__(self, metrics: Union[Dict[Text, float], float], step: int):
        if False:
            for i in range(10):
                print('nop')
        'Write metrics to summary with the given writer.\n\n    Args:\n      metrics: a dictionary of metrics values. Prefer dictionary.\n      step: integer. The training step.\n    '
        if not isinstance(metrics, dict):
            logging.warning('Warning: summary writer prefer metrics as dictionary.')
            metrics = {'metric': metrics}
        with self._writer.as_default():
            for (k, v) in metrics.items():
                tf.summary.scalar(k, v, step=step)
            self._writer.flush()

class DistributedExecutor(object):
    """Interface to train and eval models with tf.distribute.Strategy.

  Arguments:
    strategy: an instance of tf.distribute.Strategy.
    params: Model configuration needed to run distribution strategy.
    model_fn: Keras model function. Signature:
      (params: ParamsDict) -> tf.keras.models.Model.
    loss_fn: loss function. Signature:
      (y_true: Tensor, y_pred: Tensor) -> Tensor
    metric_fn: metric function. Signature: () -> tf.keras.metrics.Metric.
    is_multi_host: Set to True when using multi hosts for training, like multi
      worker GPU or TPU pod (slice). Otherwise, False.
  """

    def __init__(self, strategy, params, model_fn, loss_fn, is_multi_host=False):
        if False:
            return 10
        self._params = params
        self._model_fn = model_fn
        self._loss_fn = loss_fn
        self._strategy = strategy
        self._checkpoint_name = 'ctl_step_{step}.ckpt'
        self._is_multi_host = is_multi_host

    @property
    def checkpoint_name(self):
        if False:
            return 10
        'Returns default checkpoint name.'
        return self._checkpoint_name

    @checkpoint_name.setter
    def checkpoint_name(self, name):
        if False:
            i = 10
            return i + 15
        'Sets default summary writer for the current thread.'
        self._checkpoint_name = name

    def loss_fn(self):
        if False:
            return 10
        return self._loss_fn()

    def model_fn(self, params):
        if False:
            for i in range(10):
                print('nop')
        return self._model_fn(params)

    def _save_config(self, model_dir):
        if False:
            i = 10
            return i + 15
        'Save parameters to config files if model_dir is defined.'
        logging.info('Save config to model_dir %s.', model_dir)
        if model_dir:
            if not tf.io.gfile.exists(model_dir):
                tf.io.gfile.makedirs(model_dir)
            self._params.lock()
            params_dict.save_params_dict_to_yaml(self._params, model_dir + '/params.yaml')
        else:
            logging.warning('model_dir is empty, so skip the save config.')

    def _get_input_iterator(self, input_fn: Callable[..., tf.data.Dataset], strategy: tf.distribute.Strategy) -> Optional[Iterator[Any]]:
        if False:
            i = 10
            return i + 15
        'Returns distributed dataset iterator.\n\n    Args:\n      input_fn: (params: dict) -> tf.data.Dataset.\n      strategy: an instance of tf.distribute.Strategy.\n\n    Returns:\n      An iterator that yields input tensors.\n    '
        if input_fn is None:
            return None
        if self._is_multi_host:
            return iter(strategy.experimental_distribute_datasets_from_function(input_fn))
        else:
            input_data = input_fn()
            return iter(strategy.experimental_distribute_dataset(input_data))

    def _create_replicated_step(self, strategy, model, loss_fn, optimizer, metric=None):
        if False:
            for i in range(10):
                print('nop')

        def _replicated_step(inputs):
            if False:
                i = 10
                return i + 15
            'Replicated training step.'
            (inputs, labels) = inputs
            with tf.GradientTape() as tape:
                outputs = model(inputs, training=True)
                prediction_loss = loss_fn(labels, outputs)
                loss = tf.reduce_mean(prediction_loss)
                loss = loss / strategy.num_replicas_in_sync
                if isinstance(metric, tf.keras.metrics.Metric):
                    metric.update_state(labels, outputs)
                else:
                    logging.error('train metric is not an instance of tf.keras.metrics.Metric.')
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss
        return _replicated_step

    def _create_train_step(self, strategy, model, loss_fn, optimizer, metric=None):
        if False:
            for i in range(10):
                print('nop')
        'Creates a distributed training step.\n\n      Args:\n        strategy: an instance of tf.distribute.Strategy.\n        model: (Tensor, bool) -> Tensor. model function.\n        loss_fn: (y_true: Tensor, y_pred: Tensor) -> Tensor.\n        optimizer: tf.keras.optimizers.Optimizer.\n        iterator: an iterator that yields input tensors.\n        metric: tf.keras.metrics.Metric subclass.\n\n      Returns:\n        The training step callable.\n    '
        _replicated_step = self._create_replicated_step(strategy, model, loss_fn, optimizer, metric)

        @tf.function
        def train_step(iterator, num_steps):
            if False:
                for i in range(10):
                    print('nop')
            'Performs a distributed training step.\n\n      Args:\n        iterator: an iterator that yields input tensors.\n\n      Returns:\n        The loss tensor.\n      '
            if not isinstance(num_steps, tf.Tensor):
                raise ValueError('steps should be an Tensor. Python object may cause retracing.')
            per_replica_losses = strategy.experimental_run_v2(_replicated_step, args=(next(iterator),))
            for _ in tf.range(num_steps - 1):
                per_replica_losses = strategy.experimental_run_v2(_replicated_step, args=(next(iterator),))
            loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
            return loss
        return train_step

    def _create_test_step(self, strategy, model, metric):
        if False:
            return 10
        'Creates a distributed test step.'

        @tf.function
        def test_step(iterator):
            if False:
                print('Hello World!')
            'Calculates evaluation metrics on distributed devices.'
            if not metric:
                logging.info('Skip test_step because metric is None (%s)', metric)
                return (None, None)
            if not isinstance(metric, tf.keras.metrics.Metric):
                raise ValueError('Metric must be an instance of tf.keras.metrics.Metric for running in test_step. Actual {}'.format(metric))

            def _test_step_fn(inputs):
                if False:
                    i = 10
                    return i + 15
                'Replicated accuracy calculation.'
                (inputs, labels) = inputs
                model_outputs = model(inputs, training=False)
                metric.update_state(labels, model_outputs)
                return (labels, model_outputs)
            return strategy.experimental_run_v2(_test_step_fn, args=(next(iterator),))
        return test_step

    def train(self, train_input_fn: Callable[[params_dict.ParamsDict], tf.data.Dataset], eval_input_fn: Callable[[params_dict.ParamsDict], tf.data.Dataset]=None, model_dir: Text=None, total_steps: int=1, iterations_per_loop: int=1, train_metric_fn: Callable[[], Any]=None, eval_metric_fn: Callable[[], Any]=None, summary_writer_fn: Callable[[Text, Text], SummaryWriter]=SummaryWriter, init_checkpoint: Callable[[tf.keras.Model], Any]=None, custom_callbacks: List[tf.keras.callbacks.Callback]=None, save_config: bool=True):
        if False:
            i = 10
            return i + 15
        'Runs distributed training.\n\n    Args:\n      train_input_fn: (params: dict) -> tf.data.Dataset training data input\n        function.\n      eval_input_fn: (Optional) same type as train_input_fn. If not None, will\n        trigger evaluting metric on eval data. If None, will not run eval step.\n      model_dir: the folder path for model checkpoints.\n      total_steps: total training steps.\n      iterations_per_loop: train steps per loop. After each loop, this job will\n        update metrics like loss and save checkpoint.\n      train_metric_fn: metric_fn for evaluation in train_step.\n      eval_metric_fn: metric_fn for evaluation in test_step.\n      summary_writer_fn: function to create summary writer.\n      init_checkpoint: function to load checkpoint.\n      custom_callbacks: A list of Keras Callbacks objects to run during\n        training. More specifically, `on_batch_begin()`, `on_batch_end()`,\n        methods are invoked during training.\n      save_config: bool. Whether to save params to model_dir.\n\n    Returns:\n      The training loss and eval metrics.\n    '
        assert train_input_fn is not None
        if train_metric_fn and (not callable(train_metric_fn)):
            raise ValueError('if `train_metric_fn` is specified, train_metric_fn must be a callable.')
        if eval_metric_fn and (not callable(eval_metric_fn)):
            raise ValueError('if `eval_metric_fn` is specified, eval_metric_fn must be a callable.')
        train_metric_fn = train_metric_fn or _no_metric
        eval_metric_fn = eval_metric_fn or _no_metric
        if custom_callbacks and iterations_per_loop != 1:
            logging.error('It is sematically wrong to run callbacks when iterations_per_loop is not one (%s)', iterations_per_loop)

        def _run_callbacks_on_batch_begin(batch):
            if False:
                return 10
            'Runs custom callbacks at the start of every step.'
            if not custom_callbacks:
                return
            for callback in custom_callbacks:
                if callback:
                    callback.on_batch_begin(batch)

        def _run_callbacks_on_batch_end(batch):
            if False:
                print('Hello World!')
            'Runs custom callbacks at the end of every step.'
            if not custom_callbacks:
                return
            for callback in custom_callbacks:
                if callback:
                    callback.on_batch_end(batch)
        if save_config:
            self._save_config(model_dir)
        if FLAGS.save_checkpoint_freq:
            save_freq = FLAGS.save_checkpoint_freq
        else:
            save_freq = iterations_per_loop
        params = self._params
        strategy = self._strategy
        train_iterator = self._get_input_iterator(train_input_fn, strategy)
        train_loss = None
        eval_metric_result = None
        with strategy.scope():
            model = self.model_fn(params.as_dict())
            if not hasattr(model, 'optimizer'):
                raise ValueError('User should set optimizer attribute to model inside `model_fn`.')
            optimizer = model.optimizer
            checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
            latest_checkpoint_file = tf.train.latest_checkpoint(model_dir)
            initial_step = 0
            if latest_checkpoint_file:
                logging.info('Checkpoint file %s found and restoring from checkpoint', latest_checkpoint_file)
                checkpoint.restore(latest_checkpoint_file)
                initial_step = optimizer.iterations.numpy()
                logging.info('Loading from checkpoint file completed. Init step %d', initial_step)
            elif init_checkpoint:
                logging.info('Restoring from init checkpoint function')
                init_checkpoint(model)
                logging.info('Loading from init checkpoint file completed')
            current_step = optimizer.iterations.numpy()
            checkpoint_name = self.checkpoint_name
            eval_metric = eval_metric_fn()
            train_metric = train_metric_fn()
            train_summary_writer = summary_writer_fn(model_dir, 'eval_train')
            test_summary_writer = summary_writer_fn(model_dir, 'eval_test')
        train_step = self._create_train_step(strategy=strategy, model=model, loss_fn=self.loss_fn(), optimizer=optimizer, metric=train_metric)
        test_step = None
        if eval_input_fn and eval_metric:
            test_step = self._create_test_step(strategy, model, metric=eval_metric)
        logging.info('Training started')
        last_save_checkpoint_step = current_step
        while current_step < total_steps:
            num_steps = _steps_to_run(current_step, total_steps, iterations_per_loop)
            _run_callbacks_on_batch_begin(current_step)
            train_loss = train_step(train_iterator, tf.convert_to_tensor(num_steps, dtype=tf.int32))
            _run_callbacks_on_batch_end(current_step)
            current_step += num_steps
            train_loss = tf.nest.map_structure(lambda x: x.numpy().astype(float), train_loss)
            if not isinstance(train_loss, dict):
                train_loss = {'total_loss': train_loss}
            if np.isnan(train_loss['total_loss']):
                raise ValueError('total loss is NaN.')
            if train_metric:
                train_metric_result = train_metric.result()
                if isinstance(train_metric, tf.keras.metrics.Metric):
                    train_metric_result = tf.nest.map_structure(lambda x: x.numpy().astype(float), train_metric_result)
                if not isinstance(train_metric_result, dict):
                    train_metric_result = {'metric': train_metric_result}
                train_metric_result.update(train_loss)
            else:
                train_metric_result = train_loss
            if callable(optimizer.lr):
                train_metric_result.update({'learning_rate': optimizer.lr(current_step).numpy()})
            else:
                train_metric_result.update({'learning_rate': optimizer.lr.numpy()})
            logging.info('Train Step: %d/%d  / loss = %s / training metric = %s', current_step, total_steps, train_loss, train_metric_result)
            train_summary_writer(metrics=train_metric_result, step=optimizer.iterations)
            if save_freq > 0 and current_step < total_steps and (current_step - last_save_checkpoint_step >= save_freq):
                _save_checkpoint(checkpoint, model_dir, checkpoint_name.format(step=current_step))
                last_save_checkpoint_step = current_step
            if test_step:
                eval_iterator = self._get_input_iterator(eval_input_fn, strategy)
                eval_metric_result = self._run_evaluation(test_step, current_step, eval_metric, eval_iterator)
                logging.info('Step: %s evalation metric = %s.', current_step, eval_metric_result)
                test_summary_writer(metrics=eval_metric_result, step=optimizer.iterations)
            if eval_metric and current_step < total_steps:
                eval_metric.reset_states()
            if train_metric and current_step < total_steps:
                train_metric.reset_states()
        if last_save_checkpoint_step < total_steps:
            _save_checkpoint(checkpoint, model_dir, checkpoint_name.format(step=current_step))
        if test_step:
            logging.info('Running final evaluation after training is complete.')
            eval_iterator = self._get_input_iterator(eval_input_fn, strategy)
            eval_metric_result = self._run_evaluation(test_step, current_step, eval_metric, eval_iterator)
            logging.info('Final evaluation metric = %s.', eval_metric_result)
            test_summary_writer(metrics=eval_metric_result, step=optimizer.iterations)
        return (train_loss, eval_metric_result)

    def _run_evaluation(self, test_step, current_training_step, metric, test_iterator):
        if False:
            print('Hello World!')
        'Runs validation steps and aggregate metrics.'
        if not test_iterator or not metric:
            logging.warning('Both test_iterator (%s) and metrics (%s) must not be None.', test_iterator, metric)
            return None
        logging.info('Running evaluation after step: %s.', current_training_step)
        while True:
            try:
                test_step(test_iterator)
            except (StopIteration, tf.errors.OutOfRangeError):
                break
        metric_result = metric.result()
        if isinstance(metric, tf.keras.metrics.Metric):
            metric_result = metric_result.numpy().astype(float)
        logging.info('Step: [%d] Validation metric = %f', current_training_step, metric_result)
        return metric_result

    def evaluate_from_model_dir(self, model_dir: Text, eval_input_fn: Callable[[params_dict.ParamsDict], tf.data.Dataset], eval_metric_fn: Callable[[], Any], total_steps: int=-1, eval_timeout: int=None, min_eval_interval: int=180, summary_writer_fn: Callable[[Text, Text], SummaryWriter]=SummaryWriter):
        if False:
            while True:
                i = 10
        'Runs distributed evaluation on model folder.\n\n    Args:\n      eval_input_fn: (Optional) same type as train_input_fn. If not None, will\n        trigger evaluting metric on eval data. If None, will not run eval step.\n      eval_metric_fn: metric_fn for evaluation in test_step.\n      model_dir: the folder for storing model checkpoints.\n      total_steps: total training steps. If the current step reaches the\n        total_steps, the evaluation loop will stop.\n      eval_timeout: The maximum number of seconds to wait between checkpoints.\n        If left as None, then the process will wait indefinitely. Used by\n        tf.train.checkpoints_iterator.\n      min_eval_interval: The minimum number of seconds between yielding\n        checkpoints. Used by tf.train.checkpoints_iterator.\n      summary_writer_fn: function to create summary writer.\n\n    Returns:\n      Eval metrics dictionary of the last checkpoint.\n    '
        if not model_dir:
            raise ValueError('model_dir must be set.')

        def terminate_eval():
            if False:
                i = 10
                return i + 15
            tf.logging.info('Terminating eval after %d seconds of no checkpoints' % eval_timeout)
            return True
        summary_writer = summary_writer_fn(model_dir, 'eval')
        for checkpoint_path in tf.train.checkpoints_iterator(model_dir, min_interval_secs=min_eval_interval, timeout=eval_timeout, timeout_fn=terminate_eval):
            (eval_metric_result, current_step) = self.evaluate_checkpoint(checkpoint_path=checkpoint_path, eval_input_fn=eval_input_fn, eval_metric_fn=eval_metric_fn, summary_writer=summary_writer)
            if total_steps > 0 and current_step >= total_steps:
                logging.info('Evaluation finished after training step %d', current_step)
                break
        return eval_metric_result

    def evaluate_checkpoint(self, checkpoint_path: Text, eval_input_fn: Callable[[params_dict.ParamsDict], tf.data.Dataset], eval_metric_fn: Callable[[], Any], summary_writer: SummaryWriter=None):
        if False:
            i = 10
            return i + 15
        'Runs distributed evaluation on the one checkpoint.\n\n    Args:\n      eval_input_fn: (Optional) same type as train_input_fn. If not None, will\n        trigger evaluting metric on eval data. If None, will not run eval step.\n      eval_metric_fn: metric_fn for evaluation in test_step.\n      checkpoint_path: the checkpoint to evaluate.\n      summary_writer_fn: function to create summary writer.\n\n    Returns:\n      Eval metrics dictionary of the last checkpoint.\n    '
        if not callable(eval_metric_fn):
            raise ValueError('if `eval_metric_fn` is specified, eval_metric_fn must be a callable.')
        params = self._params
        strategy = self._strategy
        with strategy.scope():
            model = self.model_fn(params.as_dict())
            checkpoint = tf.train.Checkpoint(model=model)
            eval_metric = eval_metric_fn()
            assert eval_metric, 'eval_metric does not exist'
            test_step = self._create_test_step(strategy, model, metric=eval_metric)
            logging.info('Starting to evaluate.')
            if not checkpoint_path:
                raise ValueError('checkpoint path is empty')
            reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
            current_step = reader.get_tensor('optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE')
            logging.info('Checkpoint file %s found and restoring from checkpoint', checkpoint_path)
            checkpoint.restore(checkpoint_path)
            eval_iterator = self._get_input_iterator(eval_input_fn, strategy)
            eval_metric_result = self._run_evaluation(test_step, current_step, eval_metric, eval_iterator)
            logging.info('Step: %s evalation metric = %s.', current_step, eval_metric_result)
            summary_writer(metrics=eval_metric_result, step=current_step)
            eval_metric.reset_states()
        return (eval_metric_result, current_step)

    def predict(self):
        if False:
            print('Hello World!')
        return NotImplementedError('Unimplmented function.')

class ExecutorBuilder(object):
    """Builder of DistributedExecutor.

  Example 1: Builds an executor with supported Strategy.
    builder = ExecutorBuilder(
        strategy_type='tpu',
        strategy_config={'tpu': '/bns/xxx'})
    dist_executor = builder.build_executor(
        params=params,
        model_fn=my_model_fn,
        loss_fn=my_loss_fn,
        metric_fn=my_metric_fn)

  Example 2: Builds an executor with customized Strategy.
    builder = ExecutorBuilder()
    builder.strategy = <some customized Strategy>
    dist_executor = builder.build_executor(
        params=params,
        model_fn=my_model_fn,
        loss_fn=my_loss_fn,
        metric_fn=my_metric_fn)

  Example 3: Builds a customized executor with customized Strategy.
    class MyDistributedExecutor(DistributedExecutor):
      # implementation ...

    builder = ExecutorBuilder()
    builder.strategy = <some customized Strategy>
    dist_executor = builder.build_executor(
        class_ctor=MyDistributedExecutor,
        params=params,
        model_fn=my_model_fn,
        loss_fn=my_loss_fn,
        metric_fn=my_metric_fn)

  Args:
    strategy_type: string. One of 'tpu', 'mirrored', 'multi_worker_mirrored'. If
      None. User is responsible to set the strategy before calling
      build_executor(...).
    strategy_config: necessary config for constructing the proper Strategy.
      Check strategy_flags_dict() for examples of the structure.
  """

    def __init__(self, strategy_type=None, strategy_config=None):
        if False:
            while True:
                i = 10
        self._strategy_config = strategy_config
        self._strategy = self._build_strategy(strategy_type)

    @property
    def strategy(self):
        if False:
            while True:
                i = 10
        'Returns default checkpoint name.'
        return self._strategy

    @strategy.setter
    def strategy(self, new_strategy):
        if False:
            i = 10
            return i + 15
        'Sets default summary writer for the current thread.'
        self._strategy = new_strategy

    def _build_strategy(self, strategy_type):
        if False:
            for i in range(10):
                print('nop')
        "Builds tf.distribute.Strategy instance.\n\n    Args:\n      strategy_type: string. One of 'tpu', 'one_device_gpu', 'mirrored', 'multi_worker_mirrored'.\n\n    Returns:\n      An tf.distribute.Strategy object. Returns None if strategy_type is None.\n    "
        if strategy_type is None:
            return None
        if strategy_type == 'tpu':
            return self._build_tpu_strategy()
        elif strategy_type == 'one_device_gpu':
            return tf.distribute.OneDeviceStrategy('device:GPU:0')
        elif strategy_type == 'mirrored':
            return self._build_mirrored_strategy()
        elif strategy_type == 'multi_worker_mirrored':
            return self._build_multiworker_mirrored_strategy()
        else:
            raise NotImplementedError('Unsupport accelerator type "%s"' % strategy_type)

    def _build_mirrored_strategy(self):
        if False:
            return 10
        'Builds a MirroredStrategy object.'
        return tf.distribute.MirroredStrategy()

    def _build_tpu_strategy(self):
        if False:
            while True:
                i = 10
        'Builds a TPUStrategy object.'
        tpu = self._strategy_config.tpu
        logging.info('Use TPU at %s', tpu if tpu is not None else '')
        cluster_resolver = tpu_lib.tpu_initialize(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
        return strategy

    def _build_multiworker_mirrored_strategy(self):
        if False:
            i = 10
            return i + 15
        'Builds a MultiWorkerMirroredStrategy object.'
        worker_hosts = self._strategy_config.worker_hosts
        if worker_hosts is not None:
            worker_hosts = worker_hosts.split(',')
            task_index = self._strategy_config.task_index
            os.environ['TF_CONFIG'] = json.dumps({'cluster': {'worker': worker_hosts}, 'task': {'type': 'worker', 'index': task_index}})
        multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        return multiworker_strategy

    def build_executor(self, class_ctor=DistributedExecutor, params=None, model_fn=None, loss_fn=None, **kwargs):
        if False:
            while True:
                i = 10
        'Creates an executor according to strategy type.\n\n    See doc string of the DistributedExecutor.__init__ for more information of\n    the\n    input arguments.\n\n    Args:\n      class_ctor: A constructor of executor (default: DistributedExecutor).\n      params: ParamsDict, all the model parameters and runtime parameters.\n      model_fn: Keras model function.\n      loss_fn: loss function.\n      **kwargs: other arguments to the executor constructor.\n\n    Returns:\n      An instance of DistributedExecutor or its subclass.\n    '
        if self._strategy is None:
            raise ValueError('`strategy` should not be None. You need to specify `strategy_type` in the builder contructor or directly set the `strategy` property of the builder.')
        return class_ctor(strategy=self._strategy, params=params, model_fn=model_fn, loss_fn=loss_fn, **kwargs)