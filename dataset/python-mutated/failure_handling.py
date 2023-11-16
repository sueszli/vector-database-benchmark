"""Module for `PreemptionCheckpointHandler`.

This is currently under development and the API is subject to change.

PreemptionCheckpointHandler reduces loss of training progress caused by
termination (preemption or maintenance) of workers in multi-worker synchronous
training and avoid surfacing an error indistinguishable from application errors
to the job scheduler or users.
"""
import os
import signal
import sys
import threading
import time
from tensorflow.core.distributed_runtime.preemption import gen_check_preemption_op
from tensorflow.python.checkpoint import checkpoint as checkpoint_lib
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute.failure_handling import failure_handling_util
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
_INITIAL_RUN_COUNT_KEY = 'RUN_TO_CHECKPOINT'
_FINAL_RUN_COUNT_KEY = 'LAST_RUN_TO_CHECKPOINT'
_PREEMPTION_WORKER_KEY = 'TERMINATED_WORKER'
_ACKNOWLEDGE_KEY = 'RECEIVED_SIGNAL'
_ITERATION_VARIABLE = 'checkpointed_runs'
_STOP_WATCHING_CLUSTER_VALUE = 'STOP_WATCHER'
PREEMPTION_KEY = 'TF_DEFAULT_PREEMPTION_NOTICE_KEY'

def _non_chief_checkpoint_dir(checkpoint_dir, task_id):
    if False:
        while True:
            i = 10
    'Returns a directory for non-chief worker to save checkpoint.'
    dirpath = os.path.dirname(checkpoint_dir)
    base = os.path.basename(checkpoint_dir)
    base_dirpath = 'workertemp_' + str(task_id)
    dirpath = os.path.join(dirpath, base_dirpath)
    file_io.recursive_create_dir_v2(dirpath)
    return os.path.join(dirpath, base)

@tf_export('distribute.experimental.TerminationConfig', v1=[])
class TerminationConfig(object):
    """Customization of `PreemptionCheckpointHandler` for various platforms.

  A `TerminationConfig` can be created and passed to a
  `tf.distribute.experimental.PreemptionCheckpointHandler` to provide
  customization based on the platform. It can deliver three pieces of
  information:

  * How to decide if there is a termination event soon

  The form of termination notification and how to fetch it vary across
  platforms. Thus `PreemptionCheckpointHandler` may take a user-defined
  function, `termination_watcher_fn`, and execute it repeatedly to check for
  termination notification. `termination_watcher_fn` should be a function
  that returns `True` if a termination notification is available and
  `False` otherwise. The function should be lightweight and non-blocking so that
  resources can be cleaned up properly if no termination signal is ever raised
  until training finishes.

  * How to exit the program

  A user can configure this through the `exit_fn`, which
  `PreemptionCheckpointHandler` executes after saving the checkpoint to exit the
  training program gracefully. For `tf.distribute.MultiWorkerMirroredStrategy`,
  a restart is necessary to reset the program's state. However, having a
  customized `exit_fn` may facilitate the restart and smoothen the training
  experience. How so? Maybe the platform has an agreement to a `RESTART_CODE`
  recognized as a program auto-restart signal, or maybe the user has a
  coordinating script that starts up the training, in which they can configure
  the program to auto-restart if it ever exits with this `RESTART_CODE`. In both
  cases, configuring the `exit_fn` to be `sys.exit(RESTART_CODE)` makes the
  training seamless.

  * How long does `PreemptionCheckpointHandler` have from receiving a
  termination event notice till the actual termination

  Some platforms have a gap time as long as one hour or so. In these cases,
  there is the option to utilize this gap time for training as much as possible
  before saving a checkpoint and exiting. This can be achieved by passing the
  `grace_period` argument a nonzero value. Note, for a user with a grace period
  that is not multiple times longer than their checkpoint writing time (e.g.,
  three times or more), we advise not to configure this argument, in which case
  `PreemptionCheckpointHandler` will directly save a checkpoint and exit.


  **The default behavior**:

  * For Google Borg Platform:
      * Automatically know how to detect preemption signal
      * Exit with a platform-recognized restart code
      * Save a checkpoint and exit immediately

  * For Google Cloud Platform:
      * Automatically know how to detect maintenance signal.
      * Exit with a code (User may configure this)
      * Automatically utilized the extended training period before save and exit

  * For Other platform:
      * If `termination_watcher_fn` is `None`, we will treat `signal.SIGTERM` as
      a termination signal.
      * If `exit_fn` is not configured, we exit the program with an arbitrary
      code.
      * If `grace_period` is not configured, we will wrap up the current
      training step, save a checkpoint, and exit the program as soon as we
      receive the termination signal.
  """

    def __init__(self, termination_watcher_fn=None, exit_fn=None, grace_period=None, save_fn=None):
        if False:
            return 10
        "Creates a `TerminationConfig` object.\n\n    Args:\n      termination_watcher_fn: a function to execute repeatedly that returns\n        `True` if a preemption signal is available and False otherwise. The\n        function cannot block until a preemption signal is available, which\n        prevents proper cleanup of the program. A change is **NOT** recommended\n        for users on Google Borg or Google Cloud Platform.\n      exit_fn: a function to execute after a checkpoint is saved and before the\n        preemption happens. Usually, it should be in the form of\n        `lambda: sys.exit(RESTART_CODE)`, where `RESTART_CODE` varies by\n        platform. A change is **NOT** recommended for users on Google Borg.\n        Users on Google Cloud Platform may configure it to use a customized\n        `RESTART_CODE`.\n      grace_period: the length of time between receiving a preemption signal and\n        the actual preemption. A change is **NOT** recommended for users on\n        Google Borg, Google Cloud Platform, or users with a short grace period.\n      save_fn: an optional function letting you configure how to save a\n        checkpoint. This is useful if you'd like to pass extra argument to\n        `tf.train.CheckpointManager.save` or `tf.train.Checkpoint.save`. By\n        default, if not configured, the API will save checkpoint without extra\n        arguments.\n    "
        self.termination_watcher_fn = termination_watcher_fn
        self.exit_fn = exit_fn
        self.grace_period = grace_period
        self.save_fn = save_fn

class GcpGpuTerminationConfig(TerminationConfig):
    """Configurations for GCP GPU VM."""

    def __init__(self, termination_watcher_fn=None, exit_fn=None, grace_period=None, save_fn=None):
        if False:
            while True:
                i = 10
        self.termination_watcher_fn = termination_watcher_fn or failure_handling_util.termination_watcher_function_gce
        self.exit_fn = exit_fn or failure_handling_util.gce_exit_fn
        self.grace_period = grace_period if grace_period or grace_period == 0 else failure_handling_util.GRACE_PERIOD_GCE
        self.save_fn = save_fn

class GcpCpuTerminationConfig(TerminationConfig):
    """Configurations for GCP CPU VM."""

    def __init__(self, termination_watcher_fn=None, exit_fn=None, grace_period=None, save_fn=None):
        if False:
            print('Hello World!')
        self.termination_watcher_fn = termination_watcher_fn or failure_handling_util.termination_watcher_function_gce
        self.exit_fn = exit_fn or failure_handling_util.gce_exit_fn
        self.grace_period = grace_period or 0
        self.save_fn = save_fn

class BorgTerminationConfig(TerminationConfig):
    """Configurations for Borg."""

    def __init__(self, termination_watcher_fn=None, exit_fn=None, grace_period=None, save_fn=None):
        if False:
            while True:
                i = 10
        self.termination_watcher_fn = termination_watcher_fn
        default_exit_fn = lambda : sys.exit(42)
        self.exit_fn = exit_fn or default_exit_fn
        self.grace_period = grace_period or 0
        self.save_fn = save_fn

class BorgTPUTerminationConfig(TerminationConfig):
    """Configurations for Borg."""

    def __init__(self, termination_watcher_fn=None, exit_fn=None, grace_period=None, save_fn=None):
        if False:
            return 10
        self.termination_watcher_fn = termination_watcher_fn
        self.exit_fn = exit_fn or failure_handling_util.default_tpu_exit_fn
        self.grace_period = grace_period or 0
        self.save_fn = save_fn

def _complete_config_for_environment(platform_device, termination_config):
    if False:
        return 10
    'Complete un-filled fields of TerminationConfig based on platform.'
    if not termination_config:
        termination_config = TerminationConfig()
    if platform_device is failure_handling_util.PlatformDevice.GCE_GPU:
        return GcpGpuTerminationConfig(termination_config.termination_watcher_fn, termination_config.exit_fn, termination_config.grace_period, termination_config.save_fn)
    elif platform_device is failure_handling_util.PlatformDevice.GCE_CPU:
        return GcpCpuTerminationConfig(termination_config.termination_watcher_fn, termination_config.exit_fn, termination_config.grace_period, termination_config.save_fn)
    elif platform_device is failure_handling_util.PlatformDevice.INTERNAL_TPU:
        return BorgTPUTerminationConfig(termination_config.termination_watcher_fn, termination_config.exit_fn, termination_config.grace_period, termination_config.save_fn)
    else:
        return BorgTerminationConfig(termination_config.termination_watcher_fn, termination_config.exit_fn, termination_config.grace_period, termination_config.save_fn)

@tf_export('distribute.experimental.PreemptionCheckpointHandler', v1=[])
class PreemptionCheckpointHandler(object):
    """Preemption and error handler for synchronous training.

  Note: This API only supports use with
  `tf.distribute.MultiWorkerMirroredStrategy` and `tf.distribute.TPUStrategy`.

  A `PreemptionCheckpointHandler` coordinates all workers to save a checkpoint
  upon receiving a preemption signal. It also helps disseminate application
  error messages accurately among the cluster. When a
  `PreemptionCheckpointHandler` object is created, it restores values from
  the latest checkpoint file if any exists.

  Right after the initialization, the object starts to watch out for termination
  signal for any member in the cluster. If receiving a signal, the next time the
  worker executes `PreemptionCheckpointHandler.run`, the
  `PreemptionCheckpointHandler` will align all workers to save a checkpoint.
  Then, if an `exit_fn` is configured via
  `tf.distribute.experimental.TerminationConfig`, it will be invoked. Otherwise,
  the process will simply exit and later the platform should restart it.

  Note: We advise users of `tf.distribute.MultiWorkerMirroredStrategy` who
  choose to configure their
  own `exit_fn` in `tf.distribute.experimental.TerminationConfig` to include a
  `sys.exit(CODE_OR_MESSAGE)` in the `exit_fn` so that after the restart, all
  workers can initialize communication services correctly. For users of
  `tf.distribute.TPUStrategy`, if they do not wish to do a cluster restart but
  would like an in-process restart (i.e., keep the coordinator alive and re-do
  the steps to connect to cluster, initialize TPU system, and make the
  `TPUStrategy` object), they could configure the `exit_fn` to a no-op.

  For users of `tf.distribute.MultiWorkerMirroredStrategy`, the core API is
  `PreemptionCheckpointHandler.run`:

  ```python
  strategy = tf.distribute.MultiWorkerMirroredStrategy()

  trained_epoch = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='epoch')
  step_in_epoch = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='step_in_epoch')

  with strategy.scope():
    dataset, model, optimizer = ...

    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=model,
                                     trained_epoch=trained_epoch,
                                     step_in_epoch=step_in_epoch)

    preemption_checkpoint_handler = tf.distribute.experimental.PreemptionCheckpointHandler(cluster_resolver, checkpoint, checkpoint_dir)

  while trained_epoch.numpy() < NUM_EPOCH:

    while step_in_epoch.numpy() < STEPS_PER_EPOCH:

      # distributed_train_function contains a call to strategy.run.
      loss += preemption_checkpoint_handler.run(distributed_train_function, args=(next(iterator),))
      # For users of MultiWorkerMirroredStrategy, usually
      # STEPS_PER_TRAIN_FUNCTION = 1.
      step_in_epoch.assign_add(STEPS_PER_TRAIN_FUNCTION)
      ...

    epoch.assign_add(1)
    step_in_epoch.assign(0)
  ```

  For users of `tf.distribute.TPUStrategy`, the core APIs are
  `PreemptionCheckpointHandler.run` and
  `PreemptionCheckpointHandler.watch_preemption_scope`:

  ```python

  strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)

  # Rest of TPU init omitted, see documentation for TPUSTrategy.

  with preemption_checkpoint_handler.watch_preemption_scope():
    while trained_epoch.numpy() < NUM_EPOCH:

      while step_in_epoch.numpy() < STEPS_PER_EPOCH:

        # distributed_train_function contains a call to strategy.run.
        loss += preemption_checkpoint_handler.run(distributed_train_function, args=(next(iterator),))

        # For users of TPUStrategy, usually STEPS_PER_TRAIN_FUNCTION >> 1 since
        # clustering multiple steps within a tf.function amortizes the overhead
        # of launching a multi-device function on TPU Pod.
        step_in_epoch.assign_add(STEPS_PER_TRAIN_FUNCTION)
        ...

      epoch.assign_add(1)
      step_in_epoch.assign(0)
  ```

  Not all interruptions come with advance notice so that the
  `PreemptionCheckpointHandler` can handle them, e.g., those caused by hardware
  failure. For a user who saves checkpoints for these cases themselves outside
  the `PreemptionCheckpointHandler`, if they are using a
  `tf.train.CheckpointManager`, pass it as the
  `checkpoint_or_checkpoint_manager` argument to the
  `PreemptionCheckpointHandler`. If they do not have a
  `tf.train.CheckpointManager` but are directly working with
  `tf.train.Checkpoint`, we advise saving the checkpoints in the directory
  that's passed as the `checkpoint_dir` argument. In this way, at the program
  beginning, `PreemptionCheckpointHandler` can restore the latest checkpoint
  from the directory, no matter it's saved by the user themselves or saved by
  the `PreemptionCheckpointHandler` before preemption happens.

  **A note on the platform:**

  `PreemptionCheckpointHandler` can only handle the kind of termination with
  advance notice. For now, the API recognizes the termination signal for CPU,
  GPU, and TPU on Google Borg and CPU and GPU on the Google Cloud Platform. In
  these cases, `PreemptionCheckpointHandler` will automatically adopt the
  correct preemption/maintenance notification detection mechanism. Users of
  other platforms can configure a detection monitoring behavior through the
  `tf.distribute.experimental.TerminationConfig`. Customization for the exit
  behavior and grace period length could also be done here.
  """

    def __init__(self, cluster_resolver, checkpoint_or_checkpoint_manager, checkpoint_dir=None, termination_config=None):
        if False:
            while True:
                i = 10
        'Creates the `PreemptionCheckpointHandler`.\n\n    Args:\n      cluster_resolver: a `tf.distribute.cluster_resolver.ClusterResolver`\n        object. You may also obtain it through the `cluster_resolver` attribute\n        of the distribution strategy in use.\n      checkpoint_or_checkpoint_manager: a `tf.train.CheckpointManager` or a\n        `tf.train.Checkpoint`. If you are using a `tf.train.CheckpointManager`\n        to manage checkpoints outside the `PreemptionCheckpointHandler` for\n        backup purpose as well, pass it as `checkpoint_or_checkpoint_manager`\n        argument. Otherwise, pass a `tf.train.Checkpoint` and the\n        `PreemptionCheckpointHandler` will create\n        a `tf.train.CheckpointManager` to manage it in the `checkpoint_dir`.\n      checkpoint_dir: a directory where the `PreemptionCheckpointHandler` saves\n        and restores checkpoints. When a `PreemptionCheckpointHandler` is\n        created, the latest checkpoint in the `checkpoint_dir` will be restored.\n        (This is not needed if a `tf.train.CheckpointManager` instead of a\n        `tf.train.Checkpoint` is passed as the\n        `checkpoint_or_checkpoint_manager` argument.)\n      termination_config: optional, a\n        `tf.distribute.experimental.TerminationConfig` object to configure for a\n        platform other than Google Borg or GCP.\n    '
        if isinstance(checkpoint_or_checkpoint_manager, checkpoint_lib.Checkpoint) and (not checkpoint_dir):
            raise errors.InvalidArgumentError('When a checkpoint is passed, a checkpoint_dir must be passed as well.')
        self._cluster_resolver = cluster_resolver
        self._termination_config = termination_config
        self._checkpoint_or_checkpoint_manager = checkpoint_or_checkpoint_manager
        self._checkpoint_dir = checkpoint_dir
        self._platform_device = failure_handling_util.detect_platform()
        completed_termination_config = _complete_config_for_environment(self._platform_device, self._termination_config)
        self._termination_watcher_fn = completed_termination_config.termination_watcher_fn
        self._exit_fn = completed_termination_config.exit_fn
        self._grace_period = completed_termination_config.grace_period
        self._save_fn = completed_termination_config.save_fn
        self._local_mode = True
        if self._platform_device in (failure_handling_util.PlatformDevice.GCE_TPU, failure_handling_util.PlatformDevice.GCE_CPU):
            logging.warning('PreemptionCheckpointHandler does not support usage with TPU or CPU device on GCP.')
        elif self._platform_device == failure_handling_util.PlatformDevice.INTERNAL_TPU:
            self._initialize_for_tpu_strategy()
        else:
            if cluster_resolver and 'ps' in cluster_resolver.cluster_spec().as_dict():
                raise NotImplementedError('PreemptionCheckpointHandler does not supportusage with tf.distribute.experimental.ParameterServerStrategy.')
            self._initialize_for_mirrored_and_multi_worker_mirrored()
        logging.info('PreemptionCheckpointHandler initialized or restored.')

    def _initialize_for_tpu_strategy(self):
        if False:
            return 10
        'Makes configurations for using the handler with TPUStrategy.'
        self._is_chief = True
        self._poll_termination_signal_thread = None
        self._cluster_wise_termination_watcher_thread = None
        self._maybe_create_checkpoint_manager()
        self._read_checkpoint_manager.restore_or_initialize()
        self._run_counter = 0

    def _initialize_for_mirrored_and_multi_worker_mirrored(self):
        if False:
            for i in range(10):
                print('nop')
        'Makes configurations and start watchers for MS, MWMS, or OneDevice.'
        if not self._cluster_resolver or not self._cluster_resolver.cluster_spec().jobs:
            self._local_mode = True
            self._id_in_cluster = 'single_worker'
            self._is_chief = True
        else:
            self._local_mode = False
            self._id_in_cluster = str(multi_worker_util.id_in_cluster(self._cluster_resolver.cluster_spec(), self._cluster_resolver.task_type, self._cluster_resolver.task_id))
            self._is_chief = multi_worker_util.is_chief(cluster_spec=self._cluster_resolver.cluster_spec(), task_type=self._cluster_resolver.task_type, task_id=self._cluster_resolver.task_id)
        self._checkpointed_runs = variables.Variable(initial_value=constant_op.constant(0, dtype=dtypes.int64), trainable=False, name=_ITERATION_VARIABLE)
        self._maybe_create_checkpoint_manager()
        if not hasattr(self._write_checkpoint_manager._checkpoint, _ITERATION_VARIABLE):
            setattr(self._write_checkpoint_manager._checkpoint, _ITERATION_VARIABLE, self._checkpointed_runs)
        if not hasattr(self._read_checkpoint_manager._checkpoint, _ITERATION_VARIABLE):
            setattr(self._read_checkpoint_manager._checkpoint, _ITERATION_VARIABLE, self._checkpointed_runs)
        self._read_checkpoint_manager.restore_or_initialize()
        self._final_checkpoint_countdown = False
        self._estimated_run_time = 0
        self._run_counter = self._checkpointed_runs.numpy()
        self._received_own_sigterm = threading.Event()
        self._received_checkpoint_step = threading.Event()
        distribute_lib.distribution_strategy_input_api_counter.get_cell(self._platform_device.name, 'PreemptionCheckpointHandler').increase_by(1)
        if not self._local_mode:
            self._cluster_wise_termination_watcher_thread = threading.Thread(target=self._watch_step_to_save_key, name='PeerTerminationWatcher-%s' % self._id_in_cluster, daemon=True)
            logging.info("Start watcher for peer's signal.")
            self._cluster_wise_termination_watcher_thread.start()
        else:
            self._cluster_wise_termination_watcher_thread = None
        self._poll_termination_signal_thread = None
        if self._termination_watcher_fn:
            self._start_polling_for_termination_signal()
        else:
            self._start_watching_for_signal()

    def _maybe_create_checkpoint_manager(self):
        if False:
            print('Hello World!')
        'Create CheckpointManager(s) if a checkpoint is passed else take it.'
        if isinstance(self._checkpoint_or_checkpoint_manager, checkpoint_management.CheckpointManager):
            self._read_checkpoint_manager = self._checkpoint_or_checkpoint_manager
            self._write_checkpoint_manager = self._checkpoint_or_checkpoint_manager
            self._api_made_checkpoint_manager = False
        else:
            self._api_made_checkpoint_manager = True
            self._read_checkpoint_manager = checkpoint_management.CheckpointManager(self._checkpoint_or_checkpoint_manager, directory=self._checkpoint_dir, max_to_keep=1)
            if self._is_chief:
                self._write_checkpoint_manager = self._read_checkpoint_manager
            else:
                self._write_checkpoint_manager = checkpoint_management.CheckpointManager(self._checkpoint_or_checkpoint_manager, _non_chief_checkpoint_dir(self._checkpoint_dir, self._cluster_resolver.task_id), max_to_keep=1)

    def _start_watching_for_signal(self):
        if False:
            i = 10
            return i + 15
        logging.info('Start watcher for local signal.')
        signal.signal(signal.SIGTERM, self._sigterm_handler_fn)

    def _start_polling_for_termination_signal(self):
        if False:
            for i in range(10):
                print('nop')
        self._poll_termination_signal_thread_should_stop = threading.Event()
        self._poll_termination_signal_thread = threading.Thread(target=self._poll_termination_signal, name='WorkerTerminationSignalWatcher-%s' % self._id_in_cluster, daemon=True)
        logging.info('Start polling for termination signal.')
        self._poll_termination_signal_thread.start()

    def _poll_termination_signal(self):
        if False:
            i = 10
            return i + 15
        'Poll maintenance notice and notify peers if receiving one.'
        while True:
            if self._poll_termination_signal_thread_should_stop.is_set() or self._final_checkpoint_countdown:
                return
            if self._termination_watcher_fn():
                break
            time.sleep(1)
        self._maybe_set_received_own_sigterm()

    def _maybe_set_received_own_sigterm(self):
        if False:
            while True:
                i = 10
        'Claim earliest preemption if no one else has done it before.'
        if self._local_mode:
            logging.info('Member %s has received termination notice.', self._id_in_cluster)
            self._received_own_sigterm_time = time.time()
            self._received_own_sigterm.set()
            return
        try:
            context.context().set_config_key_value(_PREEMPTION_WORKER_KEY, self._id_in_cluster)
            logging.info('Member %s has received termination notice.', self._id_in_cluster)
            self._received_own_sigterm_time = time.time()
            self._received_own_sigterm.set()
        except errors.AlreadyExistsError:
            logging.info('Member %s has received termination notice. But some other worker has received it as well! Leaving it to them to decide when to checkpoint. ', self._id_in_cluster)
            return

    def _stop_poll_termination_signal_thread(self):
        if False:
            return 10
        if getattr(self, '_poll_termination_signal_thread', None):
            self._poll_termination_signal_thread_should_stop.set()
            self._poll_termination_signal_thread.join()
            self._poll_termination_signal_thread = None
            logging.info("Shut down watcher for one's own termination signal")

    def _stop_cluster_wise_termination_watcher_thread(self):
        if False:
            while True:
                i = 10
        'Stop the thread that is _watch_step_to_save_key.'
        if getattr(self, '_cluster_wise_termination_watcher_thread', None):
            try:
                context.context().set_config_key_value(_INITIAL_RUN_COUNT_KEY, _STOP_WATCHING_CLUSTER_VALUE)
            except (errors.AlreadyExistsError, errors.UnavailableError):
                pass
            except Exception as e:
                logging.info('Ignoring error when shutting down _stop_cluster_wise_termination_watcher_thread: ' + str(e))
            try:
                context.context().set_config_key_value(_FINAL_RUN_COUNT_KEY, _STOP_WATCHING_CLUSTER_VALUE)
            except (errors.AlreadyExistsError, errors.UnavailableError):
                pass
            except Exception as e:
                logging.info('Ignoring error when shutting down _stop_cluster_wise_termination_watcher_thread: ' + str(e))
            finally:
                self._cluster_wise_termination_watcher_thread.join()
                self._cluster_wise_termination_watcher_thread = None
                logging.info("Shut down watcher for peer's termination signal.")

    def __del__(self):
        if False:
            while True:
                i = 10
        self._stop_cluster_wise_termination_watcher_thread()
        self._stop_poll_termination_signal_thread()

    @property
    @deprecated(None, 'Track steps using a tf.Variable saved in checkpoint instead.')
    @doc_controls.do_not_generate_docs
    def total_run_calls(self):
        if False:
            print('Hello World!')
        'Returns the number of times `PreemptionCheckpointHandler.run` is called.\n\n    DEPRECATED: user should track total steps themselves, as this API provides\n    little expressivity gain but could easily be misused and incurs extra\n    synchronization cost for TPUStrategy users.\n\n    This value tracks the number of all calls to\n    `PreemptionCheckpointHandler.run` including those before the program is\n    restarted and the training is restored, by saving and reading the value in\n    the checkpoint. A user can compute their total number of iterations\n    by `PreemptionCheckpointHandler.total_run_calls *\n    number_of_steps_in_train_function`,\n    while `number_of_steps_in_train_function` should be one for\n    `tf.distribute.MultiWorkerMirroredStrategy` users. They can also use this\n    value to infer the starting epoch and step after training restores, as shown\n    in the example above.\n    '
        if self._platform_device == failure_handling_util.PlatformDevice.INTERNAL_TPU:
            raise NotImplementedError('Please create variables saved in checkpoint to keep track of steps and epochs.')
        return self._run_counter

    def run(self, distributed_train_function, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "Runs a training function with error and preemption handling.\n\n    This function handles the preemption signal from any peer in the cluster by\n    saving the training progress and exiting gracefully. It will\n    also broadcase any program error encountered during the execution of\n    `distributed_train_function` to all workers so that they can raise the same\n    error.\n\n    The `distributed_train_function` argument should be a distributed train\n    function (i.e., containing a call to `tf.distribute.Strategy.run`). For\n    `tf.distribute.MultiWorkerMirroredStrategy` users, we recommend passing in a\n    single-step `distributed_train_function` to\n    `PreemptionCheckpointHandler.run` so that the checkpoint can be saved in\n    time in case a preemption signal or maintenance notice is sent.\n\n    Besides the preemption and error handling part,\n    `PreemptionCheckpointHandler.run(distributed_train_function, *args,\n    **kwargs)` has the same effect and output as\n    `distributed_train_function(*args, **kwargs)`. `distributed_train_function`\n    can return either some or no result. The following is a shortened example:\n\n    ```python\n\n    @tf.function\n    def distributed_train_step(iterator):\n      # A distributed single-step training function.\n\n      def step_fn(inputs):\n        # A per-replica single-step training function.\n        x, y = inputs\n        ...\n        return loss\n\n      per_replica_losses = strategy.run(step_fn, args=(next(iterator),))\n      return strategy.reduce(\n          tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)\n\n    for epoch in range(preemption_handler.total_run_calls // STEPS_PER_EPOCH,\n                       EPOCHS_TO_RUN):\n      iterator = iter(multi_worker_dataset)\n      total_loss = 0.0\n      num_batches = 0\n\n      for step in range(preemption_handler.total_run_calls % STEPS_PER_EPOCH,\n                        STEPS_PER_EPOCH):\n        total_loss += preemption_handler.run(distributed_train_step)\n        num_batches += 1\n\n      train_loss = total_loss / num_batches\n      print('Epoch: %d, train_loss: %f.' %(epoch.numpy(), train_loss))\n\n      train_accuracy.reset_states()\n    ```\n\n    Args:\n      distributed_train_function: A (single-step) distributed training function.\n      *args: args for `distributed_train_function`.\n      **kwargs: kwargs for `distributed_train_function`.\n\n    Raises:\n      Program error encountered by any member in the cluster while executing the\n      `distributed_train_function`, or any error from the program error\n      propagation process.\n\n    Returns:\n      Result of running the `distributed_train_function`.\n    "
        if self._platform_device == failure_handling_util.PlatformDevice.INTERNAL_TPU:
            return self._run_for_tpu(distributed_train_function, *args, **kwargs)
        elif self._platform_device in (failure_handling_util.PlatformDevice.GCE_TPU, failure_handling_util.PlatformDevice.GCE_CPU):
            return distributed_train_function(*args, **kwargs)
        else:
            return self._run_for_multi_worker_mirrored(distributed_train_function, *args, **kwargs)

    def _run_for_tpu(self, distributed_train_function, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'PreemptionCheckpointHandler.run implementation for TPUStrategy.'
        gen_check_preemption_op.check_preemption(preemption_key=PREEMPTION_KEY)
        return distributed_train_function(*args, **kwargs)

    def _run_for_multi_worker_mirrored(self, distributed_train_function, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'PreemptionCheckpointHandler.run implementation for MWMS.'
        try:
            self._check_preemption_and_maybe_checkpoint()
            run_begin_time = time.time()
            result = distributed_train_function(*args, **kwargs)
            new_run_time = time.time() - run_begin_time
            self._run_counter += 1
            self._estimated_run_time = self._estimated_run_time + (new_run_time - self._estimated_run_time) / self._run_counter
        except errors.OpError as e:
            if not self._local_mode:
                logging.info('Propagating error to cluster: %r: %s', e, e)
                try:
                    context.context().report_error_to_cluster(e.error_code, e.message)
                except Exception as ex:
                    logging.info('Ignoring error during error propagation: %r:%s', ex, ex)
            raise
        return result

    def save_checkpoint_if_preempted(self, *args, **kwargs):
        if False:
            print('Hello World!')
        "Saves a checkpoint if a preemption signal has been made available.\n\n    This is an alternative API for `PreemptionCheckpointHandler.run` and\n    `PreemptionCheckpointHandler.watch_preemption_scope`. This method works for\n    both `tf.distribute.MultiWorkerMirroredStrategy` and\n    `tf.distribute.TPUStrategy`. However, **for TPUStrategy, this method will\n    add a synchronization point between workers and the coordinator** and thus\n    may have performance implication. If this is a concern, use the combination\n    of `PreemptionCheckpointHandler.watch_preemption_scope` and\n    `PreemptionCheckpointHandler.run` instead.\n\n    ```python\n    strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)\n    # initialization omitted\n\n    with strategy.scope():\n      # Save in the checkpoint.\n      trained_step = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='trained_step', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)\n\n      checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory, max_to_keep=1)\n      preemption_handler = tf.distribute.experimental.PreemptionCheckpointHandler(cluster_resolver, checkpoint_manager)\n\n    while trained_step.numpy() < NUM_STEPS:\n      # Train STEPS_IN_FUNCTION steps at once.\n      train_multi_step_function()\n      trained_step.assign_add(STEPS_IN_FUNCTION)\n      preemption_handler.save_checkpoint_if_preempted()\n    ```\n\n    Args:\n      *args: args for `tf.train.CheckpointManager.save()` to save checkpoint.\n      **kwargs: kwargs for `tf.train.CheckpointManager.save()` to save.\n    "
        if self._platform_device == failure_handling_util.PlatformDevice.INTERNAL_TPU:
            try:
                with context.async_scope():
                    gen_check_preemption_op.check_preemption(preemption_key=PREEMPTION_KEY)
            except errors.AbortedError as abort_error:
                if abort_error.experimental_payloads.get(b'type.googleapis.com/tensorflow.distributed_runtime.WorkerPreemption'):
                    logging.info('Clearing preemption error to save checkpoint...')
                    context.async_clear_error()
                    self._save_checkpoint(*args, **kwargs)
                    self._exit_fn()
                else:
                    raise
        elif self._platform_device in (failure_handling_util.PlatformDevice.GCE_TPU, failure_handling_util.PlatformDevice.GCE_CPU):
            return
        else:
            self._check_preemption_and_maybe_checkpoint(*args, **kwargs)
            self._run_counter += 1
            self._estimated_run_time = 0

    @tf_contextlib.contextmanager
    def watch_preemption_scope(self):
        if False:
            i = 10
            return i + 15
        'Syncs error and maybe save checkpoint for usage with TPUStrategy.\n\n    Note: Usage with `tf.distribute.MultiWorkerMirroredStrategy` does not need\n    this API.\n\n    Example usage:\n\n    ```python\n    with preemption_checkpoint_handler.watch_preemption_scope():\n      while trained_step.numpy() < NUM_STEPS:\n\n        # distributed_train_function contains a call to strategy.run.\n        loss += preemption_checkpoint_handler.run(distributed_train_function, args=(next(iterator),))\n        trained_step.assign_add(STEPS_PER_TRAIN_FUNCTION)\n    ```\n\n    In this workflow, `PreemptionCheckpointHandler.run` will flag preemption\n    signal received, and `watch_preemption_scope` will handle the preemption\n    signal by saving a checkpoint and then either exit to restart or execute a\n    user-passed `exit_fn` in `tf.distribute.experimental.TerminationConfig`. If\n    no preemption signal is received during execution of ops and function inside\n    the scope, `watch_preemption_scope` ensures the completion of all async op\n    and function execution when exiting and will raises exceptions if async\n    execution results in an error state.\n\n    Yields:\n      None\n    '
        if self._platform_device == failure_handling_util.PlatformDevice.INTERNAL_TPU:
            try:
                with context.async_scope():
                    yield
            except errors.AbortedError as abort_error:
                if abort_error.experimental_payloads.get(b'type.googleapis.com/tensorflow.distributed_runtime.WorkerPreemption'):
                    logging.info('Clearing preemption error to save checkpoint...')
                    context.async_clear_error()
                    self._save_checkpoint()
                    self._exit_fn()
                else:
                    raise
        else:
            try:
                yield
            except errors.OpError as e:
                if not self._local_mode:
                    logging.info('Propagating error to cluster: %r: %s', e, e)
                    try:
                        context.context().report_error_to_cluster(e.error_code, e.message)
                    except Exception as ex:
                        logging.info('Ignoring error during error propagation: %r:%s', ex, ex)
                raise

    def _save_checkpoint(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Saves the checkpoint and exit program.'
        distribute_lib.distribution_strategy_input_api_counter.get_cell(self._platform_device.name, 'PreemptionCheckpointHandler Saving Checkpoint').increase_by(1)
        logging.info('PreemptionCheckpointHandler: Starting saving a checkpoint.')
        if self._platform_device != failure_handling_util.PlatformDevice.INTERNAL_TPU:
            self._checkpointed_runs.assign(self.total_run_calls)
        start_time = time.monotonic()
        with checkpoint_context.preemption_save_context():
            if self._save_fn:
                self._save_fn(*args, **kwargs)
            else:
                self._write_checkpoint_manager.save(*args, **kwargs)
        end_time = time.monotonic()
        logging.info('Checkpoint finished at path %s', self._write_checkpoint_manager.directory)
        self._checkpoint_time = end_time - start_time

    def _check_preemption_and_maybe_checkpoint(self, *args, **kwargs):
        if False:
            return 10
        'Checkpoint if any worker has received a preemption signal.\n\n    This function handles preemption signal reported by any worker in the\n    cluster. The current implementation relies on the fact that all workers in a\n    MultiWorkerMirroredStrategy training cluster have a step number difference\n    maximum of 1.\n    - If the signal comes from the worker itself (i.e., where this failure\n    handler sits), the worker will notify all peers to checkpoint after they\n    finish CURRENT_STEP+1 steps, where CURRENT_STEP is the step this worker has\n    just finished. And the worker will wait for all peers to acknowledge that\n    they have received its preemption signal and the final-step number before\n    the worker proceeds on training the final step.\n    - If the signal comes from another member in the cluster but NO final-step\n    info is available, proceed on training, because it will be available after\n    finishing the next step.\n    - If the signal comes from some other member in the cluster, and final-step\n    info is available, if the worker has not finished these steps yet, keep\n    training; otherwise, checkpoint and exit with a cluster-recognized restart\n    code.\n\n    Args:\n      *args: args for `tf.train.CheckpointManager.save()` to save checkpoint.\n      **kwargs: kwargs for `tf.train.CheckpointManager.save()` to save.\n    '
        if self._platform_device == failure_handling_util.PlatformDevice.INTERNAL_TPU:
            gen_check_preemption_op.check_preemption(preemption_key=PREEMPTION_KEY)
            return
        if self._final_checkpoint_countdown:
            run_count_config_key = _FINAL_RUN_COUNT_KEY
        else:
            run_count_config_key = _INITIAL_RUN_COUNT_KEY
        if self._received_checkpoint_step.is_set():
            if self._step_to_checkpoint == str(self._run_counter):
                self._save_checkpoint(*args, **kwargs)
                if self._time_to_exit():
                    self._stop_poll_termination_signal_thread()
                    self._stop_cluster_wise_termination_watcher_thread()
                    if self._api_made_checkpoint_manager and (not self._is_chief):
                        gfile.DeleteRecursively(os.path.dirname(self._write_checkpoint_manager.directory))
                    logging.info('PreemptionCheckpointHandler: checkpoint saved. Exiting.')
                    self._exit_fn()
                else:
                    logging.info('Continue training for the grace period.')
                    self._final_checkpoint_countdown = True
                    self._received_checkpoint_step.clear()
        elif self._received_own_sigterm.is_set():
            if self._final_checkpoint_countdown:
                if self._target_time_for_termination < time.time():
                    logging.info('Grace period almost ended. Final call to save a checkpoint!')
                else:
                    return
            step_to_save_at = str(self._run_counter + 1)
            logging.info('Termination caught in main thread on preempted worker')
            if self._local_mode:
                self._step_to_checkpoint = step_to_save_at
                self._received_checkpoint_step.set()
            else:
                context.context().set_config_key_value(run_count_config_key, step_to_save_at)
                logging.info('%s set to %s', run_count_config_key, step_to_save_at)
                if not self._local_mode:
                    worker_count = multi_worker_util.worker_count(self._cluster_resolver.cluster_spec(), self._cluster_resolver.task_type)
                    for i in range(worker_count):
                        context.context().get_config_key_value(f'{_ACKNOWLEDGE_KEY}_{run_count_config_key}_{i}')
                        logging.info('Sigterm acknowledgement from replica %d received', i)
            self._setup_countdown_if_has_grace_period_and_not_already_counting_down()

    def _time_to_exit(self):
        if False:
            i = 10
            return i + 15
        'Return whether to exit: exit if no grace period or grace period ends.'
        return self._grace_period <= 0 or self._final_checkpoint_countdown

    def _setup_countdown_if_has_grace_period_and_not_already_counting_down(self):
        if False:
            while True:
                i = 10
        'Set up at the beginning of a countdown period for long grace period.'
        if self._grace_period > 0 and (not self._final_checkpoint_countdown):
            buffer_factor = 3
            self._target_time_for_termination = self._received_own_sigterm_time + self._grace_period - buffer_factor * self._estimated_run_time * 2

    def _sigterm_handler_fn(self, signum, frame):
        if False:
            while True:
                i = 10
        "Upload the to-be-preempted worker's id to coordination service."
        del signum, frame
        self._maybe_set_received_own_sigterm()

    def _watch_step_to_save_key(self):
        if False:
            print('Hello World!')
        'Watch out for step-to-save config key and acknowledge.\n\n    All workers, including the one to be preempted, execute this function to get\n    step-to-save.\n    '
        step_value = context.context().get_config_key_value(_INITIAL_RUN_COUNT_KEY)
        if step_value != _STOP_WATCHING_CLUSTER_VALUE:
            self._step_to_checkpoint = step_value
            self._received_checkpoint_step.set()
            ack_key = f'{_ACKNOWLEDGE_KEY}_{_INITIAL_RUN_COUNT_KEY}_{self._id_in_cluster}'
            context.context().set_config_key_value(ack_key, '1')
            logging.info('PreemptionCheckpointHandler: %s set, preemption awareness acknowledged', ack_key)
            if self._grace_period > 0:
                final_step_value = context.context().get_config_key_value(_FINAL_RUN_COUNT_KEY)
                if final_step_value != _STOP_WATCHING_CLUSTER_VALUE:
                    ack_key = f'{_ACKNOWLEDGE_KEY}_{_FINAL_RUN_COUNT_KEY}_{self._id_in_cluster}'
                    context.context().set_config_key_value(ack_key, '1')
                    logging.info('PreemptionCheckpointHandler: %s acknowledged, final checkpoint timing received.', ack_key)
                    self._received_checkpoint_step.set()
                    self._step_to_checkpoint = final_step_value
WorkerPreemptionHandler = PreemptionCheckpointHandler