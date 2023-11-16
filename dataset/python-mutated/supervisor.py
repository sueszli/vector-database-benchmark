"""Training helper that checkpoints models and computes summaries."""
import contextlib
import os
import time
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary as _summary
from tensorflow.python.training import coordinator
from tensorflow.python.training import saver as saver_mod
from tensorflow.python.training import session_manager as session_manager_mod
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

@tf_export(v1=['train.Supervisor'])
class Supervisor:
    """A training helper that checkpoints models and computes summaries.

  This class is deprecated. Please use
  `tf.compat.v1.train.MonitoredTrainingSession` instead.

  The Supervisor is a small wrapper around a `Coordinator`, a `Saver`,
  and a `SessionManager` that takes care of common needs of TensorFlow
  training programs.

  #### Use for a single program

  ```python
  with tf.Graph().as_default():
    ...add operations to the graph...
    # Create a Supervisor that will checkpoint the model in '/tmp/mydir'.
    sv = Supervisor(logdir='/tmp/mydir')
    # Get a TensorFlow session managed by the supervisor.
    with sv.managed_session(FLAGS.master) as sess:
      # Use the session to train the graph.
      while not sv.should_stop():
        sess.run(<my_train_op>)
  ```

  Within the `with sv.managed_session()` block all variables in the graph have
  been initialized.  In addition, a few services have been started to
  checkpoint the model and add summaries to the event log.

  If the program crashes and is restarted, the managed session automatically
  reinitialize variables from the most recent checkpoint.

  The supervisor is notified of any exception raised by one of the services.
  After an exception is raised, `should_stop()` returns `True`.  In that case
  the training loop should also stop.  This is why the training loop has to
  check for `sv.should_stop()`.

  Exceptions that indicate that the training inputs have been exhausted,
  `tf.errors.OutOfRangeError`, also cause `sv.should_stop()` to return `True`
  but are not re-raised from the `with` block: they indicate a normal
  termination.

  #### Use for multiple replicas

  To train with replicas you deploy the same program in a `Cluster`.
  One of the tasks must be identified as the *chief*: the task that handles
  initialization, checkpoints, summaries, and recovery.  The other tasks
  depend on the *chief* for these services.

  The only change you have to do to the single program code is to indicate
  if the program is running as the *chief*.

  ```python
  # Choose a task as the chief. This could be based on server_def.task_index,
  # or job_def.name, or job_def.tasks. It's entirely up to the end user.
  # But there can be only one *chief*.
  is_chief = (server_def.task_index == 0)
  server = tf.distribute.Server(server_def)

  with tf.Graph().as_default():
    ...add operations to the graph...
    # Create a Supervisor that uses log directory on a shared file system.
    # Indicate if you are the 'chief'
    sv = Supervisor(logdir='/shared_directory/...', is_chief=is_chief)
    # Get a Session in a TensorFlow server on the cluster.
    with sv.managed_session(server.target) as sess:
      # Use the session to train the graph.
      while not sv.should_stop():
        sess.run(<my_train_op>)
  ```

  In the *chief* task, the `Supervisor` works exactly as in the first example
  above.  In the other tasks `sv.managed_session()` waits for the Model to have
  been initialized before returning a session to the training code.  The
  non-chief tasks depend on the chief task for initializing the model.

  If one of the tasks crashes and restarts, `managed_session()`
  checks if the Model is initialized.  If yes, it just creates a session and
  returns it to the training code that proceeds normally.  If the model needs
  to be initialized, the chief task takes care of reinitializing it; the other
  tasks just wait for the model to have been initialized.

  NOTE: This modified program still works fine as a single program.
  The single program marks itself as the chief.

  #### What `master` string to use

  Whether you are running on your machine or in the cluster you can use the
  following values for the --master flag:

  * Specifying `''` requests an in-process session that does not use RPC.

  * Specifying `'local'` requests a session that uses the RPC-based
    "Master interface" to run TensorFlow programs. See
    `tf.train.Server.create_local_server` for
    details.

  * Specifying `'grpc://hostname:port'` requests a session that uses
    the RPC interface to a specific host, and also allows the in-process
    master to access remote tensorflow workers. Often, it is
    appropriate to pass `server.target` (for some `tf.distribute.Server`
    named `server).

  #### Advanced use

  ##### Launching additional services

  `managed_session()` launches the Checkpoint and Summary services (threads).
  If you need more services to run you can simply launch them in the block
  controlled by `managed_session()`.

  Example: Start a thread to print losses.  We want this thread to run
  every 60 seconds, so we launch it with `sv.loop()`.

  ```python
  ...
  sv = Supervisor(logdir='/tmp/mydir')
  with sv.managed_session(FLAGS.master) as sess:
    sv.loop(60, print_loss, (sess, ))
    while not sv.should_stop():
      sess.run(my_train_op)
  ```

  ##### Launching fewer services

  `managed_session()` launches the "summary" and "checkpoint" threads which use
  either the optionally `summary_op` and `saver` passed to the constructor, or
  default ones created automatically by the supervisor.  If you want to run
  your own summary and checkpointing logic, disable these services by passing
  `None` to the `summary_op` and `saver` parameters.

  Example: Create summaries manually every 100 steps in the chief.

  ```python
  # Create a Supervisor with no automatic summaries.
  sv = Supervisor(logdir='/tmp/mydir', is_chief=is_chief, summary_op=None)
  # As summary_op was None, managed_session() does not start the
  # summary thread.
  with sv.managed_session(FLAGS.master) as sess:
    for step in range(1000000):
      if sv.should_stop():
        break
      if is_chief and step % 100 == 0:
        # Create the summary every 100 chief steps.
        sv.summary_computed(sess, sess.run(my_summary_op))
      else:
        # Train normally
        sess.run(my_train_op)
  ```

  ##### Custom model initialization

  `managed_session()` only supports initializing the model by running an
  `init_op` or restoring from the latest checkpoint.  If you have special
  initialization needs, see how to specify a `local_init_op` when creating the
  supervisor.  You can also use the `SessionManager` directly to create a
  session and check if it could be initialized automatically.
  """
    USE_DEFAULT = 0

    @deprecation.deprecated(None, 'Please switch to tf.train.MonitoredTrainingSession')
    def __init__(self, graph=None, ready_op=USE_DEFAULT, ready_for_local_init_op=USE_DEFAULT, is_chief=True, init_op=USE_DEFAULT, init_feed_dict=None, local_init_op=USE_DEFAULT, logdir=None, summary_op=USE_DEFAULT, saver=USE_DEFAULT, global_step=USE_DEFAULT, save_summaries_secs=120, save_model_secs=600, recovery_wait_secs=30, stop_grace_secs=120, checkpoint_basename='model.ckpt', session_manager=None, summary_writer=USE_DEFAULT, init_fn=None, local_init_run_options=None):
        if False:
            print('Hello World!')
        "Create a `Supervisor`.\n\n    Args:\n      graph: A `Graph`.  The graph that the model will use.  Defaults to the\n        default `Graph`.  The supervisor may add operations to the graph before\n        creating a session, but the graph should not be modified by the caller\n        after passing it to the supervisor.\n      ready_op: 1-D string `Tensor`.  This tensor is evaluated by supervisors in\n        `prepare_or_wait_for_session()` to check if the model is ready to use.\n        The model is considered ready if it returns an empty array.  Defaults to\n        the tensor returned from `tf.compat.v1.report_uninitialized_variables()`\n        If `None`, the model is not checked for readiness.\n      ready_for_local_init_op: 1-D string `Tensor`.  This tensor is evaluated by\n        supervisors in `prepare_or_wait_for_session()` to check if the model is\n        ready to run the local_init_op. The model is considered ready if it\n        returns an empty array. Defaults to `None`. If `None`, the model is not\n        checked for readiness before running local_init_op.\n      is_chief: If True, create a chief supervisor in charge of initializing and\n        restoring the model.  If False, create a supervisor that relies on a\n        chief supervisor for inits and restore.\n      init_op: `Operation`.  Used by chief supervisors to initialize the model\n        when it can not be recovered.  Defaults to an `Operation` that\n        initializes all global variables.  If `None`, no initialization is done\n        automatically unless you pass a value for `init_fn`, see below.\n      init_feed_dict: A dictionary that maps `Tensor` objects to feed values.\n        This feed dictionary will be used when `init_op` is evaluated.\n      local_init_op: `Operation`. Used by all supervisors to run initializations\n        that should run for every new supervisor instance. By default these are\n        table initializers and initializers for local variables. If `None`, no\n        further per supervisor-instance initialization is done automatically.\n      logdir: A string.  Optional path to a directory where to checkpoint the\n        model and log events for the visualizer.  Used by chief supervisors. The\n        directory will be created if it does not exist.\n      summary_op: An `Operation` that returns a Summary for the event logs. Used\n        by chief supervisors if a `logdir` was specified.  Defaults to the\n        operation returned from summary.merge_all().  If `None`, summaries are\n        not computed automatically.\n      saver: A Saver object.  Used by chief supervisors if a `logdir` was\n        specified.  Defaults to the saved returned by Saver(). If `None`, the\n        model is not saved automatically.\n      global_step: An integer Tensor of size 1 that counts steps.  The value\n        from 'global_step' is used in summaries and checkpoint filenames.\n        Default to the op named 'global_step' in the graph if it exists, is of\n        rank 1, size 1, and of type tf.int32 or tf.int64.  If `None` the global\n        step is not recorded in summaries and checkpoint files.  Used by chief\n        supervisors if a `logdir` was specified.\n      save_summaries_secs: Number of seconds between the computation of\n        summaries for the event log.  Defaults to 120 seconds.  Pass 0 to\n        disable summaries.\n      save_model_secs: Number of seconds between the creation of model\n        checkpoints.  Defaults to 600 seconds.  Pass 0 to disable checkpoints.\n      recovery_wait_secs: Number of seconds between checks that the model is\n        ready.  Used by supervisors when waiting for a chief supervisor to\n        initialize or restore the model.  Defaults to 30 seconds.\n      stop_grace_secs: Grace period, in seconds, given to running threads to\n        stop when `stop()` is called.  Defaults to 120 seconds.\n      checkpoint_basename: The basename for checkpoint saving.\n      session_manager: `SessionManager`, which manages Session creation and\n        recovery. If it is `None`, a default `SessionManager` will be created\n        with the set of arguments passed in for backwards compatibility.\n      summary_writer: `SummaryWriter` to use or `USE_DEFAULT`.  Can be `None` to\n        indicate that no summaries should be written.\n      init_fn: Optional callable used to initialize the model. Called after the\n        optional `init_op` is called.  The callable must accept one argument,\n        the session being initialized.\n      local_init_run_options: RunOptions to be passed as the SessionManager\n        local_init_run_options parameter.\n\n    Returns:\n      A `Supervisor`.\n\n    Raises:\n      RuntimeError: If called with eager execution enabled.\n\n    @compatibility(eager)\n    `Supervisor`s are not supported when eager execution is enabled.\n    @end_compatibility\n    "
        if context.executing_eagerly():
            raise RuntimeError('Supervisors are incompatible with eager execution.')
        if graph is None:
            graph = ops.get_default_graph()
        with graph.as_default():
            self._init_ready_op(ready_op=ready_op, ready_for_local_init_op=ready_for_local_init_op)
            self._init_init_op(init_op=init_op, init_feed_dict=init_feed_dict)
            self._init_local_init_op(local_init_op=local_init_op)
            self._init_saver(saver=saver)
            self._init_summary_op(summary_op=summary_op)
            self._init_global_step(global_step=global_step)
        self._graph = graph
        self._meta_graph_def = meta_graph.create_meta_graph_def(graph_def=graph.as_graph_def(add_shapes=True), saver_def=self._saver.saver_def if self._saver else None)
        self._is_chief = is_chief
        self._coord = coordinator.Coordinator()
        self._recovery_wait_secs = recovery_wait_secs
        self._stop_grace_secs = stop_grace_secs
        self._init_fn = init_fn
        self._local_init_run_options = local_init_run_options
        self._logdir = None
        self._save_summaries_secs = None
        self._save_model_secs = None
        self._save_path = None
        self._summary_writer = None
        if self._is_chief:
            self._logdir = logdir
            self._save_summaries_secs = save_summaries_secs
            self._save_model_secs = save_model_secs
            if self._logdir:
                self._save_path = os.path.join(self._logdir, checkpoint_basename)
            if summary_writer is Supervisor.USE_DEFAULT:
                if self._logdir:
                    self._summary_writer = _summary.FileWriter(self._logdir)
            else:
                self._summary_writer = summary_writer
            self._graph_added_to_summary = False
        self._init_session_manager(session_manager=session_manager)
        self._verify_setup()
        graph.finalize()

    def _init_session_manager(self, session_manager=None):
        if False:
            print('Hello World!')
        if session_manager is None:
            self._session_manager = session_manager_mod.SessionManager(local_init_op=self._local_init_op, ready_op=self._ready_op, ready_for_local_init_op=self._ready_for_local_init_op, graph=self._graph, recovery_wait_secs=self._recovery_wait_secs, local_init_run_options=self._local_init_run_options)
        else:
            self._session_manager = session_manager

    def _get_first_op_from_collection(self, key):
        if False:
            i = 10
            return i + 15
        'Returns the first `Operation` from a collection.\n\n    Args:\n      key: A string collection key.\n\n    Returns:\n      The first Op found in a collection, or `None` if the collection is empty.\n    '
        try:
            op_list = ops.get_collection(key)
            if len(op_list) > 1:
                logging.info('Found %d %s operations. Returning the first one.', len(op_list), key)
            if op_list:
                return op_list[0]
        except LookupError:
            pass
        return None

    def _init_ready_op(self, ready_op=USE_DEFAULT, ready_for_local_init_op=USE_DEFAULT):
        if False:
            print('Hello World!')
        "Initializes ready_op.\n\n    Args:\n      ready_op: `Tensor` to check if the model is initialized. If it's set to\n        USE_DEFAULT, creates an op that checks all the variables are\n        initialized.\n      ready_for_local_init_op: `Tensor` to check if the model is ready to run\n        local_init_op. If it's set to USE_DEFAULT, creates an op that checks all\n        the global variables are initialized.\n    "
        if ready_op is Supervisor.USE_DEFAULT:
            ready_op = self._get_first_op_from_collection(ops.GraphKeys.READY_OP)
            if ready_op is None:
                ready_op = variables.report_uninitialized_variables()
                ops.add_to_collection(ops.GraphKeys.READY_OP, ready_op)
        self._ready_op = ready_op
        if ready_for_local_init_op is Supervisor.USE_DEFAULT:
            ready_for_local_init_op = self._get_first_op_from_collection(ops.GraphKeys.READY_FOR_LOCAL_INIT_OP)
        self._ready_for_local_init_op = ready_for_local_init_op

    def _init_init_op(self, init_op=USE_DEFAULT, init_feed_dict=None):
        if False:
            print('Hello World!')
        'Initializes init_op.\n\n    Args:\n      init_op: `Operation` to initialize the variables. If set to USE_DEFAULT,\n        create an op that initializes all variables and tables.\n      init_feed_dict: A dictionary that maps `Tensor` objects to feed values.\n        This feed dictionary will be used when `init_op` is evaluated.\n    '
        if init_op is Supervisor.USE_DEFAULT:
            init_op = self._get_first_op_from_collection(ops.GraphKeys.INIT_OP)
            if init_op is None:
                init_op = variables.global_variables_initializer()
                ops.add_to_collection(ops.GraphKeys.INIT_OP, init_op)
        self._init_op = init_op
        self._init_feed_dict = init_feed_dict

    def _init_local_init_op(self, local_init_op=USE_DEFAULT):
        if False:
            while True:
                i = 10
        'Initializes local_init_op.\n\n    Args:\n      local_init_op: `Operation` run for every new supervisor instance. If set\n        to USE_DEFAULT, use the first op from the GraphKeys.LOCAL_INIT_OP\n        collection. If the collection is empty, create an op that initializes\n        all local variables and all tables.\n    '
        if local_init_op is Supervisor.USE_DEFAULT:
            local_init_op = self._get_first_op_from_collection(ops.GraphKeys.LOCAL_INIT_OP)
            if local_init_op is None:
                op_list = [variables.local_variables_initializer(), lookup_ops.tables_initializer()]
                if op_list:
                    local_init_op = control_flow_ops.group(*op_list)
                    ops.add_to_collection(ops.GraphKeys.LOCAL_INIT_OP, local_init_op)
        self._local_init_op = local_init_op

    def _init_saver(self, saver=USE_DEFAULT):
        if False:
            return 10
        'Initializes saver.\n\n    Args:\n      saver: A `Saver` object. If set to USE_DEFAULT, create one that saves all\n        the variables.\n    '
        if saver is Supervisor.USE_DEFAULT:
            saver = self._get_first_op_from_collection(ops.GraphKeys.SAVERS)
            if saver is None and variables.global_variables():
                saver = saver_mod.Saver()
                ops.add_to_collection(ops.GraphKeys.SAVERS, saver)
        self._saver = saver

    def _init_summary_op(self, summary_op=USE_DEFAULT):
        if False:
            for i in range(10):
                print('nop')
        'Initializes summary_op.\n\n    Args:\n      summary_op: An Operation that returns a Summary for the event logs. If set\n        to USE_DEFAULT, create an op that merges all the summaries.\n    '
        if summary_op is Supervisor.USE_DEFAULT:
            summary_op = self._get_first_op_from_collection(ops.GraphKeys.SUMMARY_OP)
            if summary_op is None:
                summary_op = _summary.merge_all()
                if summary_op is not None:
                    ops.add_to_collection(ops.GraphKeys.SUMMARY_OP, summary_op)
        self._summary_op = summary_op

    def _init_global_step(self, global_step=USE_DEFAULT):
        if False:
            i = 10
            return i + 15
        'Initializes global_step.\n\n    Args:\n      global_step: An integer Tensor of size 1 that counts steps. If set to\n        USE_DEFAULT, creates global_step tensor.\n    '
        if global_step is Supervisor.USE_DEFAULT:
            global_step = self._get_first_op_from_collection(ops.GraphKeys.GLOBAL_STEP)
            if global_step is None:
                global_step = self._default_global_step_tensor()
                if global_step is not None:
                    ops.add_to_collection(ops.GraphKeys.GLOBAL_STEP, global_step)
        self._global_step = global_step

    @property
    def is_chief(self):
        if False:
            while True:
                i = 10
        'Return True if this is a chief supervisor.\n\n    Returns:\n      A bool.\n    '
        return self._is_chief

    @property
    def session_manager(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the SessionManager used by the Supervisor.\n\n    Returns:\n      A SessionManager object.\n    '
        return self._session_manager

    @property
    def coord(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the Coordinator used by the Supervisor.\n\n    The Coordinator can be useful if you want to run multiple threads\n    during your training.\n\n    Returns:\n      A Coordinator object.\n    '
        return self._coord

    @property
    def init_op(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the Init Op used by the supervisor.\n\n    Returns:\n      An Op or `None`.\n    '
        return self._init_op

    @property
    def init_feed_dict(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the feed dictionary used when evaluating the `init_op`.\n\n    Returns:\n      A feed dictionary or `None`.\n    '
        return self._init_feed_dict

    @property
    def ready_op(self):
        if False:
            while True:
                i = 10
        'Return the Ready Op used by the supervisor.\n\n    Returns:\n      An Op or `None`.\n    '
        return self._ready_op

    @property
    def ready_for_local_init_op(self):
        if False:
            while True:
                i = 10
        return self._ready_for_local_init_op

    @property
    def summary_writer(self):
        if False:
            i = 10
            return i + 15
        'Return the SummaryWriter used by the chief supervisor.\n\n    Returns:\n      A SummaryWriter.\n    '
        return self._summary_writer

    @property
    def summary_op(self):
        if False:
            while True:
                i = 10
        'Return the Summary Tensor used by the chief supervisor.\n\n    Returns:\n      A string Tensor for the summary or `None`.\n    '
        return self._summary_op

    @property
    def save_summaries_secs(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the delay between summary computations.\n\n    Returns:\n      A timestamp.\n    '
        return self._save_summaries_secs

    @property
    def global_step(self):
        if False:
            print('Hello World!')
        'Return the global_step Tensor used by the supervisor.\n\n    Returns:\n      An integer Tensor for the global_step.\n    '
        return self._global_step

    @property
    def saver(self):
        if False:
            while True:
                i = 10
        'Return the Saver used by the supervisor.\n\n    Returns:\n      A Saver object.\n    '
        return self._saver

    @property
    def save_model_secs(self):
        if False:
            return 10
        'Return the delay between checkpoints.\n\n    Returns:\n      A timestamp.\n    '
        return self._save_model_secs

    @property
    def save_path(self):
        if False:
            while True:
                i = 10
        'Return the save path used by the supervisor.\n\n    Returns:\n      A string.\n    '
        return self._save_path

    def _write_graph(self):
        if False:
            i = 10
            return i + 15
        'Writes graph_def to `logdir` and adds it to summary if applicable.'
        assert self._is_chief
        if self._logdir:
            training_util.write_graph(self._graph.as_graph_def(add_shapes=True), self._logdir, 'graph.pbtxt')
        if self._summary_writer and (not self._graph_added_to_summary):
            self._summary_writer.add_graph(self._graph)
            self._summary_writer.add_meta_graph(self._meta_graph_def)
            self._graph_added_to_summary = True

    def start_standard_services(self, sess):
        if False:
            print('Hello World!')
        "Start the standard services for 'sess'.\n\n    This starts services in the background.  The services started depend\n    on the parameters to the constructor and may include:\n\n      - A Summary thread computing summaries every save_summaries_secs.\n      - A Checkpoint thread saving the model every save_model_secs.\n      - A StepCounter thread measure step time.\n\n    Args:\n      sess: A Session.\n\n    Returns:\n      A list of threads that are running the standard services.  You can use\n      the Supervisor's Coordinator to join these threads with:\n        sv.coord.Join(<list of threads>)\n\n    Raises:\n      RuntimeError: If called with a non-chief Supervisor.\n      ValueError: If not `logdir` was passed to the constructor as the\n        services need a log directory.\n    "
        if not self._is_chief:
            raise RuntimeError('Only chief supervisor can start standard services. Because only chief supervisors can write events.')
        if not self._logdir:
            logging.warning("Standard services need a 'logdir' passed to the SessionManager")
            return
        if self._global_step is not None and self._summary_writer:
            current_step = training_util.global_step(sess, self._global_step)
            self._summary_writer.add_session_log(SessionLog(status=SessionLog.START), current_step)
        threads = []
        if self._save_summaries_secs and self._summary_writer:
            if self._summary_op is not None:
                threads.append(SVSummaryThread(self, sess))
            if self._global_step is not None:
                threads.append(SVStepCounterThread(self, sess))
        if self.saver and self._save_model_secs:
            threads.append(SVTimerCheckpointThread(self, sess))
        for t in threads:
            t.start()
        return threads

    def prepare_or_wait_for_session(self, master='', config=None, wait_for_checkpoint=False, max_wait_secs=7200, start_standard_services=True):
        if False:
            for i in range(10):
                print('nop')
        "Make sure the model is ready to be used.\n\n    Create a session on 'master', recovering or initializing the model as\n    needed, or wait for a session to be ready.  If running as the chief\n    and `start_standard_service` is set to True, also call the session\n    manager to start the standard services.\n\n    Args:\n      master: name of the TensorFlow master to use.  See the\n        `tf.compat.v1.Session` constructor for how this is interpreted.\n      config: Optional ConfigProto proto used to configure the session, which is\n        passed as-is to create the session.\n      wait_for_checkpoint: Whether we should wait for the availability of a\n        checkpoint before creating Session. Defaults to False.\n      max_wait_secs: Maximum time to wait for the session to become available.\n      start_standard_services: Whether to start the standard services and the\n        queue runners.\n\n    Returns:\n      A Session object that can be used to drive the model.\n    "
        self._coord.clear_stop()
        if self._summary_writer:
            self._summary_writer.reopen()
        if self._is_chief:
            sess = self._session_manager.prepare_session(master, init_op=self.init_op, saver=self.saver, checkpoint_dir=self._logdir, wait_for_checkpoint=wait_for_checkpoint, max_wait_secs=max_wait_secs, config=config, init_feed_dict=self._init_feed_dict, init_fn=self._init_fn)
            self._write_graph()
            if start_standard_services:
                logging.info('Starting standard services.')
                self.start_standard_services(sess)
        else:
            sess = self._session_manager.wait_for_session(master, config=config, max_wait_secs=max_wait_secs)
        if start_standard_services:
            logging.info('Starting queue runners.')
            self.start_queue_runners(sess)
        return sess

    def start_queue_runners(self, sess, queue_runners=None):
        if False:
            while True:
                i = 10
        "Start threads for `QueueRunners`.\n\n    Note that the queue runners collected in the graph key `QUEUE_RUNNERS`\n    are already started automatically when you create a session with the\n    supervisor, so unless you have non-collected queue runners to start\n    you do not need to call this explicitly.\n\n    Args:\n      sess: A `Session`.\n      queue_runners: A list of `QueueRunners`. If not specified, we'll use the\n        list of queue runners gathered in the graph under the key\n        `GraphKeys.QUEUE_RUNNERS`.\n\n    Returns:\n      The list of threads started for the `QueueRunners`.\n\n    Raises:\n      RuntimeError: If called with eager execution enabled.\n\n    @compatibility(eager)\n    Queues are not compatible with eager execution. To ingest data when eager\n    execution is enabled, use the `tf.data` API.\n    @end_compatibility\n    "
        if context.executing_eagerly():
            raise RuntimeError('Queues are not compatible with eager execution.')
        if queue_runners is None:
            queue_runners = self._graph.get_collection(ops.GraphKeys.QUEUE_RUNNERS)
        threads = []
        for qr in queue_runners:
            threads.extend(qr.create_threads(sess, coord=self._coord, daemon=True, start=True))
        return threads

    def loop(self, timer_interval_secs, target, args=None, kwargs=None):
        if False:
            print('Hello World!')
        'Start a LooperThread that calls a function periodically.\n\n    If `timer_interval_secs` is None the thread calls `target(*args, **kwargs)`\n    repeatedly.  Otherwise it calls it every `timer_interval_secs`\n    seconds.  The thread terminates when a stop is requested.\n\n    The started thread is added to the list of threads managed by the supervisor\n    so it does not need to be passed to the `stop()` method.\n\n    Args:\n      timer_interval_secs: Number. Time boundaries at which to call `target`.\n      target: A callable object.\n      args: Optional arguments to pass to `target` when calling it.\n      kwargs: Optional keyword arguments to pass to `target` when calling it.\n\n    Returns:\n      The started thread.\n    '
        looper = coordinator.LooperThread(self._coord, timer_interval_secs, target=target, args=args, kwargs=kwargs)
        looper.start()
        return looper

    def stop(self, threads=None, close_summary_writer=True, ignore_live_threads=False):
        if False:
            return 10
        'Stop the services and the coordinator.\n\n    This does not close the session.\n\n    Args:\n      threads: Optional list of threads to join with the coordinator.  If\n        `None`, defaults to the threads running the standard services, the\n        threads started for `QueueRunners`, and the threads started by the\n        `loop()` method.  To wait on additional threads, pass the list in this\n        parameter.\n      close_summary_writer: Whether to close the `summary_writer`.  Defaults to\n        `True` if the summary writer was created by the supervisor, `False`\n        otherwise.\n      ignore_live_threads: If `True` ignores threads that remain running after a\n        grace period when joining threads via the coordinator, instead of\n        raising a RuntimeError.\n    '
        self._coord.request_stop()
        try:
            self._coord.join(threads, stop_grace_period_secs=self._stop_grace_secs, ignore_live_threads=ignore_live_threads)
        finally:
            if close_summary_writer and self._summary_writer:
                self._summary_writer.add_session_log(SessionLog(status=SessionLog.STOP))
                self._summary_writer.close()
                self._graph_added_to_summary = False

    def request_stop(self, ex=None):
        if False:
            return 10
        'Request that the coordinator stop the threads.\n\n    See `Coordinator.request_stop()`.\n\n    Args:\n      ex: Optional `Exception`, or Python `exc_info` tuple as returned by\n        `sys.exc_info()`.  If this is the first call to `request_stop()` the\n        corresponding exception is recorded and re-raised from `join()`.\n    '
        self._coord.request_stop(ex=ex)

    def should_stop(self):
        if False:
            return 10
        'Check if the coordinator was told to stop.\n\n    See `Coordinator.should_stop()`.\n\n    Returns:\n      True if the coordinator was told to stop, False otherwise.\n    '
        return self._coord.should_stop()

    def stop_on_exception(self):
        if False:
            i = 10
            return i + 15
        'Context handler to stop the supervisor when an exception is raised.\n\n    See `Coordinator.stop_on_exception()`.\n\n    Returns:\n      A context handler.\n    '
        return self._coord.stop_on_exception()

    def wait_for_stop(self):
        if False:
            return 10
        'Block waiting for the coordinator to stop.'
        self._coord.wait_for_stop()

    def summary_computed(self, sess, summary, global_step=None):
        if False:
            return 10
        "Indicate that a summary was computed.\n\n    Args:\n      sess: A `Session` object.\n      summary: A Summary proto, or a string holding a serialized summary proto.\n      global_step: Int. global step this summary is associated with. If `None`,\n        it will try to fetch the current step.\n\n    Raises:\n      TypeError: if 'summary' is not a Summary proto or a string.\n      RuntimeError: if the Supervisor was created without a `logdir`.\n    "
        if not self._summary_writer:
            raise RuntimeError('Writing a summary requires a summary writer.')
        if global_step is None and self.global_step is not None:
            global_step = training_util.global_step(sess, self.global_step)
        self._summary_writer.add_summary(summary, global_step)

    def _default_global_step_tensor(self):
        if False:
            return 10
        'Returns the global_step from the default graph.\n\n    Returns:\n      The global step `Tensor` or `None`.\n    '
        try:
            gs = ops.get_default_graph().get_tensor_by_name('global_step:0')
            if gs.dtype.base_dtype in [dtypes.int32, dtypes.int64]:
                return gs
            else:
                logging.warning("Found 'global_step' is not an int type: %s", gs.dtype)
                return None
        except KeyError:
            return None

    def _verify_setup(self):
        if False:
            i = 10
            return i + 15
        'Check that all is good.\n\n    Raises:\n      ValueError: If something is not good.\n    '
        if not self._is_chief:
            for op in self._graph.get_operations():
                if op.type in ['Variable', 'VariableV2'] and (not op.device):
                    raise ValueError('When using replicas, all Variables must have their device set: %s' % op)

    @contextlib.contextmanager
    def managed_session(self, master='', config=None, start_standard_services=True, close_summary_writer=True):
        if False:
            return 10
        'Returns a context manager for a managed session.\n\n    This context manager creates and automatically recovers a session.  It\n    optionally starts the standard services that handle checkpoints and\n    summaries.  It monitors exceptions raised from the `with` block or from the\n    services and stops the supervisor as needed.\n\n    The context manager is typically used as follows:\n\n    ```python\n    def train():\n      sv = tf.compat.v1.train.Supervisor(...)\n      with sv.managed_session(<master>) as sess:\n        for step in range(..):\n          if sv.should_stop():\n            break\n          sess.run(<my training op>)\n          ...do other things needed at each training step...\n    ```\n\n    An exception raised from the `with` block or one of the service threads is\n    raised again when the block exits.  This is done after stopping all threads\n    and closing the session.  For example, an `AbortedError` exception, raised\n    in case of preemption of one of the workers in a distributed model, is\n    raised again when the block exits.\n\n    If you want to retry the training loop in case of preemption you can do it\n    as follows:\n\n    ```python\n    def main(...):\n      while True\n        try:\n          train()\n        except tf.errors.Aborted:\n          pass\n    ```\n\n    As a special case, exceptions used for control flow, such as\n    `OutOfRangeError` which reports that input queues are exhausted, are not\n    raised again from the `with` block: they indicate a clean termination of\n    the training loop and are considered normal termination.\n\n    Args:\n      master: name of the TensorFlow master to use.  See the\n        `tf.compat.v1.Session` constructor for how this is interpreted.\n      config: Optional `ConfigProto` proto used to configure the session. Passed\n        as-is to create the session.\n      start_standard_services: Whether to start the standard services, such as\n        checkpoint, summary and step counter.\n      close_summary_writer: Whether to close the summary writer when closing the\n        session.  Defaults to True.\n\n    Returns:\n      A context manager that yields a `Session` restored from the latest\n      checkpoint or initialized from scratch if not checkpoint exists.  The\n      session is closed when the `with` block exits.\n    '
        try:
            sess = self.prepare_or_wait_for_session(master=master, config=config, start_standard_services=start_standard_services)
            yield sess
        except Exception as e:
            self.request_stop(e)
        finally:
            try:
                self.stop(close_summary_writer=close_summary_writer)
            finally:
                try:
                    sess.close()
                except Exception:
                    pass

class SVSummaryThread(coordinator.LooperThread):
    """A thread to save summaries on a timer."""

    def __init__(self, sv, sess):
        if False:
            i = 10
            return i + 15
        'Create a SVSummaryThread.\n\n    Args:\n      sv: A `Supervisor`.\n      sess: A `Session`.\n    '
        super(SVSummaryThread, self).__init__(sv.coord, sv.save_summaries_secs)
        self._sv = sv
        self._sess = sess

    def run_loop(self):
        if False:
            return 10
        if self._sv.global_step is not None:
            (summary_strs, global_step) = self._sess.run([self._sv.summary_op, self._sv.global_step])
        else:
            summary_strs = self._sess.run(self._sv.summary_op)
            global_step = None
        if self._sv.summary_writer:
            logging.info('Recording summary at step %s.', global_step)
            self._sv.summary_writer.add_summary(summary_strs, global_step)

class SVStepCounterThread(coordinator.LooperThread):
    """Threads to count steps and measure their duration."""

    def __init__(self, sv, sess, step_counter=None):
        if False:
            i = 10
            return i + 15
        'Create a `SVStepCounterThread`.\n\n    Args:\n      sv: A `Supervisor`.\n      sess: A `Session`.\n      step_counter: A `Tensor` holding the step counter. By defaults, it uses\n        sv.global_step.\n    '
        super(SVStepCounterThread, self).__init__(sv.coord, sv.save_summaries_secs)
        self._sv = sv
        self._sess = sess
        self._last_time = 0.0
        self._last_step = 0
        step_counter = sv.global_step if step_counter is None else step_counter
        self._step_counter = step_counter
        self._summary_tag = '%s/sec' % self._step_counter.op.name

    def start_loop(self):
        if False:
            for i in range(10):
                print('nop')
        self._last_time = time.time()
        self._last_step = training_util.global_step(self._sess, self._step_counter)

    def run_loop(self):
        if False:
            for i in range(10):
                print('nop')
        current_step = training_util.global_step(self._sess, self._step_counter)
        added_steps = current_step - self._last_step
        self._last_step = current_step
        current_time = time.time()
        elapsed_time = current_time - self._last_time
        self._last_time = current_time
        if elapsed_time > 0.0:
            steps_per_sec = added_steps / elapsed_time
        else:
            steps_per_sec = float('inf')
        summary = Summary(value=[Summary.Value(tag=self._summary_tag, simple_value=steps_per_sec)])
        if self._sv.summary_writer:
            self._sv.summary_writer.add_summary(summary, current_step)
        logging.log_first_n(logging.INFO, '%s: %g', 10, self._summary_tag, steps_per_sec)

class SVTimerCheckpointThread(coordinator.LooperThread):
    """A thread to checkpoint on a timer."""

    def __init__(self, sv, sess):
        if False:
            while True:
                i = 10
        'Create a `SVTimerCheckpointThread`.\n\n    Args:\n      sv: A `Supervisor`.\n      sess: A `Session`.\n    '
        super(SVTimerCheckpointThread, self).__init__(sv.coord, sv.save_model_secs)
        self._sv = sv
        self._sess = sess

    def run_loop(self):
        if False:
            print('Hello World!')
        logging.info('Saving checkpoint to path %s', self._sv.save_path)
        self._sv.saver.save(self._sess, self._sv.save_path, global_step=self._sv.global_step)
        if self._sv.summary_writer and self._sv.global_step is not None:
            current_step = training_util.global_step(self._sess, self._sv.global_step)
            self._sv.summary_writer.add_session_log(SessionLog(status=SessionLog.CHECKPOINT, checkpoint_path=self._sv.save_path), current_step)
setattr(Supervisor, 'PrepareSession', Supervisor.prepare_or_wait_for_session)
setattr(Supervisor, 'StartQueueRunners', Supervisor.start_queue_runners)
setattr(Supervisor, 'StartStandardServices', Supervisor.start_standard_services)
setattr(Supervisor, 'Stop', Supervisor.stop)
setattr(Supervisor, 'RequestStop', Supervisor.request_stop)
setattr(Supervisor, 'Loop', Supervisor.loop)
setattr(Supervisor, 'ShouldStop', Supervisor.should_stop)
setattr(Supervisor, 'StopOnException', Supervisor.stop_on_exception)
setattr(Supervisor, 'WaitForStop', Supervisor.wait_for_stop)
setattr(Supervisor, 'SummaryComputed', Supervisor.summary_computed)