"""Create threads to run multiple enqueue ops."""
import threading
import weakref
from tensorflow.core.protobuf import queue_runner_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
_DEPRECATION_INSTRUCTION = 'To construct input pipelines, use the `tf.data` module.'

@tf_export(v1=['train.queue_runner.QueueRunner', 'train.QueueRunner'])
class QueueRunner:
    """Holds a list of enqueue operations for a queue, each to be run in a thread.

  Queues are a convenient TensorFlow mechanism to compute tensors
  asynchronously using multiple threads. For example in the canonical 'Input
  Reader' setup one set of threads generates filenames in a queue; a second set
  of threads read records from the files, processes them, and enqueues tensors
  on a second queue; a third set of threads dequeues these input records to
  construct batches and runs them through training operations.

  There are several delicate issues when running multiple threads that way:
  closing the queues in sequence as the input is exhausted, correctly catching
  and reporting exceptions, etc.

  The `QueueRunner`, combined with the `Coordinator`, helps handle these issues.

  @compatibility(TF2)
  QueueRunners are not compatible with eager execution. Instead, please
  use [tf.data](https://www.tensorflow.org/guide/data) to get data into your
  model.
  @end_compatibility
  """

    @deprecation.deprecated(None, _DEPRECATION_INSTRUCTION)
    def __init__(self, queue=None, enqueue_ops=None, close_op=None, cancel_op=None, queue_closed_exception_types=None, queue_runner_def=None, import_scope=None):
        if False:
            print('Hello World!')
        'Create a QueueRunner.\n\n    On construction the `QueueRunner` adds an op to close the queue.  That op\n    will be run if the enqueue ops raise exceptions.\n\n    When you later call the `create_threads()` method, the `QueueRunner` will\n    create one thread for each op in `enqueue_ops`.  Each thread will run its\n    enqueue op in parallel with the other threads.  The enqueue ops do not have\n    to all be the same op, but it is expected that they all enqueue tensors in\n    `queue`.\n\n    Args:\n      queue: A `Queue`.\n      enqueue_ops: List of enqueue ops to run in threads later.\n      close_op: Op to close the queue. Pending enqueue ops are preserved.\n      cancel_op: Op to close the queue and cancel pending enqueue ops.\n      queue_closed_exception_types: Optional tuple of Exception types that\n        indicate that the queue has been closed when raised during an enqueue\n        operation.  Defaults to `(tf.errors.OutOfRangeError,)`.  Another common\n        case includes `(tf.errors.OutOfRangeError, tf.errors.CancelledError)`,\n        when some of the enqueue ops may dequeue from other Queues.\n      queue_runner_def: Optional `QueueRunnerDef` protocol buffer. If specified,\n        recreates the QueueRunner from its contents. `queue_runner_def` and the\n        other arguments are mutually exclusive.\n      import_scope: Optional `string`. Name scope to add. Only used when\n        initializing from protocol buffer.\n\n    Raises:\n      ValueError: If both `queue_runner_def` and `queue` are both specified.\n      ValueError: If `queue` or `enqueue_ops` are not provided when not\n        restoring from `queue_runner_def`.\n      RuntimeError: If eager execution is enabled.\n    '
        if context.executing_eagerly():
            raise RuntimeError('QueueRunners are not supported when eager execution is enabled. Instead, please use tf.data to get data into your model.')
        if queue_runner_def:
            if queue or enqueue_ops:
                raise ValueError('queue_runner_def and queue are mutually exclusive.')
            self._init_from_proto(queue_runner_def, import_scope=import_scope)
        else:
            self._init_from_args(queue=queue, enqueue_ops=enqueue_ops, close_op=close_op, cancel_op=cancel_op, queue_closed_exception_types=queue_closed_exception_types)
        self._lock = threading.Lock()
        self._runs_per_session = weakref.WeakKeyDictionary()
        self._exceptions_raised = []

    def _init_from_args(self, queue=None, enqueue_ops=None, close_op=None, cancel_op=None, queue_closed_exception_types=None):
        if False:
            i = 10
            return i + 15
        'Create a QueueRunner from arguments.\n\n    Args:\n      queue: A `Queue`.\n      enqueue_ops: List of enqueue ops to run in threads later.\n      close_op: Op to close the queue. Pending enqueue ops are preserved.\n      cancel_op: Op to close the queue and cancel pending enqueue ops.\n      queue_closed_exception_types: Tuple of exception types, which indicate\n        the queue has been safely closed.\n\n    Raises:\n      ValueError: If `queue` or `enqueue_ops` are not provided when not\n        restoring from `queue_runner_def`.\n      TypeError: If `queue_closed_exception_types` is provided, but is not\n        a non-empty tuple of error types (subclasses of `tf.errors.OpError`).\n    '
        if not queue or not enqueue_ops:
            raise ValueError('Must provide queue and enqueue_ops.')
        self._queue = queue
        self._enqueue_ops = enqueue_ops
        self._close_op = close_op
        self._cancel_op = cancel_op
        if queue_closed_exception_types is not None:
            if not isinstance(queue_closed_exception_types, tuple) or not queue_closed_exception_types or (not all((issubclass(t, errors.OpError) for t in queue_closed_exception_types))):
                raise TypeError('queue_closed_exception_types, when provided, must be a tuple of tf.error types, but saw: %s' % queue_closed_exception_types)
        self._queue_closed_exception_types = queue_closed_exception_types
        if self._close_op is None:
            self._close_op = self._queue.close()
        if self._cancel_op is None:
            self._cancel_op = self._queue.close(cancel_pending_enqueues=True)
        if not self._queue_closed_exception_types:
            self._queue_closed_exception_types = (errors.OutOfRangeError,)
        else:
            self._queue_closed_exception_types = tuple(self._queue_closed_exception_types)

    def _init_from_proto(self, queue_runner_def, import_scope=None):
        if False:
            print('Hello World!')
        'Create a QueueRunner from `QueueRunnerDef`.\n\n    Args:\n      queue_runner_def: Optional `QueueRunnerDef` protocol buffer.\n      import_scope: Optional `string`. Name scope to add.\n    '
        assert isinstance(queue_runner_def, queue_runner_pb2.QueueRunnerDef)
        g = ops.get_default_graph()
        self._queue = g.as_graph_element(ops.prepend_name_scope(queue_runner_def.queue_name, import_scope))
        self._enqueue_ops = [g.as_graph_element(ops.prepend_name_scope(op, import_scope)) for op in queue_runner_def.enqueue_op_name]
        self._close_op = g.as_graph_element(ops.prepend_name_scope(queue_runner_def.close_op_name, import_scope))
        self._cancel_op = g.as_graph_element(ops.prepend_name_scope(queue_runner_def.cancel_op_name, import_scope))
        self._queue_closed_exception_types = tuple((errors.exception_type_from_error_code(code) for code in queue_runner_def.queue_closed_exception_types))
        if not self._queue_closed_exception_types:
            self._queue_closed_exception_types = (errors.OutOfRangeError,)

    @property
    def queue(self):
        if False:
            for i in range(10):
                print('nop')
        return self._queue

    @property
    def enqueue_ops(self):
        if False:
            print('Hello World!')
        return self._enqueue_ops

    @property
    def close_op(self):
        if False:
            while True:
                i = 10
        return self._close_op

    @property
    def cancel_op(self):
        if False:
            while True:
                i = 10
        return self._cancel_op

    @property
    def queue_closed_exception_types(self):
        if False:
            print('Hello World!')
        return self._queue_closed_exception_types

    @property
    def exceptions_raised(self):
        if False:
            return 10
        'Exceptions raised but not handled by the `QueueRunner` threads.\n\n    Exceptions raised in queue runner threads are handled in one of two ways\n    depending on whether or not a `Coordinator` was passed to\n    `create_threads()`:\n\n    * With a `Coordinator`, exceptions are reported to the coordinator and\n      forgotten by the `QueueRunner`.\n    * Without a `Coordinator`, exceptions are captured by the `QueueRunner` and\n      made available in this `exceptions_raised` property.\n\n    Returns:\n      A list of Python `Exception` objects.  The list is empty if no exception\n      was captured.  (No exceptions are captured when using a Coordinator.)\n    '
        return self._exceptions_raised

    @property
    def name(self):
        if False:
            print('Hello World!')
        'The string name of the underlying Queue.'
        return self._queue.name

    def _run(self, sess, enqueue_op, coord=None):
        if False:
            while True:
                i = 10
        'Execute the enqueue op in a loop, close the queue in case of error.\n\n    Args:\n      sess: A Session.\n      enqueue_op: The Operation to run.\n      coord: Optional Coordinator object for reporting errors and checking\n        for stop conditions.\n    '
        decremented = False
        try:
            enqueue_callable = sess.make_callable(enqueue_op)
            while True:
                if coord and coord.should_stop():
                    break
                try:
                    enqueue_callable()
                except self._queue_closed_exception_types:
                    with self._lock:
                        self._runs_per_session[sess] -= 1
                        decremented = True
                        if self._runs_per_session[sess] == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                logging.vlog(1, 'Ignored exception: %s', str(e))
                        return
        except Exception as e:
            if coord:
                coord.request_stop(e)
            else:
                logging.error('Exception in QueueRunner: %s', str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            if not decremented:
                with self._lock:
                    self._runs_per_session[sess] -= 1

    def _close_on_stop(self, sess, cancel_op, coord):
        if False:
            print('Hello World!')
        'Close the queue when the Coordinator requests stop.\n\n    Args:\n      sess: A Session.\n      cancel_op: The Operation to run.\n      coord: Coordinator.\n    '
        coord.wait_for_stop()
        try:
            sess.run(cancel_op)
        except Exception as e:
            logging.vlog(1, 'Ignored exception: %s', str(e))

    def create_threads(self, sess, coord=None, daemon=False, start=False):
        if False:
            return 10
        'Create threads to run the enqueue ops for the given session.\n\n    This method requires a session in which the graph was launched.  It creates\n    a list of threads, optionally starting them.  There is one thread for each\n    op passed in `enqueue_ops`.\n\n    The `coord` argument is an optional coordinator that the threads will use\n    to terminate together and report exceptions.  If a coordinator is given,\n    this method starts an additional thread to close the queue when the\n    coordinator requests a stop.\n\n    If previously created threads for the given session are still running, no\n    new threads will be created.\n\n    Args:\n      sess: A `Session`.\n      coord: Optional `Coordinator` object for reporting errors and checking\n        stop conditions.\n      daemon: Boolean.  If `True` make the threads daemon threads.\n      start: Boolean.  If `True` starts the threads.  If `False` the\n        caller must call the `start()` method of the returned threads.\n\n    Returns:\n      A list of threads.\n    '
        with self._lock:
            try:
                if self._runs_per_session[sess] > 0:
                    return []
            except KeyError:
                pass
            self._runs_per_session[sess] = len(self._enqueue_ops)
            self._exceptions_raised = []
        ret_threads = []
        for op in self._enqueue_ops:
            name = 'QueueRunnerThread-{}-{}'.format(self.name, op.name)
            ret_threads.append(threading.Thread(target=self._run, args=(sess, op, coord), name=name))
        if coord:
            name = 'QueueRunnerThread-{}-close_on_stop'.format(self.name)
            ret_threads.append(threading.Thread(target=self._close_on_stop, args=(sess, self._cancel_op, coord), name=name))
        for t in ret_threads:
            if coord:
                coord.register_thread(t)
            if daemon:
                t.daemon = True
            if start:
                t.start()
        return ret_threads

    def to_proto(self, export_scope=None):
        if False:
            print('Hello World!')
        'Converts this `QueueRunner` to a `QueueRunnerDef` protocol buffer.\n\n    Args:\n      export_scope: Optional `string`. Name scope to remove.\n\n    Returns:\n      A `QueueRunnerDef` protocol buffer, or `None` if the `Variable` is not in\n      the specified name scope.\n    '
        if export_scope is None or self.queue.name.startswith(export_scope):
            queue_runner_def = queue_runner_pb2.QueueRunnerDef()
            queue_runner_def.queue_name = ops.strip_name_scope(self.queue.name, export_scope)
            for enqueue_op in self.enqueue_ops:
                queue_runner_def.enqueue_op_name.append(ops.strip_name_scope(enqueue_op.name, export_scope))
            queue_runner_def.close_op_name = ops.strip_name_scope(self.close_op.name, export_scope)
            queue_runner_def.cancel_op_name = ops.strip_name_scope(self.cancel_op.name, export_scope)
            queue_runner_def.queue_closed_exception_types.extend([errors.error_code_from_exception_type(cls) for cls in self._queue_closed_exception_types])
            return queue_runner_def
        else:
            return None

    @staticmethod
    def from_proto(queue_runner_def, import_scope=None):
        if False:
            while True:
                i = 10
        'Returns a `QueueRunner` object created from `queue_runner_def`.'
        return QueueRunner(queue_runner_def=queue_runner_def, import_scope=import_scope)

@tf_export(v1=['train.queue_runner.add_queue_runner', 'train.add_queue_runner'])
@deprecation.deprecated(None, _DEPRECATION_INSTRUCTION)
def add_queue_runner(qr, collection=ops.GraphKeys.QUEUE_RUNNERS):
    if False:
        for i in range(10):
            print('nop')
    'Adds a `QueueRunner` to a collection in the graph.\n\n  When building a complex model that uses many queues it is often difficult to\n  gather all the queue runners that need to be run.  This convenience function\n  allows you to add a queue runner to a well known collection in the graph.\n\n  The companion method `start_queue_runners()` can be used to start threads for\n  all the collected queue runners.\n\n  @compatibility(TF2)\n  QueueRunners are not compatible with eager execution. Instead, please\n  use [tf.data](https://www.tensorflow.org/guide/data) to get data into your\n  model.\n  @end_compatibility\n\n  Args:\n    qr: A `QueueRunner`.\n    collection: A `GraphKey` specifying the graph collection to add\n      the queue runner to.  Defaults to `GraphKeys.QUEUE_RUNNERS`.\n  '
    ops.add_to_collection(collection, qr)

@tf_export(v1=['train.queue_runner.start_queue_runners', 'train.start_queue_runners'])
@deprecation.deprecated(None, _DEPRECATION_INSTRUCTION)
def start_queue_runners(sess=None, coord=None, daemon=True, start=True, collection=ops.GraphKeys.QUEUE_RUNNERS):
    if False:
        return 10
    "Starts all queue runners collected in the graph.\n\n  This is a companion method to `add_queue_runner()`.  It just starts\n  threads for all queue runners collected in the graph.  It returns\n  the list of all threads.\n\n  @compatibility(TF2)\n  QueueRunners are not compatible with eager execution. Instead, please\n  use [tf.data](https://www.tensorflow.org/guide/data) to get data into your\n  model.\n  @end_compatibility\n\n  Args:\n    sess: `Session` used to run the queue ops.  Defaults to the\n      default session.\n    coord: Optional `Coordinator` for coordinating the started threads.\n    daemon: Whether the threads should be marked as `daemons`, meaning\n      they don't block program exit.\n    start: Set to `False` to only create the threads, not start them.\n    collection: A `GraphKey` specifying the graph collection to\n      get the queue runners from.  Defaults to `GraphKeys.QUEUE_RUNNERS`.\n\n  Raises:\n    ValueError: if `sess` is None and there isn't any default session.\n    TypeError: if `sess` is not a `tf.compat.v1.Session` object.\n\n  Returns:\n    A list of threads.\n\n  Raises:\n    RuntimeError: If called with eager execution enabled.\n    ValueError: If called without a default `tf.compat.v1.Session` registered.\n  "
    if context.executing_eagerly():
        raise RuntimeError('Queues are not compatible with eager execution.')
    if sess is None:
        sess = ops.get_default_session()
        if not sess:
            raise ValueError('Cannot start queue runners: No default session is registered. Use `with sess.as_default()` or pass an explicit session to tf.start_queue_runners(sess=sess)')
    if not isinstance(sess, session.SessionInterface):
        if sess.__class__.__name__ in ['MonitoredSession', 'SingularMonitoredSession']:
            return []
        raise TypeError('sess must be a `tf.Session` object. Given class: {}'.format(sess.__class__))
    queue_runners = ops.get_collection(collection)
    if not queue_runners:
        logging.warning('`tf.train.start_queue_runners()` was called when no queue runners were defined. You can safely remove the call to this deprecated function.')
    with sess.graph.as_default():
        threads = []
        for qr in ops.get_collection(collection):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=daemon, start=start))
    return threads
ops.register_proto_function(ops.GraphKeys.QUEUE_RUNNERS, proto_type=queue_runner_pb2.QueueRunnerDef, to_proto=QueueRunner.to_proto, from_proto=QueueRunner.from_proto)