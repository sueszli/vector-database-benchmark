"""Writes events to disk in a logdir."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import gfile

class EventFileWriterV2(object):
    """Writes `Event` protocol buffers to an event file via the graph.

  The `EventFileWriterV2` class is backed by the summary file writer in the v2
  summary API (currently in tf.contrib.summary), so it uses a shared summary
  writer resource and graph ops to write events.

  As with the original EventFileWriter, this class will asynchronously write
  Event protocol buffers to the backing file. The Event file is encoded using
  the tfrecord format, which is similar to RecordIO.
  """

    def __init__(self, session, logdir, max_queue=10, flush_secs=120, filename_suffix=''):
        if False:
            print('Hello World!')
        "Creates an `EventFileWriterV2` and an event file to write to.\n\n    On construction, this calls `tf.contrib.summary.create_file_writer` within\n    the graph from `session.graph` to look up a shared summary writer resource\n    for `logdir` if one exists, and create one if not. Creating the summary\n    writer resource in turn creates a new event file in `logdir` to be filled\n    with `Event` protocol buffers passed to `add_event`. Graph ops to control\n    this writer resource are added to `session.graph` during this init call;\n    stateful methods on this class will call `session.run()` on these ops.\n\n    Note that because the underlying resource is shared, it is possible that\n    other parts of the code using the same session may interact independently\n    with the resource, e.g. by flushing or even closing it. It is the caller's\n    responsibility to avoid any undesirable sharing in this regard.\n\n    The remaining arguments to the constructor (`flush_secs`, `max_queue`, and\n    `filename_suffix`) control the construction of the shared writer resource\n    if one is created. If an existing resource is reused, these arguments have\n    no effect.  See `tf.contrib.summary.create_file_writer` for details.\n\n    Args:\n      session: A `tf.compat.v1.Session`. Session that will hold shared writer\n        resource. The writer ops will be added to session.graph during this\n        init call.\n      logdir: A string. Directory where event file will be written.\n      max_queue: Integer. Size of the queue for pending events and summaries.\n      flush_secs: Number. How often, in seconds, to flush the\n        pending events and summaries to disk.\n      filename_suffix: A string. Every event file's name is suffixed with\n        `filename_suffix`.\n    "
        self._session = session
        self._logdir = logdir
        self._closed = False
        gfile.MakeDirs(self._logdir)
        with self._session.graph.as_default():
            with ops.name_scope('filewriter'):
                file_writer = summary_ops_v2.create_file_writer(logdir=self._logdir, max_queue=max_queue, flush_millis=flush_secs * 1000, filename_suffix=filename_suffix)
                with summary_ops_v2.always_record_summaries(), file_writer.as_default():
                    self._event_placeholder = array_ops.placeholder_with_default(constant_op.constant('unused', dtypes.string), shape=[])
                    self._add_event_op = summary_ops_v2.import_event(self._event_placeholder)
                self._init_op = file_writer.init()
                self._flush_op = file_writer.flush()
                self._close_op = file_writer.close()
            self._session.run(self._init_op)

    def get_logdir(self):
        if False:
            while True:
                i = 10
        'Returns the directory where event file will be written.'
        return self._logdir

    def reopen(self):
        if False:
            return 10
        'Reopens the EventFileWriter.\n\n    Can be called after `close()` to add more events in the same directory.\n    The events will go into a new events file.\n\n    Does nothing if the EventFileWriter was not closed.\n    '
        if self._closed:
            self._closed = False
            self._session.run(self._init_op)

    def add_event(self, event):
        if False:
            i = 10
            return i + 15
        'Adds an event to the event file.\n\n    Args:\n      event: An `Event` protocol buffer.\n    '
        if not self._closed:
            event_pb = event.SerializeToString()
            self._session.run(self._add_event_op, feed_dict={self._event_placeholder: event_pb})

    def flush(self):
        if False:
            i = 10
            return i + 15
        'Flushes the event file to disk.\n\n    Call this method to make sure that all pending events have been written to\n    disk.\n    '
        self._session.run(self._flush_op)

    def close(self):
        if False:
            while True:
                i = 10
        'Flushes the event file to disk and close the file.\n\n    Call this method when you do not need the summary writer anymore.\n    '
        if not self._closed:
            self.flush()
            self._session.run(self._close_op)
            self._closed = True