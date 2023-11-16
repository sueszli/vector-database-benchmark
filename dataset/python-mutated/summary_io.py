"""Reads Summaries from and writes Summaries to event files."""
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.summary.writer.writer import FileWriter as _FileWriter
from tensorflow.python.summary.writer.writer_cache import FileWriterCache as SummaryWriterCache
from tensorflow.python.util.deprecation import deprecated

class SummaryWriter(_FileWriter):

    @deprecated('2016-11-30', 'Please switch to tf.summary.FileWriter. The interface and behavior is the same; this is just a rename.')
    def __init__(self, logdir, graph=None, max_queue=10, flush_secs=120, graph_def=None):
        if False:
            return 10
        "Creates a `SummaryWriter` and an event file.\n\n    This class is deprecated, and should be replaced with tf.summary.FileWriter.\n\n    On construction the summary writer creates a new event file in `logdir`.\n    This event file will contain `Event` protocol buffers constructed when you\n    call one of the following functions: `add_summary()`, `add_session_log()`,\n    `add_event()`, or `add_graph()`.\n\n    If you pass a `Graph` to the constructor it is added to\n    the event file. (This is equivalent to calling `add_graph()` later).\n\n    TensorBoard will pick the graph from the file and display it graphically so\n    you can interactively explore the graph you built. You will usually pass\n    the graph from the session in which you launched it:\n\n    ```python\n    ...create a graph...\n    # Launch the graph in a session.\n    sess = tf.compat.v1.Session()\n    # Create a summary writer, add the 'graph' to the event file.\n    writer = tf.compat.v1.summary.FileWriter(<some-directory>, sess.graph)\n    ```\n\n    The other arguments to the constructor control the asynchronous writes to\n    the event file:\n\n    *  `flush_secs`: How often, in seconds, to flush the added summaries\n       and events to disk.\n    *  `max_queue`: Maximum number of summaries or events pending to be\n       written to disk before one of the 'add' calls block.\n\n    Args:\n      logdir: A string. Directory where event file will be written.\n      graph: A `Graph` object, such as `sess.graph`.\n      max_queue: Integer. Size of the queue for pending events and summaries.\n      flush_secs: Number. How often, in seconds, to flush the\n        pending events and summaries to disk.\n      graph_def: DEPRECATED: Use the `graph` argument instead.\n    "
        super(SummaryWriter, self).__init__(logdir, graph, max_queue, flush_secs, graph_def)