"""Utilities to test summaries."""
import os
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import gfile

def events_from_file(filepath):
    if False:
        while True:
            i = 10
    'Returns all events in a single event file.\n\n  Args:\n    filepath: Path to the event file.\n\n  Returns:\n    A list of all tf.compat.v1.Event protos in the event file.\n  '
    records = list(tf_record.tf_record_iterator(filepath))
    result = []
    for r in records:
        event = event_pb2.Event()
        event.ParseFromString(r)
        result.append(event)
    return result

def events_from_logdir(logdir):
    if False:
        for i in range(10):
            print('nop')
    'Returns all events in the single eventfile in logdir.\n\n  Args:\n    logdir: The directory in which the single event file is sought.\n\n  Returns:\n    A list of all tf.compat.v1.Event protos from the single event file.\n\n  Raises:\n    AssertionError: If logdir does not contain exactly one file.\n  '
    assert gfile.Exists(logdir)
    files = gfile.ListDirectory(logdir)
    assert len(files) == 1, 'Found not exactly one file in logdir: %s' % files
    return events_from_file(os.path.join(logdir, files[0]))