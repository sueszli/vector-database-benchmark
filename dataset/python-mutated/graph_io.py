"""Utility functions for reading/writing graphs."""
import os
import os.path
import sys
from google.protobuf import text_format
from tensorflow.python.framework import byte_swap_tensor
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.tf_export import tf_export

@tf_export('io.write_graph', v1=['io.write_graph', 'train.write_graph'])
def write_graph(graph_or_graph_def, logdir, name, as_text=True):
    if False:
        print('Hello World!')
    "Writes a graph proto to a file.\n\n  The graph is written as a text proto unless `as_text` is `False`.\n\n  ```python\n  v = tf.Variable(0, name='my_variable')\n  sess = tf.compat.v1.Session()\n  tf.io.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')\n  ```\n\n  or\n\n  ```python\n  v = tf.Variable(0, name='my_variable')\n  sess = tf.compat.v1.Session()\n  tf.io.write_graph(sess.graph, '/tmp/my-model', 'train.pbtxt')\n  ```\n\n  Args:\n    graph_or_graph_def: A `Graph` or a `GraphDef` protocol buffer.\n    logdir: Directory where to write the graph. This can refer to remote\n      filesystems, such as Google Cloud Storage (GCS).\n    name: Filename for the graph.\n    as_text: If `True`, writes the graph as an ASCII proto.\n\n  Returns:\n    The path of the output proto file.\n  "
    if isinstance(graph_or_graph_def, ops.Graph):
        graph_def = graph_or_graph_def.as_graph_def()
    else:
        graph_def = graph_or_graph_def
    if sys.byteorder == 'big':
        if hasattr(graph_def, 'node'):
            byte_swap_tensor.swap_tensor_content_in_graph_node(graph_def, 'big', 'little')
        else:
            byte_swap_tensor.swap_tensor_content_in_graph_function(graph_def, 'big', 'little')
    if not logdir.startswith('gs:'):
        file_io.recursive_create_dir(logdir)
    path = os.path.join(logdir, name)
    if as_text:
        file_io.atomic_write_string_to_file(path, text_format.MessageToString(graph_def, float_format=''))
    else:
        file_io.atomic_write_string_to_file(path, graph_def.SerializeToString(deterministic=True))
    return path