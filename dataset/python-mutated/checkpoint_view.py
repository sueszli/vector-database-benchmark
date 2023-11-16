"""Manages a Checkpoint View."""
import collections
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import trackable_view
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export

@tf_export('train.CheckpointView', v1=[])
class CheckpointView(object):
    """Gathers and serializes a checkpoint view.

  This is for loading specific portions of a module from a
  checkpoint, and be able to compare two modules by matching components.

  Example usage:

  >>> class SimpleModule(tf.Module):
  ...   def __init__(self, name=None):
  ...     super().__init__(name=name)
  ...     self.a_var = tf.Variable(5.0)
  ...     self.b_var = tf.Variable(4.0)
  ...     self.vars = [tf.Variable(1.0), tf.Variable(2.0)]

  >>> root = SimpleModule(name="root")
  >>> root.leaf = SimpleModule(name="leaf")
  >>> ckpt = tf.train.Checkpoint(root)
  >>> save_path = ckpt.save('/tmp/tf_ckpts')
  >>> checkpoint_view = tf.train.CheckpointView(save_path)

  Pass `node_id=0` to `tf.train.CheckpointView.children()` to get the dictionary
  of all children directly linked to the checkpoint root.

  >>> for name, node_id in checkpoint_view.children(0).items():
  ...   print(f"- name: '{name}', node_id: {node_id}")
  - name: 'a_var', node_id: 1
  - name: 'b_var', node_id: 2
  - name: 'vars', node_id: 3
  - name: 'leaf', node_id: 4
  - name: 'root', node_id: 0
  - name: 'save_counter', node_id: 5

  """

    def __init__(self, save_path):
        if False:
            return 10
        'Configure the checkpoint view.\n\n    Args:\n      save_path: The path to the checkpoint.\n\n    Raises:\n      ValueError: If the save_path does not lead to a TF2 checkpoint.\n    '
        reader = py_checkpoint_reader.NewCheckpointReader(save_path)
        try:
            object_graph_string = reader.get_tensor(base.OBJECT_GRAPH_PROTO_KEY)
        except errors_impl.NotFoundError as not_found_error:
            raise ValueError(f'The specified checkpoint "{save_path}" does not appear to be object-based (saved with TF2) since it is missing the key "{base.OBJECT_GRAPH_PROTO_KEY}". Likely it was created with the TF1 name-based saver and does not contain an object dependency graph.') from not_found_error
        object_graph_proto = trackable_object_graph_pb2.TrackableObjectGraph()
        object_graph_proto.ParseFromString(object_graph_string)
        self._object_graph_proto = object_graph_proto

    def children(self, node_id):
        if False:
            for i in range(10):
                print('nop')
        'Returns all child trackables attached to obj.\n\n    Args:\n      node_id: Id of the node to return its children.\n\n    Returns:\n      Dictionary of all children attached to the object with name to node_id.\n    '
        return {child.local_name: child.node_id for child in self._object_graph_proto.nodes[node_id].children}

    def descendants(self):
        if False:
            i = 10
            return i + 15
        'Returns a list of trackables by node_id attached to obj.'
        return list(self._descendants_with_paths().keys())

    def _descendants_with_paths(self):
        if False:
            while True:
                i = 10
        'Returns a dict of descendants by node_id and paths to node.\n\n    The names returned by this private method are subject to change.\n    '
        all_nodes_with_paths = {}
        to_visit = collections.deque([0])
        all_nodes_with_paths[0] = 'root'
        path = all_nodes_with_paths.get(0)
        while to_visit:
            node_id = to_visit.popleft()
            obj = self._object_graph_proto.nodes[node_id]
            for child in obj.children:
                if child.node_id == 0 or child.node_id in all_nodes_with_paths.keys():
                    continue
                path = all_nodes_with_paths.get(node_id)
                if child.node_id not in all_nodes_with_paths.keys():
                    to_visit.append(child.node_id)
                all_nodes_with_paths[child.node_id] = path + '.' + child.local_name
        return all_nodes_with_paths

    def match(self, obj):
        if False:
            for i in range(10):
                print('nop')
        'Returns all matching trackables between CheckpointView and Trackable.\n\n    Matching trackables represents trackables with the same name and position in\n    graph.\n\n    Args:\n      obj: `Trackable` root.\n\n    Returns:\n      Dictionary containing all overlapping trackables that maps `node_id` to\n      `Trackable`.\n\n    Example usage:\n\n    >>> class SimpleModule(tf.Module):\n    ...   def __init__(self, name=None):\n    ...     super().__init__(name=name)\n    ...     self.a_var = tf.Variable(5.0)\n    ...     self.b_var = tf.Variable(4.0)\n    ...     self.vars = [tf.Variable(1.0), tf.Variable(2.0)]\n\n    >>> root = SimpleModule(name="root")\n    >>> leaf = root.leaf = SimpleModule(name="leaf")\n    >>> leaf.leaf3 = tf.Variable(6.0, name="leaf3")\n    >>> leaf.leaf4 = tf.Variable(7.0, name="leaf4")\n    >>> ckpt = tf.train.Checkpoint(root)\n    >>> save_path = ckpt.save(\'/tmp/tf_ckpts\')\n    >>> checkpoint_view = tf.train.CheckpointView(save_path)\n\n    >>> root2 = SimpleModule(name="root")\n    >>> leaf2 = root2.leaf2 = SimpleModule(name="leaf2")\n    >>> leaf2.leaf3 = tf.Variable(6.0)\n    >>> leaf2.leaf4 = tf.Variable(7.0)\n\n    Pass `node_id=0` to `tf.train.CheckpointView.children()` to get the\n    dictionary of all children directly linked to the checkpoint root.\n\n    >>> checkpoint_view_match = checkpoint_view.match(root2).items()\n    >>> for item in checkpoint_view_match:\n    ...   print(item)\n    (0, ...)\n    (1, <tf.Variable \'Variable:0\' shape=() dtype=float32, numpy=5.0>)\n    (2, <tf.Variable \'Variable:0\' shape=() dtype=float32, numpy=4.0>)\n    (3, ListWrapper([<tf.Variable \'Variable:0\' shape=() dtype=float32,\n    numpy=1.0>, <tf.Variable \'Variable:0\' shape=() dtype=float32, numpy=2.0>]))\n    (6, <tf.Variable \'Variable:0\' shape=() dtype=float32, numpy=1.0>)\n    (7, <tf.Variable \'Variable:0\' shape=() dtype=float32, numpy=2.0>)\n\n    '
        if not isinstance(obj, base.Trackable):
            raise ValueError(f'Expected a Trackable, got {obj} of type {type(obj)}.')
        overlapping_nodes = {}
        overlapping_nodes[0] = obj
        to_visit = collections.deque([(0, obj)])
        visited = set()
        view = trackable_view.TrackableView(obj)
        while to_visit:
            (current_node_id, current_trackable) = to_visit.popleft()
            trackable_children = view.children(current_trackable)
            for (child_name, child_node_id) in self.children(current_node_id).items():
                if child_node_id in visited or child_node_id == 0:
                    continue
                if child_name in trackable_children:
                    current_assignment = overlapping_nodes.get(child_node_id)
                    if current_assignment is None:
                        overlapping_nodes[child_node_id] = trackable_children[child_name]
                        to_visit.append((child_node_id, trackable_children[child_name]))
                    elif current_assignment is not trackable_children[child_name]:
                        logging.warning(f'Inconsistent references when matching the checkpoint into this object graph. The referenced objects are: ({current_assignment} and {trackable_children[child_name]}).')
            visited.add(current_node_id)
        return overlapping_nodes

    def diff(self, obj):
        if False:
            i = 10
            return i + 15
        'Returns diff between CheckpointView and Trackable.\n\n    This method is intended to be used to compare the object stored in a\n    checkpoint vs a live model in Python. For example, if checkpoint\n    restoration fails the `assert_consumed()` or\n    `assert_existing_objects_matched()` checks, you can use this to list out\n    the objects/checkpoint nodes which were not restored.\n\n    Example Usage:\n\n    >>> class SimpleModule(tf.Module):\n    ...   def __init__(self, name=None):\n    ...     super().__init__(name=name)\n    ...     self.a_var = tf.Variable(5.0)\n    ...     self.b_var = tf.Variable(4.0)\n    ...     self.vars = [tf.Variable(1.0), tf.Variable(2.0)]\n\n    >>> root = SimpleModule(name="root")\n    >>> leaf = root.leaf = SimpleModule(name="leaf")\n    >>> leaf.leaf3 = tf.Variable(6.0, name="leaf3")\n    >>> leaf.leaf4 = tf.Variable(7.0, name="leaf4")\n    >>> ckpt = tf.train.Checkpoint(root)\n    >>> save_path = ckpt.save(\'/tmp/tf_ckpts\')\n    >>> checkpoint_view = tf.train.CheckpointView(save_path)\n\n    >>> root2 = SimpleModule(name="root")\n    >>> leaf2 = root2.leaf2 = SimpleModule(name="leaf2")\n    >>> leaf2.leaf3 = tf.Variable(6.0)\n    >>> leaf2.leaf4 = tf.Variable(7.0)\n\n    Pass `node_id=0` to `tf.train.CheckpointView.children()` to get the\n    dictionary of all children directly linked to the checkpoint root.\n\n    >>> checkpoint_view_diff = checkpoint_view.diff(root2)\n    >>> checkpoint_view_match = checkpoint_view_diff[0].items()\n    >>> for item in checkpoint_view_match:\n    ...   print(item)\n    (0, ...)\n    (1, <tf.Variable \'Variable:0\' shape=() dtype=float32, numpy=5.0>)\n    (2, <tf.Variable \'Variable:0\' shape=() dtype=float32, numpy=4.0>)\n    (3, ListWrapper([<tf.Variable \'Variable:0\' shape=() dtype=float32,\n    numpy=1.0>, <tf.Variable \'Variable:0\' shape=() dtype=float32, numpy=2.0>]))\n    (6, <tf.Variable \'Variable:0\' shape=() dtype=float32, numpy=1.0>)\n    (7, <tf.Variable \'Variable:0\' shape=() dtype=float32, numpy=2.0>)\n\n    >>> only_in_checkpoint_view = checkpoint_view_diff[1]\n    >>> print(only_in_checkpoint_view)\n    [4, 5, 8, 9, 10, 11, 12, 13, 14]\n\n    >>> only_in_trackable = checkpoint_view_diff[2]\n    >>> print(only_in_trackable)\n    [..., <tf.Variable \'Variable:0\' shape=() dtype=float32, numpy=5.0>,\n    <tf.Variable \'Variable:0\' shape=() dtype=float32, numpy=4.0>,\n    ListWrapper([<tf.Variable \'Variable:0\' shape=() dtype=float32, numpy=1.0>,\n    <tf.Variable \'Variable:0\' shape=() dtype=float32, numpy=2.0>]),\n    <tf.Variable \'Variable:0\' shape=() dtype=float32, numpy=6.0>,\n    <tf.Variable \'Variable:0\' shape=() dtype=float32, numpy=7.0>,\n    <tf.Variable \'Variable:0\' shape=() dtype=float32, numpy=1.0>,\n    <tf.Variable \'Variable:0\' shape=() dtype=float32, numpy=2.0>]\n\n    Args:\n      obj: `Trackable` root.\n\n    Returns:\n      Tuple of (\n      - Overlaps: Dictionary containing all overlapping trackables that maps\n      `node_id` to `Trackable`, same as CheckpointView.match().\n      - Only in CheckpointView: List of `node_id` that only exist in\n      CheckpointView.\n      - Only in Trackable: List of `Trackable` that only exist in Trackable.\n      )\n\n    '
        overlapping_nodes = self.match(obj)
        only_in_checkpoint_view = []
        only_in_trackable = []
        for node_id in self.descendants():
            if node_id not in overlapping_nodes.keys():
                only_in_checkpoint_view.append(node_id)
        for trackable in trackable_view.TrackableView(obj).descendants():
            if trackable not in object_identity.ObjectIdentitySet(overlapping_nodes.values()):
                only_in_trackable.append(trackable)
        return (overlapping_nodes, only_in_checkpoint_view, only_in_trackable)