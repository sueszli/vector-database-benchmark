"""Manages a Trackable object graph."""
import collections
import weakref
from tensorflow.python.trackable import base
from tensorflow.python.trackable import converter
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export

@tf_export('train.TrackableView', v1=[])
class TrackableView(object):
    """Gathers and serializes a trackable view.

  Example usage:

  >>> class SimpleModule(tf.Module):
  ...   def __init__(self, name=None):
  ...     super().__init__(name=name)
  ...     self.a_var = tf.Variable(5.0)
  ...     self.b_var = tf.Variable(4.0)
  ...     self.vars = [tf.Variable(1.0), tf.Variable(2.0)]

  >>> root = SimpleModule(name="root")
  >>> root.leaf = SimpleModule(name="leaf")
  >>> trackable_view = tf.train.TrackableView(root)

  Pass root to tf.train.TrackableView.children() to get the dictionary of all
  children directly linked to root by name.
  >>> trackable_view_children = trackable_view.children(root)
  >>> for item in trackable_view_children.items():
  ...   print(item)
  ('a_var', <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=5.0>)
  ('b_var', <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=4.0>)
  ('vars', ListWrapper([<tf.Variable 'Variable:0' shape=() dtype=float32,
  numpy=1.0>, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>]))
  ('leaf', ...)

  """

    def __init__(self, root):
        if False:
            return 10
        'Configure the trackable view.\n\n    Args:\n      root: A `Trackable` object whose variables (including the variables of\n        dependencies, recursively) should be saved. May be a weak reference.\n    '
        self._root_ref = root if isinstance(root, weakref.ref) else weakref.ref(root)

    @classmethod
    def children(cls, obj, save_type=base.SaveType.CHECKPOINT, **kwargs):
        if False:
            return 10
        "Returns all child trackables attached to obj.\n\n    Args:\n      obj: A `Trackable` object.\n      save_type: A string, can be 'savedmodel' or 'checkpoint'.\n      **kwargs: kwargs to use when retrieving the object's children.\n\n    Returns:\n      Dictionary of all children attached to the object with name to trackable.\n    "
        obj._maybe_initialize_trackable()
        children = {}
        for (name, ref) in obj._trackable_children(save_type, **kwargs).items():
            ref = converter.convert_to_trackable(ref, parent=obj)
            children[name] = ref
        return children

    @property
    def root(self):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self._root_ref, weakref.ref):
            derefed = self._root_ref()
            assert derefed is not None
            return derefed
        else:
            return self._root_ref

    def descendants(self):
        if False:
            print('Hello World!')
        'Returns a list of all nodes from self.root using a breadth first traversal.'
        return self._descendants_with_paths()[0]

    def _descendants_with_paths(self):
        if False:
            print('Hello World!')
        'Returns a list of all nodes and its paths from self.root using a breadth first traversal.'
        bfs_sorted = []
        to_visit = collections.deque([self.root])
        node_paths = object_identity.ObjectIdentityDictionary()
        node_paths[self.root] = ()
        while to_visit:
            current_trackable = to_visit.popleft()
            bfs_sorted.append(current_trackable)
            for (name, dependency) in self.children(current_trackable).items():
                if dependency not in node_paths:
                    node_paths[dependency] = node_paths[current_trackable] + (base.TrackableReference(name, dependency),)
                    to_visit.append(dependency)
        return (bfs_sorted, node_paths)