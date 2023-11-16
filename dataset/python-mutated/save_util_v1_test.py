"""Tests for the checkpoint/util.py."""
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.eager import test
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import object_identity

class TrackableWithRegisteredSaver(autotrackable.AutoTrackable):
    pass
registration.register_checkpoint_saver(name='RegisteredSaver', predicate=lambda x: isinstance(x, TrackableWithRegisteredSaver), save_fn=lambda trackables, file_prefix: [], restore_fn=lambda trackables, merged_prefix: None)

class SerializationTest(test.TestCase):

    def test_serialize_gathered_objects(self):
        if False:
            for i in range(10):
                print('nop')
        root = autotrackable.AutoTrackable()
        root.v = variables.Variable(1.0)
        root.registered = TrackableWithRegisteredSaver()
        (named_saveable_objects, _, _, registered_savers) = save_util_v1.serialize_gathered_objects(graph_view.ObjectGraphView(root))
        self.assertLen(named_saveable_objects, 1)
        self.assertIs(named_saveable_objects[0].op, root.v)
        self.assertDictEqual({'Custom.RegisteredSaver': {'registered': root.registered}}, registered_savers)

    def test_serialize_gathered_objects_with_map(self):
        if False:
            i = 10
            return i + 15
        root = autotrackable.AutoTrackable()
        root.v = variables.Variable(1.0)
        root.registered = TrackableWithRegisteredSaver()
        copy_of_registered = TrackableWithRegisteredSaver()
        copy_of_v = variables.Variable(1.0)
        object_map = object_identity.ObjectIdentityDictionary()
        object_map[root.registered] = copy_of_registered
        object_map[root.v] = copy_of_v
        (named_saveable_objects, _, _, registered_savers) = save_util_v1.serialize_gathered_objects(graph_view.ObjectGraphView(root), object_map)
        self.assertLen(named_saveable_objects, 1)
        self.assertIsNot(named_saveable_objects[0].op, root.v)
        self.assertIs(named_saveable_objects[0].op, copy_of_v)
        ret_value = registered_savers['Custom.RegisteredSaver']['registered']
        self.assertIsNot(root.registered, ret_value)
        self.assertIs(copy_of_registered, ret_value)
if __name__ == '__main__':
    test.main()