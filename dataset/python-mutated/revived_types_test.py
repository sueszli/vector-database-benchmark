"""Tests for revived type matching."""
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.platform import test
from tensorflow.python.saved_model import revived_types
from tensorflow.python.trackable import autotrackable

class CustomTestClass(autotrackable.AutoTrackable):

    def __init__(self, version):
        if False:
            print('Hello World!')
        self.version = version
revived_types.register_revived_type('test_type', lambda obj: isinstance(obj, CustomTestClass), versions=[revived_types.VersionedTypeRegistration(object_factory=lambda _: CustomTestClass(1), version=1, min_producer_version=1, min_consumer_version=1), revived_types.VersionedTypeRegistration(object_factory=lambda _: CustomTestClass(2), version=2, min_producer_version=2, min_consumer_version=1), revived_types.VersionedTypeRegistration(object_factory=lambda _: CustomTestClass(3), version=3, min_producer_version=3, min_consumer_version=2), revived_types.VersionedTypeRegistration(object_factory=lambda _: CustomTestClass(4), version=4, min_producer_version=4, min_consumer_version=2, bad_consumers=[3])])

class RegistrationMatchingTest(test.TestCase):

    def test_save_typecheck(self):
        if False:
            i = 10
            return i + 15
        self.assertIs(revived_types.serialize(autotrackable.AutoTrackable()), None)

    def test_load_identifier_not_found(self):
        if False:
            for i in range(10):
                print('nop')
        nothing_matches = revived_types.deserialize(saved_object_graph_pb2.SavedUserObject(identifier='_unregistered_type', version=versions_pb2.VersionDef(producer=1, min_consumer=1, bad_consumers=[])))
        self.assertIs(nothing_matches, None)

    def test_most_recent_version_saved(self):
        if False:
            for i in range(10):
                print('nop')
        serialized = revived_types.serialize(CustomTestClass(None))
        self.assertEqual([3], serialized.version.bad_consumers)
        (deserialized, _) = revived_types.deserialize(serialized)
        self.assertIsInstance(deserialized, CustomTestClass)
        self.assertEqual(4, deserialized.version)

    def test_min_consumer_version(self):
        if False:
            return 10
        nothing_matches = revived_types.deserialize(saved_object_graph_pb2.SavedUserObject(identifier='test_type', version=versions_pb2.VersionDef(producer=5, min_consumer=5, bad_consumers=[])))
        self.assertIs(nothing_matches, None)

    def test_bad_versions(self):
        if False:
            i = 10
            return i + 15
        (deserialized, _) = revived_types.deserialize(saved_object_graph_pb2.SavedUserObject(identifier='test_type', version=versions_pb2.VersionDef(producer=5, min_consumer=1, bad_consumers=[4, 3])))
        self.assertEqual(2, deserialized.version)

    def test_min_producer_version(self):
        if False:
            i = 10
            return i + 15
        (deserialized, _) = revived_types.deserialize(saved_object_graph_pb2.SavedUserObject(identifier='test_type', version=versions_pb2.VersionDef(producer=3, min_consumer=0, bad_consumers=[])))
        self.assertEqual(3, deserialized.version)
if __name__ == '__main__':
    test.main()