"""Unit tests for runtime_util.py."""
import unittest
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime import runtime_util
from streamlit.runtime.runtime_util import is_cacheable_msg, serialize_forward_msg
from tests.streamlit.message_mocks import create_dataframe_msg
from tests.testutil import patch_config_options

class RuntimeUtilTest(unittest.TestCase):

    def test_should_cache_msg(self):
        if False:
            return 10
        'Test runtime_util.should_cache_msg'
        with patch_config_options({'global.minCachedMessageSize': 0}):
            self.assertTrue(is_cacheable_msg(create_dataframe_msg([1, 2, 3])))
        with patch_config_options({'global.minCachedMessageSize': 1000}):
            self.assertFalse(is_cacheable_msg(create_dataframe_msg([1, 2, 3])))

    def test_should_limit_msg_size(self):
        if False:
            i = 10
            return i + 15
        max_message_size_mb = 50
        runtime_util._max_message_size_bytes = None
        with patch_config_options({'server.maxMessageSize': max_message_size_mb}):
            large_msg = create_dataframe_msg([1, 2, 3])
            large_msg.delta.new_element.markdown.body = 'X' * (max_message_size_mb + 10) * 1000 * 1000
            large_msg_copy = ForwardMsg()
            large_msg_copy.CopyFrom(large_msg)
            deserialized_msg = ForwardMsg()
            deserialized_msg.ParseFromString(serialize_forward_msg(large_msg_copy))
            self.assertEqual(deserialized_msg.metadata, large_msg.metadata)
            self.assertNotEqual(deserialized_msg, large_msg)
            self.assertTrue('exceeds the message size limit' in deserialized_msg.delta.new_element.exception.message)