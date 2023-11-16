from __future__ import absolute_import
import re
from unittest2 import TestCase
import st2common.util.queues as queue_utils

class TestQueueUtils(TestCase):

    def test_get_queue_name(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ValueError, queue_utils.get_queue_name, queue_name_base=None, queue_name_suffix=None)
        self.assertRaises(ValueError, queue_utils.get_queue_name, queue_name_base='', queue_name_suffix=None)
        self.assertEqual(queue_utils.get_queue_name(queue_name_base='st2.test.watch', queue_name_suffix=None), 'st2.test.watch')
        self.assertEqual(queue_utils.get_queue_name(queue_name_base='st2.test.watch', queue_name_suffix=''), 'st2.test.watch')
        queue_name = queue_utils.get_queue_name(queue_name_base='st2.test.watch', queue_name_suffix='foo', add_random_uuid_to_suffix=True)
        pattern = re.compile('st2.test.watch.foo-\\w')
        self.assertTrue(re.match(pattern, queue_name))
        queue_name = queue_utils.get_queue_name(queue_name_base='st2.test.watch', queue_name_suffix='foo', add_random_uuid_to_suffix=False)
        self.assertEqual(queue_name, 'st2.test.watch.foo')