from __future__ import absolute_import
import unittest
from st2common.transport.connection_retry_wrapper import ClusterRetryContext
from six.moves import range

class TestClusterRetryContext(unittest.TestCase):

    def test_single_node_cluster_retry(self):
        if False:
            return 10
        retry_context = ClusterRetryContext(cluster_size=1)
        (should_stop, wait) = retry_context.test_should_stop()
        self.assertFalse(should_stop, 'Not done trying.')
        self.assertEqual(wait, 10)
        (should_stop, wait) = retry_context.test_should_stop()
        self.assertFalse(should_stop, 'Not done trying.')
        self.assertEqual(wait, 10)
        (should_stop, wait) = retry_context.test_should_stop()
        self.assertTrue(should_stop, 'Done trying.')
        self.assertEqual(wait, -1)

    def test_should_stop_second_channel_open_error_should_be_non_fatal(self):
        if False:
            print('Hello World!')
        retry_context = ClusterRetryContext(cluster_size=1)
        e = Exception("(504) CHANNEL_ERROR - second 'channel.open' seen")
        (should_stop, wait) = retry_context.test_should_stop(e=e)
        self.assertFalse(should_stop)
        self.assertEqual(wait, -1)
        e = Exception("CHANNEL_ERROR - second 'channel.open' seen")
        (should_stop, wait) = retry_context.test_should_stop(e=e)
        self.assertFalse(should_stop)
        self.assertEqual(wait, -1)

    def test_multiple_node_cluster_retry(self):
        if False:
            while True:
                i = 10
        cluster_size = 3
        last_index = cluster_size * 2
        retry_context = ClusterRetryContext(cluster_size=cluster_size)
        for i in range(last_index + 1):
            (should_stop, wait) = retry_context.test_should_stop()
            if i == last_index:
                self.assertTrue(should_stop, 'Done trying.')
                self.assertEqual(wait, -1)
            else:
                self.assertFalse(should_stop, 'Not done trying.')
                if (i + 1) % cluster_size == 0:
                    self.assertEqual(wait, 10)
                else:
                    self.assertEqual(wait, 0)

    def test_zero_node_cluster_retry(self):
        if False:
            print('Hello World!')
        retry_context = ClusterRetryContext(cluster_size=0)
        (should_stop, wait) = retry_context.test_should_stop()
        self.assertTrue(should_stop, 'Done trying.')
        self.assertEqual(wait, -1)