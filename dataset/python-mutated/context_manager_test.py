"""Tests for tensorflow_models.object_detection.utils.context_manager."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from object_detection.utils import context_manager

class ContextManagerTest(tf.test.TestCase):

    def test_identity_context_manager(self):
        if False:
            while True:
                i = 10
        with context_manager.IdentityContextManager() as identity_context:
            self.assertIsNone(identity_context)
if __name__ == '__main__':
    tf.test.main()