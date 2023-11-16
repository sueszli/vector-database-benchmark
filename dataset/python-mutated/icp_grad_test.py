"""Tests for icp grad."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import icp_grad
import icp_test
import tensorflow as tf
from tensorflow.python.ops import gradient_checker

class IcpOpGradTest(icp_test.IcpOpTestBase):

    def test_grad_transform(self):
        if False:
            while True:
                i = 10
        with self.test_session():
            cloud_source = self.small_cloud
            cloud_target = cloud_source + [0.05, 0, 0]
            ego_motion = self.identity_transform
            (transform, unused_residual) = self._run_icp(cloud_source, ego_motion, cloud_target)
            err = gradient_checker.compute_gradient_error(ego_motion, ego_motion.shape.as_list(), transform, transform.shape.as_list())
        self.assertGreater(err, 0.001)

    def test_grad_transform_same_ego_motion(self):
        if False:
            for i in range(10):
                print('nop')
        with self.test_session():
            cloud_source = self.small_cloud
            cloud_target = cloud_source + [0.1, 0, 0]
            ego_motion = tf.constant([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
            (transform, unused_residual) = self._run_icp(cloud_source, ego_motion, cloud_target)
            err = gradient_checker.compute_gradient_error(ego_motion, ego_motion.shape.as_list(), transform, transform.shape.as_list())
        self.assertGreater(err, 0.001)

    def test_grad_residual(self):
        if False:
            print('Hello World!')
        with self.test_session():
            cloud_source = self.small_cloud
            cloud_target = cloud_source + [0.05, 0, 0]
            ego_motion = self.identity_transform
            (unused_transform, residual) = self._run_icp(cloud_source, ego_motion, cloud_target)
            err = gradient_checker.compute_gradient_error(cloud_source, cloud_source.shape.as_list(), residual, residual.shape.as_list())
        self.assertGreater(err, 0.001)
if __name__ == '__main__':
    tf.test.main()