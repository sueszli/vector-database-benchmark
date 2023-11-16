import open3d as o3d
import numpy as np
import pytest
import mltest
pytestmark = mltest.default_marks

@mltest.parametrize.ml_gpu_only
def test_three_nn(ml):
    if False:
        for i in range(10):
            print('nop')
    values0 = mltest.fetch_numpy('https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/three_nn/values0.npy')
    values1 = mltest.fetch_numpy('https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/three_nn/values1.npy')
    (ans0, ans1) = mltest.run_op(ml, ml.device, True, ml.ops.three_nn, values0, values1)
    expected0 = mltest.fetch_numpy('https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/three_nn/out0.npy')
    expected1 = mltest.fetch_numpy('https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/three_nn/out1.npy')
    np.testing.assert_equal(ans0, expected0)
    np.testing.assert_equal(ans1, expected1)