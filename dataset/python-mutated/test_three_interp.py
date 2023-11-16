import open3d as o3d
import numpy as np
import pytest
import mltest
pytestmark = mltest.default_marks

@mltest.parametrize.ml_gpu_only
def test_three_interp(ml):
    if False:
        for i in range(10):
            print('nop')
    values0 = mltest.fetch_numpy('https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/three_interp/values0.npy')
    values1 = mltest.fetch_numpy('https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/three_interp/values1.npy')
    values2 = mltest.fetch_numpy('https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/three_interp/values2.npy')
    ans = mltest.run_op(ml, ml.device, True, ml.ops.three_interpolate, values0, values1, values2)
    expected = mltest.fetch_numpy('https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/three_interp/out.npy')
    np.testing.assert_equal(ans, expected)