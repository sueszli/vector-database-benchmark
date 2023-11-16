import open3d as o3d
import numpy as np
import time
import pytest

@pytest.mark.parametrize('input_array, expect_exception', [(np.ones((0, 3), dtype=np.float64), False), (np.ones((2, 4), dtype=np.float64), True), ([[1, 2, 3], [4, 5, 6]], False), ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], False), (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64), False), (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32), False), (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32), False), (np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.float64)[:, 0:6:2], False), (np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float64).T, False), (np.asfortranarray(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)), False)])
def test_Vector3dVector(input_array, expect_exception):
    if False:
        print('Hello World!')

    def run_test(input_array):
        if False:
            i = 10
            return i + 15
        open3d_array = o3d.utility.Vector3dVector(input_array)
        output_array = np.asarray(open3d_array)
        np.testing.assert_allclose(input_array, output_array)
    if expect_exception:
        with pytest.raises(Exception):
            run_test(input_array)
    else:
        run_test(input_array)

@pytest.mark.parametrize('input_array, expect_exception', [(np.ones((0, 3), dtype=np.int32), False), (np.ones((2, 4), dtype=np.int32), True), ([[1, 2, 3], [4, 5, 6]], False), ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], False), (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64), False), (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32), False), (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32), False), (np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int32)[:, 0:6:2], False), (np.array([[1, 4], [2, 5], [3, 6]], dtype=np.int32).T, False), (np.asfortranarray(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)), False)])
def test_Vector3iVector(input_array, expect_exception):
    if False:
        i = 10
        return i + 15

    def run_test(input_array):
        if False:
            return 10
        open3d_array = o3d.utility.Vector3iVector(input_array)
        output_array = np.asarray(open3d_array)
        np.testing.assert_allclose(input_array, output_array)
    if expect_exception:
        with pytest.raises(Exception):
            run_test(input_array)
    else:
        run_test(input_array)

@pytest.mark.parametrize('input_array, expect_exception', [(np.ones((0, 2), dtype=np.int32), False), (np.ones((10, 3), dtype=np.int32), True), ([[1, 2], [4, 5]], False), ([[1.0, 2.0], [4.0, 5.0]], False), (np.array([[1, 2], [4, 5]], dtype=np.float64), False), (np.array([[1, 2], [4, 5]], dtype=np.int32), False), (np.array([[1, 2], [4, 5]], dtype=np.int32), False), (np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int32)[:, 0:6:3], False), (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32).T, False), (np.asfortranarray(np.array([[1, 2], [4, 5]], dtype=np.int32)), False)])
def test_Vector2iVector(input_array, expect_exception):
    if False:
        print('Hello World!')

    def run_test(input_array):
        if False:
            return 10
        open3d_array = o3d.utility.Vector2iVector(input_array)
        output_array = np.asarray(open3d_array)
        np.testing.assert_allclose(input_array, output_array)
    if expect_exception:
        with pytest.raises(Exception):
            run_test(input_array)
    else:
        run_test(input_array)

@pytest.mark.parametrize('input_array, expect_exception', [(np.ones((0, 4, 4), dtype=np.float64), False), (np.ones((10, 3), dtype=np.float64), True), (np.ones((10, 3, 3), dtype=np.float64), True), ([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]], False), (np.random.randint(10, size=(10, 4, 4)).astype(np.float64), False), (np.random.randint(10, size=(10, 4, 4)).astype(np.int32), False), (np.random.random((10, 8, 8)).astype(np.float64)[:, 0:8:2, 0:8:2], False), (np.asfortranarray(np.array(np.random.random((10, 4, 4)), dtype=np.float64)), False)])
def test_Matrix4dVector(input_array, expect_exception):
    if False:
        while True:
            i = 10

    def run_test(input_array):
        if False:
            return 10
        open3d_array = o3d.utility.Matrix4dVector(input_array)
        output_array = np.asarray(open3d_array)
        np.testing.assert_allclose(input_array, output_array)
    if expect_exception:
        with pytest.raises(Exception):
            run_test(input_array)
    else:
        run_test(input_array)

def test_benchmark():
    if False:
        for i in range(10):
            print('nop')
    vector_size = int(2000000.0)
    x = np.random.randint(10, size=(vector_size, 3)).astype(np.float64)
    print('\no3d.utility.Vector3dVector:', x.shape)
    start_time = time.time()
    y = o3d.utility.Vector3dVector(x)
    print('open3d -> numpy: %.6fs' % (time.time() - start_time))
    start_time = time.time()
    z = np.asarray(y)
    print('numpy -> open3d: %.6fs' % (time.time() - start_time))
    np.testing.assert_allclose(x, z)
    print('\no3d.utility.Vector3iVector:', x.shape)
    x = np.random.randint(10, size=(vector_size, 3)).astype(np.int32)
    start_time = time.time()
    y = o3d.utility.Vector3iVector(x)
    print('open3d -> numpy: %.6fs' % (time.time() - start_time))
    start_time = time.time()
    z = np.asarray(y)
    print('numpy -> open3d: %.6fs' % (time.time() - start_time))
    np.testing.assert_allclose(x, z)
    print('\no3d.utility.Vector2iVector:', x.shape)
    x = np.random.randint(10, size=(vector_size, 2)).astype(np.int32)
    start_time = time.time()
    y = o3d.utility.Vector2iVector(x)
    print('open3d -> numpy: %.6fs' % (time.time() - start_time))
    start_time = time.time()
    z = np.asarray(y)
    print('numpy -> open3d: %.6fs' % (time.time() - start_time))
    np.testing.assert_allclose(x, z)