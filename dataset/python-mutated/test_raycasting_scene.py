import open3d as o3d
import numpy as np
import pytest

def test_cast_rays():
    if False:
        for i in range(10):
            print('nop')
    vertices = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=o3d.core.float32)
    triangles = o3d.core.Tensor([[0, 1, 2]], dtype=o3d.core.uint32)
    scene = o3d.t.geometry.RaycastingScene()
    geom_id = scene.add_triangles(vertices, triangles)
    rays = o3d.core.Tensor([[0.2, 0.1, 1, 0, 0, -1], [10, 10, 10, 1, 0, 0]], dtype=o3d.core.float32)
    ans = scene.cast_rays(rays)
    assert geom_id == ans['geometry_ids'][0]
    assert np.isclose(ans['t_hit'][0].item(), 1.0)
    assert o3d.t.geometry.RaycastingScene.INVALID_ID == ans['geometry_ids'][1]
    assert np.isinf(ans['t_hit'][1].item())

def test_cast_lots_of_rays():
    if False:
        print('Hello World!')
    vertices = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=o3d.core.float32)
    triangles = o3d.core.Tensor([[0, 1, 2]], dtype=o3d.core.uint32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(vertices, triangles)
    rs = np.random.RandomState(123)
    rays = o3d.core.Tensor.from_numpy(rs.rand(7654321, 6).astype(np.float32))
    _ = scene.cast_rays(rays)

def test_test_occlusions():
    if False:
        i = 10
        return i + 15
    vertices = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=o3d.core.float32)
    triangles = o3d.core.Tensor([[0, 1, 2]], dtype=o3d.core.uint32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(vertices, triangles)
    rays = o3d.core.Tensor([[0.2, 0.1, 1, 0, 0, -1], [10, 10, 10, 1, 0, 0]], dtype=o3d.core.float32)
    ans = scene.test_occlusions(rays)
    assert ans[0] == True
    assert ans[1] == False
    ans = scene.test_occlusions(rays, tfar=0.5)
    assert ans.any() == False
    ans = scene.test_occlusions(rays, tnear=1.5)
    assert ans.any() == False

def test_test_lots_of_occlusions():
    if False:
        i = 10
        return i + 15
    vertices = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=o3d.core.float32)
    triangles = o3d.core.Tensor([[0, 1, 2]], dtype=o3d.core.uint32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(vertices, triangles)
    rs = np.random.RandomState(123)
    rays = o3d.core.Tensor.from_numpy(rs.rand(7654321, 6).astype(np.float32))
    _ = scene.test_occlusions(rays)

def test_add_triangle_mesh():
    if False:
        return 10
    cube = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_box())
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)
    rays = o3d.core.Tensor([[0.5, 0.5, -1, 0, 0, 1], [0.5, 0.5, 0.5, 0, 0, 1], [10, 10, 10, 1, 0, 0]], dtype=o3d.core.float32)
    ans = scene.count_intersections(rays)
    np.testing.assert_equal(ans.numpy(), [2, 1, 0])

def test_count_intersections():
    if False:
        for i in range(10):
            print('nop')
    cube = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_box())
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)
    rays = o3d.core.Tensor([[0.5, 0.5, -1, 0, 0, 1], [0.5, 0.5, 0.5, 0, 0, 1], [10, 10, 10, 1, 0, 0]], dtype=o3d.core.float32)
    ans = scene.count_intersections(rays)
    np.testing.assert_equal(ans.numpy(), [2, 1, 0])

def test_count_lots_of_intersections():
    if False:
        while True:
            i = 10
    cube = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_box())
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)
    rs = np.random.RandomState(123)
    rays = o3d.core.Tensor.from_numpy(rs.rand(1234567, 6).astype(np.float32))
    _ = scene.count_intersections(rays)

def test_list_intersections():
    if False:
        while True:
            i = 10
    cube = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_box())
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)
    rays = o3d.core.Tensor([[0.5, 0.5, -1, 0, 0, 1], [0.5, 0.5, 0.5, 0, 0, 1], [10, 10, 10, 1, 0, 0]], dtype=o3d.core.float32)
    ans = scene.list_intersections(rays)
    np.testing.assert_allclose(ans['t_hit'].numpy(), np.array([1.0, 2.0, 0.5]), rtol=1e-06, atol=1e-06)

def test_list_lots_of_intersections():
    if False:
        return 10
    cube = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_box())
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)
    rs = np.random.RandomState(123)
    rays = o3d.core.Tensor.from_numpy(rs.rand(123456, 6).astype(np.float32))
    _ = scene.list_intersections(rays)

def test_compute_closest_points():
    if False:
        print('Hello World!')
    vertices = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=o3d.core.float32)
    triangles = o3d.core.Tensor([[0, 1, 2]], dtype=o3d.core.uint32)
    scene = o3d.t.geometry.RaycastingScene()
    geom_id = scene.add_triangles(vertices, triangles)
    query_points = o3d.core.Tensor([[0.2, 0.1, 1], [10, 10, 10]], dtype=o3d.core.float32)
    ans = scene.compute_closest_points(query_points)
    assert (geom_id == ans['geometry_ids']).all()
    assert (0 == ans['primitive_ids']).all()
    np.testing.assert_allclose(ans['points'].numpy(), np.array([[0.2, 0.1, 0.0], [1, 1, 0]]), rtol=1e-06, atol=1e-06)

def test_compute_lots_of_closest_points():
    if False:
        while True:
            i = 10
    vertices = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=o3d.core.float32)
    triangles = o3d.core.Tensor([[0, 1, 2]], dtype=o3d.core.uint32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(vertices, triangles)
    rs = np.random.RandomState(123)
    query_points = o3d.core.Tensor.from_numpy(rs.rand(1234567, 3).astype(np.float32))
    _ = scene.compute_closest_points(query_points)

def test_compute_distance():
    if False:
        return 10
    cube = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_box())
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)
    query_points = o3d.core.Tensor([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5], [0, 0, 0]], dtype=o3d.core.float32)
    ans = scene.compute_distance(query_points)
    np.testing.assert_allclose(ans.numpy(), [0.5, np.sqrt(3 * 0.5 ** 2), 0.0])

def test_compute_signed_distance():
    if False:
        print('Hello World!')
    cube = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_box())
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)
    query_points = o3d.core.Tensor([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5], [0, 0, 0]], dtype=o3d.core.float32)
    ans = scene.compute_signed_distance(query_points)
    np.testing.assert_allclose(ans.numpy(), [-0.5, np.sqrt(3 * 0.5 ** 2), 0.0])

def test_compute_occupancy():
    if False:
        return 10
    cube = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_box())
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)
    query_points = o3d.core.Tensor([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]], dtype=o3d.core.float32)
    ans = scene.compute_occupancy(query_points)
    np.testing.assert_allclose(ans.numpy(), [1.0, 0.0])

@pytest.mark.parametrize('shape', ([11], [1, 2, 3], [32, 14]))
def test_output_shapes(shape):
    if False:
        while True:
            i = 10
    vertices = o3d.core.Tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=o3d.core.float32)
    triangles = o3d.core.Tensor([[0, 1, 2]], dtype=o3d.core.uint32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(vertices, triangles)
    rs = np.random.RandomState(123)
    rays = o3d.core.Tensor.from_numpy(rs.uniform(size=shape + [6]).astype(np.float32))
    query_points = o3d.core.Tensor.from_numpy(rs.uniform(size=shape + [3]).astype(np.float32))
    ans = scene.count_intersections(rays)
    assert list(ans.shape) == shape
    ans = scene.compute_distance(query_points)
    assert list(ans.shape) == shape
    ans = scene.compute_signed_distance(query_points)
    assert list(ans.shape) == shape
    ans = scene.compute_occupancy(query_points)
    assert list(ans.shape) == shape
    last_dim = {'t_hit': [], 'geometry_ids': [], 'primitive_ids': [], 'primitive_uvs': [2], 'primitive_normals': [3], 'points': [3], 'ray_ids': [], 'ray_splits': []}
    ans = scene.cast_rays(rays)
    for (k, v) in ans.items():
        expected_shape = shape + last_dim[k]
        assert list(v.shape) == expected_shape, 'shape mismatch: expected {} but got {} for {}'.format(expected_shape, list(v.shape), k)
    ans = scene.compute_closest_points(query_points)
    for (k, v) in ans.items():
        expected_shape = shape + last_dim[k]
        assert list(v.shape) == expected_shape, 'shape mismatch: expected {} but got {} for {}'.format(expected_shape, list(v.shape), k)
    ans = scene.list_intersections(rays)
    nx = np.sum(scene.count_intersections(rays).numpy()).tolist()
    for (k, v) in ans.items():
        if k == 'ray_splits':
            alt_shape = [np.prod(rays.shape[:-1]) + 1]
        else:
            alt_shape = [nx]
        expected_shape = np.append(alt_shape, last_dim[k]).tolist()
        assert list(v.shape) == expected_shape, 'shape mismatch: expected {} but got {} for {}'.format(expected_shape, list(v.shape), k)

def test_sphere_wrong_occupancy():
    if False:
        for i in range(10):
            print('nop')
    mesh = o3d.geometry.TriangleMesh.create_sphere(0.8)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    min_bound = mesh.vertex.positions.min(0).numpy() * 1.1
    max_bound = mesh.vertex.positions.max(0).numpy() * 1.1
    xyz_range = np.linspace(min_bound, max_bound, num=6)
    query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)
    occupancy = scene.compute_occupancy(query_points)
    expected = np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
    np.testing.assert_equal(occupancy.numpy(), expected)
    occupancy_3samples = scene.compute_occupancy(query_points, nsamples=3)
    np.testing.assert_equal(occupancy_3samples.numpy(), expected)