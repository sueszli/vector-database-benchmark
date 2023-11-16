import numpy as np
import pytest
from vispy import scene
from vispy.geometry import create_sphere
from vispy.testing import TestingCanvas, requires_application, run_tests_if_main, requires_pyopengl

@requires_pyopengl()
@requires_application()
def test_mesh_normals():
    if False:
        for i in range(10):
            print('nop')
    size = (45, 40)
    with TestingCanvas(size=size, bgcolor='k') as c:
        v = c.central_widget.add_view(border_width=0)
        v.camera = 'arcball'
        mdata = create_sphere(radius=1.0)
        mesh = scene.visuals.Mesh(meshdata=mdata, shading=None, color=(0.1, 0.1, 0.1, 1.0))
        v.add(mesh)
        rendered_without_normals = c.render()
        assert np.all(rendered_without_normals[..., 0:3] < 32)
        face_normals = scene.visuals.MeshNormals(mdata, primitive='face', color=(1, 0, 0))
        face_normals.parent = mesh
        rendered_with_face_normals = c.render()
        face_normals.parent = None
        assert np.sum(rendered_with_face_normals[..., 0] > 128) > 64
        pytest.raises(AssertionError, np.testing.assert_allclose, rendered_without_normals, rendered_with_face_normals)
        vertex_normals = scene.visuals.MeshNormals(mdata, primitive='vertex', color=(0, 1, 0))
        vertex_normals.parent = mesh
        rendered_with_vertex_normals = c.render()
        vertex_normals.parent = None
        assert np.sum(rendered_with_vertex_normals[..., 1] > 128) > 64
        pytest.raises(AssertionError, np.testing.assert_allclose, rendered_without_normals, rendered_with_vertex_normals)
        pytest.raises(AssertionError, np.testing.assert_allclose, rendered_with_face_normals, rendered_with_vertex_normals)

@requires_pyopengl()
@requires_application()
def test_mesh_normals_length_scalar():
    if False:
        print('Hello World!')
    size = (45, 40)
    with TestingCanvas(size=size, bgcolor='k') as c:
        v = c.central_widget.add_view(border_width=0)
        v.camera = 'arcball'
        mdata = create_sphere(radius=1.0)
        mesh = scene.visuals.Mesh(meshdata=mdata, shading=None, color=(0.1, 0.1, 0.1, 1.0))
        v.add(mesh)
        length = 0.5
        normals_0_5 = scene.visuals.MeshNormals(mdata, color=(1, 0, 0), length=length)
        normals_0_5.parent = mesh
        rendered_length_0_5 = c.render()
        normals_0_5.parent = None
        length = 1.0
        normals_1_0 = scene.visuals.MeshNormals(mdata, color=(1, 0, 0), length=length)
        normals_1_0.parent = mesh
        rendered_length_1_0 = c.render()
        normals_1_0.parent = None
        n_pixels_0_5 = np.sum(rendered_length_0_5[..., 0] > 128)
        n_pixels_1_0 = np.sum(rendered_length_1_0[..., 0] > 128)
        assert n_pixels_1_0 > n_pixels_0_5

@requires_pyopengl()
@requires_application()
@pytest.mark.parametrize('primitive', ['face', 'vertex'])
def test_mesh_normals_length_array(primitive):
    if False:
        for i in range(10):
            print('nop')
    size = (45, 40)
    with TestingCanvas(size=size, bgcolor='k') as c:
        v = c.central_widget.add_view(border_width=0)
        v.camera = 'arcball'
        v.camera.fov = 90
        meshdata = create_sphere(radius=1.0)
        mesh = scene.visuals.Mesh(meshdata=meshdata, shading=None, color=(0.1, 0.1, 0.1, 1.0))
        v.add(mesh)
        if primitive == 'face':
            n_normals = len(meshdata.get_faces())
        elif primitive == 'vertex':
            n_normals = len(meshdata.get_vertices())
        lengths_0_5 = np.full(n_normals, 0.5, dtype=float)
        normals_0_5 = scene.visuals.MeshNormals(meshdata, primitive=primitive, color=(1, 0, 0), length=lengths_0_5)
        normals_0_5.parent = mesh
        rendered_lengths_0_5 = c.render()
        normals_0_5.parent = None
        lengths_1_0 = np.full(n_normals, 1.0, dtype=float)
        normals_1_0 = scene.visuals.MeshNormals(meshdata, primitive=primitive, color=(1, 0, 0), length=lengths_1_0)
        normals_1_0.parent = mesh
        rendered_lengths_1_0 = c.render()
        normals_1_0.parent = None
        n_pixels_0_5 = np.sum(rendered_lengths_0_5[..., 0] > 128)
        n_pixels_1_0 = np.sum(rendered_lengths_1_0[..., 0] > 128)
        assert n_pixels_1_0 > n_pixels_0_5
        lengths_ramp = np.linspace(0.5, 1.0, n_normals, dtype=float)
        normals_ramp = scene.visuals.MeshNormals(meshdata, primitive=primitive, color=(1, 0, 0), length=lengths_ramp)
        normals_ramp.parent = mesh
        rendered_lengths_ramp = c.render()
        normals_ramp.parent = None
        n_pixels_ramp = np.sum(rendered_lengths_ramp[..., 0] > 128)
        assert n_pixels_0_5 < n_pixels_ramp < n_pixels_1_0

@requires_pyopengl()
@requires_application()
def test_mesh_normals_length_scale():
    if False:
        while True:
            i = 10
    size = (45, 40)
    with TestingCanvas(size=size, bgcolor='k') as c:
        v = c.central_widget.add_view(border_width=0)
        v.camera = 'arcball'
        meshdata = create_sphere(radius=1.0)
        mesh = scene.visuals.Mesh(meshdata=meshdata, shading=None, color=(0.1, 0.1, 0.1, 1.0))
        v.add(mesh)
        length = 1.0
        length_scale_up = 2.0
        length_scale_down = 0.5
        normals = scene.visuals.MeshNormals(meshdata, color=(1, 0, 0), length=length)
        normals.parent = mesh
        rendered_length_default = c.render()
        normals.parent = None
        normals_scaled_up = scene.visuals.MeshNormals(meshdata, color=(1, 0, 0), length=length, length_scale=length_scale_up)
        normals_scaled_up.parent = mesh
        rendered_length_scaled_up = c.render()
        normals_scaled_up.parent = None
        normals_scaled_down = scene.visuals.MeshNormals(meshdata, color=(1, 0, 0), length=length, length_scale=length_scale_down)
        normals_scaled_down.parent = mesh
        rendered_length_scaled_down = c.render()
        normals_scaled_down.parent = None
        n_pixels_default = np.sum(rendered_length_default[..., 0] > 128)
        n_pixels_scaled_up = np.sum(rendered_length_scaled_up[..., 0] > 128)
        n_pixels_scaled_down = np.sum(rendered_length_scaled_down[..., 0] > 128)
        assert n_pixels_scaled_down < n_pixels_default < n_pixels_scaled_up

@requires_pyopengl()
@requires_application()
@pytest.mark.parametrize('length_method', ['median_edge', 'max_extent'])
def test_mesh_normals_length_method(length_method):
    if False:
        while True:
            i = 10
    size = (45, 40)
    with TestingCanvas(size=size, bgcolor='k') as c:
        v = c.central_widget.add_view(border_width=0)
        v.camera = 'arcball'
        meshdata = create_sphere(radius=1.0)
        mesh = scene.visuals.Mesh(meshdata=meshdata, shading=None, color=(0.1, 0.1, 0.1, 1.0))
        v.add(mesh)
        normals = scene.visuals.MeshNormals(meshdata, color=(1, 0, 0), length_method=length_method)
        normals.parent = mesh
        _ = c.render()

def test_mesh_normals_empty():
    if False:
        i = 10
        return i + 15
    mesh = scene.visuals.Mesh()
    scene.visuals.MeshNormals(mesh.mesh_data)
run_tests_if_main()