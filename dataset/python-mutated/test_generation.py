import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from vispy.testing import run_tests_if_main
from vispy.geometry import create_box, create_cube, create_cylinder, create_sphere, create_plane

def test_box():
    if False:
        i = 10
        return i + 15
    'Test box function'
    (vertices, filled, outline) = create_box()
    assert_array_equal(np.arange(len(vertices)), np.unique(filled))
    assert_array_equal(np.arange(len(vertices)), np.unique(outline))

def test_cube():
    if False:
        return 10
    'Test cube function'
    (vertices, filled, outline) = create_cube()
    assert_array_equal(np.arange(len(vertices)), np.unique(filled))
    assert_array_equal(np.arange(len(vertices)), np.unique(outline))

def test_sphere():
    if False:
        i = 10
        return i + 15
    'Test sphere function'
    md = create_sphere(rows=10, cols=20, radius=10, method='latitude')
    radii = np.sqrt((md.get_vertices() ** 2).sum(axis=1))
    assert radii.dtype.type is np.float32
    assert_allclose(radii, np.ones_like(radii) * 10, atol=1e-06)
    md = create_sphere(subdivisions=5, radius=10, method='ico')
    radii = np.sqrt((md.get_vertices() ** 2).sum(axis=1))
    assert radii.dtype.type is np.float32
    assert_allclose(radii, np.ones_like(radii) * 10, atol=1e-06)
    md = create_sphere(rows=20, cols=20, depth=20, radius=10, method='cube')
    radii = np.sqrt((md.get_vertices() ** 2).sum(axis=1))
    assert radii.dtype.type is np.float32
    assert_allclose(radii, np.ones_like(radii) * 10, atol=1e-06)

def test_cylinder():
    if False:
        return 10
    'Test cylinder function'
    md = create_cylinder(10, 20, radius=[10, 10])
    radii = np.sqrt((md.get_vertices()[:, :2] ** 2).sum(axis=1))
    assert_allclose(radii, np.ones_like(radii) * 10)

def test_plane():
    if False:
        for i in range(10):
            print('nop')
    'Test plane function'
    (vertices, filled, outline) = create_plane()
    assert_array_equal(np.arange(len(vertices)), np.unique(filled))
    assert_array_equal(np.arange(len(vertices)), np.unique(outline))
run_tests_if_main()