import numpy as np
import pytest
from numpy.testing import assert_allclose
from skimage.draw import ellipsoid, ellipsoid_stats
from skimage.measure import marching_cubes, mesh_surface_area

def test_marching_cubes_isotropic():
    if False:
        for i in range(10):
            print('nop')
    ellipsoid_isotropic = ellipsoid(6, 10, 16, levelset=True)
    (_, surf) = ellipsoid_stats(6, 10, 16)
    (verts, faces) = marching_cubes(ellipsoid_isotropic, 0.0, method='lorensen')[:2]
    surf_calc = mesh_surface_area(verts, faces)
    assert surf > surf_calc and surf_calc > surf * 0.99
    (verts, faces) = marching_cubes(ellipsoid_isotropic, 0.0)[:2]
    surf_calc = mesh_surface_area(verts, faces)
    assert surf > surf_calc and surf_calc > surf * 0.99

def test_marching_cubes_anisotropic():
    if False:
        for i in range(10):
            print('nop')
    spacing = np.array([1.0, 10 / 6.0, 16 / 6.0])
    ellipsoid_anisotropic = ellipsoid(6, 10, 16, spacing=spacing, levelset=True)
    (_, surf) = ellipsoid_stats(6, 10, 16)
    (verts, faces) = marching_cubes(ellipsoid_anisotropic, 0.0, spacing=spacing, method='lorensen')[:2]
    surf_calc = mesh_surface_area(verts, faces)
    assert surf > surf_calc and surf_calc > surf * 0.985
    (verts, faces) = marching_cubes(ellipsoid_anisotropic, 0.0, spacing=spacing)[:2]
    surf_calc = mesh_surface_area(verts, faces)
    assert surf > surf_calc and surf_calc > surf * 0.985
    with pytest.raises(ValueError):
        (verts, faces) = marching_cubes(ellipsoid_anisotropic, 0.0, spacing=spacing, mask=np.array([]))[:2]
    marching_cubes(ellipsoid_anisotropic, 0, spacing=spacing, allow_degenerate=False)

def test_invalid_input():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError):
        marching_cubes(np.zeros((2, 2, 1)), 0, method='lorensen')
    with pytest.raises(ValueError):
        marching_cubes(np.zeros((2, 2, 1)), 1, method='lorensen')
    with pytest.raises(ValueError):
        marching_cubes(np.ones((3, 3, 3)), 1, spacing=(1, 2), method='lorensen')
    with pytest.raises(ValueError):
        marching_cubes(np.zeros((20, 20)), 0, method='lorensen')
    with pytest.raises(ValueError):
        marching_cubes(np.zeros((2, 2, 1)), 0)
    with pytest.raises(ValueError):
        marching_cubes(np.zeros((2, 2, 1)), 1)
    with pytest.raises(ValueError):
        marching_cubes(np.ones((3, 3, 3)), 1, spacing=(1, 2))
    with pytest.raises(ValueError):
        marching_cubes(np.zeros((20, 20)), 0)
    ellipsoid_isotropic = ellipsoid(6, 10, 16, levelset=True)
    with pytest.raises(ValueError):
        marching_cubes(ellipsoid_isotropic, 0.0, method='abcd')

def test_both_algs_same_result_ellipse():
    if False:
        i = 10
        return i + 15
    sphere_small = ellipsoid(1, 1, 1, levelset=True)
    (vertices1, faces1) = marching_cubes(sphere_small, 0, allow_degenerate=False)[:2]
    (vertices2, faces2) = marching_cubes(sphere_small, 0, allow_degenerate=False, method='lorensen')[:2]
    assert _same_mesh(vertices1, faces1, vertices2, faces2)

def _same_mesh(vertices1, faces1, vertices2, faces2, tol=1e-10):
    if False:
        while True:
            i = 10
    'Compare two meshes, using a certain tolerance and invariant to\n    the order of the faces.\n    '
    triangles1 = vertices1[np.array(faces1)]
    triangles2 = vertices2[np.array(faces2)]
    triang1 = [np.concatenate(sorted(t, key=lambda x: tuple(x))) for t in triangles1]
    triang2 = [np.concatenate(sorted(t, key=lambda x: tuple(x))) for t in triangles2]
    triang1 = np.array(sorted([tuple(x) for x in triang1]))
    triang2 = np.array(sorted([tuple(x) for x in triang2]))
    return triang1.shape == triang2.shape and np.allclose(triang1, triang2, 0, tol)

def test_both_algs_same_result_donut():
    if False:
        i = 10
        return i + 15
    n = 48
    (a, b) = (2.5 / n, -1.25)
    vol = np.empty((n, n, n), 'float32')
    for iz in range(vol.shape[0]):
        for iy in range(vol.shape[1]):
            for ix in range(vol.shape[2]):
                (z, y, x) = (float(iz) * a + b, float(iy) * a + b, float(ix) * a + b)
                vol[iz, iy, ix] = (((8 * x) ** 2 + (8 * y - 2) ** 2 + (8 * z) ** 2 + 16 - 1.85 * 1.85) * ((8 * x) ** 2 + (8 * y - 2) ** 2 + (8 * z) ** 2 + 16 - 1.85 * 1.85) - 64 * ((8 * x) ** 2 + (8 * y - 2) ** 2)) * (((8 * x) ** 2 + (8 * y - 2 + 4) * (8 * y - 2 + 4) + (8 * z) ** 2 + 16 - 1.85 * 1.85) * ((8 * x) ** 2 + (8 * y - 2 + 4) * (8 * y - 2 + 4) + (8 * z) ** 2 + 16 - 1.85 * 1.85) - 64 * ((8 * y - 2 + 4) * (8 * y - 2 + 4) + (8 * z) ** 2)) + 1025
    (vertices1, faces1) = marching_cubes(vol, 0, method='lorensen')[:2]
    (vertices2, faces2) = marching_cubes(vol, 0)[:2]
    assert not _same_mesh(vertices1, faces1, vertices2, faces2)

def test_masked_marching_cubes():
    if False:
        while True:
            i = 10
    ellipsoid_scalar = ellipsoid(6, 10, 16, levelset=True)
    mask = np.ones_like(ellipsoid_scalar, dtype=bool)
    mask[:10, :, :] = False
    mask[:, :, 20:] = False
    (ver, faces, _, _) = marching_cubes(ellipsoid_scalar, 0, mask=mask)
    area = mesh_surface_area(ver, faces)
    assert_allclose(area, 299.56878662109375, rtol=0.01)

def test_masked_marching_cubes_empty():
    if False:
        while True:
            i = 10
    ellipsoid_scalar = ellipsoid(6, 10, 16, levelset=True)
    mask = np.array([])
    with pytest.raises(ValueError):
        _ = marching_cubes(ellipsoid_scalar, 0, mask=mask)

def test_masked_marching_cubes_all_true():
    if False:
        for i in range(10):
            print('nop')
    ellipsoid_scalar = ellipsoid(6, 10, 16, levelset=True)
    mask = np.ones_like(ellipsoid_scalar, dtype=bool)
    (ver_m, faces_m, _, _) = marching_cubes(ellipsoid_scalar, 0, mask=mask)
    (ver, faces, _, _) = marching_cubes(ellipsoid_scalar, 0, mask=mask)
    assert_allclose(ver_m, ver, rtol=1e-05)
    assert_allclose(faces_m, faces, rtol=1e-05)