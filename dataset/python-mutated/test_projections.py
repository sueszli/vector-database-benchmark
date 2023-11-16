"""Test sky projections defined in WCS Paper II"""
import os
import unittest.mock as mk
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
from astropy import units as u
from astropy import wcs
from astropy.io import fits
from astropy.modeling import projections
from astropy.modeling.parameters import InputParameterError
from astropy.tests.helper import assert_quantity_allclose
from astropy.utils.data import get_pkg_data_filename

def test_new_wcslib_projections():
    if False:
        while True:
            i = 10
    assert not set(wcs.PRJ_CODES).symmetric_difference(projections.projcodes + projections._NOT_SUPPORTED_PROJ_CODES)

def test_Projection_properties():
    if False:
        print('Hello World!')
    projection = projections.Sky2Pix_PlateCarree()
    assert projection.n_inputs == 2
    assert projection.n_outputs == 2
PIX_COORDINATES = [-10, 30]
MAPS_DIR = os.path.join(os.pardir, os.pardir, 'wcs', 'tests', 'data', 'maps')
pars = [(x,) for x in projections.projcodes]
pars.remove(('XPH',))

@pytest.mark.parametrize(('code',), pars)
def test_Sky2Pix(code):
    if False:
        for i in range(10):
            print('nop')
    'Check astropy model eval against wcslib eval'
    wcs_map = os.path.join(MAPS_DIR, f'1904-66_{code}.hdr')
    test_file = get_pkg_data_filename(wcs_map)
    header = fits.Header.fromfile(test_file, endcard=False, padding=False)
    params = []
    for i in range(3):
        key = f'PV2_{i + 1}'
        if key in header:
            params.append(header[key])
    w = wcs.WCS(header)
    w.wcs.crval = [0.0, 0.0]
    w.wcs.crpix = [0, 0]
    w.wcs.cdelt = [1, 1]
    wcslibout = w.wcs.p2s([PIX_COORDINATES], 1)
    wcs_pix = w.wcs.s2p(wcslibout['world'], 1)['pixcrd']
    model = getattr(projections, 'Sky2Pix_' + code)
    tinv = model(*params)
    (x, y) = tinv(wcslibout['phi'], wcslibout['theta'])
    assert_almost_equal(np.asarray(x), wcs_pix[:, 0])
    assert_almost_equal(np.asarray(y), wcs_pix[:, 1])
    assert isinstance(tinv.prjprm, wcs.Prjprm)

@pytest.mark.parametrize(('code',), pars)
def test_Pix2Sky(code):
    if False:
        print('Hello World!')
    'Check astropy model eval against wcslib eval'
    wcs_map = os.path.join(MAPS_DIR, f'1904-66_{code}.hdr')
    test_file = get_pkg_data_filename(wcs_map)
    header = fits.Header.fromfile(test_file, endcard=False, padding=False)
    params = []
    for i in range(3):
        key = f'PV2_{i + 1}'
        if key in header:
            params.append(header[key])
    w = wcs.WCS(header)
    w.wcs.crval = [0.0, 0.0]
    w.wcs.crpix = [0, 0]
    w.wcs.cdelt = [1, 1]
    wcslibout = w.wcs.p2s([PIX_COORDINATES], 1)
    wcs_phi = wcslibout['phi']
    wcs_theta = wcslibout['theta']
    model = getattr(projections, 'Pix2Sky_' + code)
    tanprj = model(*params)
    (phi, theta) = tanprj(*PIX_COORDINATES)
    assert_almost_equal(np.asarray(phi), wcs_phi)
    assert_almost_equal(np.asarray(theta), wcs_theta)

@pytest.mark.parametrize(('code',), pars)
def test_Sky2Pix_unit(code):
    if False:
        while True:
            i = 10
    'Check astropy model eval against wcslib eval'
    wcs_map = os.path.join(MAPS_DIR, f'1904-66_{code}.hdr')
    test_file = get_pkg_data_filename(wcs_map)
    header = fits.Header.fromfile(test_file, endcard=False, padding=False)
    params = []
    for i in range(3):
        key = f'PV2_{i + 1}'
        if key in header:
            params.append(header[key])
    w = wcs.WCS(header)
    w.wcs.crval = [0.0, 0.0]
    w.wcs.crpix = [0, 0]
    w.wcs.cdelt = [1, 1]
    wcslibout = w.wcs.p2s([PIX_COORDINATES], 1)
    wcs_pix = w.wcs.s2p(wcslibout['world'], 1)['pixcrd']
    model = getattr(projections, 'Sky2Pix_' + code)
    tinv = model(*params)
    (x, y) = tinv(wcslibout['phi'] * u.deg, wcslibout['theta'] * u.deg)
    assert_quantity_allclose(x, wcs_pix[:, 0] * u.deg)
    assert_quantity_allclose(y, wcs_pix[:, 1] * u.deg)

@pytest.mark.parametrize(('code',), pars)
def test_Pix2Sky_unit(code):
    if False:
        return 10
    'Check astropy model eval against wcslib eval'
    wcs_map = os.path.join(MAPS_DIR, f'1904-66_{code}.hdr')
    test_file = get_pkg_data_filename(wcs_map)
    header = fits.Header.fromfile(test_file, endcard=False, padding=False)
    params = []
    for i in range(3):
        key = f'PV2_{i + 1}'
        if key in header:
            params.append(header[key])
    w = wcs.WCS(header)
    w.wcs.crval = [0.0, 0.0]
    w.wcs.crpix = [0, 0]
    w.wcs.cdelt = [1, 1]
    wcslibout = w.wcs.p2s([PIX_COORDINATES], 1)
    wcs_phi = wcslibout['phi']
    wcs_theta = wcslibout['theta']
    model = getattr(projections, 'Pix2Sky_' + code)
    tanprj = model(*params)
    (phi, theta) = tanprj(*PIX_COORDINATES * u.deg)
    assert_quantity_allclose(phi, wcs_phi * u.deg)
    assert_quantity_allclose(theta, wcs_theta * u.deg)
    (phi, theta) = tanprj(*(PIX_COORDINATES * u.deg).to(u.rad))
    assert_quantity_allclose(phi, wcs_phi * u.deg)
    assert_quantity_allclose(theta, wcs_theta * u.deg)
    (phi, theta) = tanprj(*(PIX_COORDINATES * u.deg).to(u.arcmin))
    assert_quantity_allclose(phi, wcs_phi * u.deg)
    assert_quantity_allclose(theta, wcs_theta * u.deg)

@pytest.mark.parametrize(('code',), pars)
def test_projection_default(code):
    if False:
        return 10
    'Check astropy model eval with default parameters'
    model = getattr(projections, 'Sky2Pix_' + code)
    tinv = model()
    (x, y) = tinv(45, 45)
    model = getattr(projections, 'Pix2Sky_' + code)
    tinv = model()
    (x, y) = tinv(0, 0)

class TestZenithalPerspective:
    """Test Zenithal Perspective projection"""

    def setup_class(self):
        if False:
            print('Hello World!')
        ID = 'AZP'
        wcs_map = os.path.join(MAPS_DIR, f'1904-66_{ID}.hdr')
        test_file = get_pkg_data_filename(wcs_map)
        header = fits.Header.fromfile(test_file, endcard=False, padding=False)
        self.wazp = wcs.WCS(header)
        self.wazp.wcs.crpix = np.array([0.0, 0.0])
        self.wazp.wcs.crval = np.array([0.0, 0.0])
        self.wazp.wcs.cdelt = np.array([1.0, 1.0])
        self.pv_kw = [kw[2] for kw in self.wazp.wcs.get_pv()]
        self.azp = projections.Pix2Sky_ZenithalPerspective(*self.pv_kw)

    def test_AZP_p2s(self):
        if False:
            while True:
                i = 10
        wcslibout = self.wazp.wcs.p2s([[-10, 30]], 1)
        wcs_phi = wcslibout['phi']
        wcs_theta = wcslibout['theta']
        (phi, theta) = self.azp(-10, 30)
        assert_almost_equal(np.asarray(phi), wcs_phi)
        assert_almost_equal(np.asarray(theta), wcs_theta)

    def test_AZP_s2p(self):
        if False:
            return 10
        wcslibout = self.wazp.wcs.p2s([[-10, 30]], 1)
        wcs_pix = self.wazp.wcs.s2p(wcslibout['world'], 1)['pixcrd']
        (x, y) = self.azp.inverse(wcslibout['phi'], wcslibout['theta'])
        assert_almost_equal(np.asarray(x), wcs_pix[:, 0])
        assert_almost_equal(np.asarray(y), wcs_pix[:, 1])

    def test_validate(self):
        if False:
            for i in range(10):
                print('nop')
        MESSAGE = 'Zenithal perspective projection is not defined for mu = -1'
        with pytest.raises(InputParameterError, match=MESSAGE):
            projections.Pix2Sky_ZenithalPerspective(-1)
        with pytest.raises(InputParameterError, match=MESSAGE):
            projections.Sky2Pix_ZenithalPerspective(-1)
        with pytest.raises(InputParameterError, match=MESSAGE):
            projections.Pix2Sky_SlantZenithalPerspective(-1)
        with pytest.raises(InputParameterError, match=MESSAGE):
            projections.Sky2Pix_SlantZenithalPerspective(-1)

class TestCylindricalPerspective:
    """Test cylindrical perspective projection"""

    def setup_class(self):
        if False:
            return 10
        ID = 'CYP'
        wcs_map = os.path.join(MAPS_DIR, f'1904-66_{ID}.hdr')
        test_file = get_pkg_data_filename(wcs_map)
        header = fits.Header.fromfile(test_file, endcard=False, padding=False)
        self.wazp = wcs.WCS(header)
        self.wazp.wcs.crpix = np.array([0.0, 0.0])
        self.wazp.wcs.crval = np.array([0.0, 0.0])
        self.wazp.wcs.cdelt = np.array([1.0, 1.0])
        self.pv_kw = [kw[2] for kw in self.wazp.wcs.get_pv()]
        self.azp = projections.Pix2Sky_CylindricalPerspective(*self.pv_kw)

    def test_CYP_p2s(self):
        if False:
            while True:
                i = 10
        wcslibout = self.wazp.wcs.p2s([[-10, 30]], 1)
        wcs_phi = wcslibout['phi']
        wcs_theta = wcslibout['theta']
        (phi, theta) = self.azp(-10, 30)
        assert_almost_equal(np.asarray(phi), wcs_phi)
        assert_almost_equal(np.asarray(theta), wcs_theta)

    def test_CYP_s2p(self):
        if False:
            i = 10
            return i + 15
        wcslibout = self.wazp.wcs.p2s([[-10, 30]], 1)
        wcs_pix = self.wazp.wcs.s2p(wcslibout['world'], 1)['pixcrd']
        (x, y) = self.azp.inverse(wcslibout['phi'], wcslibout['theta'])
        assert_almost_equal(np.asarray(x), wcs_pix[:, 0])
        assert_almost_equal(np.asarray(y), wcs_pix[:, 1])

    def test_validate(self):
        if False:
            return 10
        MESSAGE = 'CYP projection is not defined for .*'
        MESSAGE0 = 'CYP projection is not defined for mu = -lambda'
        MESSAGE1 = 'CYP projection is not defined for lambda = -mu'
        with pytest.raises(InputParameterError, match=MESSAGE):
            projections.Pix2Sky_CylindricalPerspective(1, -1)
        with pytest.raises(InputParameterError, match=MESSAGE):
            projections.Pix2Sky_CylindricalPerspective(-1, 1)
        model = projections.Pix2Sky_CylindricalPerspective()
        with pytest.raises(InputParameterError, match=MESSAGE0):
            model.mu = -1
        with pytest.raises(InputParameterError, match=MESSAGE1):
            model.lam = -1
        with pytest.raises(InputParameterError, match=MESSAGE):
            projections.Sky2Pix_CylindricalPerspective(1, -1)
        with pytest.raises(InputParameterError, match=MESSAGE):
            projections.Sky2Pix_CylindricalPerspective(-1, 1)
        model = projections.Sky2Pix_CylindricalPerspective()
        with pytest.raises(InputParameterError, match=MESSAGE0):
            model.mu = -1
        with pytest.raises(InputParameterError, match=MESSAGE1):
            model.lam = -1

def test_AffineTransformation2D():
    if False:
        print('Hello World!')
    model = projections.AffineTransformation2D(matrix=[[2, 0], [0, 2]], translation=[1, 1])
    rect = [[0, 0], [1, 0], [0, 3], [1, 3]]
    (x, y) = zip(*rect)
    new_rect = np.vstack(model(x, y)).T
    assert np.all(new_rect == [[1, 1], [3, 1], [1, 7], [3, 7]])
    MESSAGE = 'Expected transformation matrix to be a 2x2 array'
    with pytest.raises(InputParameterError, match=MESSAGE):
        model.matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    MESSAGE = 'Expected translation vector to be a 2 element row or column vector array'
    with pytest.raises(InputParameterError, match=MESSAGE):
        model.translation = [1, 2, 3]
    with pytest.raises(InputParameterError, match=MESSAGE):
        model.translation = [[1], [2]]
    with pytest.raises(InputParameterError, match=MESSAGE):
        model.translation = [[1, 2, 3]]
    a = np.array([[1], [2], [3], [4]])
    b = a.ravel()
    with mk.patch.object(np, 'vstack', autospec=True, side_effect=[a, b]) as mk_vstack:
        MESSAGE = 'Incompatible input shapes'
        with pytest.raises(ValueError, match=MESSAGE):
            model(x, y)
        with pytest.raises(ValueError, match=MESSAGE):
            model(x, y)
        assert mk_vstack.call_count == 2
    x = np.array([1, 2])
    y = np.array([1, 2, 3])
    MESSAGE = 'Expected input arrays to have the same shape'
    with pytest.raises(ValueError, match=MESSAGE):
        model.evaluate(x, y, model.matrix, model.translation)

def test_AffineTransformation2D_inverse():
    if False:
        i = 10
        return i + 15
    model1 = projections.AffineTransformation2D(matrix=[[1, 1], [1, 1]])
    MESSAGE = 'Transformation matrix is singular; .* model does not have an inverse'
    with pytest.raises(InputParameterError, match=MESSAGE):
        model1.inverse
    model2 = projections.AffineTransformation2D(matrix=[[1.2, 3.4], [5.6, 7.8]], translation=[9.1, 10.11])
    rect = [[0, 0], [1, 0], [0, 3], [1, 3]]
    (x, y) = zip(*rect)
    (x_new, y_new) = model2.inverse(*model2(x, y))
    assert_allclose([x, y], [x_new, y_new], atol=1e-10)
    model3 = projections.AffineTransformation2D(matrix=[[1.2, 3.4], [5.6, 7.8]] * u.m, translation=[9.1, 10.11] * u.m)
    (x_new, y_new) = model3.inverse(*model3(x * u.m, y * u.m))
    assert_allclose([x, y], [x_new, y_new], atol=1e-10)
    model4 = projections.AffineTransformation2D(matrix=[[1.2, 3.4], [5.6, 7.8]] * u.m, translation=[9.1, 10.11] * u.km)
    MESSAGE = 'matrix and translation must have the same units'
    with pytest.raises(ValueError, match=MESSAGE):
        model4.inverse(*model4(x * u.m, y * u.m))

def test_c_projection_striding():
    if False:
        for i in range(10):
            print('nop')
    coords = np.arange(10).reshape((5, 2))
    model = projections.Sky2Pix_ZenithalPerspective(2, 30)
    (phi, theta) = model(coords[:, 0], coords[:, 1])
    assert_almost_equal(phi, [0.0, 2.2790416, 4.4889294, 6.6250643, 8.68301])
    assert_almost_equal(theta, [-76.4816918, -75.3594654, -74.1256332, -72.784558, -71.3406629])

def test_c_projections_shaped():
    if False:
        while True:
            i = 10
    (nx, ny) = (5, 2)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    (xv, yv) = np.meshgrid(x, y)
    model = projections.Pix2Sky_TAN()
    (phi, theta) = model(xv, yv)
    assert_allclose(phi, [[0.0, 90.0, 90.0, 90.0, 90.0], [180.0, 165.96375653, 153.43494882, 143.13010235, 135.0]])
    assert_allclose(theta, [[90.0, 89.75000159, 89.50001269, 89.25004283, 89.00010152], [89.00010152, 88.96933478, 88.88210788, 88.75019826, 88.58607353]])

def test_affine_with_quantities():
    if False:
        print('Hello World!')
    x = 1
    y = 2
    xdeg = (x * u.pix).to(u.deg, equivalencies=u.pixel_scale(2.5 * u.deg / u.pix))
    ydeg = (y * u.pix).to(u.deg, equivalencies=u.pixel_scale(2.5 * u.deg / u.pix))
    xpix = x * u.pix
    ypix = y * u.pix
    qaff = projections.AffineTransformation2D(matrix=[[1, 2], [2, 1]] * u.deg)
    MESSAGE = 'To use AffineTransformation with quantities, both matrix and unit need to be quantities'
    with pytest.raises(ValueError, match=MESSAGE):
        (qx1, qy1) = qaff(xpix, ypix, equivalencies={'x': u.pixel_scale(2.5 * u.deg / u.pix), 'y': u.pixel_scale(2.5 * u.deg / u.pix)})
    qaff = projections.AffineTransformation2D(matrix=[[1, 2], [2, 1]] * u.deg, translation=[1, 2] * u.deg)
    (qx1, qy1) = qaff(xpix, ypix, equivalencies={'x': u.pixel_scale(2.5 * u.deg / u.pix), 'y': u.pixel_scale(2.5 * u.deg / u.pix)})
    aff = projections.AffineTransformation2D(matrix=[[1, 2], [2, 1]], translation=[1, 2])
    (x1, y1) = aff(xdeg.value, ydeg.value)
    assert_quantity_allclose(qx1, x1 * u.deg)
    assert_quantity_allclose(qy1, y1 * u.deg)
    pc = np.array([[0.86585778922708, 0.50029020461607], [-0.50029020461607, 0.86585778922708]])
    cdelt = np.array([[1, 3.0683055555556e-05], [3.0966944444444e-05, 1]])
    matrix = cdelt * pc
    qaff = projections.AffineTransformation2D(matrix=matrix * u.deg, translation=[0, 0] * u.deg)
    inv_matrix = np.linalg.inv(matrix)
    inv_qaff = projections.AffineTransformation2D(matrix=inv_matrix * u.pix, translation=[0, 0] * u.pix)
    qaff.inverse = inv_qaff
    (qx1, qy1) = qaff(xpix, ypix, equivalencies={'x': u.pixel_scale(1 * u.deg / u.pix), 'y': u.pixel_scale(1 * u.deg / u.pix)})
    (x1, y1) = qaff.inverse(qx1, qy1, equivalencies={'x': u.pixel_scale(1 * u.deg / u.pix), 'y': u.pixel_scale(1 * u.deg / u.pix)})
    assert_quantity_allclose(x1, xpix)
    assert_quantity_allclose(y1, ypix)

def test_Pix2Sky_ZenithalPerspective_inverse():
    if False:
        for i in range(10):
            print('nop')
    model = projections.Pix2Sky_ZenithalPerspective(2, 30)
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_ZenithalPerspective)
    assert inverse.mu == model.mu == 2
    assert_allclose(inverse.gamma, model.gamma)
    assert_allclose(inverse.gamma, 30)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_ZenithalPerspective_inverse():
    if False:
        return 10
    model = projections.Sky2Pix_ZenithalPerspective(2, 30)
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_AZP)
    assert inverse.mu == model.mu == 2
    assert_allclose(inverse.gamma, model.gamma)
    assert_allclose(inverse.gamma, 30)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_SlantZenithalPerspective_inverse():
    if False:
        print('Hello World!')
    model = projections.Pix2Sky_SlantZenithalPerspective(2, 30, 40)
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_SlantZenithalPerspective)
    assert inverse.mu == model.mu == 2
    assert_allclose(inverse.phi0, model.phi0)
    assert_allclose(inverse.theta0, model.theta0)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_SlantZenithalPerspective_inverse():
    if False:
        print('Hello World!')
    model = projections.Sky2Pix_SlantZenithalPerspective(2, 30, 40)
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_SlantZenithalPerspective)
    assert inverse.mu == model.mu == 2
    assert_allclose(inverse.phi0, model.phi0)
    assert_allclose(inverse.theta0, model.theta0)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_Gnomonic_inverse():
    if False:
        return 10
    model = projections.Pix2Sky_Gnomonic()
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_Gnomonic)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_Gnomonic_inverse():
    if False:
        for i in range(10):
            print('nop')
    model = projections.Sky2Pix_Gnomonic()
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_Gnomonic)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_Stereographic_inverse():
    if False:
        while True:
            i = 10
    model = projections.Pix2Sky_Stereographic()
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_Stereographic)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_Stereographic_inverse():
    if False:
        while True:
            i = 10
    model = projections.Sky2Pix_Stereographic()
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_Stereographic)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_SlantOrthographic_inverse():
    if False:
        i = 10
        return i + 15
    model = projections.Pix2Sky_SlantOrthographic(2, 30)
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_SlantOrthographic)
    assert inverse.xi == model.xi == 2
    assert inverse.eta == model.eta == 30
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-08)
    assert_allclose(b, y, atol=1e-08)

def test_Sky2Pix_SlantOrthographic_inverse():
    if False:
        while True:
            i = 10
    model = projections.Sky2Pix_SlantOrthographic(2, 30)
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_SlantOrthographic)
    assert inverse.xi == model.xi == 2
    assert inverse.eta == model.eta == 30
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-08)
    assert_allclose(b, y, atol=1e-08)

def test_Pix2Sky_ZenithalEquidistant_inverse():
    if False:
        while True:
            i = 10
    model = projections.Pix2Sky_ZenithalEquidistant()
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_ZenithalEquidistant)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_ZenithalEquidistant_inverse():
    if False:
        return 10
    model = projections.Sky2Pix_ZenithalEquidistant()
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_ZenithalEquidistant)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_ZenithalEqualArea_inverse():
    if False:
        return 10
    model = projections.Pix2Sky_ZenithalEqualArea()
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_ZenithalEqualArea)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_ZenithalEqualArea_inverse():
    if False:
        for i in range(10):
            print('nop')
    model = projections.Sky2Pix_ZenithalEqualArea()
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_ZenithalEqualArea)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_Airy_inverse():
    if False:
        i = 10
        return i + 15
    model = projections.Pix2Sky_Airy(30)
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_Airy)
    assert inverse.theta_b == model.theta_b == 30
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_Airy_inverse():
    if False:
        while True:
            i = 10
    model = projections.Sky2Pix_Airy(30)
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_Airy)
    assert inverse.theta_b == model.theta_b == 30
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_CylindricalPerspective_inverse():
    if False:
        return 10
    model = projections.Pix2Sky_CylindricalPerspective(2, 30)
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_CylindricalPerspective)
    assert inverse.mu == model.mu == 2
    assert inverse.lam == model.lam == 30
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_CylindricalPerspective_inverse():
    if False:
        return 10
    model = projections.Sky2Pix_CylindricalPerspective(2, 30)
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_CylindricalPerspective)
    assert inverse.mu == model.mu == 2
    assert inverse.lam == model.lam == 30
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_CylindricalEqualArea_inverse():
    if False:
        i = 10
        return i + 15
    model = projections.Pix2Sky_CylindricalEqualArea(0.567)
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_CylindricalEqualArea)
    assert inverse.lam == model.lam == 0.567

def test_Sky2Pix_CylindricalEqualArea_inverse():
    if False:
        i = 10
        return i + 15
    model = projections.Sky2Pix_CylindricalEqualArea(0.765)
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_CylindricalEqualArea)
    assert inverse.lam == model.lam == 0.765

def test_Pix2Sky_PlateCarree_inverse():
    if False:
        print('Hello World!')
    model = projections.Pix2Sky_PlateCarree()
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_PlateCarree)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_PlateCarree_inverse():
    if False:
        i = 10
        return i + 15
    model = projections.Sky2Pix_PlateCarree()
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_PlateCarree)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_Mercator_inverse():
    if False:
        while True:
            i = 10
    model = projections.Pix2Sky_Mercator()
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_Mercator)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_Mercator_inverse():
    if False:
        print('Hello World!')
    model = projections.Sky2Pix_Mercator()
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_Mercator)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_SansonFlamsteed_inverse():
    if False:
        for i in range(10):
            print('nop')
    model = projections.Pix2Sky_SansonFlamsteed()
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_SansonFlamsteed)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_SansonFlamsteed_inverse():
    if False:
        for i in range(10):
            print('nop')
    model = projections.Sky2Pix_SansonFlamsteed()
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_SansonFlamsteed)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_Parabolic_inverse():
    if False:
        print('Hello World!')
    model = projections.Pix2Sky_Parabolic()
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_Parabolic)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_Parabolic_inverse():
    if False:
        print('Hello World!')
    model = projections.Sky2Pix_Parabolic()
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_Parabolic)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_Molleweide_inverse():
    if False:
        return 10
    model = projections.Pix2Sky_Molleweide()
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_Molleweide)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_Molleweide_inverse():
    if False:
        return 10
    model = projections.Sky2Pix_Molleweide()
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_Molleweide)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_HammerAitoff_inverse():
    if False:
        for i in range(10):
            print('nop')
    model = projections.Pix2Sky_HammerAitoff()
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_HammerAitoff)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_HammerAitoff_inverse():
    if False:
        while True:
            i = 10
    model = projections.Sky2Pix_HammerAitoff()
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_HammerAitoff)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_ConicPerspective_inverse():
    if False:
        for i in range(10):
            print('nop')
    model = projections.Pix2Sky_ConicPerspective(2, 30)
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_ConicPerspective)
    assert inverse.sigma == model.sigma == 2
    assert_allclose(inverse.delta, model.delta)
    assert_allclose(inverse.delta, 30)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_ConicPerspective_inverse():
    if False:
        while True:
            i = 10
    model = projections.Sky2Pix_ConicPerspective(2, 30)
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_ConicPerspective)
    assert inverse.sigma == model.sigma == 2
    assert_allclose(inverse.delta, model.delta)
    assert_allclose(inverse.delta, 30)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_ConicEqualArea_inverse():
    if False:
        print('Hello World!')
    model = projections.Pix2Sky_ConicEqualArea(2, 30)
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_ConicEqualArea)
    assert inverse.sigma == model.sigma == 2
    assert_allclose(inverse.delta, model.delta)
    assert_allclose(inverse.delta, 30)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_ConicEqualArea_inverse():
    if False:
        i = 10
        return i + 15
    model = projections.Sky2Pix_ConicEqualArea(2, 30)
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_ConicEqualArea)
    assert inverse.sigma == model.sigma == 2
    assert_allclose(inverse.delta, model.delta)
    assert_allclose(inverse.delta, 30)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_ConicEquidistant_inverse():
    if False:
        for i in range(10):
            print('nop')
    model = projections.Pix2Sky_ConicEquidistant(2, 30)
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_ConicEquidistant)
    assert inverse.sigma == model.sigma == 2
    assert_allclose(inverse.delta, model.delta)
    assert_allclose(inverse.delta, 30)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_ConicEquidistant_inverse():
    if False:
        print('Hello World!')
    model = projections.Sky2Pix_ConicEquidistant(2, 30)
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_ConicEquidistant)
    assert inverse.sigma == model.sigma == 2
    assert_allclose(inverse.delta, model.delta)
    assert_allclose(inverse.delta, 30)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_ConicOrthomorphic_inverse():
    if False:
        print('Hello World!')
    model = projections.Pix2Sky_ConicOrthomorphic(2, 30)
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_ConicOrthomorphic)
    assert inverse.sigma == model.sigma == 2
    assert_allclose(inverse.delta, model.delta)
    assert_allclose(inverse.delta, 30)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_ConicOrthomorphic_inverse():
    if False:
        for i in range(10):
            print('nop')
    model = projections.Sky2Pix_ConicOrthomorphic(2, 30)
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_ConicOrthomorphic)
    assert inverse.sigma == model.sigma == 2
    assert_allclose(inverse.delta, model.delta)
    assert_allclose(inverse.delta, 30)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_BonneEqualArea_inverse():
    if False:
        for i in range(10):
            print('nop')
    model = projections.Pix2Sky_BonneEqualArea(2)
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_BonneEqualArea)
    assert inverse.theta1 == model.theta1 == 2
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_BonneEqualArea_inverse():
    if False:
        return 10
    model = projections.Sky2Pix_BonneEqualArea(2)
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_BonneEqualArea)
    assert inverse.theta1 == model.theta1 == 2
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_Polyconic_inverse():
    if False:
        i = 10
        return i + 15
    model = projections.Pix2Sky_Polyconic()
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_Polyconic)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_Polyconic_inverse():
    if False:
        i = 10
        return i + 15
    model = projections.Sky2Pix_Polyconic()
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_Polyconic)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_TangentialSphericalCube_inverse():
    if False:
        return 10
    model = projections.Pix2Sky_TangentialSphericalCube()
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_TangentialSphericalCube)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_TangentialSphericalCube_inverse():
    if False:
        while True:
            i = 10
    model = projections.Sky2Pix_TangentialSphericalCube()
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_TangentialSphericalCube)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_COBEQuadSphericalCube_inverse():
    if False:
        print('Hello World!')
    model = projections.Pix2Sky_COBEQuadSphericalCube()
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_COBEQuadSphericalCube)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=0.001)
    assert_allclose(b, y, atol=0.001)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=0.001)
    assert_allclose(b, y, atol=0.001)

def test_Sky2Pix_COBEQuadSphericalCube_inverse():
    if False:
        return 10
    model = projections.Sky2Pix_COBEQuadSphericalCube()
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_COBEQuadSphericalCube)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=0.001)
    assert_allclose(b, y, atol=0.001)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=0.001)
    assert_allclose(b, y, atol=0.001)

def test_Pix2Sky_QuadSphericalCube_inverse():
    if False:
        return 10
    model = projections.Pix2Sky_QuadSphericalCube()
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_QuadSphericalCube)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_QuadSphericalCube_inverse():
    if False:
        return 10
    model = projections.Sky2Pix_QuadSphericalCube()
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_QuadSphericalCube)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_HEALPix_inverse():
    if False:
        return 10
    model = projections.Pix2Sky_HEALPix(2, 30)
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_HEALPix)
    assert inverse.H == model.H == 2
    assert inverse.X == model.X == 30
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_HEALPix_inverse():
    if False:
        i = 10
        return i + 15
    model = projections.Sky2Pix_HEALPix(2, 30)
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_HEALPix)
    assert inverse.H == model.H == 2
    assert inverse.X == model.X == 30
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Pix2Sky_HEALPixPolar_inverse():
    if False:
        while True:
            i = 10
    model = projections.Pix2Sky_HEALPixPolar()
    inverse = model.inverse
    assert isinstance(inverse, projections.Sky2Pix_HEALPixPolar)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)

def test_Sky2Pix_HEALPixPolar_inverse():
    if False:
        i = 10
        return i + 15
    model = projections.Sky2Pix_HEALPixPolar()
    inverse = model.inverse
    assert isinstance(inverse, projections.Pix2Sky_HEALPixPolar)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    (a, b) = model(*inverse(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)
    (a, b) = inverse(*model(x, y))
    assert_allclose(a, x, atol=1e-12)
    assert_allclose(b, y, atol=1e-12)