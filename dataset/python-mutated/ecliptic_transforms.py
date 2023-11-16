"""
Contains the transformation functions for getting to/from ecliptic systems.
"""
import erfa
from astropy import units as u
from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates.errors import UnitsError
from astropy.coordinates.matrix_utilities import matrix_transpose, rotation_matrix
from astropy.coordinates.transformations import AffineTransform, DynamicMatrixTransform, FunctionTransformWithFiniteDifference
from .ecliptic import BarycentricMeanEcliptic, BarycentricTrueEcliptic, CustomBarycentricEcliptic, GeocentricMeanEcliptic, GeocentricTrueEcliptic, HeliocentricEclipticIAU76, HeliocentricMeanEcliptic, HeliocentricTrueEcliptic
from .gcrs import GCRS
from .icrs import ICRS
from .utils import EQUINOX_J2000, get_jd12, get_offset_sun_from_barycenter

def _mean_ecliptic_rotation_matrix(equinox):
    if False:
        return 10
    return erfa.ecm06(*get_jd12(equinox, 'tt'))

def _true_ecliptic_rotation_matrix(equinox):
    if False:
        return 10
    (jd1, jd2) = get_jd12(equinox, 'tt')
    (gamb, phib, psib, epsa) = erfa.pfw06(jd1, jd2)
    (dpsi, deps) = erfa.nut06a(jd1, jd2)
    rnpb = erfa.fw2m(gamb, phib, psib + dpsi, epsa + deps)
    obl = erfa.obl06(jd1, jd2) + deps
    return rotation_matrix(obl << u.radian, 'x') @ rnpb

def _obliquity_only_rotation_matrix(obl=erfa.obl80(EQUINOX_J2000.jd1, EQUINOX_J2000.jd2) * u.radian):
    if False:
        for i in range(10):
            print('nop')
    return rotation_matrix(obl, 'x')

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, GCRS, GeocentricMeanEcliptic, finite_difference_frameattr_name='equinox')
def gcrs_to_geoecliptic(gcrs_coo, to_frame):
    if False:
        i = 10
        return i + 15
    gcrs_coo2 = gcrs_coo.transform_to(GCRS(obstime=to_frame.obstime))
    rmat = _mean_ecliptic_rotation_matrix(to_frame.equinox)
    newrepr = gcrs_coo2.cartesian.transform(rmat)
    return to_frame.realize_frame(newrepr)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, GeocentricMeanEcliptic, GCRS)
def geoecliptic_to_gcrs(from_coo, gcrs_frame):
    if False:
        while True:
            i = 10
    rmat = _mean_ecliptic_rotation_matrix(from_coo.equinox)
    newrepr = from_coo.cartesian.transform(matrix_transpose(rmat))
    gcrs = GCRS(newrepr, obstime=from_coo.obstime)
    return gcrs.transform_to(gcrs_frame)

@frame_transform_graph.transform(DynamicMatrixTransform, ICRS, BarycentricMeanEcliptic)
def icrs_to_baryecliptic(from_coo, to_frame):
    if False:
        i = 10
        return i + 15
    return _mean_ecliptic_rotation_matrix(to_frame.equinox)

@frame_transform_graph.transform(DynamicMatrixTransform, BarycentricMeanEcliptic, ICRS)
def baryecliptic_to_icrs(from_coo, to_frame):
    if False:
        i = 10
        return i + 15
    return matrix_transpose(icrs_to_baryecliptic(to_frame, from_coo))
_NEED_ORIGIN_HINT = 'The input {0} coordinates do not have length units. This probably means you created coordinates with lat/lon but no distance.  Heliocentric<->ICRS transforms cannot function in this case because there is an origin shift.'

@frame_transform_graph.transform(AffineTransform, ICRS, HeliocentricMeanEcliptic)
def icrs_to_helioecliptic(from_coo, to_frame):
    if False:
        print('Hello World!')
    if not u.m.is_equivalent(from_coo.cartesian.x.unit):
        raise UnitsError(_NEED_ORIGIN_HINT.format(from_coo.__class__.__name__))
    ssb_from_sun = get_offset_sun_from_barycenter(to_frame.obstime, reverse=True, include_velocity=bool(from_coo.data.differentials))
    rmat = _mean_ecliptic_rotation_matrix(to_frame.equinox)
    return (rmat, ssb_from_sun.transform(rmat))

@frame_transform_graph.transform(AffineTransform, HeliocentricMeanEcliptic, ICRS)
def helioecliptic_to_icrs(from_coo, to_frame):
    if False:
        for i in range(10):
            print('nop')
    if not u.m.is_equivalent(from_coo.cartesian.x.unit):
        raise UnitsError(_NEED_ORIGIN_HINT.format(from_coo.__class__.__name__))
    rmat = _mean_ecliptic_rotation_matrix(from_coo.equinox)
    sun_from_ssb = get_offset_sun_from_barycenter(from_coo.obstime, include_velocity=bool(from_coo.data.differentials))
    return (matrix_transpose(rmat), sun_from_ssb)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, GCRS, GeocentricTrueEcliptic, finite_difference_frameattr_name='equinox')
def gcrs_to_true_geoecliptic(gcrs_coo, to_frame):
    if False:
        print('Hello World!')
    gcrs_coo2 = gcrs_coo.transform_to(GCRS(obstime=to_frame.obstime))
    rmat = _true_ecliptic_rotation_matrix(to_frame.equinox)
    newrepr = gcrs_coo2.cartesian.transform(rmat)
    return to_frame.realize_frame(newrepr)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, GeocentricTrueEcliptic, GCRS)
def true_geoecliptic_to_gcrs(from_coo, gcrs_frame):
    if False:
        for i in range(10):
            print('nop')
    rmat = _true_ecliptic_rotation_matrix(from_coo.equinox)
    newrepr = from_coo.cartesian.transform(matrix_transpose(rmat))
    gcrs = GCRS(newrepr, obstime=from_coo.obstime)
    return gcrs.transform_to(gcrs_frame)

@frame_transform_graph.transform(DynamicMatrixTransform, ICRS, BarycentricTrueEcliptic)
def icrs_to_true_baryecliptic(from_coo, to_frame):
    if False:
        for i in range(10):
            print('nop')
    return _true_ecliptic_rotation_matrix(to_frame.equinox)

@frame_transform_graph.transform(DynamicMatrixTransform, BarycentricTrueEcliptic, ICRS)
def true_baryecliptic_to_icrs(from_coo, to_frame):
    if False:
        for i in range(10):
            print('nop')
    return matrix_transpose(icrs_to_true_baryecliptic(to_frame, from_coo))

@frame_transform_graph.transform(AffineTransform, ICRS, HeliocentricTrueEcliptic)
def icrs_to_true_helioecliptic(from_coo, to_frame):
    if False:
        for i in range(10):
            print('nop')
    if not u.m.is_equivalent(from_coo.cartesian.x.unit):
        raise UnitsError(_NEED_ORIGIN_HINT.format(from_coo.__class__.__name__))
    ssb_from_sun = get_offset_sun_from_barycenter(to_frame.obstime, reverse=True, include_velocity=bool(from_coo.data.differentials))
    rmat = _true_ecliptic_rotation_matrix(to_frame.equinox)
    return (rmat, ssb_from_sun.transform(rmat))

@frame_transform_graph.transform(AffineTransform, HeliocentricTrueEcliptic, ICRS)
def true_helioecliptic_to_icrs(from_coo, to_frame):
    if False:
        while True:
            i = 10
    if not u.m.is_equivalent(from_coo.cartesian.x.unit):
        raise UnitsError(_NEED_ORIGIN_HINT.format(from_coo.__class__.__name__))
    rmat = _true_ecliptic_rotation_matrix(from_coo.equinox)
    sun_from_ssb = get_offset_sun_from_barycenter(from_coo.obstime, include_velocity=bool(from_coo.data.differentials))
    return (matrix_transpose(rmat), sun_from_ssb)

@frame_transform_graph.transform(AffineTransform, HeliocentricEclipticIAU76, ICRS)
def ecliptic_to_iau76_icrs(from_coo, to_frame):
    if False:
        while True:
            i = 10
    rmat = _obliquity_only_rotation_matrix()
    sun_from_ssb = get_offset_sun_from_barycenter(from_coo.obstime, include_velocity=bool(from_coo.data.differentials))
    return (matrix_transpose(rmat), sun_from_ssb)

@frame_transform_graph.transform(AffineTransform, ICRS, HeliocentricEclipticIAU76)
def icrs_to_iau76_ecliptic(from_coo, to_frame):
    if False:
        while True:
            i = 10
    ssb_from_sun = get_offset_sun_from_barycenter(to_frame.obstime, reverse=True, include_velocity=bool(from_coo.data.differentials))
    rmat = _obliquity_only_rotation_matrix()
    return (rmat, ssb_from_sun.transform(rmat))

@frame_transform_graph.transform(DynamicMatrixTransform, ICRS, CustomBarycentricEcliptic)
def icrs_to_custombaryecliptic(from_coo, to_frame):
    if False:
        i = 10
        return i + 15
    return _obliquity_only_rotation_matrix(to_frame.obliquity)

@frame_transform_graph.transform(DynamicMatrixTransform, CustomBarycentricEcliptic, ICRS)
def custombaryecliptic_to_icrs(from_coo, to_frame):
    if False:
        i = 10
        return i + 15
    return icrs_to_custombaryecliptic(to_frame, from_coo).T
frame_transform_graph._add_merged_transform(GeocentricMeanEcliptic, ICRS, GeocentricMeanEcliptic)
frame_transform_graph._add_merged_transform(GeocentricTrueEcliptic, ICRS, GeocentricTrueEcliptic)
frame_transform_graph._add_merged_transform(HeliocentricMeanEcliptic, ICRS, HeliocentricMeanEcliptic)
frame_transform_graph._add_merged_transform(HeliocentricTrueEcliptic, ICRS, HeliocentricTrueEcliptic)
frame_transform_graph._add_merged_transform(HeliocentricEclipticIAU76, ICRS, HeliocentricEclipticIAU76)
frame_transform_graph._add_merged_transform(BarycentricMeanEcliptic, ICRS, BarycentricMeanEcliptic)
frame_transform_graph._add_merged_transform(BarycentricTrueEcliptic, ICRS, BarycentricTrueEcliptic)
frame_transform_graph._add_merged_transform(CustomBarycentricEcliptic, ICRS, CustomBarycentricEcliptic)