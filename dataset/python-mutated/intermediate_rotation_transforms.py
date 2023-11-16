"""
Contains the transformation functions for getting to/from ITRS, TEME, GCRS, and CIRS.
These are distinct from the ICRS and AltAz functions because they are just
rotations without aberration corrections or offsets.
"""
import erfa
import numpy as np
from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_transpose
from astropy.coordinates.transformations import FunctionTransformWithFiniteDifference
from .cirs import CIRS
from .equatorial import TEME, TETE
from .gcrs import GCRS, PrecessedGeocentric
from .icrs import ICRS
from .itrs import ITRS
from .utils import get_jd12, get_polar_motion

def teme_to_itrs_mat(time):
    if False:
        for i in range(10):
            print('nop')
    gst = erfa.gmst82(*get_jd12(time, 'ut1'))
    (xp, yp) = get_polar_motion(time)
    pmmat = erfa.pom00(xp, yp, 0)
    return erfa.c2tcio(np.eye(3), gst, pmmat)

def gcrs_to_cirs_mat(time):
    if False:
        print('Hello World!')
    return erfa.c2i06a(*get_jd12(time, 'tt'))

def cirs_to_itrs_mat(time):
    if False:
        i = 10
        return i + 15
    (xp, yp) = get_polar_motion(time)
    sp = erfa.sp00(*get_jd12(time, 'tt'))
    pmmat = erfa.pom00(xp, yp, sp)
    era = erfa.era00(*get_jd12(time, 'ut1'))
    return erfa.c2tcio(np.eye(3), era, pmmat)

def tete_to_itrs_mat(time, rbpn=None):
    if False:
        return 10
    'Compute the polar motion p-matrix at the given time.\n\n    If the nutation-precession matrix is already known, it should be passed in,\n    as this is by far the most expensive calculation.\n    '
    (xp, yp) = get_polar_motion(time)
    sp = erfa.sp00(*get_jd12(time, 'tt'))
    pmmat = erfa.pom00(xp, yp, sp)
    (ujd1, ujd2) = get_jd12(time, 'ut1')
    (jd1, jd2) = get_jd12(time, 'tt')
    if rbpn is None:
        gast = erfa.gst06a(ujd1, ujd2, jd1, jd2)
    else:
        gast = erfa.gst06(ujd1, ujd2, jd1, jd2, rbpn)
    return erfa.c2tcio(np.eye(3), gast, pmmat)

def gcrs_precession_mat(equinox):
    if False:
        while True:
            i = 10
    (gamb, phib, psib, epsa) = erfa.pfw06(*get_jd12(equinox, 'tt'))
    return erfa.fw2m(gamb, phib, psib, epsa)

def get_location_gcrs(location, obstime, ref_to_itrs, gcrs_to_ref):
    if False:
        print('Hello World!')
    'Create a GCRS frame at the location and obstime.\n\n    The reference frame z axis must point to the Celestial Intermediate Pole\n    (as is the case for CIRS and TETE).\n\n    This function is here to avoid location.get_gcrs(obstime), which would\n    recalculate matrices that are already available below (and return a GCRS\n    coordinate, rather than a frame with obsgeoloc and obsgeovel).  Instead,\n    it uses the private method that allows passing in the matrices.\n\n    '
    (obsgeoloc, obsgeovel) = location._get_gcrs_posvel(obstime, ref_to_itrs, gcrs_to_ref)
    return GCRS(obstime=obstime, obsgeoloc=obsgeoloc, obsgeovel=obsgeovel)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, GCRS, TETE)
def gcrs_to_tete(gcrs_coo, tete_frame):
    if False:
        while True:
            i = 10
    rbpn = erfa.pnm06a(*get_jd12(tete_frame.obstime, 'tt'))
    loc_gcrs = get_location_gcrs(tete_frame.location, tete_frame.obstime, tete_to_itrs_mat(tete_frame.obstime, rbpn=rbpn), rbpn)
    gcrs_coo2 = gcrs_coo.transform_to(loc_gcrs)
    crepr = gcrs_coo2.cartesian.transform(rbpn)
    return tete_frame.realize_frame(crepr)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, TETE, GCRS)
def tete_to_gcrs(tete_coo, gcrs_frame):
    if False:
        print('Hello World!')
    rbpn = erfa.pnm06a(*get_jd12(tete_coo.obstime, 'tt'))
    newrepr = tete_coo.cartesian.transform(matrix_transpose(rbpn))
    loc_gcrs = get_location_gcrs(tete_coo.location, tete_coo.obstime, tete_to_itrs_mat(tete_coo.obstime, rbpn=rbpn), rbpn)
    gcrs = loc_gcrs.realize_frame(newrepr)
    return gcrs.transform_to(gcrs_frame)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, TETE, ITRS)
def tete_to_itrs(tete_coo, itrs_frame):
    if False:
        while True:
            i = 10
    tete_coo2 = tete_coo.transform_to(TETE(obstime=itrs_frame.obstime, location=itrs_frame.location))
    pmat = tete_to_itrs_mat(itrs_frame.obstime)
    crepr = tete_coo2.cartesian.transform(pmat)
    return itrs_frame.realize_frame(crepr)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ITRS, TETE)
def itrs_to_tete(itrs_coo, tete_frame):
    if False:
        for i in range(10):
            print('nop')
    pmat = tete_to_itrs_mat(itrs_coo.obstime)
    newrepr = itrs_coo.cartesian.transform(matrix_transpose(pmat))
    tete = TETE(newrepr, obstime=itrs_coo.obstime, location=itrs_coo.location)
    return tete.transform_to(tete_frame)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, GCRS, CIRS)
def gcrs_to_cirs(gcrs_coo, cirs_frame):
    if False:
        for i in range(10):
            print('nop')
    pmat = gcrs_to_cirs_mat(cirs_frame.obstime)
    loc_gcrs = get_location_gcrs(cirs_frame.location, cirs_frame.obstime, cirs_to_itrs_mat(cirs_frame.obstime), pmat)
    gcrs_coo2 = gcrs_coo.transform_to(loc_gcrs)
    crepr = gcrs_coo2.cartesian.transform(pmat)
    return cirs_frame.realize_frame(crepr)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, CIRS, GCRS)
def cirs_to_gcrs(cirs_coo, gcrs_frame):
    if False:
        print('Hello World!')
    pmat = gcrs_to_cirs_mat(cirs_coo.obstime)
    newrepr = cirs_coo.cartesian.transform(matrix_transpose(pmat))
    loc_gcrs = get_location_gcrs(cirs_coo.location, cirs_coo.obstime, cirs_to_itrs_mat(cirs_coo.obstime), pmat)
    gcrs = loc_gcrs.realize_frame(newrepr)
    return gcrs.transform_to(gcrs_frame)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, CIRS, ITRS)
def cirs_to_itrs(cirs_coo, itrs_frame):
    if False:
        print('Hello World!')
    cirs_coo2 = cirs_coo.transform_to(CIRS(obstime=itrs_frame.obstime, location=itrs_frame.location))
    pmat = cirs_to_itrs_mat(itrs_frame.obstime)
    crepr = cirs_coo2.cartesian.transform(pmat)
    return itrs_frame.realize_frame(crepr)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ITRS, CIRS)
def itrs_to_cirs(itrs_coo, cirs_frame):
    if False:
        while True:
            i = 10
    pmat = cirs_to_itrs_mat(itrs_coo.obstime)
    newrepr = itrs_coo.cartesian.transform(matrix_transpose(pmat))
    cirs = CIRS(newrepr, obstime=itrs_coo.obstime, location=itrs_coo.location)
    return cirs.transform_to(cirs_frame)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, GCRS, PrecessedGeocentric)
def gcrs_to_precessedgeo(from_coo, to_frame):
    if False:
        while True:
            i = 10
    gcrs_coo = from_coo.transform_to(GCRS(obstime=to_frame.obstime, obsgeoloc=to_frame.obsgeoloc, obsgeovel=to_frame.obsgeovel))
    pmat = gcrs_precession_mat(to_frame.equinox)
    crepr = gcrs_coo.cartesian.transform(pmat)
    return to_frame.realize_frame(crepr)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, PrecessedGeocentric, GCRS)
def precessedgeo_to_gcrs(from_coo, to_frame):
    if False:
        print('Hello World!')
    pmat = gcrs_precession_mat(from_coo.equinox)
    crepr = from_coo.cartesian.transform(matrix_transpose(pmat))
    gcrs_coo = GCRS(crepr, obstime=from_coo.obstime, obsgeoloc=from_coo.obsgeoloc, obsgeovel=from_coo.obsgeovel)
    return gcrs_coo.transform_to(to_frame)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, TEME, ITRS)
def teme_to_itrs(teme_coo, itrs_frame):
    if False:
        for i in range(10):
            print('nop')
    pmat = teme_to_itrs_mat(teme_coo.obstime)
    crepr = teme_coo.cartesian.transform(pmat)
    itrs = ITRS(crepr, obstime=teme_coo.obstime)
    return itrs.transform_to(itrs_frame)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ITRS, TEME)
def itrs_to_teme(itrs_coo, teme_frame):
    if False:
        i = 10
        return i + 15
    itrs_coo2 = itrs_coo.transform_to(ITRS(obstime=teme_frame.obstime))
    pmat = teme_to_itrs_mat(teme_frame.obstime)
    newrepr = itrs_coo2.cartesian.transform(matrix_transpose(pmat))
    return teme_frame.realize_frame(newrepr)
frame_transform_graph._add_merged_transform(ITRS, CIRS, ITRS)
frame_transform_graph._add_merged_transform(PrecessedGeocentric, GCRS, PrecessedGeocentric)
frame_transform_graph._add_merged_transform(TEME, ITRS, TEME)
frame_transform_graph._add_merged_transform(TETE, ICRS, TETE)