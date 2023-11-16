"""
Contains the transformation functions for getting from ICRS/HCRS to CIRS and
anything in between (currently that means GCRS).
"""
import numpy as np
from astropy import units as u
from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates.erfa_astrom import erfa_astrom
from astropy.coordinates.representation import CartesianRepresentation, SphericalRepresentation, UnitSphericalRepresentation
from astropy.coordinates.transformations import AffineTransform, FunctionTransformWithFiniteDifference
from .cirs import CIRS
from .gcrs import GCRS
from .hcrs import HCRS
from .icrs import ICRS
from .utils import atciqz, aticq, get_offset_sun_from_barycenter

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ICRS, CIRS)
def icrs_to_cirs(icrs_coo, cirs_frame):
    if False:
        for i in range(10):
            print('nop')
    astrom = erfa_astrom.get().apco(cirs_frame)
    if icrs_coo.data.get_name() == 'unitspherical' or icrs_coo.data.to_cartesian().x.unit == u.one:
        srepr = icrs_coo.spherical
        (cirs_ra, cirs_dec) = atciqz(srepr.without_differentials(), astrom)
        newrep = UnitSphericalRepresentation(lat=u.Quantity(cirs_dec, u.radian, copy=False), lon=u.Quantity(cirs_ra, u.radian, copy=False), copy=False)
    else:
        astrom_eb = CartesianRepresentation(astrom['eb'], unit=u.au, xyz_axis=-1, copy=False)
        newcart = icrs_coo.cartesian - astrom_eb
        srepr = newcart.represent_as(SphericalRepresentation)
        (cirs_ra, cirs_dec) = atciqz(srepr.without_differentials(), astrom)
        newrep = SphericalRepresentation(lat=u.Quantity(cirs_dec, u.radian, copy=False), lon=u.Quantity(cirs_ra, u.radian, copy=False), distance=srepr.distance, copy=False)
    return cirs_frame.realize_frame(newrep)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, CIRS, ICRS)
def cirs_to_icrs(cirs_coo, icrs_frame):
    if False:
        while True:
            i = 10
    astrom = erfa_astrom.get().apco(cirs_coo)
    srepr = cirs_coo.represent_as(SphericalRepresentation)
    (i_ra, i_dec) = aticq(srepr.without_differentials(), astrom)
    if cirs_coo.data.get_name() == 'unitspherical' or cirs_coo.data.to_cartesian().x.unit == u.one:
        newrep = UnitSphericalRepresentation(lat=u.Quantity(i_dec, u.radian, copy=False), lon=u.Quantity(i_ra, u.radian, copy=False), copy=False)
    else:
        intermedrep = SphericalRepresentation(lat=u.Quantity(i_dec, u.radian, copy=False), lon=u.Quantity(i_ra, u.radian, copy=False), distance=srepr.distance, copy=False)
        astrom_eb = CartesianRepresentation(astrom['eb'], unit=u.au, xyz_axis=-1, copy=False)
        newrep = intermedrep + astrom_eb
    return icrs_frame.realize_frame(newrep)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ICRS, GCRS)
def icrs_to_gcrs(icrs_coo, gcrs_frame):
    if False:
        return 10
    astrom = erfa_astrom.get().apcs(gcrs_frame)
    if icrs_coo.data.get_name() == 'unitspherical' or icrs_coo.data.to_cartesian().x.unit == u.one:
        srepr = icrs_coo.represent_as(SphericalRepresentation)
        (gcrs_ra, gcrs_dec) = atciqz(srepr.without_differentials(), astrom)
        newrep = UnitSphericalRepresentation(lat=u.Quantity(gcrs_dec, u.radian, copy=False), lon=u.Quantity(gcrs_ra, u.radian, copy=False), copy=False)
    else:
        astrom_eb = CartesianRepresentation(astrom['eb'], unit=u.au, xyz_axis=-1, copy=False)
        newcart = icrs_coo.cartesian - astrom_eb
        srepr = newcart.represent_as(SphericalRepresentation)
        (gcrs_ra, gcrs_dec) = atciqz(srepr.without_differentials(), astrom)
        newrep = SphericalRepresentation(lat=u.Quantity(gcrs_dec, u.radian, copy=False), lon=u.Quantity(gcrs_ra, u.radian, copy=False), distance=srepr.distance, copy=False)
    return gcrs_frame.realize_frame(newrep)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, GCRS, ICRS)
def gcrs_to_icrs(gcrs_coo, icrs_frame):
    if False:
        print('Hello World!')
    astrom = erfa_astrom.get().apcs(gcrs_coo)
    srepr = gcrs_coo.represent_as(SphericalRepresentation)
    (i_ra, i_dec) = aticq(srepr.without_differentials(), astrom)
    if gcrs_coo.data.get_name() == 'unitspherical' or gcrs_coo.data.to_cartesian().x.unit == u.one:
        newrep = UnitSphericalRepresentation(lat=u.Quantity(i_dec, u.radian, copy=False), lon=u.Quantity(i_ra, u.radian, copy=False), copy=False)
    else:
        intermedrep = SphericalRepresentation(lat=u.Quantity(i_dec, u.radian, copy=False), lon=u.Quantity(i_ra, u.radian, copy=False), distance=srepr.distance, copy=False)
        astrom_eb = CartesianRepresentation(astrom['eb'], unit=u.au, xyz_axis=-1, copy=False)
        newrep = intermedrep + astrom_eb
    return icrs_frame.realize_frame(newrep)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, GCRS, HCRS)
def gcrs_to_hcrs(gcrs_coo, hcrs_frame):
    if False:
        for i in range(10):
            print('nop')
    if np.any(gcrs_coo.obstime != hcrs_frame.obstime):
        frameattrs = gcrs_coo.get_frame_attr_defaults()
        frameattrs['obstime'] = hcrs_frame.obstime
        gcrs_coo = gcrs_coo.transform_to(GCRS(**frameattrs))
    astrom = erfa_astrom.get().apcs(gcrs_coo)
    srepr = gcrs_coo.represent_as(SphericalRepresentation)
    (i_ra, i_dec) = aticq(srepr.without_differentials(), astrom)
    i_ra = u.Quantity(i_ra, u.radian, copy=False)
    i_dec = u.Quantity(i_dec, u.radian, copy=False)
    if gcrs_coo.data.get_name() == 'unitspherical' or gcrs_coo.data.to_cartesian().x.unit == u.one:
        newrep = UnitSphericalRepresentation(lat=i_dec, lon=i_ra, copy=False)
    else:
        intermedrep = SphericalRepresentation(lat=i_dec, lon=i_ra, distance=srepr.distance, copy=False)
        eh = astrom['eh'] * astrom['em'][..., np.newaxis]
        eh = CartesianRepresentation(eh, unit=u.au, xyz_axis=-1, copy=False)
        newrep = intermedrep.to_cartesian() + eh
    return hcrs_frame.realize_frame(newrep)
_NEED_ORIGIN_HINT = 'The input {0} coordinates do not have length units. This probably means you created coordinates with lat/lon but no distance.  Heliocentric<->ICRS transforms cannot function in this case because there is an origin shift.'

@frame_transform_graph.transform(AffineTransform, HCRS, ICRS)
def hcrs_to_icrs(hcrs_coo, icrs_frame):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(hcrs_coo.data, UnitSphericalRepresentation):
        raise u.UnitsError(_NEED_ORIGIN_HINT.format(hcrs_coo.__class__.__name__))
    return (None, get_offset_sun_from_barycenter(hcrs_coo.obstime, include_velocity=bool(hcrs_coo.data.differentials)))

@frame_transform_graph.transform(AffineTransform, ICRS, HCRS)
def icrs_to_hcrs(icrs_coo, hcrs_frame):
    if False:
        return 10
    if isinstance(icrs_coo.data, UnitSphericalRepresentation):
        raise u.UnitsError(_NEED_ORIGIN_HINT.format(icrs_coo.__class__.__name__))
    return (None, get_offset_sun_from_barycenter(hcrs_frame.obstime, reverse=True, include_velocity=bool(icrs_coo.data.differentials)))
frame_transform_graph._add_merged_transform(CIRS, ICRS, CIRS)
frame_transform_graph._add_merged_transform(GCRS, ICRS, GCRS)
frame_transform_graph._add_merged_transform(HCRS, ICRS, HCRS)