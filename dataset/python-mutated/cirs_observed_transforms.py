"""
Contains the transformation functions for getting to "observed" systems from CIRS.
"""
import erfa
import numpy as np
from astropy import units as u
from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates.erfa_astrom import erfa_astrom
from astropy.coordinates.representation import SphericalRepresentation, UnitSphericalRepresentation
from astropy.coordinates.transformations import FunctionTransformWithFiniteDifference
from .altaz import AltAz
from .cirs import CIRS
from .hadec import HADec
from .utils import PIOVER2

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, CIRS, AltAz)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, CIRS, HADec)
def cirs_to_observed(cirs_coo, observed_frame):
    if False:
        return 10
    if np.any(observed_frame.location != cirs_coo.location) or np.any(cirs_coo.obstime != observed_frame.obstime):
        cirs_coo = cirs_coo.transform_to(CIRS(obstime=observed_frame.obstime, location=observed_frame.location))
    is_unitspherical = isinstance(cirs_coo.data, UnitSphericalRepresentation) or cirs_coo.cartesian.x.unit == u.one
    usrepr = cirs_coo.represent_as(UnitSphericalRepresentation)
    cirs_ra = usrepr.lon.to_value(u.radian)
    cirs_dec = usrepr.lat.to_value(u.radian)
    astrom = erfa_astrom.get().apio(observed_frame)
    if isinstance(observed_frame, AltAz):
        (lon, zen, _, _, _) = erfa.atioq(cirs_ra, cirs_dec, astrom)
        lat = PIOVER2 - zen
    else:
        (_, _, lon, lat, _) = erfa.atioq(cirs_ra, cirs_dec, astrom)
    if is_unitspherical:
        rep = UnitSphericalRepresentation(lat=u.Quantity(lat, u.radian, copy=False), lon=u.Quantity(lon, u.radian, copy=False), copy=False)
    else:
        rep = SphericalRepresentation(lat=u.Quantity(lat, u.radian, copy=False), lon=u.Quantity(lon, u.radian, copy=False), distance=cirs_coo.distance, copy=False)
    return observed_frame.realize_frame(rep)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, AltAz, CIRS)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, HADec, CIRS)
def observed_to_cirs(observed_coo, cirs_frame):
    if False:
        for i in range(10):
            print('nop')
    usrepr = observed_coo.represent_as(UnitSphericalRepresentation)
    lon = usrepr.lon.to_value(u.radian)
    lat = usrepr.lat.to_value(u.radian)
    if isinstance(observed_coo, AltAz):
        coord_type = 'A'
        lat = PIOVER2 - lat
    else:
        coord_type = 'H'
    astrom = erfa_astrom.get().apio(observed_coo)
    (cirs_ra, cirs_dec) = erfa.atoiq(coord_type, lon, lat, astrom) << u.radian
    if isinstance(observed_coo.data, UnitSphericalRepresentation) or observed_coo.cartesian.x.unit == u.one:
        distance = None
    else:
        distance = observed_coo.distance
    cirs_at_aa_time = CIRS(ra=cirs_ra, dec=cirs_dec, distance=distance, obstime=observed_coo.obstime, location=observed_coo.location)
    return cirs_at_aa_time.transform_to(cirs_frame)