import numpy as np
from astropy import units as u
from astropy.coordinates import representation as r
from astropy.coordinates.attributes import EarthLocationAttribute, QuantityAttribute, TimeAttribute
from astropy.coordinates.baseframe import BaseCoordinateFrame, RepresentationMapping, base_doc
from astropy.utils.decorators import format_doc
__all__ = ['AltAz']
_90DEG = 90 * u.deg
doc_components = "\n    az : `~astropy.coordinates.Angle`, optional, keyword-only\n        The Azimuth for this object (``alt`` must also be given and\n        ``representation`` must be None).\n    alt : `~astropy.coordinates.Angle`, optional, keyword-only\n        The Altitude for this object (``az`` must also be given and\n        ``representation`` must be None).\n    distance : `~astropy.units.Quantity` ['length'], optional, keyword-only\n        The Distance for this object along the line-of-sight.\n\n    pm_az_cosalt : `~astropy.units.Quantity` ['angular speed'], optional, keyword-only\n        The proper motion in azimuth (including the ``cos(alt)`` factor) for\n        this object (``pm_alt`` must also be given).\n    pm_alt : `~astropy.units.Quantity` ['angular speed'], optional, keyword-only\n        The proper motion in altitude for this object (``pm_az_cosalt`` must\n        also be given).\n    radial_velocity : `~astropy.units.Quantity` ['speed'], optional, keyword-only\n        The radial velocity of this object."
doc_footer = '\n    Other parameters\n    ----------------\n    obstime : `~astropy.time.Time`\n        The time at which the observation is taken.  Used for determining the\n        position and orientation of the Earth.\n    location : `~astropy.coordinates.EarthLocation`\n        The location on the Earth.  This can be specified either as an\n        `~astropy.coordinates.EarthLocation` object or as anything that can be\n        transformed to an `~astropy.coordinates.ITRS` frame.\n    pressure : `~astropy.units.Quantity` [\'pressure\']\n        The atmospheric pressure as an `~astropy.units.Quantity` with pressure\n        units.  This is necessary for performing refraction corrections.\n        Setting this to 0 (the default) will disable refraction calculations\n        when transforming to/from this frame.\n    temperature : `~astropy.units.Quantity` [\'temperature\']\n        The ground-level temperature as an `~astropy.units.Quantity` in\n        deg C.  This is necessary for performing refraction corrections.\n    relative_humidity : `~astropy.units.Quantity` [\'dimensionless\'] or number\n        The relative humidity as a dimensionless quantity between 0 to 1.\n        This is necessary for performing refraction corrections.\n    obswl : `~astropy.units.Quantity` [\'length\']\n        The average wavelength of observations as an `~astropy.units.Quantity`\n         with length units.  This is necessary for performing refraction\n         corrections.\n\n    Notes\n    -----\n    The refraction model is based on that implemented in ERFA, which is fast\n    but becomes inaccurate for altitudes below about 5 degrees.  Near and below\n    altitudes of 0, it can even give meaningless answers, and in this case\n    transforming to AltAz and back to another frame can give highly discrepant\n    results.  For much better numerical stability, leave the ``pressure`` at\n    ``0`` (the default), thereby disabling the refraction correction and\n    yielding "topocentric" horizontal coordinates.\n    '

@format_doc(base_doc, components=doc_components, footer=doc_footer)
class AltAz(BaseCoordinateFrame):
    """
    A coordinate or frame in the Altitude-Azimuth system (Horizontal
    coordinates) with respect to the WGS84 ellipsoid.  Azimuth is oriented
    East of North (i.e., N=0, E=90 degrees).  Altitude is also known as
    elevation angle, so this frame is also in the Azimuth-Elevation system.

    This frame is assumed to *include* refraction effects if the ``pressure``
    frame attribute is non-zero.

    The frame attributes are listed under **Other Parameters**, which are
    necessary for transforming from AltAz to some other system.
    """
    frame_specific_representation_info = {r.SphericalRepresentation: [RepresentationMapping('lon', 'az'), RepresentationMapping('lat', 'alt')]}
    default_representation = r.SphericalRepresentation
    default_differential = r.SphericalCosLatDifferential
    obstime = TimeAttribute(default=None)
    location = EarthLocationAttribute(default=None)
    pressure = QuantityAttribute(default=0, unit=u.hPa)
    temperature = QuantityAttribute(default=0, unit=u.deg_C)
    relative_humidity = QuantityAttribute(default=0, unit=u.dimensionless_unscaled)
    obswl = QuantityAttribute(default=1 * u.micron, unit=u.micron)

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)

    @property
    def secz(self):
        if False:
            return 10
        '\n        Secant of the zenith angle for this coordinate, a common estimate of\n        the airmass.\n        '
        return 1 / np.sin(self.alt)

    @property
    def zen(self):
        if False:
            return 10
        '\n        The zenith angle (or zenith distance / co-altitude) for this coordinate.\n        '
        return _90DEG.to(self.alt.unit) - self.alt