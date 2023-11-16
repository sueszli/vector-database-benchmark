from astropy import units as u
from astropy.coordinates import representation as r
from astropy.coordinates.attributes import EarthLocationAttribute, QuantityAttribute, TimeAttribute
from astropy.coordinates.baseframe import BaseCoordinateFrame, RepresentationMapping, base_doc
from astropy.utils.decorators import format_doc
__all__ = ['HADec']
doc_components = "\n    ha : `~astropy.coordinates.Angle`, optional, keyword-only\n        The Hour Angle for this object (``dec`` must also be given and\n        ``representation`` must be None).\n    dec : `~astropy.coordinates.Angle`, optional, keyword-only\n        The Declination for this object (``ha`` must also be given and\n        ``representation`` must be None).\n    distance : `~astropy.units.Quantity` ['length'], optional, keyword-only\n        The Distance for this object along the line-of-sight.\n\n    pm_ha_cosdec : `~astropy.units.Quantity` ['angular speed'], optional, keyword-only\n        The proper motion in hour angle (including the ``cos(dec)`` factor) for\n        this object (``pm_dec`` must also be given).\n    pm_dec : `~astropy.units.Quantity` ['angular speed'], optional, keyword-only\n        The proper motion in declination for this object (``pm_ha_cosdec`` must\n        also be given).\n    radial_velocity : `~astropy.units.Quantity` ['speed'], optional, keyword-only\n        The radial velocity of this object."
doc_footer = '\n    Other parameters\n    ----------------\n    obstime : `~astropy.time.Time`\n        The time at which the observation is taken.  Used for determining the\n        position and orientation of the Earth.\n    location : `~astropy.coordinates.EarthLocation`\n        The location on the Earth.  This can be specified either as an\n        `~astropy.coordinates.EarthLocation` object or as anything that can be\n        transformed to an `~astropy.coordinates.ITRS` frame.\n    pressure : `~astropy.units.Quantity` [\'pressure\']\n        The atmospheric pressure as an `~astropy.units.Quantity` with pressure\n        units.  This is necessary for performing refraction corrections.\n        Setting this to 0 (the default) will disable refraction calculations\n        when transforming to/from this frame.\n    temperature : `~astropy.units.Quantity` [\'temperature\']\n        The ground-level temperature as an `~astropy.units.Quantity` in\n        deg C.  This is necessary for performing refraction corrections.\n    relative_humidity : `~astropy.units.Quantity` [\'dimensionless\'] or number.\n        The relative humidity as a dimensionless quantity between 0 to 1.\n        This is necessary for performing refraction corrections.\n    obswl : `~astropy.units.Quantity` [\'length\']\n        The average wavelength of observations as an `~astropy.units.Quantity`\n         with length units.  This is necessary for performing refraction\n         corrections.\n\n    Notes\n    -----\n    The refraction model is based on that implemented in ERFA, which is fast\n    but becomes inaccurate for altitudes below about 5 degrees.  Near and below\n    altitudes of 0, it can even give meaningless answers, and in this case\n    transforming to HADec and back to another frame can give highly discrepant\n    results.  For much better numerical stability, leave the ``pressure`` at\n    ``0`` (the default), thereby disabling the refraction correction and\n    yielding "topocentric" equatorial coordinates.\n    '

@format_doc(base_doc, components=doc_components, footer=doc_footer)
class HADec(BaseCoordinateFrame):
    """
    A coordinate or frame in the Hour Angle-Declination system (Equatorial
    coordinates) with respect to the WGS84 ellipsoid.  Hour Angle is oriented
    with respect to upper culmination such that the hour angle is negative to
    the East and positive to the West.

    This frame is assumed to *include* refraction effects if the ``pressure``
    frame attribute is non-zero.

    The frame attributes are listed under **Other Parameters**, which are
    necessary for transforming from HADec to some other system.
    """
    frame_specific_representation_info = {r.SphericalRepresentation: [RepresentationMapping('lon', 'ha', u.hourangle), RepresentationMapping('lat', 'dec')]}
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
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        if self.has_data:
            self._set_data_lon_wrap_angle(self.data)

    @staticmethod
    def _set_data_lon_wrap_angle(data):
        if False:
            i = 10
            return i + 15
        if hasattr(data, 'lon'):
            data.lon.wrap_angle = 180.0 * u.deg
        return data

    def represent_as(self, base, s='base', in_frame_units=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure the wrap angle for any spherical\n        representations.\n        '
        data = super().represent_as(base, s, in_frame_units=in_frame_units)
        self._set_data_lon_wrap_angle(data)
        return data