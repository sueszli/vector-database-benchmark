"""
================================================================
Convert a radial velocity to the Galactic Standard of Rest (GSR)
================================================================

Radial or line-of-sight velocities of sources are often reported in a
Heliocentric or Solar-system barycentric reference frame. A common
transformation incorporates the projection of the Sun's motion along the
line-of-sight to the target, hence transforming it to a Galactic rest frame
instead (sometimes referred to as the Galactic Standard of Rest, GSR). This
transformation depends on the assumptions about the orientation of the Galactic
frame relative to the bary- or Heliocentric frame. It also depends on the
assumed solar velocity vector. Here we'll demonstrate how to perform this
transformation using a sky position and barycentric radial-velocity.


*By: Adrian Price-Whelan*

*License: BSD*


"""
import astropy.coordinates as coord
import astropy.units as u
coord.galactocentric_frame_defaults.set('latest')
icrs = coord.SkyCoord(ra=258.58356362 * u.deg, dec=14.55255619 * u.deg, radial_velocity=-16.1 * u.km / u.s, frame='icrs')
v_sun = coord.Galactocentric().galcen_v_sun.to_cartesian()
gal = icrs.transform_to(coord.Galactic)
cart_data = gal.data.to_cartesian()
unit_vector = cart_data / cart_data.norm()
v_proj = v_sun.dot(unit_vector)
rv_gsr = icrs.radial_velocity + v_proj
print(rv_gsr)

def rv_to_gsr(c, v_sun=None):
    if False:
        i = 10
        return i + 15
    'Transform a barycentric radial velocity to the Galactic Standard of Rest\n    (GSR).\n\n    The input radial velocity must be passed in as a\n\n    Parameters\n    ----------\n    c : `~astropy.coordinates.BaseCoordinateFrame` subclass instance\n        The radial velocity, associated with a sky coordinates, to be\n        transformed.\n    v_sun : `~astropy.units.Quantity`, optional\n        The 3D velocity of the solar system barycenter in the GSR frame.\n        Defaults to the same solar motion as in the\n        `~astropy.coordinates.Galactocentric` frame.\n\n    Returns\n    -------\n    v_gsr : `~astropy.units.Quantity`\n        The input radial velocity transformed to a GSR frame.\n\n    '
    if v_sun is None:
        v_sun = coord.Galactocentric().galcen_v_sun.to_cartesian()
    gal = c.transform_to(coord.Galactic)
    cart_data = gal.data.to_cartesian()
    unit_vector = cart_data / cart_data.norm()
    v_proj = v_sun.dot(unit_vector)
    return c.radial_velocity + v_proj
rv_gsr = rv_to_gsr(icrs)
print(rv_gsr)