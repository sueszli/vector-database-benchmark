import numpy as np
from astropy import units as u
from astropy.coordinates import earth_orientation as earth
from astropy.coordinates.attributes import TimeAttribute
from astropy.coordinates.baseframe import base_doc, frame_transform_graph
from astropy.coordinates.representation import CartesianRepresentation, UnitSphericalRepresentation
from astropy.coordinates.transformations import DynamicMatrixTransform, FunctionTransformWithFiniteDifference
from astropy.utils.decorators import format_doc
from .baseradec import BaseRADecFrame, doc_components
from .utils import EQUINOX_B1950
__all__ = ['FK4', 'FK4NoETerms']
doc_footer_fk4 = '\n    Other parameters\n    ----------------\n    equinox : `~astropy.time.Time`\n        The equinox of this frame.\n    obstime : `~astropy.time.Time`\n        The time this frame was observed.  If ``None``, will be the same as\n        ``equinox``.\n'

@format_doc(base_doc, components=doc_components, footer=doc_footer_fk4)
class FK4(BaseRADecFrame):
    """
    A coordinate or frame in the FK4 system.

    Note that this is a barycentric version of FK4 - that is, the origin for
    this frame is the Solar System Barycenter, *not* the Earth geocenter.

    The frame attributes are listed under **Other Parameters**.
    """
    equinox = TimeAttribute(default=EQUINOX_B1950)
    obstime = TimeAttribute(default=None, secondary_attribute='equinox')

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, FK4, FK4)
def fk4_to_fk4(fk4coord1, fk4frame2):
    if False:
        print('Hello World!')
    fnoe_w_eqx1 = fk4coord1.transform_to(FK4NoETerms(equinox=fk4coord1.equinox))
    fnoe_w_eqx2 = fnoe_w_eqx1.transform_to(FK4NoETerms(equinox=fk4frame2.equinox))
    return fnoe_w_eqx2.transform_to(fk4frame2)

@format_doc(base_doc, components=doc_components, footer=doc_footer_fk4)
class FK4NoETerms(BaseRADecFrame):
    """
    A coordinate or frame in the FK4 system, but with the E-terms of aberration
    removed.

    The frame attributes are listed under **Other Parameters**.
    """
    equinox = TimeAttribute(default=EQUINOX_B1950)
    obstime = TimeAttribute(default=None, secondary_attribute='equinox')

    @staticmethod
    def _precession_matrix(oldequinox, newequinox):
        if False:
            i = 10
            return i + 15
        "\n        Compute and return the precession matrix for FK4 using Newcomb's method.\n        Used inside some of the transformation functions.\n\n        Parameters\n        ----------\n        oldequinox : `~astropy.time.Time`\n            The equinox to precess from.\n        newequinox : `~astropy.time.Time`\n            The equinox to precess to.\n\n        Returns\n        -------\n        newcoord : array\n            The precession matrix to transform to the new equinox\n        "
        return earth._precession_matrix_besselian(oldequinox.byear, newequinox.byear)

@frame_transform_graph.transform(DynamicMatrixTransform, FK4NoETerms, FK4NoETerms)
def fk4noe_to_fk4noe(fk4necoord1, fk4neframe2):
    if False:
        i = 10
        return i + 15
    return fk4necoord1._precession_matrix(fk4necoord1.equinox, fk4neframe2.equinox)

def fk4_e_terms(equinox):
    if False:
        return 10
    '\n    Return the e-terms of aberration vector.\n\n    Parameters\n    ----------\n    equinox : Time object\n        The equinox for which to compute the e-terms\n    '
    k = 0.0056932
    k = np.radians(k)
    e = earth.eccentricity(equinox.jd)
    g = earth.mean_lon_of_perigee(equinox.jd)
    g = np.radians(g)
    o = earth.obliquity(equinox.jd, algorithm=1980)
    o = np.radians(o)
    return (e * k * np.sin(g), -e * k * np.cos(g) * np.cos(o), -e * k * np.cos(g) * np.sin(o))

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, FK4, FK4NoETerms)
def fk4_to_fk4_no_e(fk4coord, fk4noeframe):
    if False:
        return 10
    rep = fk4coord.cartesian
    d_orig = rep.norm()
    rep /= d_orig
    eterms_a = CartesianRepresentation(u.Quantity(fk4_e_terms(fk4coord.equinox), u.dimensionless_unscaled, copy=False), copy=False)
    rep = rep - eterms_a + eterms_a.dot(rep) * rep
    d_new = rep.norm()
    rep *= d_orig / d_new
    if isinstance(fk4coord.data, UnitSphericalRepresentation):
        rep = rep.represent_as(UnitSphericalRepresentation)
    newobstime = fk4coord._obstime if fk4noeframe._obstime is None else fk4noeframe._obstime
    fk4noe = FK4NoETerms(rep, equinox=fk4coord.equinox, obstime=newobstime)
    if fk4coord.equinox != fk4noeframe.equinox:
        fk4noe = fk4noe.transform_to(fk4noeframe)
    return fk4noe

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, FK4NoETerms, FK4)
def fk4_no_e_to_fk4(fk4noecoord, fk4frame):
    if False:
        print('Hello World!')
    if fk4noecoord.equinox != fk4frame.equinox:
        fk4noe_w_fk4equinox = FK4NoETerms(equinox=fk4frame.equinox, obstime=fk4noecoord.obstime)
        fk4noecoord = fk4noecoord.transform_to(fk4noe_w_fk4equinox)
    rep = fk4noecoord.cartesian
    d_orig = rep.norm()
    rep /= d_orig
    eterms_a = CartesianRepresentation(u.Quantity(fk4_e_terms(fk4noecoord.equinox), u.dimensionless_unscaled, copy=False), copy=False)
    rep0 = rep.copy()
    for _ in range(10):
        rep = (eterms_a + rep0) / (1.0 + eterms_a.dot(rep))
    d_new = rep.norm()
    rep *= d_orig / d_new
    if isinstance(fk4noecoord.data, UnitSphericalRepresentation):
        rep = rep.represent_as(UnitSphericalRepresentation)
    return fk4frame.realize_frame(rep)