from astropy import units as u
from astropy.coordinates import representation as r
from astropy.coordinates.attributes import DifferentialAttribute
from astropy.coordinates.baseframe import BaseCoordinateFrame, RepresentationMapping, base_doc, frame_transform_graph
from astropy.coordinates.transformations import AffineTransform
from astropy.time import Time
from astropy.utils.decorators import format_doc
from .baseradec import BaseRADecFrame
from .baseradec import doc_components as doc_components_radec
from .galactic import Galactic
from .icrs import ICRS
J2000 = Time('J2000')
v_bary_Schoenrich2010 = r.CartesianDifferential([11.1, 12.24, 7.25] * u.km / u.s)
__all__ = ['LSR', 'GalacticLSR', 'LSRK', 'LSRD']
doc_footer_lsr = '\n    Other parameters\n    ----------------\n    v_bary : `~astropy.coordinates.CartesianDifferential`\n        The velocity of the solar system barycenter with respect to the LSR, in\n        Galactic cartesian velocity components.\n'

@format_doc(base_doc, components=doc_components_radec, footer=doc_footer_lsr)
class LSR(BaseRADecFrame):
    """A coordinate or frame in the Local Standard of Rest (LSR).

    This coordinate frame is axis-aligned and co-spatial with
    `~astropy.coordinates.ICRS`, but has a velocity offset relative to the
    solar system barycenter to remove the peculiar motion of the sun relative
    to the LSR. Roughly, the LSR is the mean velocity of the stars in the solar
    neighborhood, but the precise definition of which depends on the study. As
    defined in Schönrich et al. (2010): "The LSR is the rest frame at the
    location of the Sun of a star that would be on a circular orbit in the
    gravitational potential one would obtain by azimuthally averaging away
    non-axisymmetric features in the actual Galactic potential." No such orbit
    truly exists, but it is still a commonly used velocity frame.

    We use default values from Schönrich et al. (2010) for the barycentric
    velocity relative to the LSR, which is defined in Galactic (right-handed)
    cartesian velocity components
    :math:`(U, V, W) = (11.1, 12.24, 7.25)~{{\\rm km}}~{{\\rm s}}^{{-1}}`. These
    values are customizable via the ``v_bary`` argument which specifies the
    velocity of the solar system barycenter with respect to the LSR.

    The frame attributes are listed under **Other Parameters**.

    """
    v_bary = DifferentialAttribute(default=v_bary_Schoenrich2010, allowed_classes=[r.CartesianDifferential])

@frame_transform_graph.transform(AffineTransform, ICRS, LSR)
def icrs_to_lsr(icrs_coord, lsr_frame):
    if False:
        while True:
            i = 10
    v_bary_gal = Galactic(lsr_frame.v_bary.to_cartesian())
    v_bary_icrs = v_bary_gal.transform_to(icrs_coord)
    v_offset = v_bary_icrs.data.represent_as(r.CartesianDifferential)
    offset = r.CartesianRepresentation([0, 0, 0] * u.au, differentials=v_offset)
    return (None, offset)

@frame_transform_graph.transform(AffineTransform, LSR, ICRS)
def lsr_to_icrs(lsr_coord, icrs_frame):
    if False:
        print('Hello World!')
    v_bary_gal = Galactic(lsr_coord.v_bary.to_cartesian())
    v_bary_icrs = v_bary_gal.transform_to(icrs_frame)
    v_offset = v_bary_icrs.data.represent_as(r.CartesianDifferential)
    offset = r.CartesianRepresentation([0, 0, 0] * u.au, differentials=-v_offset)
    return (None, offset)
doc_components_gal = "\n    l : `~astropy.coordinates.Angle`, optional, keyword-only\n        The Galactic longitude for this object (``b`` must also be given and\n        ``representation`` must be None).\n    b : `~astropy.coordinates.Angle`, optional, keyword-only\n        The Galactic latitude for this object (``l`` must also be given and\n        ``representation`` must be None).\n    distance : `~astropy.units.Quantity` ['length'], optional, keyword-only\n        The Distance for this object along the line-of-sight.\n        (``representation`` must be None).\n\n    pm_l_cosb : `~astropy.units.Quantity` ['angular speed'], optional, keyword-only\n        The proper motion in Galactic longitude (including the ``cos(b)`` term)\n        for this object (``pm_b`` must also be given).\n    pm_b : `~astropy.units.Quantity` ['angular speed'], optional, keyword-only\n        The proper motion in Galactic latitude for this object (``pm_l_cosb``\n        must also be given).\n    radial_velocity : `~astropy.units.Quantity` ['speed'], optional, keyword-only\n        The radial velocity of this object.\n"

@format_doc(base_doc, components=doc_components_gal, footer=doc_footer_lsr)
class GalacticLSR(BaseCoordinateFrame):
    """A coordinate or frame in the Local Standard of Rest (LSR), axis-aligned
    to the Galactic frame.

    This coordinate frame is axis-aligned and co-spatial with
    `~astropy.coordinates.ICRS`, but has a velocity offset relative to the
    solar system barycenter to remove the peculiar motion of the sun relative
    to the LSR. Roughly, the LSR is the mean velocity of the stars in the solar
    neighborhood, but the precise definition of which depends on the study. As
    defined in Schönrich et al. (2010): "The LSR is the rest frame at the
    location of the Sun of a star that would be on a circular orbit in the
    gravitational potential one would obtain by azimuthally averaging away
    non-axisymmetric features in the actual Galactic potential." No such orbit
    truly exists, but it is still a commonly used velocity frame.

    We use default values from Schönrich et al. (2010) for the barycentric
    velocity relative to the LSR, which is defined in Galactic (right-handed)
    cartesian velocity components
    :math:`(U, V, W) = (11.1, 12.24, 7.25)~{{\\rm km}}~{{\\rm s}}^{{-1}}`. These
    values are customizable via the ``v_bary`` argument which specifies the
    velocity of the solar system barycenter with respect to the LSR.

    The frame attributes are listed under **Other Parameters**.

    """
    frame_specific_representation_info = {r.SphericalRepresentation: [RepresentationMapping('lon', 'l'), RepresentationMapping('lat', 'b')]}
    default_representation = r.SphericalRepresentation
    default_differential = r.SphericalCosLatDifferential
    v_bary = DifferentialAttribute(default=v_bary_Schoenrich2010)

@frame_transform_graph.transform(AffineTransform, Galactic, GalacticLSR)
def galactic_to_galacticlsr(galactic_coord, lsr_frame):
    if False:
        print('Hello World!')
    v_bary_gal = Galactic(lsr_frame.v_bary.to_cartesian())
    v_offset = v_bary_gal.data.represent_as(r.CartesianDifferential)
    offset = r.CartesianRepresentation([0, 0, 0] * u.au, differentials=v_offset)
    return (None, offset)

@frame_transform_graph.transform(AffineTransform, GalacticLSR, Galactic)
def galacticlsr_to_galactic(lsr_coord, galactic_frame):
    if False:
        return 10
    v_bary_gal = Galactic(lsr_coord.v_bary.to_cartesian())
    v_offset = v_bary_gal.data.represent_as(r.CartesianDifferential)
    offset = r.CartesianRepresentation([0, 0, 0] * u.au, differentials=-v_offset)
    return (None, offset)

class LSRK(BaseRADecFrame):
    """A coordinate or frame in the Kinematic Local Standard of Rest (LSR).

    This frame is defined as having a velocity of 20 km/s towards RA=270 Dec=30
    (B1900) relative to the solar system Barycenter. This is defined in:

        Gordon 1975, Methods of Experimental Physics: Volume 12:
        Astrophysics, Part C: Radio Observations - Section 6.1.5.

    This coordinate frame is axis-aligned and co-spatial with
    `~astropy.coordinates.ICRS`, but has a velocity offset relative to the
    solar system barycenter to remove the peculiar motion of the sun relative
    to the LSRK.

    """
V_OFFSET_LSRK = r.CartesianDifferential([0.28999706839034606, -17.317264789717928, 10.00141199546947] * u.km / u.s)
ICRS_LSRK_OFFSET = r.CartesianRepresentation([0, 0, 0] * u.au, differentials=V_OFFSET_LSRK)
LSRK_ICRS_OFFSET = r.CartesianRepresentation([0, 0, 0] * u.au, differentials=-V_OFFSET_LSRK)

@frame_transform_graph.transform(AffineTransform, ICRS, LSRK)
def icrs_to_lsrk(icrs_coord, lsr_frame):
    if False:
        print('Hello World!')
    return (None, ICRS_LSRK_OFFSET)

@frame_transform_graph.transform(AffineTransform, LSRK, ICRS)
def lsrk_to_icrs(lsr_coord, icrs_frame):
    if False:
        for i in range(10):
            print('nop')
    return (None, LSRK_ICRS_OFFSET)

class LSRD(BaseRADecFrame):
    """A coordinate or frame in the Dynamical Local Standard of Rest (LSRD).

    This frame is defined as a velocity of U=9 km/s, V=12 km/s,
    and W=7 km/s in Galactic coordinates or 16.552945 km/s
    towards l=53.13 b=25.02. This is defined in:

       Delhaye 1965, Solar Motion and Velocity Distribution of
       Common Stars.

    This coordinate frame is axis-aligned and co-spatial with
    `~astropy.coordinates.ICRS`, but has a velocity offset relative to the
    solar system barycenter to remove the peculiar motion of the sun relative
    to the LSRD.

    """
V_OFFSET_LSRD = r.CartesianDifferential([-0.6382306360182073, -14.585424483191094, 7.8011572411006815] * u.km / u.s)
ICRS_LSRD_OFFSET = r.CartesianRepresentation([0, 0, 0] * u.au, differentials=V_OFFSET_LSRD)
LSRD_ICRS_OFFSET = r.CartesianRepresentation([0, 0, 0] * u.au, differentials=-V_OFFSET_LSRD)

@frame_transform_graph.transform(AffineTransform, ICRS, LSRD)
def icrs_to_lsrd(icrs_coord, lsr_frame):
    if False:
        print('Hello World!')
    return (None, ICRS_LSRD_OFFSET)

@frame_transform_graph.transform(AffineTransform, LSRD, ICRS)
def lsrd_to_icrs(lsr_coord, icrs_frame):
    if False:
        return 10
    return (None, LSRD_ICRS_OFFSET)
frame_transform_graph._add_merged_transform(LSR, ICRS, LSR)
frame_transform_graph._add_merged_transform(GalacticLSR, Galactic, GalacticLSR)