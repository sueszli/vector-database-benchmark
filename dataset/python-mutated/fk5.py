from astropy.coordinates import earth_orientation as earth
from astropy.coordinates.attributes import TimeAttribute
from astropy.coordinates.baseframe import base_doc, frame_transform_graph
from astropy.coordinates.transformations import DynamicMatrixTransform
from astropy.utils.decorators import format_doc
from .baseradec import BaseRADecFrame, doc_components
from .utils import EQUINOX_J2000
__all__ = ['FK5']
doc_footer = '\n    Other parameters\n    ----------------\n    equinox : `~astropy.time.Time`\n        The equinox of this frame.\n'

@format_doc(base_doc, components=doc_components, footer=doc_footer)
class FK5(BaseRADecFrame):
    """
    A coordinate or frame in the FK5 system.

    Note that this is a barycentric version of FK5 - that is, the origin for
    this frame is the Solar System Barycenter, *not* the Earth geocenter.

    The frame attributes are listed under **Other Parameters**.
    """
    equinox = TimeAttribute(default=EQUINOX_J2000)

    @staticmethod
    def _precession_matrix(oldequinox, newequinox):
        if False:
            print('Hello World!')
        '\n        Compute and return the precession matrix for FK5 based on Capitaine et\n        al. 2003/IAU2006.  Used inside some of the transformation functions.\n\n        Parameters\n        ----------\n        oldequinox : `~astropy.time.Time`\n            The equinox to precess from.\n        newequinox : `~astropy.time.Time`\n            The equinox to precess to.\n\n        Returns\n        -------\n        newcoord : array\n            The precession matrix to transform to the new equinox\n        '
        return earth.precession_matrix_Capitaine(oldequinox, newequinox)

@frame_transform_graph.transform(DynamicMatrixTransform, FK5, FK5)
def fk5_to_fk5(fk5coord1, fk5frame2):
    if False:
        for i in range(10):
            print('nop')
    return fk5coord1._precession_matrix(fk5coord1.equinox, fk5frame2.equinox)