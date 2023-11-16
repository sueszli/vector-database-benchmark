from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_transpose, rotation_matrix
from astropy.coordinates.transformations import DynamicMatrixTransform
from .fk5 import FK5
from .icrs import ICRS
from .utils import EQUINOX_J2000

def _icrs_to_fk5_matrix():
    if False:
        i = 10
        return i + 15
    '\n    B-matrix from USNO circular 179.  Used by the ICRS->FK5 transformation\n    functions.\n    '
    eta0 = -19.9 / 3600000.0
    xi0 = 9.1 / 3600000.0
    da0 = -22.9 / 3600000.0
    return rotation_matrix(-eta0, 'x') @ rotation_matrix(xi0, 'y') @ rotation_matrix(da0, 'z')
_ICRS_TO_FK5_J2000_MAT = _icrs_to_fk5_matrix()

@frame_transform_graph.transform(DynamicMatrixTransform, ICRS, FK5)
def icrs_to_fk5(icrscoord, fk5frame):
    if False:
        i = 10
        return i + 15
    pmat = fk5frame._precession_matrix(EQUINOX_J2000, fk5frame.equinox)
    return pmat @ _ICRS_TO_FK5_J2000_MAT

@frame_transform_graph.transform(DynamicMatrixTransform, FK5, ICRS)
def fk5_to_icrs(fk5coord, icrsframe):
    if False:
        i = 10
        return i + 15
    pmat = fk5coord._precession_matrix(fk5coord.equinox, EQUINOX_J2000)
    return matrix_transpose(_ICRS_TO_FK5_J2000_MAT) @ pmat