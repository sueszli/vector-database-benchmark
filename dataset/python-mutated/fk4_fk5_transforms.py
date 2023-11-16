import numpy as np
from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_transpose
from astropy.coordinates.transformations import DynamicMatrixTransform
from .fk4 import FK4NoETerms
from .fk5 import FK5
from .utils import EQUINOX_B1950, EQUINOX_J2000
_B1950_TO_J2000_M = np.array([[0.9999256794956877, -0.0111814832204662, -0.0048590038153592], [0.0111814832391717, +0.9999374848933135, -2.71625947142e-05], [0.0048590037723143, -2.7170293744e-05, +0.9999881946023742]])
_FK4_CORR = np.array([[-0.0026455262, -1.1539918689, +2.111134619], [+1.1540628161, -0.0129042997, +0.0236021478], [-2.1112979048, -0.0056024448, +0.0102587734]]) * 1e-06

def _fk4_B_matrix(obstime):
    if False:
        print('Hello World!')
    '\n    This is a correction term in the FK4 transformations because FK4 is a\n    rotating system - see Murray 89 eqn 29.\n    '
    T = (obstime.jyear - 1950.0) / 100.0
    if getattr(T, 'shape', ()):
        T.shape += (1, 1)
    return _B1950_TO_J2000_M + _FK4_CORR * T

@frame_transform_graph.transform(DynamicMatrixTransform, FK4NoETerms, FK5)
def fk4_no_e_to_fk5(fk4noecoord, fk5frame):
    if False:
        i = 10
        return i + 15
    B = _fk4_B_matrix(fk4noecoord.obstime)
    pmat1 = fk4noecoord._precession_matrix(fk4noecoord.equinox, EQUINOX_B1950)
    pmat2 = fk5frame._precession_matrix(EQUINOX_J2000, fk5frame.equinox)
    return pmat2 @ B @ pmat1

@frame_transform_graph.transform(DynamicMatrixTransform, FK5, FK4NoETerms)
def fk5_to_fk4_no_e(fk5coord, fk4noeframe):
    if False:
        for i in range(10):
            print('nop')
    B = matrix_transpose(_fk4_B_matrix(fk4noeframe.obstime))
    pmat1 = fk5coord._precession_matrix(fk5coord.equinox, EQUINOX_J2000)
    pmat2 = fk4noeframe._precession_matrix(EQUINOX_B1950, fk4noeframe.equinox)
    return pmat2 @ B @ pmat1