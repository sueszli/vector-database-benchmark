from sympy.core.function import diff
from sympy.core.singleton import S
from sympy.integrals.integrals import integrate
from sympy.physics.vector import Vector, express
from sympy.physics.vector.frame import _check_frame
from sympy.physics.vector.vector import _check_vector
__all__ = ['curl', 'divergence', 'gradient', 'is_conservative', 'is_solenoidal', 'scalar_potential', 'scalar_potential_difference']

def curl(vect, frame):
    if False:
        i = 10
        return i + 15
    "\n    Returns the curl of a vector field computed wrt the coordinate\n    symbols of the given frame.\n\n    Parameters\n    ==========\n\n    vect : Vector\n        The vector operand\n\n    frame : ReferenceFrame\n        The reference frame to calculate the curl in\n\n    Examples\n    ========\n\n    >>> from sympy.physics.vector import ReferenceFrame\n    >>> from sympy.physics.vector import curl\n    >>> R = ReferenceFrame('R')\n    >>> v1 = R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z\n    >>> curl(v1, R)\n    0\n    >>> v2 = R[0]*R[1]*R[2]*R.x\n    >>> curl(v2, R)\n    R_x*R_y*R.y - R_x*R_z*R.z\n\n    "
    _check_vector(vect)
    if vect == 0:
        return Vector(0)
    vect = express(vect, frame, variables=True)
    vectx = vect.dot(frame.x)
    vecty = vect.dot(frame.y)
    vectz = vect.dot(frame.z)
    outvec = Vector(0)
    outvec += (diff(vectz, frame[1]) - diff(vecty, frame[2])) * frame.x
    outvec += (diff(vectx, frame[2]) - diff(vectz, frame[0])) * frame.y
    outvec += (diff(vecty, frame[0]) - diff(vectx, frame[1])) * frame.z
    return outvec

def divergence(vect, frame):
    if False:
        i = 10
        return i + 15
    "\n    Returns the divergence of a vector field computed wrt the coordinate\n    symbols of the given frame.\n\n    Parameters\n    ==========\n\n    vect : Vector\n        The vector operand\n\n    frame : ReferenceFrame\n        The reference frame to calculate the divergence in\n\n    Examples\n    ========\n\n    >>> from sympy.physics.vector import ReferenceFrame\n    >>> from sympy.physics.vector import divergence\n    >>> R = ReferenceFrame('R')\n    >>> v1 = R[0]*R[1]*R[2] * (R.x+R.y+R.z)\n    >>> divergence(v1, R)\n    R_x*R_y + R_x*R_z + R_y*R_z\n    >>> v2 = 2*R[1]*R[2]*R.y\n    >>> divergence(v2, R)\n    2*R_z\n\n    "
    _check_vector(vect)
    if vect == 0:
        return S.Zero
    vect = express(vect, frame, variables=True)
    vectx = vect.dot(frame.x)
    vecty = vect.dot(frame.y)
    vectz = vect.dot(frame.z)
    out = S.Zero
    out += diff(vectx, frame[0])
    out += diff(vecty, frame[1])
    out += diff(vectz, frame[2])
    return out

def gradient(scalar, frame):
    if False:
        print('Hello World!')
    "\n    Returns the vector gradient of a scalar field computed wrt the\n    coordinate symbols of the given frame.\n\n    Parameters\n    ==========\n\n    scalar : sympifiable\n        The scalar field to take the gradient of\n\n    frame : ReferenceFrame\n        The frame to calculate the gradient in\n\n    Examples\n    ========\n\n    >>> from sympy.physics.vector import ReferenceFrame\n    >>> from sympy.physics.vector import gradient\n    >>> R = ReferenceFrame('R')\n    >>> s1 = R[0]*R[1]*R[2]\n    >>> gradient(s1, R)\n    R_y*R_z*R.x + R_x*R_z*R.y + R_x*R_y*R.z\n    >>> s2 = 5*R[0]**2*R[2]\n    >>> gradient(s2, R)\n    10*R_x*R_z*R.x + 5*R_x**2*R.z\n\n    "
    _check_frame(frame)
    outvec = Vector(0)
    scalar = express(scalar, frame, variables=True)
    for (i, x) in enumerate(frame):
        outvec += diff(scalar, frame[i]) * x
    return outvec

def is_conservative(field):
    if False:
        i = 10
        return i + 15
    "\n    Checks if a field is conservative.\n\n    Parameters\n    ==========\n\n    field : Vector\n        The field to check for conservative property\n\n    Examples\n    ========\n\n    >>> from sympy.physics.vector import ReferenceFrame\n    >>> from sympy.physics.vector import is_conservative\n    >>> R = ReferenceFrame('R')\n    >>> is_conservative(R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z)\n    True\n    >>> is_conservative(R[2] * R.y)\n    False\n\n    "
    if field == Vector(0):
        return True
    frame = list(field.separate())[0]
    return curl(field, frame).simplify() == Vector(0)

def is_solenoidal(field):
    if False:
        print('Hello World!')
    "\n    Checks if a field is solenoidal.\n\n    Parameters\n    ==========\n\n    field : Vector\n        The field to check for solenoidal property\n\n    Examples\n    ========\n\n    >>> from sympy.physics.vector import ReferenceFrame\n    >>> from sympy.physics.vector import is_solenoidal\n    >>> R = ReferenceFrame('R')\n    >>> is_solenoidal(R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z)\n    True\n    >>> is_solenoidal(R[1] * R.y)\n    False\n\n    "
    if field == Vector(0):
        return True
    frame = list(field.separate())[0]
    return divergence(field, frame).simplify() is S.Zero

def scalar_potential(field, frame):
    if False:
        while True:
            i = 10
    "\n    Returns the scalar potential function of a field in a given frame\n    (without the added integration constant).\n\n    Parameters\n    ==========\n\n    field : Vector\n        The vector field whose scalar potential function is to be\n        calculated\n\n    frame : ReferenceFrame\n        The frame to do the calculation in\n\n    Examples\n    ========\n\n    >>> from sympy.physics.vector import ReferenceFrame\n    >>> from sympy.physics.vector import scalar_potential, gradient\n    >>> R = ReferenceFrame('R')\n    >>> scalar_potential(R.z, R) == R[2]\n    True\n    >>> scalar_field = 2*R[0]**2*R[1]*R[2]\n    >>> grad_field = gradient(scalar_field, R)\n    >>> scalar_potential(grad_field, R)\n    2*R_x**2*R_y*R_z\n\n    "
    if not is_conservative(field):
        raise ValueError('Field is not conservative')
    if field == Vector(0):
        return S.Zero
    _check_frame(frame)
    field = express(field, frame, variables=True)
    dimensions = list(frame)
    temp_function = integrate(field.dot(dimensions[0]), frame[0])
    for (i, dim) in enumerate(dimensions[1:]):
        partial_diff = diff(temp_function, frame[i + 1])
        partial_diff = field.dot(dim) - partial_diff
        temp_function += integrate(partial_diff, frame[i + 1])
    return temp_function

def scalar_potential_difference(field, frame, point1, point2, origin):
    if False:
        i = 10
        return i + 15
    "\n    Returns the scalar potential difference between two points in a\n    certain frame, wrt a given field.\n\n    If a scalar field is provided, its values at the two points are\n    considered. If a conservative vector field is provided, the values\n    of its scalar potential function at the two points are used.\n\n    Returns (potential at position 2) - (potential at position 1)\n\n    Parameters\n    ==========\n\n    field : Vector/sympyfiable\n        The field to calculate wrt\n\n    frame : ReferenceFrame\n        The frame to do the calculations in\n\n    point1 : Point\n        The initial Point in given frame\n\n    position2 : Point\n        The second Point in the given frame\n\n    origin : Point\n        The Point to use as reference point for position vector\n        calculation\n\n    Examples\n    ========\n\n    >>> from sympy.physics.vector import ReferenceFrame, Point\n    >>> from sympy.physics.vector import scalar_potential_difference\n    >>> R = ReferenceFrame('R')\n    >>> O = Point('O')\n    >>> P = O.locatenew('P', R[0]*R.x + R[1]*R.y + R[2]*R.z)\n    >>> vectfield = 4*R[0]*R[1]*R.x + 2*R[0]**2*R.y\n    >>> scalar_potential_difference(vectfield, R, O, P, O)\n    2*R_x**2*R_y\n    >>> Q = O.locatenew('O', 3*R.x + R.y + 2*R.z)\n    >>> scalar_potential_difference(vectfield, R, P, Q, O)\n    -2*R_x**2*R_y + 18\n\n    "
    _check_frame(frame)
    if isinstance(field, Vector):
        scalar_fn = scalar_potential(field, frame)
    else:
        scalar_fn = field
    position1 = express(point1.pos_from(origin), frame, variables=True)
    position2 = express(point2.pos_from(origin), frame, variables=True)
    subs_dict1 = {}
    subs_dict2 = {}
    for (i, x) in enumerate(frame):
        subs_dict1[frame[i]] = x.dot(position1)
        subs_dict2[frame[i]] = x.dot(position2)
    return scalar_fn.subs(subs_dict2) - scalar_fn.subs(subs_dict1)