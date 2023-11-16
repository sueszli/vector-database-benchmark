import numpy as np
from scipy.spatial.transform import Rotation as Rot

def rot_mat_2d(angle):
    if False:
        print('Hello World!')
    '\n    Create 2D rotation matrix from an angle\n\n    Parameters\n    ----------\n    angle :\n\n    Returns\n    -------\n    A 2D rotation matrix\n\n    Examples\n    --------\n    >>> angle_mod(-4.0)\n\n\n    '
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]

def angle_mod(x, zero_2_2pi=False, degree=False):
    if False:
        while True:
            i = 10
    '\n    Angle modulo operation\n    Default angle modulo range is [-pi, pi)\n\n    Parameters\n    ----------\n    x : float or array_like\n        A angle or an array of angles. This array is flattened for\n        the calculation. When an angle is provided, a float angle is returned.\n    zero_2_2pi : bool, optional\n        Change angle modulo range to [0, 2pi)\n        Default is False.\n    degree : bool, optional\n        If True, then the given angles are assumed to be in degrees.\n        Default is False.\n\n    Returns\n    -------\n    ret : float or ndarray\n        an angle or an array of modulated angle.\n\n    Examples\n    --------\n    >>> angle_mod(-4.0)\n    2.28318531\n\n    >>> angle_mod([-4.0])\n    np.array(2.28318531)\n\n    >>> angle_mod([-150.0, 190.0, 350], degree=True)\n    array([-150., -170.,  -10.])\n\n    >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)\n    array([300.])\n\n    '
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False
    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)
    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi
    if degree:
        mod_angle = np.rad2deg(mod_angle)
    if is_float:
        return mod_angle.item()
    else:
        return mod_angle