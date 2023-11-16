"""Representation of a pose in :math:`SE(2)`.

"""
import math
import numpy as np
from ..util import neg_pi_to_pi

class PoseSE2(np.ndarray):
    """A representation of a pose in :math:`SE(2)`.

    Parameters
    ----------
    position : np.ndarray, list
        The position in :math:`\\mathbb{R}^2`
    orientation : float
        The angle of the pose (in radians)

    """

    def __new__(cls, position, orientation):
        if False:
            print('Hello World!')
        obj = np.array([position[0], position[1], neg_pi_to_pi(orientation)], dtype=float).view(cls)
        return obj

    def copy(self):
        if False:
            print('Hello World!')
        'Return a copy of the pose.\n\n        Returns\n        -------\n        PoseSE2\n            A copy of the pose\n\n        '
        return PoseSE2(self[:2], self[2])

    def to_array(self):
        if False:
            i = 10
            return i + 15
        'Return the pose as a numpy array.\n\n        Returns\n        -------\n        np.ndarray\n            The pose as a numpy array\n\n        '
        return np.array(self)

    def to_compact(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the pose as a compact numpy array.\n\n        Returns\n        -------\n        np.ndarray\n            The pose as a compact numpy array\n\n        '
        return np.array(self)

    def to_matrix(self):
        if False:
            print('Hello World!')
        'Return the pose as an :math:`SE(2)` matrix.\n\n        Returns\n        -------\n        np.ndarray\n            The pose as an :math:`SE(2)` matrix\n\n        '
        return np.array([[np.cos(self[2]), -np.sin(self[2]), self[0]], [np.sin(self[2]), np.cos(self[2]), self[1]], [0.0, 0.0, 1.0]], dtype=float)

    @classmethod
    def from_matrix(cls, matrix):
        if False:
            print('Hello World!')
        'Return the pose as an :math:`SE(2)` matrix.\n\n        Parameters\n        ----------\n        matrix : np.ndarray\n            The :math:`SE(2)` matrix that will be converted to a `PoseSE2` instance\n\n        Returns\n        -------\n        PoseSE2\n            The matrix as a `PoseSE2` object\n\n        '
        return cls([matrix[0, 2], matrix[1, 2]], math.atan2(matrix[1, 0], matrix[0, 0]))

    @property
    def position(self):
        if False:
            while True:
                i = 10
        "Return the pose's position.\n\n        Returns\n        -------\n        np.ndarray\n            The position portion of the pose\n\n        "
        return np.array(self[:2])

    @property
    def orientation(self):
        if False:
            for i in range(10):
                print('nop')
        "Return the pose's orientation.\n\n        Returns\n        -------\n        float\n            The angle of the pose\n\n        "
        return self[2]

    @property
    def inverse(self):
        if False:
            i = 10
            return i + 15
        "Return the pose's inverse.\n\n        Returns\n        -------\n        PoseSE2\n            The pose's inverse\n\n        "
        return PoseSE2([-self[0] * np.cos(self[2]) - self[1] * np.sin(self[2]), self[0] * np.sin(self[2]) - self[1] * np.cos(self[2])], -self[2])

    def __add__(self, other):
        if False:
            print('Hello World!')
        'Add poses (i.e., pose composition): :math:`p_1 \\oplus p_2`.\n\n        Parameters\n        ----------\n        other : PoseSE2\n            The other pose\n\n        Returns\n        -------\n        PoseSE2\n            The result of pose composition\n\n        '
        return PoseSE2([self[0] + other[0] * np.cos(self[2]) - other[1] * np.sin(self[2]), self[1] + other[0] * np.sin(self[2]) + other[1] * np.cos(self[2])], neg_pi_to_pi(self[2] + other[2]))

    def __sub__(self, other):
        if False:
            i = 10
            return i + 15
        'Subtract poses (i.e., inverse pose composition): :math:`p_1 \\ominus p_2`.\n\n        Parameters\n        ----------\n        other : PoseSE2\n            The other pose\n\n        Returns\n        -------\n        PoseSE2\n            The result of inverse pose composition\n\n        '
        return PoseSE2([(self[0] - other[0]) * np.cos(other[2]) + (self[1] - other[1]) * np.sin(other[2]), (other[0] - self[0]) * np.sin(other[2]) + (self[1] - other[1]) * np.cos(other[2])], neg_pi_to_pi(self[2] - other[2]))