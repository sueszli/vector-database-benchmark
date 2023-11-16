from __future__ import annotations
import math
import numpy as np
from scipy.spatial.transform import Rotation
from pyrr import Matrix44
from manimlib.constants import DEGREES, RADIANS
from manimlib.constants import FRAME_SHAPE
from manimlib.constants import DOWN, LEFT, ORIGIN, OUT, RIGHT, UP
from manimlib.mobject.mobject import Mobject
from manimlib.utils.space_ops import normalize
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from manimlib.typing import Vect3

class CameraFrame(Mobject):

    def __init__(self, frame_shape: tuple[float, float]=FRAME_SHAPE, center_point: Vect3=ORIGIN, fovy: float=45 * DEGREES, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.uniforms['orientation'] = Rotation.identity().as_quat()
        self.uniforms['fovy'] = fovy
        self.default_orientation = Rotation.identity()
        self.view_matrix = np.identity(4)
        self.camera_location = OUT
        self.set_points(np.array([ORIGIN, LEFT, RIGHT, DOWN, UP]))
        self.set_width(frame_shape[0], stretch=True)
        self.set_height(frame_shape[1], stretch=True)
        self.move_to(center_point)

    def set_orientation(self, rotation: Rotation):
        if False:
            for i in range(10):
                print('nop')
        self.uniforms['orientation'][:] = rotation.as_quat()
        return self

    def get_orientation(self):
        if False:
            while True:
                i = 10
        return Rotation.from_quat(self.uniforms['orientation'])

    def make_orientation_default(self):
        if False:
            print('Hello World!')
        self.default_orientation = self.get_orientation()
        return self

    def to_default_state(self):
        if False:
            print('Hello World!')
        self.set_shape(*FRAME_SHAPE)
        self.center()
        self.set_orientation(self.default_orientation)
        return self

    def get_euler_angles(self) -> np.ndarray:
        if False:
            while True:
                i = 10
        orientation = self.get_orientation()
        if all(orientation.as_quat() == [0, 0, 0, 1]):
            return np.zeros(3)
        return orientation.as_euler('zxz')[::-1]

    def get_theta(self):
        if False:
            while True:
                i = 10
        return self.get_euler_angles()[0]

    def get_phi(self):
        if False:
            i = 10
            return i + 15
        return self.get_euler_angles()[1]

    def get_gamma(self):
        if False:
            return 10
        return self.get_euler_angles()[2]

    def get_scale(self):
        if False:
            print('Hello World!')
        return self.get_height() / FRAME_SHAPE[1]

    def get_inverse_camera_rotation_matrix(self):
        if False:
            while True:
                i = 10
        return self.get_orientation().as_matrix().T

    def get_view_matrix(self, refresh=False):
        if False:
            print('Hello World!')
        "\n        Returns a 4x4 for the affine transformation mapping a point\n        into the camera's internal coordinate system\n        "
        if self._data_has_changed:
            shift = np.identity(4)
            rotation = np.identity(4)
            scale_mat = np.identity(4)
            shift[:3, 3] = -self.get_center()
            rotation[:3, :3] = self.get_inverse_camera_rotation_matrix()
            scale = self.get_scale()
            if scale > 0:
                scale_mat[:3, :3] /= self.get_scale()
            self.view_matrix = np.dot(scale_mat, np.dot(rotation, shift))
        return self.view_matrix

    def get_inv_view_matrix(self):
        if False:
            i = 10
            return i + 15
        return np.linalg.inv(self.get_view_matrix())

    @Mobject.affects_data
    def interpolate(self, *args, **kwargs):
        if False:
            return 10
        super().interpolate(*args, **kwargs)

    @Mobject.affects_data
    def rotate(self, angle: float, axis: np.ndarray=OUT, **kwargs):
        if False:
            while True:
                i = 10
        rot = Rotation.from_rotvec(angle * normalize(axis))
        self.set_orientation(rot * self.get_orientation())
        return self

    def set_euler_angles(self, theta: float | None=None, phi: float | None=None, gamma: float | None=None, units: float=RADIANS):
        if False:
            while True:
                i = 10
        eulers = self.get_euler_angles()
        for (i, var) in enumerate([theta, phi, gamma]):
            if var is not None:
                eulers[i] = var * units
        if all(eulers == 0):
            rot = Rotation.identity()
        else:
            rot = Rotation.from_euler('zxz', eulers[::-1])
        self.set_orientation(rot)
        return self

    def reorient(self, theta_degrees: float | None=None, phi_degrees: float | None=None, gamma_degrees: float | None=None):
        if False:
            print('Hello World!')
        '\n        Shortcut for set_euler_angles, defaulting to taking\n        in angles in degrees\n        '
        self.set_euler_angles(theta_degrees, phi_degrees, gamma_degrees, units=DEGREES)
        return self

    def set_theta(self, theta: float):
        if False:
            for i in range(10):
                print('nop')
        return self.set_euler_angles(theta=theta)

    def set_phi(self, phi: float):
        if False:
            while True:
                i = 10
        return self.set_euler_angles(phi=phi)

    def set_gamma(self, gamma: float):
        if False:
            while True:
                i = 10
        return self.set_euler_angles(gamma=gamma)

    def increment_theta(self, dtheta: float):
        if False:
            for i in range(10):
                print('nop')
        self.rotate(dtheta, OUT)
        return self

    def increment_phi(self, dphi: float):
        if False:
            while True:
                i = 10
        self.rotate(dphi, self.get_inverse_camera_rotation_matrix()[0])
        return self

    def increment_gamma(self, dgamma: float):
        if False:
            i = 10
            return i + 15
        self.rotate(dgamma, self.get_inverse_camera_rotation_matrix()[2])
        return self

    @Mobject.affects_data
    def set_focal_distance(self, focal_distance: float):
        if False:
            i = 10
            return i + 15
        self.uniforms['fovy'] = 2 * math.atan(0.5 * self.get_height() / focal_distance)
        return self

    @Mobject.affects_data
    def set_field_of_view(self, field_of_view: float):
        if False:
            i = 10
            return i + 15
        self.uniforms['fovy'] = field_of_view
        return self

    def get_shape(self):
        if False:
            print('Hello World!')
        return (self.get_width(), self.get_height())

    def get_aspect_ratio(self):
        if False:
            while True:
                i = 10
        (width, height) = self.get_shape()
        return width / height

    def get_center(self) -> np.ndarray:
        if False:
            while True:
                i = 10
        return self.get_points()[0]

    def get_width(self) -> float:
        if False:
            while True:
                i = 10
        points = self.get_points()
        return points[2, 0] - points[1, 0]

    def get_height(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        points = self.get_points()
        return points[4, 1] - points[3, 1]

    def get_focal_distance(self) -> float:
        if False:
            while True:
                i = 10
        return 0.5 * self.get_height() / math.tan(0.5 * self.uniforms['fovy'])

    def get_field_of_view(self) -> float:
        if False:
            i = 10
            return i + 15
        return self.uniforms['fovy']

    def get_implied_camera_location(self) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        if self._data_has_changed:
            to_camera = self.get_inverse_camera_rotation_matrix()[2]
            dist = self.get_focal_distance()
            self.camera_location = self.get_center() + dist * to_camera
        return self.camera_location

    def to_fixed_frame_point(self, point: Vect3, relative: bool=False):
        if False:
            while True:
                i = 10
        view = self.get_view_matrix()
        point4d = [*point, 0 if relative else 1]
        return np.dot(point4d, view.T)[:3]

    def from_fixed_frame_point(self, point: Vect3, relative: bool=False):
        if False:
            i = 10
            return i + 15
        inv_view = self.get_inv_view_matrix()
        point4d = [*point, 0 if relative else 1]
        return np.dot(point4d, inv_view.T)[:3]