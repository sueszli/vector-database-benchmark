from __future__ import annotations
from kornia.core import Tensor, diag
from kornia.geometry.vector import Vector2, Vector3

class Z1Projection:

    def project(self, points: Vector3) -> Vector2:
        if False:
            i = 10
            return i + 15
        'Project one or more Vector3 from the camera frame into the canonical z=1 plane through perspective\n        division.\n\n        Args:\n            points: Vector3 representing the points to project.\n\n        Returns:\n            Vector2 representing the projected points.\n\n        Example:\n            >>> points = Vector3.from_coords(1., 2., 3.)\n            >>> Z1Projection().project(points)\n            x: 0.3333333432674408\n            y: 0.6666666865348816\n        '
        xy = points.data[..., :2]
        z = points.z
        uv = (xy.T @ diag(z).inverse()).T if len(z.shape) else xy.T * 1 / z
        return Vector2(uv)

    def unproject(self, points: Vector2, depth: Tensor | float) -> Vector3:
        if False:
            i = 10
            return i + 15
        'Unproject one or more Vector2 from the canonical z=1 plane into the camera frame.\n\n        Args:\n            points: Vector2 representing the points to unproject.\n            depth: Tensor representing the depth of the points to unproject.\n\n        Returns:\n            Vector3 representing the unprojected points.\n\n        Example:\n            >>> points = Vector2.from_coords(1., 2.)\n            >>> Z1Projection().unproject(points, 3)\n            x: tensor([3.])\n            y: tensor([6.])\n            z: tensor([3.])\n        '
        if isinstance(depth, (float, int)):
            depth = Tensor([depth])
        return Vector3.from_coords(points.x * depth, points.y * depth, depth)

class OrthographicProjection:

    def project(self, points: Vector3) -> Vector2:
        if False:
            return 10
        raise NotImplementedError

    def unproject(self, points: Vector2, depth: Tensor) -> Vector3:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError