from kornia.core import Tensor
from kornia.geometry.vector import Vector2

class AffineTransform:

    def distort(self, params: Tensor, points: Vector2) -> Vector2:
        if False:
            print('Hello World!')
        'Distort one or more Vector2 points using the affine transform.\n\n        Args:\n            params: Tensor representing the affine transform parameters.\n            points: Vector2 representing the points to distort.\n\n        Returns:\n            Vector2 representing the distorted points.\n\n        Example:\n            >>> params = Tensor([1., 2., 3., 4.])\n            >>> points = Vector2.from_coords(1., 2.)\n            >>> AffineTransform().distort(params, points)\n            x: 4.0\n            y: 8.0\n        '
        (fx, fy, cx, cy) = (params[..., 0], params[..., 1], params[..., 2], params[..., 3])
        u = points.x * fx + cx
        v = points.y * fy + cy
        return Vector2.from_coords(u, v)

    def undistort(self, params: Tensor, points: Vector2) -> Vector2:
        if False:
            while True:
                i = 10
        'Undistort one or more Vector2 points using the affine transform.\n\n        Args:\n            params: Tensor representing the affine transform parameters.\n            points: Vector2 representing the points to undistort.\n\n        Returns:\n            Vector2 representing the undistorted points.\n\n        Example:\n            >>> params = Tensor([1., 2., 3., 4.])\n            >>> points = Vector2.from_coords(1., 2.)\n            >>> AffineTransform().undistort(params, points)\n            x: -2.0\n            y: -1.0\n        '
        (fx, fy, cx, cy) = (params[..., 0], params[..., 1], params[..., 2], params[..., 3])
        x = (points.x - cx) / fx
        y = (points.y - cy) / fy
        return Vector2.from_coords(x, y)

class BrownConradyTransform:

    def distort(self, params: Tensor, points: Vector2) -> Vector2:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def undistort(self, params: Tensor, points: Vector2) -> Vector2:
        if False:
            print('Hello World!')
        raise NotImplementedError

class KannalaBrandtK3Transform:

    def distort(self, params: Tensor, points: Vector2) -> Vector2:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def undistort(self, params: Tensor, points: Vector2) -> Vector2:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError