from __future__ import annotations
from enum import Enum
from typing import Any, Union
from kornia.core import Tensor, stack, zeros_like
from kornia.geometry.vector import Vector2, Vector3
from kornia.image import ImageSize
from kornia.sensors.camera.distortion_model import AffineTransform, BrownConradyTransform, KannalaBrandtK3Transform
from kornia.sensors.camera.projection_model import OrthographicProjection, Z1Projection

class CameraModelType(Enum):
    PINHOLE = 0
    BROWN_CONRADY = 1
    KANNALA_BRANDT_K3 = 2
    ORTHOGRAPHIC = 3

def get_model_from_type(model_type: CameraModelType, image_size: ImageSize, params: Tensor) -> CameraModelVariants:
    if False:
        for i in range(10):
            print('nop')
    if model_type == CameraModelType.PINHOLE:
        return PinholeModel(image_size, params)
    elif model_type == CameraModelType.BROWN_CONRADY:
        return BrownConradyModel(image_size, params)
    elif model_type == CameraModelType.KANNALA_BRANDT_K3:
        return KannalaBrandtK3(image_size, params)
    elif model_type == CameraModelType.ORTHOGRAPHIC:
        return Orthographic(image_size, params)
    else:
        raise ValueError('Invalid Camera Model Type')
CameraDistortionType = Union[AffineTransform, BrownConradyTransform, KannalaBrandtK3Transform]
CameraProjectionType = Union[Z1Projection, OrthographicProjection]

class CameraModelBase:
    """Base class to represent camera models based on distortion and projection types.

    Distortion is of 3 types:
        - Affine
        - Brown Conrady
        - Kannala Brandt K3
    Projection is of 2 types:
        - Z1
        - Orthographic

    Example:
        >>> params = torch.Tensor([328., 328., 320., 240.])
        >>> cam = CameraModelBase(BrownConradyTransform(), Z1Projection(), ImageSize(480, 640), params)
        >>> cam.params
        tensor([328., 328., 320., 240.])
    """

    def __init__(self, distortion: CameraDistortionType, projection: CameraProjectionType, image_size: ImageSize, params: Tensor) -> None:
        if False:
            return 10
        'Constructor method for CameraModelBase class.\n\n        Args:\n            distortion: Distortion type\n            projection: Projection type\n            image_size: Image size\n            params: Camera parameters of shape :math:`(B, 4)`\n                    for PINHOLE Camera, :math:`(B, 12)`\n                    for Brown Conrady, :math:`(B, 8)`\n                    for Kannala Brandt K3.\n        '
        self.distortion = distortion
        self.projection = projection
        self._image_size = image_size
        self._height = image_size.height
        self._width = image_size.width
        self._params = params

    @property
    def image_size(self) -> ImageSize:
        if False:
            for i in range(10):
                print('nop')
        'Returns the image size of the camera model.'
        return self._image_size

    @property
    def height(self) -> int | Tensor:
        if False:
            return 10
        'Returns the height of the image.'
        return self._height

    @property
    def width(self) -> int | Tensor:
        if False:
            print('Hello World!')
        'Returns the width of the image.'
        return self._width

    @property
    def params(self) -> Tensor:
        if False:
            return 10
        'Returns the camera parameters.'
        return self._params

    @property
    def fx(self) -> Tensor:
        if False:
            return 10
        'Returns the focal length in x direction.'
        return self._params[..., 0]

    @property
    def fy(self) -> Tensor:
        if False:
            return 10
        'Returns the focal length in y direction.'
        return self._params[..., 1]

    @property
    def cx(self) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        'Returns the principal point in x direction.'
        return self._params[..., 2]

    @property
    def cy(self) -> Tensor:
        if False:
            i = 10
            return i + 15
        'Returns the principal point in y direction.'
        return self._params[..., 3]

    def matrix(self) -> Tensor:
        if False:
            while True:
                i = 10
        'Returns the camera matrix.'
        raise NotImplementedError

    def K(self) -> Tensor:
        if False:
            return 10
        'Returns the camera matrix.'
        return self.matrix()

    def project(self, points: Vector3) -> Vector2:
        if False:
            for i in range(10):
                print('nop')
        'Projects 3D points to 2D camera plane.\n\n        Args:\n            points: Vector3 representing 3D points.\n\n        Returns:\n            Vector2 representing the projected 2D points.\n\n        Example:\n            >>> points = Vector3(torch.Tensor([1.0, 1.0, 1.0]))\n            >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.PINHOLE, torch.Tensor([328., 328., 320., 240.]))\n            >>> cam.project(points)\n            x: 648.0\n            y: 568.0\n        '
        return self.distortion.distort(self.params, self.projection.project(points))

    def unproject(self, points: Vector2, depth: Tensor) -> Vector3:
        if False:
            print('Hello World!')
        'Unprojects 2D points from camera plane to 3D.\n\n        Args:\n            points: Vector2 representing 2D points.\n            depth: Depth of the points.\n\n        Returns:\n            Vector3 representing the unprojected 3D points.\n\n        Example:\n            >>> points = Vector2(torch.Tensor([1.0, 1.0]))\n            >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.PINHOLE, torch.Tensor([328., 328., 320., 240.]))\n            >>> cam.unproject(points, torch.Tensor([1.0]))\n            x: tensor([-0.9726])\n            y: tensor([-0.7287])\n            z: tensor([1.])\n        '
        return self.projection.unproject(self.distortion.undistort(self.params, points), depth)

class PinholeModel(CameraModelBase):
    """Class to represent Pinhole Camera Model.

    The pinhole camera model describes the mathematical relationship between
    the coordinates of a point in three-dimensional space and its projection
    onto the image plane of an ideal pinhole camera,
    where the camera aperture is described as a point and no lenses are used to focus light.
    See more: https://en.wikipedia.org/wiki/Pinhole_camera_model

    Example:
        >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.PINHOLE, torch.Tensor([328., 328., 320., 240.]))
        >>> cam
        CameraModel(ImageSize(height=480, width=640), PinholeModel, tensor([328., 328., 320., 240.]))
    """

    def __init__(self, image_size: ImageSize, params: Tensor) -> None:
        if False:
            while True:
                i = 10
        'Constructor method for PinholeModel class.\n\n        Args:\n            image_size: Image size\n            params: Camera parameters of shape :math:`(B, 4)` of the form :math:`(fx, fy, cx, cy)`.\n        '
        if params.shape[-1] != 4 or len(params.shape) > 2:
            raise ValueError('params must be of shape (B, 4) for PINHOLE Camera')
        super().__init__(AffineTransform(), Z1Projection(), image_size, params)

    def matrix(self) -> Tensor:
        if False:
            return 10
        'Returns the camera matrix.\n\n        The matrix is of the form:\n\n        .. math::\n            \\begin{bmatrix} fx & 0 & cx \\\\\n            0 & fy & cy \\\\\n            0 & 0 & 1\\end{bmatrix}\n\n        Example:\n            >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.PINHOLE, torch.Tensor([1.0, 2.0, 3.0, 4.0]))\n            >>> cam.matrix()\n            tensor([[1., 0., 3.],\n                    [0., 2., 4.],\n                    [0., 0., 1.]])\n        '
        z = zeros_like(self.fx)
        row1 = stack((self.fx, z, self.cx), -1)
        row2 = stack((z, self.fy, self.cy), -1)
        row3 = stack((z, z, z), -1)
        K = stack((row1, row2, row3), -2)
        K[..., -1, -1] = 1.0
        return K

    def scale(self, scale_factor: Tensor) -> PinholeModel:
        if False:
            return 10
        'Scales the camera model by a scale factor.\n\n        Args:\n            scale_factor: Scale factor to scale the camera model.\n\n        Returns:\n            Scaled camera model.\n\n        Example:\n            >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.PINHOLE, torch.Tensor([328., 328., 320., 240.]))\n            >>> cam_scaled = cam.scale(2)\n            >>> cam_scaled.params\n            tensor([656., 656., 640., 480.])\n        '
        fx = self.fx * scale_factor
        fy = self.fy * scale_factor
        cx = self.cx * scale_factor
        cy = self.cy * scale_factor
        params = stack((fx, fy, cx, cy), -1)
        image_size = ImageSize(self.image_size.height * scale_factor, self.image_size.width * scale_factor)
        return PinholeModel(image_size, params)

class BrownConradyModel(CameraModelBase):
    """Brown Conrady Camera Model."""

    def __init__(self, image_size: ImageSize, params: Tensor) -> None:
        if False:
            while True:
                i = 10
        'Constructor method for BrownConradyModel class.\n\n        Args:\n            image_size: Image size\n            params: Camera parameters of shape :math:`(B, 12)` of the form :math:`(fx, fy, cx, cy, kb0, kb1, kb2, kb3,\n                    k1, k2, k3, k4)`.\n        '
        if params.shape[-1] != 12 or len(params.shape) > 2:
            raise ValueError('params must be of shape (B, 12) for BROWN_CONRADY Camera')
        super().__init__(BrownConradyTransform(), Z1Projection(), image_size, params)

class KannalaBrandtK3(CameraModelBase):
    """Kannala Brandt K3 Camera Model."""

    def __init__(self, image_size: ImageSize, params: Tensor) -> None:
        if False:
            return 10
        'Constructor method for KannalaBrandtK3 class.\n\n        Args:\n            image_size: Image size\n            params: Camera parameters of shape :math:`(B, 8)` of the form :math:`(fx, fy, cx, cy, kb0, kb1, kb2, kb3)`.\n        '
        if params.shape[-1] != 8 or len(params.shape) > 2:
            raise ValueError('params must be of shape B, 8 for KANNALA_BRANDT_K3 Camera')
        super().__init__(KannalaBrandtK3Transform(), Z1Projection(), image_size, params)

class Orthographic(CameraModelBase):
    """Orthographic Camera Model."""

    def __init__(self, image_size: ImageSize, params: Tensor) -> None:
        if False:
            return 10
        'Constructor method for Orthographic class.\n\n        Args:\n            image_size: Image size\n            params: Camera parameters of shape :math:`(B, 4)` of the form :math:`(fx, fy, cx, cy)`.\n        '
        super().__init__(AffineTransform(), OrthographicProjection(), image_size, params)
        if params.shape[-1] != 4 or len(params.shape) > 2:
            raise ValueError('params must be of shape B, 4 for ORTHOGRAPHIC Camera')
CameraModelVariants = Union[PinholeModel, BrownConradyModel, KannalaBrandtK3, Orthographic]

class CameraModel:
    """Class to represent camera models.

    Example:
        >>> # Pinhole Camera Model
        >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.PINHOLE, torch.Tensor([328., 328., 320., 240.]))
        >>> # Brown Conrady Camera Model
        >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.BROWN_CONRADY, torch.Tensor([1.0, 1.0, 1.0, 1.0,
        ... 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        >>> # Kannala Brandt K3 Camera Model
        >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.KANNALA_BRANDT_K3, torch.Tensor([1.0, 1.0, 1.0,
        ... 1.0, 1.0, 1.0, 1.0, 1.0]))
        >>> # Orthographic Camera Model
        >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.ORTHOGRAPHIC, torch.Tensor([328., 328., 320., 240.]))
        >>> cam.params
        tensor([328., 328., 320., 240.])
    """

    def __init__(self, image_size: ImageSize, model_type: CameraModelType, params: Tensor) -> None:
        if False:
            i = 10
            return i + 15
        'Constructor method for CameraModel class.\n\n        Args:\n            image_size: Image size\n            model_type: Camera model type\n            params: Camera parameters of shape :math:`(B, N)`.\n        '
        self._model = get_model_from_type(model_type, image_size, params)

    def __getattr__(self, name: str) -> Any:
        if False:
            print('Hello World!')
        return getattr(self._model, name)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'CameraModel({self.image_size}, {self._model.__class__.__name__}, {self.params})'