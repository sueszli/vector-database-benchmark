from __future__ import annotations
from typing import Any, cast
import numpy.typing as npt
from ..components import ViewCoordinatesLike
from ..datatypes.mat3x3 import Mat3x3Like
from ..datatypes.vec2d import Vec2D, Vec2DLike
from ..error_utils import _send_warning_or_raise, catch_and_log_exceptions

class PinholeExt:
    """Extension for [Pinhole][rerun.archetypes.Pinhole]."""

    def __init__(self: Any, *, image_from_camera: Mat3x3Like | None=None, resolution: Vec2DLike | None=None, camera_xyz: ViewCoordinatesLike | None=None, width: int | float | None=None, height: int | float | None=None, focal_length: float | npt.ArrayLike | None=None, principal_point: npt.ArrayLike | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new instance of the Pinhole archetype.\n\n        Parameters\n        ----------\n        image_from_camera:\n            Row-major intrinsics matrix for projecting from camera space to image space.\n            The first two axes are X=Right and Y=Down, respectively.\n            Projection is done along the positive third (Z=Forward) axis.\n            This can be specified _instead_ of `focal_length` and `principal_point`.\n        resolution:\n            Pixel resolution (usually integers) of child image space. Width and height.\n            `image_from_camera` projects onto the space spanned by `(0,0)` and `resolution - 1`.\n        camera_xyz:\n            Sets the view coordinates for the camera.\n\n            All common values are available as constants on the `components.ViewCoordinates` class.\n\n            The default is `ViewCoordinates.RDF`, i.e. X=Right, Y=Down, Z=Forward, and this is also the recommended setting.\n            This means that the camera frustum will point along the positive Z axis of the parent space,\n            and the cameras "up" direction will be along the negative Y axis of the parent space.\n\n            The camera frustum will point whichever axis is set to `F` (or the opposite of `B`).\n            When logging a depth image under this entity, this is the direction the point cloud will be projected.\n            With `RDF`, the default forward is +Z.\n\n            The frustum\'s "up" direction will be whichever axis is set to `U` (or the opposite of `D`).\n            This will match the negative Y direction of pixel space (all images are assumed to have xyz=RDF).\n            With `RDF`, the default is up is -Y.\n\n            The frustum\'s "right" direction will be whichever axis is set to `R` (or the opposite of `L`).\n            This will match the positive X direction of pixel space (all images are assumed to have xyz=RDF).\n            With `RDF`, the default right is +x.\n\n            Other common formats are `RUB` (X=Right, Y=Up, Z=Back) and `FLU` (X=Forward, Y=Left, Z=Up).\n\n            NOTE: setting this to something else than `RDF` (the default) will change the orientation of the camera frustum,\n            and make the pinhole matrix not match up with the coordinate system of the pinhole entity.\n\n            The pinhole matrix (the `image_from_camera` argument) always project along the third (Z) axis,\n            but will be re-oriented to project along the forward axis of the `camera_xyz` argument.\n        focal_length:\n            The focal length of the camera in pixels.\n            This is the diagonal of the projection matrix.\n            Set one value for symmetric cameras, or two values (X=Right, Y=Down) for anamorphic cameras.\n        principal_point:\n            The center of the camera in pixels.\n            The default is half the width and height.\n            This is the last column of the projection matrix.\n            Expects two values along the dimensions Right and Down\n        width:\n            Width of the image in pixels.\n        height:\n            Height of the image in pixels.\n        '
        with catch_and_log_exceptions(context=self.__class__.__name__):
            if resolution is None and width is not None and (height is not None):
                resolution = [width, height]
            elif resolution is not None and (width is not None or height is not None):
                _send_warning_or_raise("Can't set both resolution and width/height", 1)
            if image_from_camera is None:
                if resolution is not None:
                    res_vec = Vec2D(resolution)
                    width = cast(float, res_vec.xy[0])
                    height = cast(float, res_vec.xy[1])
                else:
                    width = None
                    height = None
                if focal_length is None:
                    if height is None or width is None:
                        raise ValueError('Either image_from_camera or focal_length must be set')
                    else:
                        _send_warning_or_raise('Either image_from_camera or focal_length must be set', 1)
                        focal_length = (width * height) ** 0.5
                if principal_point is None:
                    if height is not None and width is not None:
                        principal_point = [width / 2, height / 2]
                    else:
                        raise ValueError('Must provide one of principal_point, resolution, or width/height')
                if type(focal_length) in (int, float):
                    fl_x = focal_length
                    fl_y = focal_length
                else:
                    try:
                        fl_x = focal_length[0]
                        fl_y = focal_length[1]
                    except Exception:
                        raise ValueError('Expected focal_length to be one or two floats')
                try:
                    u_cen = principal_point[0]
                    v_cen = principal_point[1]
                except Exception:
                    raise ValueError('Expected principal_point to be one or two floats')
                image_from_camera = [[fl_x, 0, u_cen], [0, fl_y, v_cen], [0, 0, 1]]
            else:
                if focal_length is not None:
                    _send_warning_or_raise('Both image_from_camera and focal_length set', 1)
                if principal_point is not None:
                    _send_warning_or_raise('Both image_from_camera and principal_point set', 1)
            self.__attrs_init__(image_from_camera=image_from_camera, resolution=resolution, camera_xyz=camera_xyz)
            return
        self.__attrs_clear__()