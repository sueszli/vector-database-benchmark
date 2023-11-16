from __future__ import annotations
import moderngl
import numpy as np
from manimlib.constants import GREY
from manimlib.constants import OUT
from manimlib.mobject.mobject import Mobject
from manimlib.utils.bezier import integer_interpolate
from manimlib.utils.bezier import interpolate
from manimlib.utils.images import get_full_raster_image_path
from manimlib.utils.iterables import listify
from manimlib.utils.iterables import resize_with_interpolation
from manimlib.utils.space_ops import normalize_along_axis
from manimlib.utils.space_ops import cross
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, Iterable, Sequence, Tuple
    from manimlib.camera.camera import Camera
    from manimlib.typing import ManimColor, Vect3, Vect3Array, Self

class Surface(Mobject):
    render_primitive: int = moderngl.TRIANGLES
    shader_folder: str = 'surface'
    shader_dtype: np.dtype = np.dtype([('point', np.float32, (3,)), ('normal', np.float32, (3,)), ('rgba', np.float32, (4,))])

    def __init__(self, color: ManimColor=GREY, shading: Tuple[float, float, float]=(0.3, 0.2, 0.4), depth_test: bool=True, u_range: Tuple[float, float]=(0.0, 1.0), v_range: Tuple[float, float]=(0.0, 1.0), resolution: Tuple[int, int]=(101, 101), prefered_creation_axis: int=1, epsilon: float=1e-05, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.u_range = u_range
        self.v_range = v_range
        self.resolution = resolution
        self.prefered_creation_axis = prefered_creation_axis
        self.epsilon = epsilon
        super().__init__(**kwargs, color=color, shading=shading, depth_test=depth_test)
        self.compute_triangle_indices()

    def init_uniforms(self):
        if False:
            return 10
        super().init_uniforms()
        self.uniforms['clip_plane'] = np.zeros(4)

    def uv_func(self, u: float, v: float) -> tuple[float, float, float]:
        if False:
            while True:
                i = 10
        return (u, v, 0.0)

    @Mobject.affects_data
    def init_points(self):
        if False:
            for i in range(10):
                print('nop')
        dim = self.dim
        (nu, nv) = self.resolution
        u_range = np.linspace(*self.u_range, nu)
        v_range = np.linspace(*self.v_range, nv)
        uv_grid = np.array([[[u, v] for v in v_range] for u in u_range])
        uv_plus_du = uv_grid.copy()
        uv_plus_du[:, :, 0] += self.epsilon
        uv_plus_dv = uv_grid.copy()
        uv_plus_dv[:, :, 1] += self.epsilon
        (points, du_points, dv_points) = [np.apply_along_axis(lambda p: self.uv_func(*p), 2, grid).reshape((nu * nv, dim)) for grid in (uv_grid, uv_plus_du, uv_plus_dv)]
        self.set_points(points)
        self.data['normal'] = normalize_along_axis(cross((du_points - points) / self.epsilon, (dv_points - points) / self.epsilon), 1)

    def apply_points_function(self, *args, **kwargs) -> Self:
        if False:
            for i in range(10):
                print('nop')
        super().apply_points_function(*args, **kwargs)
        self.get_unit_normals()
        return self

    def compute_triangle_indices(self) -> np.ndarray:
        if False:
            return 10
        (nu, nv) = self.resolution
        if nu == 0 or nv == 0:
            self.triangle_indices = np.zeros(0, dtype=int)
            return self.triangle_indices
        index_grid = np.arange(nu * nv).reshape((nu, nv))
        indices = np.zeros(6 * (nu - 1) * (nv - 1), dtype=int)
        indices[0::6] = index_grid[:-1, :-1].flatten()
        indices[1::6] = index_grid[+1:, :-1].flatten()
        indices[2::6] = index_grid[:-1, +1:].flatten()
        indices[3::6] = index_grid[:-1, +1:].flatten()
        indices[4::6] = index_grid[+1:, :-1].flatten()
        indices[5::6] = index_grid[+1:, +1:].flatten()
        self.triangle_indices = indices
        return self.triangle_indices

    def get_triangle_indices(self) -> np.ndarray:
        if False:
            print('Hello World!')
        return self.triangle_indices

    def get_unit_normals(self) -> Vect3Array:
        if False:
            while True:
                i = 10
        (nu, nv) = self.resolution
        indices = np.arange(nu * nv)
        left = indices - 1
        right = indices + 1
        up = indices - nv
        down = indices + nv
        left[0] = indices[0]
        right[-1] = indices[-1]
        up[:nv] = indices[:nv]
        down[-nv:] = indices[-nv:]
        points = self.get_points()
        crosses = cross(points[right] - points[left], points[up] - points[down])
        self.data['normal'] = normalize_along_axis(crosses, 1)
        return self.data['normal']

    @Mobject.affects_data
    def pointwise_become_partial(self, smobject: 'Surface', a: float, b: float, axis: int | None=None) -> Self:
        if False:
            print('Hello World!')
        assert isinstance(smobject, Surface)
        if axis is None:
            axis = self.prefered_creation_axis
        if a <= 0 and b >= 1:
            self.match_points(smobject)
            return self
        (nu, nv) = smobject.resolution
        self.data['point'][:] = self.get_partial_points_array(self.data['point'], a, b, (nu, nv, 3), axis=axis)
        return self

    def get_partial_points_array(self, points: Vect3Array, a: float, b: float, resolution: Sequence[int], axis: int) -> Vect3Array:
        if False:
            i = 10
            return i + 15
        if len(points) == 0:
            return points
        (nu, nv) = resolution[:2]
        points = points.reshape(resolution)
        max_index = resolution[axis] - 1
        (lower_index, lower_residue) = integer_interpolate(0, max_index, a)
        (upper_index, upper_residue) = integer_interpolate(0, max_index, b)
        if axis == 0:
            points[:lower_index] = interpolate(points[lower_index], points[lower_index + 1], lower_residue)
            points[upper_index + 1:] = interpolate(points[upper_index], points[upper_index + 1], upper_residue)
        else:
            shape = (nu, 1, resolution[2])
            points[:, :lower_index] = interpolate(points[:, lower_index], points[:, lower_index + 1], lower_residue).reshape(shape)
            points[:, upper_index + 1:] = interpolate(points[:, upper_index], points[:, upper_index + 1], upper_residue).reshape(shape)
        return points.reshape((nu * nv, *resolution[2:]))

    @Mobject.affects_data
    def sort_faces_back_to_front(self, vect: Vect3=OUT) -> Self:
        if False:
            print('Hello World!')
        tri_is = self.triangle_indices
        points = self.get_points()
        dots = (points[tri_is[::3]] * vect).sum(1)
        indices = np.argsort(dots)
        for k in range(3):
            tri_is[k::3] = tri_is[k::3][indices]
        return self

    def always_sort_to_camera(self, camera: Camera) -> Self:
        if False:
            return 10

        def updater(surface: Surface):
            if False:
                i = 10
                return i + 15
            vect = camera.get_location() - surface.get_center()
            surface.sort_faces_back_to_front(vect)
        self.add_updater(updater)
        return self

    def set_clip_plane(self, vect: Vect3 | None=None, threshold: float | None=None) -> Self:
        if False:
            return 10
        if vect is not None:
            self.uniforms['clip_plane'][:3] = vect
        if threshold is not None:
            self.uniforms['clip_plane'][3] = threshold
        return self

    def deactivate_clip_plane(self) -> Self:
        if False:
            return 10
        self.uniforms['clip_plane'][:] = 0
        return self

    def get_shader_vert_indices(self) -> np.ndarray:
        if False:
            return 10
        return self.get_triangle_indices()

class ParametricSurface(Surface):

    def __init__(self, uv_func: Callable[[float, float], Iterable[float]], u_range: tuple[float, float]=(0, 1), v_range: tuple[float, float]=(0, 1), **kwargs):
        if False:
            while True:
                i = 10
        self.passed_uv_func = uv_func
        super().__init__(u_range=u_range, v_range=v_range, **kwargs)

    def uv_func(self, u, v):
        if False:
            print('Hello World!')
        return self.passed_uv_func(u, v)

class SGroup(Surface):

    def __init__(self, *parametric_surfaces: Surface, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(resolution=(0, 0), **kwargs)
        self.add(*parametric_surfaces)

    def init_points(self):
        if False:
            i = 10
            return i + 15
        pass

class TexturedSurface(Surface):
    shader_folder: str = 'textured_surface'
    shader_dtype: Sequence[Tuple[str, type, Tuple[int]]] = [('point', np.float32, (3,)), ('normal', np.float32, (3,)), ('im_coords', np.float32, (2,)), ('opacity', np.float32, (1,))]

    def __init__(self, uv_surface: Surface, image_file: str, dark_image_file: str | None=None, **kwargs):
        if False:
            while True:
                i = 10
        if not isinstance(uv_surface, Surface):
            raise Exception('uv_surface must be of type Surface')
        if dark_image_file is None:
            dark_image_file = image_file
            self.num_textures = 1
        else:
            self.num_textures = 2
        texture_paths = {'LightTexture': get_full_raster_image_path(image_file), 'DarkTexture': get_full_raster_image_path(dark_image_file)}
        self.uv_surface = uv_surface
        self.uv_func = uv_surface.uv_func
        self.u_range: Tuple[float, float] = uv_surface.u_range
        self.v_range: Tuple[float, float] = uv_surface.v_range
        self.resolution: Tuple[int, int] = uv_surface.resolution
        super().__init__(texture_paths=texture_paths, shading=tuple(uv_surface.shading), **kwargs)

    @Mobject.affects_data
    def init_points(self):
        if False:
            print('Hello World!')
        surf = self.uv_surface
        (nu, nv) = surf.resolution
        self.resize_points(surf.get_num_points())
        self.resolution = surf.resolution
        self.data['point'][:] = surf.data['point']
        self.data['normal'][:] = surf.data['normal']
        self.data['opacity'][:, 0] = surf.data['rgba'][:, 3]
        self.data['im_coords'] = np.array([[u, v] for u in np.linspace(0, 1, nu) for v in np.linspace(1, 0, nv)])

    def init_uniforms(self):
        if False:
            i = 10
            return i + 15
        super().init_uniforms()
        self.uniforms['num_textures'] = self.num_textures

    @Mobject.affects_data
    def set_opacity(self, opacity: float | Iterable[float]) -> Self:
        if False:
            print('Hello World!')
        op_arr = np.array(listify(opacity))
        self.data['opacity'][:, 0] = resize_with_interpolation(op_arr, len(self.data))
        return self

    def set_color(self, color: ManimColor | Iterable[ManimColor] | None, opacity: float | Iterable[float] | None=None, recurse: bool=True) -> Self:
        if False:
            print('Hello World!')
        if opacity is not None:
            self.set_opacity(opacity)
        return self

    def pointwise_become_partial(self, tsmobject: 'TexturedSurface', a: float, b: float, axis: int=1) -> Self:
        if False:
            print('Hello World!')
        super().pointwise_become_partial(tsmobject, a, b, axis)
        im_coords = self.data['im_coords']
        im_coords[:] = tsmobject.data['im_coords']
        if a <= 0 and b >= 1:
            return self
        (nu, nv) = tsmobject.resolution
        im_coords[:] = self.get_partial_points_array(im_coords, a, b, (nu, nv, 2), axis)
        return self