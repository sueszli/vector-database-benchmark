from __future__ import annotations
import copy
from functools import wraps
import itertools as it
import os
import pickle
import random
import sys
import moderngl
import numbers
import numpy as np
from manimlib.constants import DEFAULT_MOBJECT_TO_EDGE_BUFFER
from manimlib.constants import DEFAULT_MOBJECT_TO_MOBJECT_BUFFER
from manimlib.constants import DOWN, IN, LEFT, ORIGIN, OUT, RIGHT, UP
from manimlib.constants import FRAME_X_RADIUS, FRAME_Y_RADIUS
from manimlib.constants import MED_SMALL_BUFF
from manimlib.constants import TAU
from manimlib.constants import WHITE
from manimlib.event_handler import EVENT_DISPATCHER
from manimlib.event_handler.event_listner import EventListener
from manimlib.event_handler.event_type import EventType
from manimlib.logger import log
from manimlib.shader_wrapper import ShaderWrapper
from manimlib.utils.color import color_gradient
from manimlib.utils.color import color_to_rgb
from manimlib.utils.color import get_colormap_list
from manimlib.utils.color import rgb_to_hex
from manimlib.utils.iterables import arrays_match
from manimlib.utils.iterables import array_is_constant
from manimlib.utils.iterables import batch_by_property
from manimlib.utils.iterables import list_update
from manimlib.utils.iterables import listify
from manimlib.utils.iterables import resize_array
from manimlib.utils.iterables import resize_preserving_order
from manimlib.utils.iterables import resize_with_interpolation
from manimlib.utils.bezier import integer_interpolate
from manimlib.utils.bezier import interpolate
from manimlib.utils.paths import straight_path
from manimlib.utils.simple_functions import get_parameters
from manimlib.utils.shaders import get_colormap_code
from manimlib.utils.space_ops import angle_of_vector
from manimlib.utils.space_ops import get_norm
from manimlib.utils.space_ops import rotation_matrix_transpose
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, Iterable, Iterator, Union, Tuple, Optional
    import numpy.typing as npt
    from manimlib.typing import ManimColor, Vect3, Vect4, Vect3Array, UniformDict, Self
    from moderngl.context import Context
    TimeBasedUpdater = Callable[['Mobject', float], 'Mobject' | None]
    NonTimeUpdater = Callable[['Mobject'], 'Mobject' | None]
    Updater = Union[TimeBasedUpdater, NonTimeUpdater]

class Mobject(object):
    """
    Mathematical Object
    """
    dim: int = 3
    shader_folder: str = ''
    render_primitive: int = moderngl.TRIANGLE_STRIP
    shader_dtype: np.dtype = np.dtype([('point', np.float32, (3,)), ('rgba', np.float32, (4,))])
    aligned_data_keys = ['point']
    pointlike_data_keys = ['point']

    def __init__(self, color: ManimColor=WHITE, opacity: float=1.0, shading: Tuple[float, float, float]=(0.0, 0.0, 0.0), texture_paths: dict[str, str] | None=None, is_fixed_in_frame: bool=False, depth_test: bool=False):
        if False:
            for i in range(10):
                print('nop')
        self.color = color
        self.opacity = opacity
        self.shading = shading
        self.texture_paths = texture_paths
        self._is_fixed_in_frame = is_fixed_in_frame
        self.depth_test = depth_test
        self.submobjects: list[Mobject] = []
        self.parents: list[Mobject] = []
        self.family: list[Mobject] = [self]
        self.locked_data_keys: set[str] = set()
        self.const_data_keys: set[str] = set()
        self.locked_uniform_keys: set[str] = set()
        self.needs_new_bounding_box: bool = True
        self._is_animating: bool = False
        self.saved_state = None
        self.target = None
        self.bounding_box: Vect3Array = np.zeros((3, 3))
        self._shaders_initialized: bool = False
        self._data_has_changed: bool = True
        self.shader_code_replacements: dict[str, str] = dict()
        self.init_data()
        self._data_defaults = np.ones(1, dtype=self.data.dtype)
        self.init_uniforms()
        self.init_updaters()
        self.init_event_listners()
        self.init_points()
        self.init_colors()
        if self.depth_test:
            self.apply_depth_test()

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.__class__.__name__

    def __add__(self, other: Mobject) -> Mobject:
        if False:
            print('Hello World!')
        assert isinstance(other, Mobject)
        return self.get_group_class()(self, other)

    def __mul__(self, other: int) -> Mobject:
        if False:
            while True:
                i = 10
        assert isinstance(other, int)
        return self.replicate(other)

    def init_data(self, length: int=0):
        if False:
            print('Hello World!')
        self.data = np.zeros(length, dtype=self.shader_dtype)

    def init_uniforms(self):
        if False:
            for i in range(10):
                print('nop')
        self.uniforms: UniformDict = {'is_fixed_in_frame': float(self._is_fixed_in_frame), 'shading': np.array(self.shading, dtype=float)}

    def init_colors(self):
        if False:
            while True:
                i = 10
        self.set_color(self.color, self.opacity)

    def init_points(self):
        if False:
            return 10
        pass

    def set_uniforms(self, uniforms: dict) -> Self:
        if False:
            i = 10
            return i + 15
        for (key, value) in uniforms.items():
            if isinstance(value, np.ndarray):
                value = value.copy()
            self.uniforms[key] = value
        return self

    @property
    def animate(self) -> _AnimationBuilder:
        if False:
            for i in range(10):
                print('nop')
        return _AnimationBuilder(self)

    def note_changed_data(self, recurse_up: bool=True) -> Self:
        if False:
            for i in range(10):
                print('nop')
        self._data_has_changed = True
        if recurse_up:
            for mob in self.parents:
                mob.note_changed_data()
        return self

    def affects_data(func: Callable):
        if False:
            for i in range(10):
                print('nop')

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if False:
                print('Hello World!')
            func(self, *args, **kwargs)
            self.note_changed_data()
        return wrapper

    def affects_family_data(func: Callable):
        if False:
            i = 10
            return i + 15

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if False:
                return 10
            func(self, *args, **kwargs)
            for mob in self.family_members_with_points():
                mob.note_changed_data()
            return self
        return wrapper

    @affects_data
    def set_data(self, data: np.ndarray) -> Self:
        if False:
            i = 10
            return i + 15
        assert data.dtype == self.data.dtype
        self.resize_points(len(data))
        self.data[:] = data
        return self

    @affects_data
    def resize_points(self, new_length: int, resize_func: Callable[[np.ndarray, int], np.ndarray]=resize_array) -> Self:
        if False:
            return 10
        if new_length == 0:
            if len(self.data) > 0:
                self._data_defaults[:1] = self.data[:1]
        elif self.get_num_points() == 0:
            self.data = self._data_defaults.copy()
        self.data = resize_func(self.data, new_length)
        self.refresh_bounding_box()
        return self

    @affects_data
    def set_points(self, points: Vect3Array | list[Vect3]) -> Self:
        if False:
            while True:
                i = 10
        self.resize_points(len(points), resize_func=resize_preserving_order)
        self.data['point'][:] = points
        return self

    @affects_data
    def append_points(self, new_points: Vect3Array) -> Self:
        if False:
            print('Hello World!')
        n = self.get_num_points()
        self.resize_points(n + len(new_points))
        self.data[n:] = self.data[n - 1]
        self.data['point'][n:] = new_points
        self.refresh_bounding_box()
        return self

    @affects_family_data
    def reverse_points(self) -> Self:
        if False:
            i = 10
            return i + 15
        for mob in self.get_family():
            mob.data = mob.data[::-1]
        return self

    @affects_family_data
    def apply_points_function(self, func: Callable[[np.ndarray], np.ndarray], about_point: Vect3 | None=None, about_edge: Vect3=ORIGIN, works_on_bounding_box: bool=False) -> Self:
        if False:
            for i in range(10):
                print('nop')
        if about_point is None and about_edge is not None:
            about_point = self.get_bounding_box_point(about_edge)
        for mob in self.get_family():
            arrs = []
            if mob.has_points():
                for key in mob.pointlike_data_keys:
                    arrs.append(mob.data[key])
            if works_on_bounding_box:
                arrs.append(mob.get_bounding_box())
            for arr in arrs:
                if about_point is None:
                    arr[:] = func(arr)
                else:
                    arr[:] = func(arr - about_point) + about_point
        if not works_on_bounding_box:
            self.refresh_bounding_box(recurse_down=True)
        else:
            for parent in self.parents:
                parent.refresh_bounding_box()
        return self

    def match_points(self, mobject: Mobject) -> Self:
        if False:
            print('Hello World!')
        self.set_points(mobject.get_points())
        return self

    def get_points(self) -> Vect3Array:
        if False:
            return 10
        return self.data['point']

    def clear_points(self) -> Self:
        if False:
            print('Hello World!')
        self.resize_points(0)
        return self

    def get_num_points(self) -> int:
        if False:
            while True:
                i = 10
        return len(self.get_points())

    def get_all_points(self) -> Vect3Array:
        if False:
            i = 10
            return i + 15
        if self.submobjects:
            return np.vstack([sm.get_points() for sm in self.get_family()])
        else:
            return self.get_points()

    def has_points(self) -> bool:
        if False:
            while True:
                i = 10
        return len(self.get_points()) > 0

    def get_bounding_box(self) -> Vect3Array:
        if False:
            return 10
        if self.needs_new_bounding_box:
            self.bounding_box[:] = self.compute_bounding_box()
            self.needs_new_bounding_box = False
        return self.bounding_box

    def compute_bounding_box(self) -> Vect3Array:
        if False:
            print('Hello World!')
        all_points = np.vstack([self.get_points(), *(mob.get_bounding_box() for mob in self.get_family()[1:] if mob.has_points())])
        if len(all_points) == 0:
            return np.zeros((3, self.dim))
        else:
            mins = all_points.min(0)
            maxs = all_points.max(0)
            mids = (mins + maxs) / 2
            return np.array([mins, mids, maxs])

    def refresh_bounding_box(self, recurse_down: bool=False, recurse_up: bool=True) -> Self:
        if False:
            while True:
                i = 10
        for mob in self.get_family(recurse_down):
            mob.needs_new_bounding_box = True
        if recurse_up:
            for parent in self.parents:
                parent.refresh_bounding_box()
        return self

    def are_points_touching(self, points: Vect3Array, buff: float=0) -> np.ndarray:
        if False:
            return 10
        bb = self.get_bounding_box()
        mins = bb[0] - buff
        maxs = bb[2] + buff
        return ((points >= mins) * (points <= maxs)).all(1)

    def is_point_touching(self, point: Vect3, buff: float=0) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.are_points_touching(np.array(point, ndmin=2), buff)[0]

    def is_touching(self, mobject: Mobject, buff: float=0.01) -> bool:
        if False:
            return 10
        bb1 = self.get_bounding_box()
        bb2 = mobject.get_bounding_box()
        return not any(((bb2[2] < bb1[0] - buff).any(), (bb2[0] > bb1[2] + buff).any()))

    def __getitem__(self, value: int | slice) -> Self:
        if False:
            while True:
                i = 10
        if isinstance(value, slice):
            GroupClass = self.get_group_class()
            return GroupClass(*self.split().__getitem__(value))
        return self.split().__getitem__(value)

    def __iter__(self) -> Iterator[Self]:
        if False:
            i = 10
            return i + 15
        return iter(self.split())

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        return len(self.split())

    def split(self) -> list[Self]:
        if False:
            print('Hello World!')
        return self.submobjects

    @affects_data
    def assemble_family(self) -> Self:
        if False:
            while True:
                i = 10
        sub_families = (sm.get_family() for sm in self.submobjects)
        self.family = [self, *it.chain(*sub_families)]
        self.refresh_has_updater_status()
        self.refresh_bounding_box()
        for parent in self.parents:
            parent.assemble_family()
        return self

    def get_family(self, recurse: bool=True) -> list[Self]:
        if False:
            i = 10
            return i + 15
        if recurse:
            return self.family
        else:
            return [self]

    def family_members_with_points(self) -> list[Self]:
        if False:
            for i in range(10):
                print('nop')
        return [m for m in self.family if len(m.data) > 0]

    def get_ancestors(self, extended: bool=False) -> list[Mobject]:
        if False:
            return 10
        '\n        Returns parents, grandparents, etc.\n        Order of result should be from higher members of the hierarchy down.\n\n        If extended is set to true, it includes the ancestors of all family members,\n        e.g. any other parents of a submobject\n        '
        ancestors = []
        to_process = list(self.get_family(recurse=extended))
        excluded = set(to_process)
        while to_process:
            for p in to_process.pop().parents:
                if p not in excluded:
                    ancestors.append(p)
                    to_process.append(p)
        ancestors.reverse()
        return list(dict.fromkeys(ancestors))

    def add(self, *mobjects: Mobject) -> Self:
        if False:
            i = 10
            return i + 15
        if self in mobjects:
            raise Exception('Mobject cannot contain self')
        for mobject in mobjects:
            if mobject not in self.submobjects:
                self.submobjects.append(mobject)
            if self not in mobject.parents:
                mobject.parents.append(self)
        self.assemble_family()
        return self

    def remove(self, *to_remove: Mobject, reassemble: bool=True, recurse: bool=True) -> Self:
        if False:
            i = 10
            return i + 15
        for parent in self.get_family(recurse):
            for child in to_remove:
                if child in parent.submobjects:
                    parent.submobjects.remove(child)
                if parent in child.parents:
                    child.parents.remove(parent)
            if reassemble:
                parent.assemble_family()
        return self

    def clear(self) -> Self:
        if False:
            print('Hello World!')
        self.remove(*self.submobjects, recurse=False)
        return self

    def add_to_back(self, *mobjects: Mobject) -> Self:
        if False:
            print('Hello World!')
        self.set_submobjects(list_update(mobjects, self.submobjects))
        return self

    def replace_submobject(self, index: int, new_submob: Mobject) -> Self:
        if False:
            print('Hello World!')
        old_submob = self.submobjects[index]
        if self in old_submob.parents:
            old_submob.parents.remove(self)
        self.submobjects[index] = new_submob
        new_submob.parents.append(self)
        self.assemble_family()
        return self

    def insert_submobject(self, index: int, new_submob: Mobject) -> Self:
        if False:
            i = 10
            return i + 15
        self.submobjects.insert(index, new_submob)
        self.assemble_family()
        return self

    def set_submobjects(self, submobject_list: list[Mobject]) -> Self:
        if False:
            for i in range(10):
                print('nop')
        if self.submobjects == submobject_list:
            return self
        self.clear()
        self.add(*submobject_list)
        return self

    def digest_mobject_attrs(self) -> Self:
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensures all attributes which are mobjects are included\n        in the submobjects list.\n        '
        mobject_attrs = [x for x in list(self.__dict__.values()) if isinstance(x, Mobject)]
        self.set_submobjects(list_update(self.submobjects, mobject_attrs))
        return self

    def arrange(self, direction: Vect3=RIGHT, center: bool=True, **kwargs) -> Self:
        if False:
            print('Hello World!')
        for (m1, m2) in zip(self.submobjects, self.submobjects[1:]):
            m2.next_to(m1, direction, **kwargs)
        if center:
            self.center()
        return self

    def arrange_in_grid(self, n_rows: int | None=None, n_cols: int | None=None, buff: float | None=None, h_buff: float | None=None, v_buff: float | None=None, buff_ratio: float | None=None, h_buff_ratio: float=0.5, v_buff_ratio: float=0.5, aligned_edge: Vect3=ORIGIN, fill_rows_first: bool=True) -> Self:
        if False:
            for i in range(10):
                print('nop')
        submobs = self.submobjects
        if n_rows is None and n_cols is None:
            n_rows = int(np.sqrt(len(submobs)))
        if n_rows is None:
            n_rows = len(submobs) // n_cols
        if n_cols is None:
            n_cols = len(submobs) // n_rows
        if buff is not None:
            h_buff = buff
            v_buff = buff
        else:
            if buff_ratio is not None:
                v_buff_ratio = buff_ratio
                h_buff_ratio = buff_ratio
            if h_buff is None:
                h_buff = h_buff_ratio * self[0].get_width()
            if v_buff is None:
                v_buff = v_buff_ratio * self[0].get_height()
        x_unit = h_buff + max([sm.get_width() for sm in submobs])
        y_unit = v_buff + max([sm.get_height() for sm in submobs])
        for (index, sm) in enumerate(submobs):
            if fill_rows_first:
                (x, y) = (index % n_cols, index // n_cols)
            else:
                (x, y) = (index // n_rows, index % n_rows)
            sm.move_to(ORIGIN, aligned_edge)
            sm.shift(x * x_unit * RIGHT + y * y_unit * DOWN)
        self.center()
        return self

    def arrange_to_fit_dim(self, length: float, dim: int, about_edge=ORIGIN) -> Self:
        if False:
            print('Hello World!')
        ref_point = self.get_bounding_box_point(about_edge)
        n_submobs = len(self.submobjects)
        if n_submobs <= 1:
            return
        total_length = sum((sm.length_over_dim(dim) for sm in self.submobjects))
        buff = (length - total_length) / (n_submobs - 1)
        vect = np.zeros(self.dim)
        vect[dim] = 1
        x = 0
        for submob in self.submobjects:
            submob.set_coord(x, dim, -vect)
            x += submob.length_over_dim(dim) + buff
        self.move_to(ref_point, about_edge)
        return self

    def arrange_to_fit_width(self, width: float, about_edge=ORIGIN) -> Self:
        if False:
            return 10
        return self.arrange_to_fit_dim(width, 0, about_edge)

    def arrange_to_fit_height(self, height: float, about_edge=ORIGIN) -> Self:
        if False:
            while True:
                i = 10
        return self.arrange_to_fit_dim(height, 1, about_edge)

    def arrange_to_fit_depth(self, depth: float, about_edge=ORIGIN) -> Self:
        if False:
            i = 10
            return i + 15
        return self.arrange_to_fit_dim(depth, 2, about_edge)

    def sort(self, point_to_num_func: Callable[[np.ndarray], float]=lambda p: p[0], submob_func: Callable[[Mobject]] | None=None) -> Self:
        if False:
            while True:
                i = 10
        if submob_func is not None:
            self.submobjects.sort(key=submob_func)
        else:
            self.submobjects.sort(key=lambda m: point_to_num_func(m.get_center()))
        self.assemble_family()
        return self

    def shuffle(self, recurse: bool=False) -> Self:
        if False:
            print('Hello World!')
        if recurse:
            for submob in self.submobjects:
                submob.shuffle(recurse=True)
        random.shuffle(self.submobjects)
        self.assemble_family()
        return self

    def reverse_submobjects(self) -> Self:
        if False:
            return 10
        self.submobjects.reverse()
        self.assemble_family()
        return self

    def stash_mobject_pointers(func: Callable):
        if False:
            for i in range(10):
                print('nop')

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            uncopied_attrs = ['parents', 'target', 'saved_state']
            stash = dict()
            for attr in uncopied_attrs:
                if hasattr(self, attr):
                    value = getattr(self, attr)
                    stash[attr] = value
                    null_value = [] if isinstance(value, list) else None
                    setattr(self, attr, null_value)
            result = func(self, *args, **kwargs)
            self.__dict__.update(stash)
            return result
        return wrapper

    @stash_mobject_pointers
    def serialize(self) -> bytes:
        if False:
            while True:
                i = 10
        return pickle.dumps(self)

    def deserialize(self, data: bytes) -> Self:
        if False:
            return 10
        self.become(pickle.loads(data))
        return self

    def deepcopy(self) -> Self:
        if False:
            print('Hello World!')
        result = copy.deepcopy(self)
        result._shaders_initialized = False
        result._data_has_changed = True
        return result

    def copy(self, deep: bool=False) -> Self:
        if False:
            return 10
        if deep:
            return self.deepcopy()
        result = copy.copy(self)
        result.parents = []
        result.target = None
        result.saved_state = None
        result.uniforms = {key: value.copy() if isinstance(value, np.ndarray) else value for (key, value) in self.uniforms.items()}
        result.submobjects = [sm.copy() for sm in self.submobjects]
        for sm in result.submobjects:
            sm.parents = [result]
        result.family = [result, *it.chain(*(sm.get_family() for sm in result.submobjects))]
        result.non_time_updaters = list(self.non_time_updaters)
        result.time_based_updaters = list(self.time_based_updaters)
        result._data_has_changed = True
        result._shaders_initialized = False
        family = self.get_family()
        for (attr, value) in self.__dict__.items():
            if isinstance(value, Mobject) and value is not self:
                if value in family:
                    setattr(result, attr, result.family[self.family.index(value)])
            elif isinstance(value, np.ndarray):
                setattr(result, attr, value.copy())
        return result

    def generate_target(self, use_deepcopy: bool=False) -> Self:
        if False:
            return 10
        self.target = self.copy(deep=use_deepcopy)
        self.target.saved_state = self.saved_state
        return self.target

    def save_state(self, use_deepcopy: bool=False) -> Self:
        if False:
            i = 10
            return i + 15
        self.saved_state = self.copy(deep=use_deepcopy)
        self.saved_state.target = self.target
        return self

    def restore(self) -> Self:
        if False:
            while True:
                i = 10
        if not hasattr(self, 'saved_state') or self.saved_state is None:
            raise Exception('Trying to restore without having saved')
        self.become(self.saved_state)
        return self

    def save_to_file(self, file_path: str) -> Self:
        if False:
            i = 10
            return i + 15
        with open(file_path, 'wb') as fp:
            fp.write(self.serialize())
        log.info(f'Saved mobject to {file_path}')
        return self

    @staticmethod
    def load(file_path) -> Mobject:
        if False:
            return 10
        if not os.path.exists(file_path):
            log.error(f'No file found at {file_path}')
            sys.exit(2)
        with open(file_path, 'rb') as fp:
            mobject = pickle.load(fp)
        return mobject

    def become(self, mobject: Mobject, match_updaters=False) -> Self:
        if False:
            i = 10
            return i + 15
        '\n        Edit all data and submobjects to be idential\n        to another mobject\n        '
        self.align_family(mobject)
        family1 = self.get_family()
        family2 = mobject.get_family()
        for (sm1, sm2) in zip(family1, family2):
            sm1.set_data(sm2.data)
            sm1.set_uniforms(sm2.uniforms)
            sm1.bounding_box[:] = sm2.bounding_box
            sm1.shader_folder = sm2.shader_folder
            sm1.texture_paths = sm2.texture_paths
            sm1.depth_test = sm2.depth_test
            sm1.render_primitive = sm2.render_primitive
            sm1.needs_new_bounding_box = sm2.needs_new_bounding_box
        for (attr, value) in list(mobject.__dict__.items()):
            if isinstance(value, Mobject) and value in family2:
                setattr(self, attr, family1[family2.index(value)])
        if match_updaters:
            self.match_updaters(mobject)
        return self

    def looks_identical(self, mobject: Mobject) -> bool:
        if False:
            return 10
        fam1 = self.family_members_with_points()
        fam2 = mobject.family_members_with_points()
        if len(fam1) != len(fam2):
            return False
        for (m1, m2) in zip(fam1, fam2):
            if m1.get_num_points() != m2.get_num_points():
                return False
            if not m1.data.dtype == m2.data.dtype:
                return False
            for key in m1.data.dtype.names:
                if not np.isclose(m1.data[key], m2.data[key]).all():
                    return False
            if set(m1.uniforms).difference(m2.uniforms):
                return False
            for key in m1.uniforms:
                value1 = m1.uniforms[key]
                value2 = m2.uniforms[key]
                if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray) and (not value1.size == value2.size):
                    return False
                if not np.isclose(value1, value2).all():
                    return False
        return True

    def has_same_shape_as(self, mobject: Mobject) -> bool:
        if False:
            return 10
        (points1, points2) = ((m.get_all_points() - m.get_center()) / m.get_height() for m in (self, mobject))
        if len(points1) != len(points2):
            return False
        return bool(np.isclose(points1, points2, atol=self.get_width() * 0.01).all())

    def replicate(self, n: int) -> Self:
        if False:
            print('Hello World!')
        group_class = self.get_group_class()
        return group_class(*(self.copy() for _ in range(n)))

    def get_grid(self, n_rows: int, n_cols: int, height: float | None=None, width: float | None=None, group_by_rows: bool=False, group_by_cols: bool=False, **kwargs) -> Self:
        if False:
            while True:
                i = 10
        '\n        Returns a new mobject containing multiple copies of this one\n        arranged in a grid\n        '
        total = n_rows * n_cols
        grid = self.replicate(total)
        if group_by_cols:
            kwargs['fill_rows_first'] = False
        grid.arrange_in_grid(n_rows, n_cols, **kwargs)
        if height is not None:
            grid.set_height(height)
        if width is not None:
            grid.set_height(width)
        group_class = self.get_group_class()
        if group_by_rows:
            return group_class(*(grid[n:n + n_cols] for n in range(0, total, n_cols)))
        elif group_by_cols:
            return group_class(*(grid[n:n + n_rows] for n in range(0, total, n_rows)))
        else:
            return grid

    def init_updaters(self):
        if False:
            return 10
        self.time_based_updaters: list[TimeBasedUpdater] = []
        self.non_time_updaters: list[NonTimeUpdater] = []
        self.has_updaters: bool = False
        self.updating_suspended: bool = False

    def update(self, dt: float=0, recurse: bool=True) -> Self:
        if False:
            for i in range(10):
                print('nop')
        if not self.has_updaters or self.updating_suspended:
            return self
        for updater in self.time_based_updaters:
            updater(self, dt)
        for updater in self.non_time_updaters:
            updater(self)
        if recurse:
            for submob in self.submobjects:
                submob.update(dt, recurse)
        return self

    def get_time_based_updaters(self) -> list[TimeBasedUpdater]:
        if False:
            for i in range(10):
                print('nop')
        return self.time_based_updaters

    def has_time_based_updater(self) -> bool:
        if False:
            return 10
        return len(self.time_based_updaters) > 0

    def get_updaters(self) -> list[Updater]:
        if False:
            print('Hello World!')
        return self.time_based_updaters + self.non_time_updaters

    def get_family_updaters(self) -> list[Updater]:
        if False:
            while True:
                i = 10
        return list(it.chain(*[sm.get_updaters() for sm in self.get_family()]))

    def add_updater(self, update_function: Updater, index: int | None=None, call_updater: bool=True) -> Self:
        if False:
            for i in range(10):
                print('nop')
        if 'dt' in get_parameters(update_function):
            updater_list = self.time_based_updaters
        else:
            updater_list = self.non_time_updaters
        if index is None:
            updater_list.append(update_function)
        else:
            updater_list.insert(index, update_function)
        self.refresh_has_updater_status()
        for parent in self.parents:
            parent.has_updaters = True
        if call_updater:
            self.update(dt=0)
        return self

    def remove_updater(self, update_function: Updater) -> Self:
        if False:
            while True:
                i = 10
        for updater_list in [self.time_based_updaters, self.non_time_updaters]:
            while update_function in updater_list:
                updater_list.remove(update_function)
        self.refresh_has_updater_status()
        return self

    def clear_updaters(self, recurse: bool=True) -> Self:
        if False:
            return 10
        self.time_based_updaters = []
        self.non_time_updaters = []
        if recurse:
            for submob in self.submobjects:
                submob.clear_updaters()
        self.refresh_has_updater_status()
        return self

    def match_updaters(self, mobject: Mobject) -> Self:
        if False:
            print('Hello World!')
        self.clear_updaters()
        for updater in mobject.get_updaters():
            self.add_updater(updater)
        return self

    def suspend_updating(self, recurse: bool=True) -> Self:
        if False:
            print('Hello World!')
        self.updating_suspended = True
        if recurse:
            for submob in self.submobjects:
                submob.suspend_updating(recurse)
        return self

    def resume_updating(self, recurse: bool=True, call_updater: bool=True) -> Self:
        if False:
            print('Hello World!')
        self.updating_suspended = False
        if recurse:
            for submob in self.submobjects:
                submob.resume_updating(recurse)
        for parent in self.parents:
            parent.resume_updating(recurse=False, call_updater=False)
        if call_updater:
            self.update(dt=0, recurse=recurse)
        return self

    def refresh_has_updater_status(self) -> Self:
        if False:
            return 10
        self.has_updaters = any((mob.get_updaters() for mob in self.get_family()))
        return self

    def is_changing(self) -> bool:
        if False:
            return 10
        return self._is_animating or self.has_updaters

    def set_animating_status(self, is_animating: bool, recurse: bool=True) -> Self:
        if False:
            return 10
        for mob in (*self.get_family(recurse), *self.get_ancestors()):
            mob._is_animating = is_animating
        return self

    def shift(self, vector: Vect3) -> Self:
        if False:
            for i in range(10):
                print('nop')
        self.apply_points_function(lambda points: points + vector, about_edge=None, works_on_bounding_box=True)
        return self

    def scale(self, scale_factor: float | npt.ArrayLike, min_scale_factor: float=1e-08, about_point: Vect3 | None=None, about_edge: Vect3=ORIGIN) -> Self:
        if False:
            print('Hello World!')
        '\n        Default behavior is to scale about the center of the mobject.\n        The argument about_edge can be a vector, indicating which side of\n        the mobject to scale about, e.g., mob.scale(about_edge = RIGHT)\n        scales about mob.get_right().\n\n        Otherwise, if about_point is given a value, scaling is done with\n        respect to that point.\n        '
        if isinstance(scale_factor, numbers.Number):
            scale_factor = max(scale_factor, min_scale_factor)
        else:
            scale_factor = np.array(scale_factor).clip(min=min_scale_factor)
        self.apply_points_function(lambda points: scale_factor * points, about_point=about_point, about_edge=about_edge, works_on_bounding_box=True)
        for mob in self.get_family():
            mob._handle_scale_side_effects(scale_factor)
        return self

    def _handle_scale_side_effects(self, scale_factor):
        if False:
            i = 10
            return i + 15
        pass

    def stretch(self, factor: float, dim: int, **kwargs) -> Self:
        if False:
            i = 10
            return i + 15

        def func(points):
            if False:
                for i in range(10):
                    print('nop')
            points[:, dim] *= factor
            return points
        self.apply_points_function(func, works_on_bounding_box=True, **kwargs)
        return self

    def rotate_about_origin(self, angle: float, axis: Vect3=OUT) -> Self:
        if False:
            return 10
        return self.rotate(angle, axis, about_point=ORIGIN)

    def rotate(self, angle: float, axis: Vect3=OUT, about_point: Vect3 | None=None, **kwargs) -> Self:
        if False:
            while True:
                i = 10
        rot_matrix_T = rotation_matrix_transpose(angle, axis)
        self.apply_points_function(lambda points: np.dot(points, rot_matrix_T), about_point, **kwargs)
        return self

    def flip(self, axis: Vect3=UP, **kwargs) -> Self:
        if False:
            while True:
                i = 10
        return self.rotate(TAU / 2, axis, **kwargs)

    def apply_function(self, function: Callable[[np.ndarray], np.ndarray], **kwargs) -> Self:
        if False:
            return 10
        if len(kwargs) == 0:
            kwargs['about_point'] = ORIGIN
        self.apply_points_function(lambda points: np.array([function(p) for p in points]), **kwargs)
        return self

    def apply_function_to_position(self, function: Callable[[np.ndarray], np.ndarray]) -> Self:
        if False:
            for i in range(10):
                print('nop')
        self.move_to(function(self.get_center()))
        return self

    def apply_function_to_submobject_positions(self, function: Callable[[np.ndarray], np.ndarray]) -> Self:
        if False:
            return 10
        for submob in self.submobjects:
            submob.apply_function_to_position(function)
        return self

    def apply_matrix(self, matrix: npt.ArrayLike, **kwargs) -> Self:
        if False:
            while True:
                i = 10
        if 'about_point' not in kwargs and 'about_edge' not in kwargs:
            kwargs['about_point'] = ORIGIN
        full_matrix = np.identity(self.dim)
        matrix = np.array(matrix)
        full_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
        self.apply_points_function(lambda points: np.dot(points, full_matrix.T), **kwargs)
        return self

    def apply_complex_function(self, function: Callable[[complex], complex], **kwargs) -> Self:
        if False:
            return 10

        def R3_func(point):
            if False:
                i = 10
                return i + 15
            (x, y, z) = point
            xy_complex = function(complex(x, y))
            return [xy_complex.real, xy_complex.imag, z]
        return self.apply_function(R3_func, **kwargs)

    def wag(self, direction: Vect3=RIGHT, axis: Vect3=DOWN, wag_factor: float=1.0) -> Self:
        if False:
            while True:
                i = 10
        for mob in self.family_members_with_points():
            alphas = np.dot(mob.get_points(), np.transpose(axis))
            alphas -= min(alphas)
            alphas /= max(alphas)
            alphas = alphas ** wag_factor
            mob.set_points(mob.get_points() + np.dot(alphas.reshape((len(alphas), 1)), np.array(direction).reshape((1, mob.dim))))
        return self

    def center(self) -> Self:
        if False:
            for i in range(10):
                print('nop')
        self.shift(-self.get_center())
        return self

    def align_on_border(self, direction: Vect3, buff: float=DEFAULT_MOBJECT_TO_EDGE_BUFFER) -> Self:
        if False:
            print('Hello World!')
        '\n        Direction just needs to be a vector pointing towards side or\n        corner in the 2d plane.\n        '
        target_point = np.sign(direction) * (FRAME_X_RADIUS, FRAME_Y_RADIUS, 0)
        point_to_align = self.get_bounding_box_point(direction)
        shift_val = target_point - point_to_align - buff * np.array(direction)
        shift_val = shift_val * abs(np.sign(direction))
        self.shift(shift_val)
        return self

    def to_corner(self, corner: Vect3=LEFT + DOWN, buff: float=DEFAULT_MOBJECT_TO_EDGE_BUFFER) -> Self:
        if False:
            for i in range(10):
                print('nop')
        return self.align_on_border(corner, buff)

    def to_edge(self, edge: Vect3=LEFT, buff: float=DEFAULT_MOBJECT_TO_EDGE_BUFFER) -> Self:
        if False:
            print('Hello World!')
        return self.align_on_border(edge, buff)

    def next_to(self, mobject_or_point: Mobject | Vect3, direction: Vect3=RIGHT, buff: float=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER, aligned_edge: Vect3=ORIGIN, submobject_to_align: Mobject | None=None, index_of_submobject_to_align: int | slice | None=None, coor_mask: Vect3=np.array([1, 1, 1])) -> Self:
        if False:
            return 10
        if isinstance(mobject_or_point, Mobject):
            mob = mobject_or_point
            if index_of_submobject_to_align is not None:
                target_aligner = mob[index_of_submobject_to_align]
            else:
                target_aligner = mob
            target_point = target_aligner.get_bounding_box_point(aligned_edge + direction)
        else:
            target_point = mobject_or_point
        if submobject_to_align is not None:
            aligner = submobject_to_align
        elif index_of_submobject_to_align is not None:
            aligner = self[index_of_submobject_to_align]
        else:
            aligner = self
        point_to_align = aligner.get_bounding_box_point(aligned_edge - direction)
        self.shift((target_point - point_to_align + buff * direction) * coor_mask)
        return self

    def shift_onto_screen(self, **kwargs) -> Self:
        if False:
            while True:
                i = 10
        space_lengths = [FRAME_X_RADIUS, FRAME_Y_RADIUS]
        for vect in (UP, DOWN, LEFT, RIGHT):
            dim = np.argmax(np.abs(vect))
            buff = kwargs.get('buff', DEFAULT_MOBJECT_TO_EDGE_BUFFER)
            max_val = space_lengths[dim] - buff
            edge_center = self.get_edge_center(vect)
            if np.dot(edge_center, vect) > max_val:
                self.to_edge(vect, **kwargs)
        return self

    def is_off_screen(self) -> bool:
        if False:
            print('Hello World!')
        if self.get_left()[0] > FRAME_X_RADIUS:
            return True
        if self.get_right()[0] < -FRAME_X_RADIUS:
            return True
        if self.get_bottom()[1] > FRAME_Y_RADIUS:
            return True
        if self.get_top()[1] < -FRAME_Y_RADIUS:
            return True
        return False

    def stretch_about_point(self, factor: float, dim: int, point: Vect3) -> Self:
        if False:
            i = 10
            return i + 15
        return self.stretch(factor, dim, about_point=point)

    def stretch_in_place(self, factor: float, dim: int) -> Self:
        if False:
            while True:
                i = 10
        return self.stretch(factor, dim)

    def rescale_to_fit(self, length: float, dim: int, stretch: bool=False, **kwargs) -> Self:
        if False:
            return 10
        old_length = self.length_over_dim(dim)
        if old_length == 0:
            return self
        if stretch:
            self.stretch(length / old_length, dim, **kwargs)
        else:
            self.scale(length / old_length, **kwargs)
        return self

    def stretch_to_fit_width(self, width: float, **kwargs) -> Self:
        if False:
            i = 10
            return i + 15
        return self.rescale_to_fit(width, 0, stretch=True, **kwargs)

    def stretch_to_fit_height(self, height: float, **kwargs) -> Self:
        if False:
            while True:
                i = 10
        return self.rescale_to_fit(height, 1, stretch=True, **kwargs)

    def stretch_to_fit_depth(self, depth: float, **kwargs) -> Self:
        if False:
            for i in range(10):
                print('nop')
        return self.rescale_to_fit(depth, 2, stretch=True, **kwargs)

    def set_width(self, width: float, stretch: bool=False, **kwargs) -> Self:
        if False:
            for i in range(10):
                print('nop')
        return self.rescale_to_fit(width, 0, stretch=stretch, **kwargs)

    def set_height(self, height: float, stretch: bool=False, **kwargs) -> Self:
        if False:
            while True:
                i = 10
        return self.rescale_to_fit(height, 1, stretch=stretch, **kwargs)

    def set_depth(self, depth: float, stretch: bool=False, **kwargs) -> Self:
        if False:
            print('Hello World!')
        return self.rescale_to_fit(depth, 2, stretch=stretch, **kwargs)

    def set_max_width(self, max_width: float, **kwargs) -> Self:
        if False:
            i = 10
            return i + 15
        if self.get_width() > max_width:
            self.set_width(max_width, **kwargs)
        return self

    def set_max_height(self, max_height: float, **kwargs) -> Self:
        if False:
            for i in range(10):
                print('nop')
        if self.get_height() > max_height:
            self.set_height(max_height, **kwargs)
        return self

    def set_max_depth(self, max_depth: float, **kwargs) -> Self:
        if False:
            return 10
        if self.get_depth() > max_depth:
            self.set_depth(max_depth, **kwargs)
        return self

    def set_min_width(self, min_width: float, **kwargs) -> Self:
        if False:
            print('Hello World!')
        if self.get_width() < min_width:
            self.set_width(min_width, **kwargs)
        return self

    def set_min_height(self, min_height: float, **kwargs) -> Self:
        if False:
            return 10
        if self.get_height() < min_height:
            self.set_height(min_height, **kwargs)
        return self

    def set_min_depth(self, min_depth: float, **kwargs) -> Self:
        if False:
            while True:
                i = 10
        if self.get_depth() < min_depth:
            self.set_depth(min_depth, **kwargs)
        return self

    def set_shape(self, width: Optional[float]=None, height: Optional[float]=None, depth: Optional[float]=None, **kwargs) -> Self:
        if False:
            while True:
                i = 10
        if width is not None:
            self.set_width(width, stretch=True, **kwargs)
        if height is not None:
            self.set_height(height, stretch=True, **kwargs)
        if depth is not None:
            self.set_depth(depth, stretch=True, **kwargs)
        return self

    def set_coord(self, value: float, dim: int, direction: Vect3=ORIGIN) -> Self:
        if False:
            print('Hello World!')
        curr = self.get_coord(dim, direction)
        shift_vect = np.zeros(self.dim)
        shift_vect[dim] = value - curr
        self.shift(shift_vect)
        return self

    def set_x(self, x: float, direction: Vect3=ORIGIN) -> Self:
        if False:
            print('Hello World!')
        return self.set_coord(x, 0, direction)

    def set_y(self, y: float, direction: Vect3=ORIGIN) -> Self:
        if False:
            print('Hello World!')
        return self.set_coord(y, 1, direction)

    def set_z(self, z: float, direction: Vect3=ORIGIN) -> Self:
        if False:
            print('Hello World!')
        return self.set_coord(z, 2, direction)

    def space_out_submobjects(self, factor: float=1.5, **kwargs) -> Self:
        if False:
            return 10
        self.scale(factor, **kwargs)
        for submob in self.submobjects:
            submob.scale(1.0 / factor)
        return self

    def move_to(self, point_or_mobject: Mobject | Vect3, aligned_edge: Vect3=ORIGIN, coor_mask: Vect3=np.array([1, 1, 1])) -> Self:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(point_or_mobject, Mobject):
            target = point_or_mobject.get_bounding_box_point(aligned_edge)
        else:
            target = point_or_mobject
        point_to_align = self.get_bounding_box_point(aligned_edge)
        self.shift((target - point_to_align) * coor_mask)
        return self

    def replace(self, mobject: Mobject, dim_to_match: int=0, stretch: bool=False) -> Self:
        if False:
            i = 10
            return i + 15
        if not mobject.get_num_points() and (not mobject.submobjects):
            self.scale(0)
            return self
        if stretch:
            for i in range(self.dim):
                self.rescale_to_fit(mobject.length_over_dim(i), i, stretch=True)
        else:
            self.rescale_to_fit(mobject.length_over_dim(dim_to_match), dim_to_match, stretch=False)
        self.shift(mobject.get_center() - self.get_center())
        return self

    def surround(self, mobject: Mobject, dim_to_match: int=0, stretch: bool=False, buff: float=MED_SMALL_BUFF) -> Self:
        if False:
            i = 10
            return i + 15
        self.replace(mobject, dim_to_match, stretch)
        length = mobject.length_over_dim(dim_to_match)
        self.scale((length + buff) / length)
        return self

    def put_start_and_end_on(self, start: Vect3, end: Vect3) -> Self:
        if False:
            print('Hello World!')
        (curr_start, curr_end) = self.get_start_and_end()
        curr_vect = curr_end - curr_start
        if np.all(curr_vect == 0):
            raise Exception('Cannot position endpoints of closed loop')
        target_vect = end - start
        self.scale(get_norm(target_vect) / get_norm(curr_vect), about_point=curr_start)
        self.rotate(angle_of_vector(target_vect) - angle_of_vector(curr_vect))
        self.rotate(np.arctan2(curr_vect[2], get_norm(curr_vect[:2])) - np.arctan2(target_vect[2], get_norm(target_vect[:2])), axis=np.array([-target_vect[1], target_vect[0], 0]))
        self.shift(start - self.get_start())
        return self

    @affects_family_data
    def set_rgba_array(self, rgba_array: npt.ArrayLike, name: str='rgba', recurse: bool=False) -> Self:
        if False:
            while True:
                i = 10
        for mob in self.get_family(recurse):
            data = mob.data if mob.get_num_points() > 0 else mob._data_defaults
            data[name][:] = rgba_array
        return self

    def set_color_by_rgba_func(self, func: Callable[[Vect3], Vect4], recurse: bool=True) -> Self:
        if False:
            return 10
        '\n        Func should take in a point in R3 and output an rgba value\n        '
        for mob in self.get_family(recurse):
            rgba_array = [func(point) for point in mob.get_points()]
            mob.set_rgba_array(rgba_array)
        return self

    def set_color_by_rgb_func(self, func: Callable[[Vect3], Vect3], opacity: float=1, recurse: bool=True) -> Self:
        if False:
            for i in range(10):
                print('nop')
        '\n        Func should take in a point in R3 and output an rgb value\n        '
        for mob in self.get_family(recurse):
            rgba_array = [[*func(point), opacity] for point in mob.get_points()]
            mob.set_rgba_array(rgba_array)
        return self

    @affects_family_data
    def set_rgba_array_by_color(self, color: ManimColor | Iterable[ManimColor] | None=None, opacity: float | Iterable[float] | None=None, name: str='rgba', recurse: bool=True) -> Self:
        if False:
            for i in range(10):
                print('nop')
        for mob in self.get_family(recurse):
            data = mob.data if mob.has_points() > 0 else mob._data_defaults
            if color is not None:
                rgbs = np.array(list(map(color_to_rgb, listify(color))))
                if 1 < len(rgbs):
                    rgbs = resize_with_interpolation(rgbs, len(data))
                data[name][:, :3] = rgbs
            if opacity is not None:
                if isinstance(opacity, list):
                    opacity = resize_with_interpolation(np.array(opacity), len(data))
                data[name][:, 3] = opacity
        return self

    def set_color(self, color: ManimColor | Iterable[ManimColor] | None, opacity: float | Iterable[float] | None=None, recurse: bool=True) -> Self:
        if False:
            print('Hello World!')
        self.set_rgba_array_by_color(color, opacity, recurse=False)
        if recurse:
            for submob in self.submobjects:
                submob.set_color(color, recurse=True)
        return self

    def set_opacity(self, opacity: float | Iterable[float] | None, recurse: bool=True) -> Self:
        if False:
            print('Hello World!')
        self.set_rgba_array_by_color(color=None, opacity=opacity, recurse=False)
        if recurse:
            for submob in self.submobjects:
                submob.set_opacity(opacity, recurse=True)
        return self

    def get_color(self) -> str:
        if False:
            i = 10
            return i + 15
        return rgb_to_hex(self.data['rgba'][0, :3])

    def get_opacity(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        return self.data['rgba'][0, 3]

    def set_color_by_gradient(self, *colors: ManimColor) -> Self:
        if False:
            return 10
        if self.has_points():
            self.set_color(colors)
        else:
            self.set_submobject_colors_by_gradient(*colors)
        return self

    def set_submobject_colors_by_gradient(self, *colors: ManimColor) -> Self:
        if False:
            for i in range(10):
                print('nop')
        if len(colors) == 0:
            raise Exception('Need at least one color')
        elif len(colors) == 1:
            return self.set_color(*colors)
        mobs = self.submobjects
        new_colors = color_gradient(colors, len(mobs))
        for (mob, color) in zip(mobs, new_colors):
            mob.set_color(color)
        return self

    def fade(self, darkness: float=0.5, recurse: bool=True) -> Self:
        if False:
            print('Hello World!')
        self.set_opacity(1.0 - darkness, recurse=recurse)

    def get_shading(self) -> np.ndarray:
        if False:
            return 10
        return self.uniforms['shading']

    def set_shading(self, reflectiveness: float | None=None, gloss: float | None=None, shadow: float | None=None, recurse: bool=True) -> Self:
        if False:
            return 10
        '\n        Larger reflectiveness makes things brighter when facing the light\n        Larger shadow makes faces opposite the light darker\n        Makes parts bright where light gets reflected toward the camera\n        '
        for mob in self.get_family(recurse):
            for (i, value) in enumerate([reflectiveness, gloss, shadow]):
                if value is not None:
                    mob.uniforms['shading'][i] = value
        return self

    def get_reflectiveness(self) -> float:
        if False:
            while True:
                i = 10
        return self.get_shading()[0]

    def get_gloss(self) -> float:
        if False:
            return 10
        return self.get_shading()[1]

    def get_shadow(self) -> float:
        if False:
            i = 10
            return i + 15
        return self.get_shading()[2]

    def set_reflectiveness(self, reflectiveness: float, recurse: bool=True) -> Self:
        if False:
            for i in range(10):
                print('nop')
        self.set_shading(reflectiveness=reflectiveness, recurse=recurse)
        return self

    def set_gloss(self, gloss: float, recurse: bool=True) -> Self:
        if False:
            print('Hello World!')
        self.set_shading(gloss=gloss, recurse=recurse)
        return self

    def set_shadow(self, shadow: float, recurse: bool=True) -> Self:
        if False:
            for i in range(10):
                print('nop')
        self.set_shading(shadow=shadow, recurse=recurse)
        return self

    def add_background_rectangle(self, color: ManimColor | None=None, opacity: float=1.0, **kwargs) -> Self:
        if False:
            return 10
        from manimlib.mobject.shape_matchers import BackgroundRectangle
        self.background_rectangle = BackgroundRectangle(self, color=color, fill_opacity=opacity, **kwargs)
        self.add_to_back(self.background_rectangle)
        return self

    def add_background_rectangle_to_submobjects(self, **kwargs) -> Self:
        if False:
            for i in range(10):
                print('nop')
        for submobject in self.submobjects:
            submobject.add_background_rectangle(**kwargs)
        return self

    def add_background_rectangle_to_family_members_with_points(self, **kwargs) -> Self:
        if False:
            i = 10
            return i + 15
        for mob in self.family_members_with_points():
            mob.add_background_rectangle(**kwargs)
        return self

    def get_bounding_box_point(self, direction: Vect3) -> Vect3:
        if False:
            print('Hello World!')
        bb = self.get_bounding_box()
        indices = (np.sign(direction) + 1).astype(int)
        return np.array([bb[indices[i]][i] for i in range(3)])

    def get_edge_center(self, direction: Vect3) -> Vect3:
        if False:
            for i in range(10):
                print('nop')
        return self.get_bounding_box_point(direction)

    def get_corner(self, direction: Vect3) -> Vect3:
        if False:
            return 10
        return self.get_bounding_box_point(direction)

    def get_all_corners(self):
        if False:
            return 10
        bb = self.get_bounding_box()
        return np.array([[bb[indices[-i + 1]][i] for i in range(3)] for indices in it.product([0, 2], repeat=3)])

    def get_center(self) -> Vect3:
        if False:
            return 10
        return self.get_bounding_box()[1]

    def get_center_of_mass(self) -> Vect3:
        if False:
            i = 10
            return i + 15
        return self.get_all_points().mean(0)

    def get_boundary_point(self, direction: Vect3) -> Vect3:
        if False:
            i = 10
            return i + 15
        all_points = self.get_all_points()
        boundary_directions = all_points - self.get_center()
        norms = np.linalg.norm(boundary_directions, axis=1)
        boundary_directions /= np.repeat(norms, 3).reshape((len(norms), 3))
        index = np.argmax(np.dot(boundary_directions, np.array(direction).T))
        return all_points[index]

    def get_continuous_bounding_box_point(self, direction: Vect3) -> Vect3:
        if False:
            print('Hello World!')
        (dl, center, ur) = self.get_bounding_box()
        corner_vect = ur - center
        return center + direction / np.max(np.abs(np.true_divide(direction, corner_vect, out=np.zeros(len(direction)), where=corner_vect != 0)))

    def get_top(self) -> Vect3:
        if False:
            i = 10
            return i + 15
        return self.get_edge_center(UP)

    def get_bottom(self) -> Vect3:
        if False:
            print('Hello World!')
        return self.get_edge_center(DOWN)

    def get_right(self) -> Vect3:
        if False:
            while True:
                i = 10
        return self.get_edge_center(RIGHT)

    def get_left(self) -> Vect3:
        if False:
            for i in range(10):
                print('nop')
        return self.get_edge_center(LEFT)

    def get_zenith(self) -> Vect3:
        if False:
            while True:
                i = 10
        return self.get_edge_center(OUT)

    def get_nadir(self) -> Vect3:
        if False:
            return 10
        return self.get_edge_center(IN)

    def length_over_dim(self, dim: int) -> float:
        if False:
            print('Hello World!')
        bb = self.get_bounding_box()
        return abs((bb[2] - bb[0])[dim])

    def get_width(self) -> float:
        if False:
            return 10
        return self.length_over_dim(0)

    def get_height(self) -> float:
        if False:
            return 10
        return self.length_over_dim(1)

    def get_depth(self) -> float:
        if False:
            print('Hello World!')
        return self.length_over_dim(2)

    def get_coord(self, dim: int, direction: Vect3=ORIGIN) -> float:
        if False:
            return 10
        '\n        Meant to generalize get_x, get_y, get_z\n        '
        return self.get_bounding_box_point(direction)[dim]

    def get_x(self, direction=ORIGIN) -> float:
        if False:
            return 10
        return self.get_coord(0, direction)

    def get_y(self, direction=ORIGIN) -> float:
        if False:
            print('Hello World!')
        return self.get_coord(1, direction)

    def get_z(self, direction=ORIGIN) -> float:
        if False:
            return 10
        return self.get_coord(2, direction)

    def get_start(self) -> Vect3:
        if False:
            while True:
                i = 10
        self.throw_error_if_no_points()
        return self.get_points()[0].copy()

    def get_end(self) -> Vect3:
        if False:
            for i in range(10):
                print('nop')
        self.throw_error_if_no_points()
        return self.get_points()[-1].copy()

    def get_start_and_end(self) -> tuple[Vect3, Vect3]:
        if False:
            i = 10
            return i + 15
        self.throw_error_if_no_points()
        points = self.get_points()
        return (points[0].copy(), points[-1].copy())

    def point_from_proportion(self, alpha: float) -> Vect3:
        if False:
            print('Hello World!')
        points = self.get_points()
        (i, subalpha) = integer_interpolate(0, len(points) - 1, alpha)
        return interpolate(points[i], points[i + 1], subalpha)

    def pfp(self, alpha):
        if False:
            return 10
        'Abbreviation for point_from_proportion'
        return self.point_from_proportion(alpha)

    def get_pieces(self, n_pieces: int) -> Group:
        if False:
            for i in range(10):
                print('nop')
        template = self.copy()
        template.set_submobjects([])
        alphas = np.linspace(0, 1, n_pieces + 1)
        return Group(*[template.copy().pointwise_become_partial(self, a1, a2) for (a1, a2) in zip(alphas[:-1], alphas[1:])])

    def get_z_index_reference_point(self) -> Vect3:
        if False:
            while True:
                i = 10
        z_index_group = getattr(self, 'z_index_group', self)
        return z_index_group.get_center()

    def match_color(self, mobject: Mobject) -> Self:
        if False:
            return 10
        return self.set_color(mobject.get_color())

    def match_dim_size(self, mobject: Mobject, dim: int, **kwargs) -> Self:
        if False:
            i = 10
            return i + 15
        return self.rescale_to_fit(mobject.length_over_dim(dim), dim, **kwargs)

    def match_width(self, mobject: Mobject, **kwargs) -> Self:
        if False:
            for i in range(10):
                print('nop')
        return self.match_dim_size(mobject, 0, **kwargs)

    def match_height(self, mobject: Mobject, **kwargs) -> Self:
        if False:
            print('Hello World!')
        return self.match_dim_size(mobject, 1, **kwargs)

    def match_depth(self, mobject: Mobject, **kwargs) -> Self:
        if False:
            i = 10
            return i + 15
        return self.match_dim_size(mobject, 2, **kwargs)

    def match_coord(self, mobject_or_point: Mobject | Vect3, dim: int, direction: Vect3=ORIGIN) -> Self:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(mobject_or_point, Mobject):
            coord = mobject_or_point.get_coord(dim, direction)
        else:
            coord = mobject_or_point[dim]
        return self.set_coord(coord, dim=dim, direction=direction)

    def match_x(self, mobject_or_point: Mobject | Vect3, direction: Vect3=ORIGIN) -> Self:
        if False:
            print('Hello World!')
        return self.match_coord(mobject_or_point, 0, direction)

    def match_y(self, mobject_or_point: Mobject | Vect3, direction: Vect3=ORIGIN) -> Self:
        if False:
            i = 10
            return i + 15
        return self.match_coord(mobject_or_point, 1, direction)

    def match_z(self, mobject_or_point: Mobject | Vect3, direction: Vect3=ORIGIN) -> Self:
        if False:
            return 10
        return self.match_coord(mobject_or_point, 2, direction)

    def align_to(self, mobject_or_point: Mobject | Vect3, direction: Vect3=ORIGIN) -> Self:
        if False:
            print('Hello World!')
        "\n        Examples:\n        mob1.align_to(mob2, UP) moves mob1 vertically so that its\n        top edge lines ups with mob2's top edge.\n\n        mob1.align_to(mob2, alignment_vect = RIGHT) moves mob1\n        horizontally so that it's center is directly above/below\n        the center of mob2\n        "
        if isinstance(mobject_or_point, Mobject):
            point = mobject_or_point.get_bounding_box_point(direction)
        else:
            point = mobject_or_point
        for dim in range(self.dim):
            if direction[dim] != 0:
                self.set_coord(point[dim], dim, direction)
        return self

    def get_group_class(self):
        if False:
            return 10
        return Group

    def is_aligned_with(self, mobject: Mobject) -> bool:
        if False:
            return 10
        if len(self.data) != len(mobject.data):
            return False
        if len(self.submobjects) != len(mobject.submobjects):
            return False
        return all((sm1.is_aligned_with(sm2) for (sm1, sm2) in zip(self.submobjects, mobject.submobjects)))

    def align_data_and_family(self, mobject: Mobject) -> Self:
        if False:
            for i in range(10):
                print('nop')
        self.align_family(mobject)
        self.align_data(mobject)
        return self

    def align_data(self, mobject: Mobject) -> Self:
        if False:
            print('Hello World!')
        for (mob1, mob2) in zip(self.get_family(), mobject.get_family()):
            mob1.align_points(mob2)
        return self

    def align_points(self, mobject: Mobject) -> Self:
        if False:
            return 10
        max_len = max(self.get_num_points(), mobject.get_num_points())
        for mob in (self, mobject):
            mob.resize_points(max_len, resize_func=resize_preserving_order)
        return self

    def align_family(self, mobject: Mobject) -> Self:
        if False:
            i = 10
            return i + 15
        mob1 = self
        mob2 = mobject
        n1 = len(mob1)
        n2 = len(mob2)
        if n1 != n2:
            mob1.add_n_more_submobjects(max(0, n2 - n1))
            mob2.add_n_more_submobjects(max(0, n1 - n2))
        for (sm1, sm2) in zip(mob1.submobjects, mob2.submobjects):
            sm1.align_family(sm2)
        return self

    def push_self_into_submobjects(self) -> Self:
        if False:
            return 10
        copy = self.copy()
        copy.set_submobjects([])
        self.resize_points(0)
        self.add(copy)
        return self

    def add_n_more_submobjects(self, n: int) -> Self:
        if False:
            print('Hello World!')
        if n == 0:
            return self
        curr = len(self.submobjects)
        if curr == 0:
            null_mob = self.copy()
            null_mob.set_points([self.get_center()])
            self.set_submobjects([null_mob.copy() for k in range(n)])
            return self
        target = curr + n
        repeat_indices = np.arange(target) * curr // target
        split_factors = [(repeat_indices == i).sum() for i in range(curr)]
        new_submobs = []
        for (submob, sf) in zip(self.submobjects, split_factors):
            new_submobs.append(submob)
            for k in range(1, sf):
                new_submobs.append(submob.invisible_copy())
        self.set_submobjects(new_submobs)
        return self

    def invisible_copy(self) -> Self:
        if False:
            while True:
                i = 10
        return self.copy().set_opacity(0)

    def interpolate(self, mobject1: Mobject, mobject2: Mobject, alpha: float, path_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray]=straight_path) -> Self:
        if False:
            for i in range(10):
                print('nop')
        keys = [k for k in self.data.dtype.names if k not in self.locked_data_keys]
        if keys:
            self.note_changed_data()
        for key in keys:
            func = path_func if key in self.pointlike_data_keys else interpolate
            md1 = mobject1.data[key]
            md2 = mobject2.data[key]
            if key in self.const_data_keys:
                md1 = md1[0]
                md2 = md2[0]
            self.data[key] = func(md1, md2, alpha)
        keys = [k for k in self.uniforms if k not in self.locked_uniform_keys]
        for key in keys:
            if key not in mobject1.uniforms or key not in mobject2.uniforms:
                continue
            self.uniforms[key] = interpolate(mobject1.uniforms[key], mobject2.uniforms[key], alpha)
        self.bounding_box[:] = path_func(mobject1.bounding_box, mobject2.bounding_box, alpha)
        return self

    def pointwise_become_partial(self, mobject, a, b) -> Self:
        if False:
            while True:
                i = 10
        '\n        Set points in such a way as to become only\n        part of mobject.\n        Inputs 0 <= a < b <= 1 determine what portion\n        of mobject to become.\n        '
        return self

    def lock_data(self, keys: Iterable[str]) -> Self:
        if False:
            for i in range(10):
                print('nop')
        "\n        To speed up some animations, particularly transformations,\n        it can be handy to acknowledge which pieces of data\n        won't change during the animation so that calls to\n        interpolate can skip this, and so that it's not\n        read into the shader_wrapper objects needlessly\n        "
        if self.has_updaters:
            return self
        self.locked_data_keys = set(keys)
        return self

    def lock_uniforms(self, keys: Iterable[str]) -> Self:
        if False:
            for i in range(10):
                print('nop')
        if self.has_updaters:
            return self
        self.locked_uniform_keys = set(keys)
        return self

    def lock_matching_data(self, mobject1: Mobject, mobject2: Mobject) -> Self:
        if False:
            print('Hello World!')
        tuples = zip(self.get_family(), mobject1.get_family(), mobject2.get_family())
        for (sm, sm1, sm2) in tuples:
            if not sm.data.dtype == sm1.data.dtype == sm2.data.dtype:
                continue
            sm.lock_data((key for key in sm.data.dtype.names if arrays_match(sm1.data[key], sm2.data[key])))
            sm.lock_uniforms((key for key in self.uniforms if all(listify(mobject1.uniforms.get(key, 0) == mobject2.uniforms.get(key, 0)))))
            sm.const_data_keys = set((key for key in sm.data.dtype.names if key not in sm.locked_data_keys if all((array_is_constant(mob.data[key]) for mob in (sm, sm1, sm2)))))
        return self

    def unlock_data(self) -> Self:
        if False:
            print('Hello World!')
        for mob in self.get_family():
            mob.locked_data_keys = set()
            mob.const_data_keys = set()
            mob.locked_uniform_keys = set()
        return self

    def affects_shader_info_id(func: Callable):
        if False:
            print('Hello World!')

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if False:
                print('Hello World!')
            result = func(self, *args, **kwargs)
            self.refresh_shader_wrapper_id()
            return result
        return wrapper

    @affects_shader_info_id
    def set_uniform(self, recurse: bool=True, **new_uniforms) -> Self:
        if False:
            i = 10
            return i + 15
        for mob in self.get_family(recurse):
            mob.uniforms.update(new_uniforms)
        return self

    @affects_shader_info_id
    def fix_in_frame(self, recurse: bool=True) -> Self:
        if False:
            i = 10
            return i + 15
        self.set_uniform(recurse, is_fixed_in_frame=1.0)
        return self

    @affects_shader_info_id
    def unfix_from_frame(self, recurse: bool=True) -> Self:
        if False:
            i = 10
            return i + 15
        self.set_uniform(recurse, is_fixed_in_frame=0.0)
        return self

    def is_fixed_in_frame(self) -> bool:
        if False:
            return 10
        return bool(self.uniforms['is_fixed_in_frame'])

    @affects_shader_info_id
    def apply_depth_test(self, recurse: bool=True) -> Self:
        if False:
            i = 10
            return i + 15
        for mob in self.get_family(recurse):
            mob.depth_test = True
        return self

    @affects_shader_info_id
    def deactivate_depth_test(self, recurse: bool=True) -> Self:
        if False:
            for i in range(10):
                print('nop')
        for mob in self.get_family(recurse):
            mob.depth_test = False
        return self

    @affects_data
    def replace_shader_code(self, old: str, new: str) -> Self:
        if False:
            print('Hello World!')
        self.shader_code_replacements[old] = new
        self._shaders_initialized = False
        for mob in self.get_ancestors():
            mob._shaders_initialized = False
        return self

    def set_color_by_code(self, glsl_code: str) -> Self:
        if False:
            while True:
                i = 10
        '\n        Takes a snippet of code and inserts it into a\n        context which has the following variables:\n        vec4 color, vec3 point, vec3 unit_normal.\n        The code should change the color variable\n        '
        self.replace_shader_code('///// INSERT COLOR FUNCTION HERE /////', glsl_code)
        return self

    def set_color_by_xyz_func(self, glsl_snippet: str, min_value: float=-5.0, max_value: float=5.0, colormap: str='viridis') -> Self:
        if False:
            for i in range(10):
                print('nop')
        '\n        Pass in a glsl expression in terms of x, y and z which returns\n        a float.\n        '
        for char in 'xyz':
            glsl_snippet = glsl_snippet.replace(char, 'point.' + char)
        rgb_list = get_colormap_list(colormap)
        self.set_color_by_code('color.rgb = float_to_color({}, {}, {}, {});'.format(glsl_snippet, float(min_value), float(max_value), get_colormap_code(rgb_list)))
        return self

    def init_shader_data(self, ctx: Context):
        if False:
            print('Hello World!')
        self.shader_indices = np.zeros(0)
        self.shader_wrapper = ShaderWrapper(ctx=ctx, vert_data=self.data, shader_folder=self.shader_folder, texture_paths=self.texture_paths, depth_test=self.depth_test, render_primitive=self.render_primitive)

    def refresh_shader_wrapper_id(self):
        if False:
            print('Hello World!')
        if self._shaders_initialized:
            self.shader_wrapper.refresh_id()
        return self

    def get_shader_wrapper(self, ctx: Context) -> ShaderWrapper:
        if False:
            while True:
                i = 10
        if not self._shaders_initialized:
            self.init_shader_data(ctx)
            self._shaders_initialized = True
        self.shader_wrapper.vert_data = self.get_shader_data()
        self.shader_wrapper.vert_indices = self.get_shader_vert_indices()
        self.shader_wrapper.bind_to_mobject_uniforms(self.get_uniforms())
        self.shader_wrapper.depth_test = self.depth_test
        for (old, new) in self.shader_code_replacements.items():
            self.shader_wrapper.replace_code(old, new)
        return self.shader_wrapper

    def get_shader_wrapper_list(self, ctx: Context) -> list[ShaderWrapper]:
        if False:
            for i in range(10):
                print('nop')
        shader_wrappers = it.chain([self.get_shader_wrapper(ctx)], *[sm.get_shader_wrapper_list(ctx) for sm in self.submobjects])
        batches = batch_by_property(shader_wrappers, lambda sw: sw.get_id())
        result = []
        for (wrapper_group, sid) in batches:
            shader_wrapper = wrapper_group[0]
            if not shader_wrapper.is_valid():
                continue
            shader_wrapper.combine_with(*wrapper_group[1:])
            if len(shader_wrapper.vert_data) > 0:
                result.append(shader_wrapper)
        return result

    def get_shader_data(self):
        if False:
            print('Hello World!')
        return self.data

    def get_uniforms(self):
        if False:
            return 10
        return self.uniforms

    def get_shader_vert_indices(self):
        if False:
            for i in range(10):
                print('nop')
        return self.shader_indices

    def render(self, ctx: Context, camera_uniforms: dict):
        if False:
            i = 10
            return i + 15
        if self._data_has_changed:
            self.shader_wrappers = self.get_shader_wrapper_list(ctx)
            for shader_wrapper in self.shader_wrappers:
                shader_wrapper.generate_vao()
            self._data_has_changed = False
        for shader_wrapper in self.shader_wrappers:
            shader_wrapper.update_program_uniforms(camera_uniforms)
            shader_wrapper.pre_render()
            shader_wrapper.render()
    '\n        Event handling follows the Event Bubbling model of DOM in javascript.\n        Return false to stop the event bubbling.\n        To learn more visit https://www.quirksmode.org/js/events_order.html\n\n        Event Callback Argument is a callable function taking two arguments:\n            1. Mobject\n            2. EventData\n    '

    def init_event_listners(self):
        if False:
            return 10
        self.event_listners: list[EventListener] = []

    def add_event_listner(self, event_type: EventType, event_callback: Callable[[Mobject, dict[str]]]):
        if False:
            return 10
        event_listner = EventListener(self, event_type, event_callback)
        self.event_listners.append(event_listner)
        EVENT_DISPATCHER.add_listner(event_listner)
        return self

    def remove_event_listner(self, event_type: EventType, event_callback: Callable[[Mobject, dict[str]]]):
        if False:
            for i in range(10):
                print('nop')
        event_listner = EventListener(self, event_type, event_callback)
        while event_listner in self.event_listners:
            self.event_listners.remove(event_listner)
        EVENT_DISPATCHER.remove_listner(event_listner)
        return self

    def clear_event_listners(self, recurse: bool=True):
        if False:
            i = 10
            return i + 15
        self.event_listners = []
        if recurse:
            for submob in self.submobjects:
                submob.clear_event_listners(recurse=recurse)
        return self

    def get_event_listners(self):
        if False:
            while True:
                i = 10
        return self.event_listners

    def get_family_event_listners(self):
        if False:
            return 10
        return list(it.chain(*[sm.get_event_listners() for sm in self.get_family()]))

    def get_has_event_listner(self):
        if False:
            while True:
                i = 10
        return any((mob.get_event_listners() for mob in self.get_family()))

    def add_mouse_motion_listner(self, callback):
        if False:
            i = 10
            return i + 15
        self.add_event_listner(EventType.MouseMotionEvent, callback)

    def remove_mouse_motion_listner(self, callback):
        if False:
            return 10
        self.remove_event_listner(EventType.MouseMotionEvent, callback)

    def add_mouse_press_listner(self, callback):
        if False:
            while True:
                i = 10
        self.add_event_listner(EventType.MousePressEvent, callback)

    def remove_mouse_press_listner(self, callback):
        if False:
            return 10
        self.remove_event_listner(EventType.MousePressEvent, callback)

    def add_mouse_release_listner(self, callback):
        if False:
            print('Hello World!')
        self.add_event_listner(EventType.MouseReleaseEvent, callback)

    def remove_mouse_release_listner(self, callback):
        if False:
            for i in range(10):
                print('nop')
        self.remove_event_listner(EventType.MouseReleaseEvent, callback)

    def add_mouse_drag_listner(self, callback):
        if False:
            print('Hello World!')
        self.add_event_listner(EventType.MouseDragEvent, callback)

    def remove_mouse_drag_listner(self, callback):
        if False:
            while True:
                i = 10
        self.remove_event_listner(EventType.MouseDragEvent, callback)

    def add_mouse_scroll_listner(self, callback):
        if False:
            while True:
                i = 10
        self.add_event_listner(EventType.MouseScrollEvent, callback)

    def remove_mouse_scroll_listner(self, callback):
        if False:
            for i in range(10):
                print('nop')
        self.remove_event_listner(EventType.MouseScrollEvent, callback)

    def add_key_press_listner(self, callback):
        if False:
            return 10
        self.add_event_listner(EventType.KeyPressEvent, callback)

    def remove_key_press_listner(self, callback):
        if False:
            print('Hello World!')
        self.remove_event_listner(EventType.KeyPressEvent, callback)

    def add_key_release_listner(self, callback):
        if False:
            while True:
                i = 10
        self.add_event_listner(EventType.KeyReleaseEvent, callback)

    def remove_key_release_listner(self, callback):
        if False:
            return 10
        self.remove_event_listner(EventType.KeyReleaseEvent, callback)

    def throw_error_if_no_points(self):
        if False:
            while True:
                i = 10
        if not self.has_points():
            message = 'Cannot call Mobject.{} ' + 'for a Mobject with no points'
            caller_name = sys._getframe(1).f_code.co_name
            raise Exception(message.format(caller_name))

class Group(Mobject):

    def __init__(self, *mobjects: Mobject, **kwargs):
        if False:
            return 10
        if not all([isinstance(m, Mobject) for m in mobjects]):
            raise Exception('All submobjects must be of type Mobject')
        Mobject.__init__(self, **kwargs)
        self.add(*mobjects)
        if any((m.is_fixed_in_frame() for m in mobjects)):
            self.fix_in_frame()

    def __add__(self, other: Mobject | Group) -> Self:
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(other, Mobject)
        return self.add(other)

class Point(Mobject):

    def __init__(self, location: Vect3=ORIGIN, artificial_width: float=1e-06, artificial_height: float=1e-06, **kwargs):
        if False:
            return 10
        self.artificial_width = artificial_width
        self.artificial_height = artificial_height
        super().__init__(**kwargs)
        self.set_location(location)

    def get_width(self) -> float:
        if False:
            return 10
        return self.artificial_width

    def get_height(self) -> float:
        if False:
            while True:
                i = 10
        return self.artificial_height

    def get_location(self) -> Vect3:
        if False:
            print('Hello World!')
        return self.get_points()[0].copy()

    def get_bounding_box_point(self, *args, **kwargs) -> Vect3:
        if False:
            while True:
                i = 10
        return self.get_location()

    def set_location(self, new_loc: npt.ArrayLike) -> Self:
        if False:
            print('Hello World!')
        self.set_points(np.array(new_loc, ndmin=2, dtype=float))
        return self

class _AnimationBuilder:

    def __init__(self, mobject: Mobject):
        if False:
            while True:
                i = 10
        self.mobject = mobject
        self.overridden_animation = None
        self.mobject.generate_target()
        self.is_chaining = False
        self.methods: list[Callable] = []
        self.anim_args = {}
        self.can_pass_args = True

    def __getattr__(self, method_name: str):
        if False:
            while True:
                i = 10
        method = getattr(self.mobject.target, method_name)
        self.methods.append(method)
        has_overridden_animation = hasattr(method, '_override_animate')
        if self.is_chaining and has_overridden_animation or self.overridden_animation:
            raise NotImplementedError('Method chaining is currently not supported for ' + 'overridden animations')

        def update_target(*method_args, **method_kwargs):
            if False:
                return 10
            if has_overridden_animation:
                self.overridden_animation = method._override_animate(self.mobject, *method_args, **method_kwargs)
            else:
                method(*method_args, **method_kwargs)
            return self
        self.is_chaining = True
        return update_target

    def __call__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        return self.set_anim_args(**kwargs)

    def set_anim_args(self, **kwargs):
        if False:
            print('Hello World!')
        '\n        You can change the args of :class:`~manimlib.animation.transform.Transform`, such as\n\n        - ``run_time``\n        - ``time_span``\n        - ``rate_func``\n        - ``lag_ratio``\n        - ``path_arc``\n        - ``path_func``\n\n        and so on.\n        '
        if not self.can_pass_args:
            raise ValueError('Animation arguments can only be passed by calling ``animate`` ' + 'or ``set_anim_args`` and can only be passed once')
        self.anim_args = kwargs
        self.can_pass_args = False
        return self

    def build(self):
        if False:
            return 10
        from manimlib.animation.transform import _MethodAnimation
        if self.overridden_animation:
            return self.overridden_animation
        return _MethodAnimation(self.mobject, self.methods, **self.anim_args)

def override_animate(method):
    if False:
        for i in range(10):
            print('nop')

    def decorator(animation_method):
        if False:
            for i in range(10):
                print('nop')
        method._override_animate = animation_method
        return animation_method
    return decorator