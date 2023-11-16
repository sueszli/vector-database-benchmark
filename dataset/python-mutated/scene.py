from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
from typing_extensions import Self
from .. import binding
from ..awaitable_response import AwaitableResponse, NullResponse
from ..dataclasses import KWONLY_SLOTS
from ..element import Element
from ..events import GenericEventArguments, SceneClickEventArguments, SceneClickHit, SceneDragEventArguments, handle_event
from .scene_object3d import Object3D

@dataclass(**KWONLY_SLOTS)
class SceneCamera:
    x: float = 0
    y: float = -3
    z: float = 5
    look_at_x: float = 0
    look_at_y: float = 0
    look_at_z: float = 0
    up_x: float = 0
    up_y: float = 0
    up_z: float = 1

@dataclass(**KWONLY_SLOTS)
class SceneObject:
    id: str = 'scene'

class Scene(Element, component='scene.js', libraries=['lib/tween/tween.umd.js'], exposed_libraries=['lib/three/three.module.js', 'lib/three/modules/CSS2DRenderer.js', 'lib/three/modules/CSS3DRenderer.js', 'lib/three/modules/DragControls.js', 'lib/three/modules/OrbitControls.js', 'lib/three/modules/STLLoader.js']):
    from .scene_objects import Box as box
    from .scene_objects import Curve as curve
    from .scene_objects import Cylinder as cylinder
    from .scene_objects import Extrusion as extrusion
    from .scene_objects import Group as group
    from .scene_objects import Line as line
    from .scene_objects import PointCloud as point_cloud
    from .scene_objects import QuadraticBezierTube as quadratic_bezier_tube
    from .scene_objects import Ring as ring
    from .scene_objects import Sphere as sphere
    from .scene_objects import SpotLight as spot_light
    from .scene_objects import Stl as stl
    from .scene_objects import Text as text
    from .scene_objects import Text3d as text3d
    from .scene_objects import Texture as texture

    def __init__(self, width: int=400, height: int=300, grid: bool=True, on_click: Optional[Callable[..., Any]]=None, on_drag_start: Optional[Callable[..., Any]]=None, on_drag_end: Optional[Callable[..., Any]]=None, drag_constraints: str='') -> None:
        if False:
            i = 10
            return i + 15
        "3D Scene\n\n        Display a 3D scene using `three.js <https://threejs.org/>`_.\n        Currently NiceGUI supports boxes, spheres, cylinders/cones, extrusions, straight lines, curves and textured meshes.\n        Objects can be translated, rotated and displayed with different color, opacity or as wireframes.\n        They can also be grouped to apply joint movements.\n\n        :param width: width of the canvas\n        :param height: height of the canvas\n        :param grid: whether to display a grid\n        :param on_click: callback to execute when a 3D object is clicked\n        :param on_drag_start: callback to execute when a 3D object is dragged\n        :param on_drag_end: callback to execute when a 3D object is dropped\n        :param drag_constraints: comma-separated JavaScript expression for constraining positions of dragged objects (e.g. ``'x = 0, z = y / 2'``)\n        "
        super().__init__()
        self._props['width'] = width
        self._props['height'] = height
        self._props['grid'] = grid
        self.objects: Dict[str, Object3D] = {}
        self.stack: List[Union[Object3D, SceneObject]] = [SceneObject()]
        self.camera: SceneCamera = SceneCamera()
        self._click_handler = on_click
        self._drag_start_handler = on_drag_start
        self._drag_end_handler = on_drag_end
        self.is_initialized = False
        self.on('init', self._handle_init)
        self.on('click3d', self._handle_click)
        self.on('dragstart', self._handle_drag)
        self.on('dragend', self._handle_drag)
        self._props['drag_constraints'] = drag_constraints

    def __enter__(self) -> Self:
        if False:
            for i in range(10):
                print('nop')
        Object3D.current_scene = self
        super().__enter__()
        return self

    def __getattribute__(self, name: str) -> Any:
        if False:
            while True:
                i = 10
        attribute = super().__getattribute__(name)
        if isinstance(attribute, type) and issubclass(attribute, Object3D):
            Object3D.current_scene = self
        return attribute

    def _handle_init(self, e: GenericEventArguments) -> None:
        if False:
            return 10
        self.is_initialized = True
        with self.client.individual_target(e.args['socket_id']):
            self.move_camera(duration=0)
            for obj in self.objects.values():
                obj.send()

    def run_method(self, name: str, *args: Any, timeout: float=1, check_interval: float=0.01) -> AwaitableResponse:
        if False:
            i = 10
            return i + 15
        'Run a method on the client.\n\n        :param name: name of the method\n        :param args: arguments to pass to the method\n        '
        if not self.is_initialized:
            return NullResponse()
        return super().run_method(name, *args, timeout=timeout, check_interval=check_interval)

    def _handle_click(self, e: GenericEventArguments) -> None:
        if False:
            print('Hello World!')
        arguments = SceneClickEventArguments(sender=self, client=self.client, click_type=e.args['click_type'], button=e.args['button'], alt=e.args['alt_key'], ctrl=e.args['ctrl_key'], meta=e.args['meta_key'], shift=e.args['shift_key'], hits=[SceneClickHit(object_id=hit['object_id'], object_name=hit['object_name'], x=hit['point']['x'], y=hit['point']['y'], z=hit['point']['z']) for hit in e.args['hits']])
        handle_event(self._click_handler, arguments)

    def _handle_drag(self, e: GenericEventArguments) -> None:
        if False:
            return 10
        arguments = SceneDragEventArguments(sender=self, client=self.client, type=e.args['type'], object_id=e.args['object_id'], object_name=e.args['object_name'], x=e.args['x'], y=e.args['y'], z=e.args['z'])
        if arguments.type == 'dragend':
            self.objects[arguments.object_id].move(arguments.x, arguments.y, arguments.z)
        handle_event(self._drag_start_handler if arguments.type == 'dragstart' else self._drag_end_handler, arguments)

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        return len(self.objects)

    def move_camera(self, x: Optional[float]=None, y: Optional[float]=None, z: Optional[float]=None, look_at_x: Optional[float]=None, look_at_y: Optional[float]=None, look_at_z: Optional[float]=None, up_x: Optional[float]=None, up_y: Optional[float]=None, up_z: Optional[float]=None, duration: float=0.5) -> None:
        if False:
            print('Hello World!')
        'Move the camera to a new position.\n\n        :param x: camera x position\n        :param y: camera y position\n        :param z: camera z position\n        :param look_at_x: camera look-at x position\n        :param look_at_y: camera look-at y position\n        :param look_at_z: camera look-at z position\n        :param up_x: x component of the camera up vector\n        :param up_y: y component of the camera up vector\n        :param up_z: z component of the camera up vector\n        :param duration: duration of the movement in seconds (default: `0.5`)\n        '
        self.camera.x = self.camera.x if x is None else x
        self.camera.y = self.camera.y if y is None else y
        self.camera.z = self.camera.z if z is None else z
        self.camera.look_at_x = self.camera.look_at_x if look_at_x is None else look_at_x
        self.camera.look_at_y = self.camera.look_at_y if look_at_y is None else look_at_y
        self.camera.look_at_z = self.camera.look_at_z if look_at_z is None else look_at_z
        self.camera.up_x = self.camera.up_x if up_x is None else up_x
        self.camera.up_y = self.camera.up_y if up_y is None else up_y
        self.camera.up_z = self.camera.up_z if up_z is None else up_z
        self.run_method('move_camera', self.camera.x, self.camera.y, self.camera.z, self.camera.look_at_x, self.camera.look_at_y, self.camera.look_at_z, self.camera.up_x, self.camera.up_y, self.camera.up_z, duration)

    def _handle_delete(self) -> None:
        if False:
            while True:
                i = 10
        binding.remove(list(self.objects.values()), Object3D)
        super()._handle_delete()

    def delete_objects(self, predicate: Callable[[Object3D], bool]=lambda _: True) -> None:
        if False:
            i = 10
            return i + 15
        'Remove objects from the scene.\n\n        :param predicate: function which returns `True` for objects which should be deleted\n        '
        for obj in list(self.objects.values()):
            if predicate(obj):
                obj.delete()

    def clear(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Remove all objects from the scene.'
        super().clear()
        self.delete_objects()