from __future__ import annotations
from attrs import define, field
from .. import components
from .._baseclasses import Archetype
from .points3d_ext import Points3DExt
__all__ = ['Points3D']

@define(str=False, repr=False, init=False)
class Points3D(Points3DExt, Archetype):
    """
    **Archetype**: A 3D point cloud with positions and optional colors, radii, labels, etc.

    Example
    -------
    ### Randomly distributed 3D points with varying color and radius:
    ```python
    import rerun as rr
    from numpy.random import default_rng

    rr.init("rerun_example_points3d_random", spawn=True)
    rng = default_rng(12345)

    positions = rng.uniform(-5, 5, size=[10, 3])
    colors = rng.uniform(0, 255, size=[10, 3])
    radii = rng.uniform(0, 1, size=[10])

    rr.log("random", rr.Points3D(positions, colors=colors, radii=radii))
    ```
    <center>
    <picture>
      <source media="(max-width: 480px)" srcset="https://static.rerun.io/point3d_random/7e94e1806d2c381943748abbb3bedb68d564de24/480w.png">
      <source media="(max-width: 768px)" srcset="https://static.rerun.io/point3d_random/7e94e1806d2c381943748abbb3bedb68d564de24/768w.png">
      <source media="(max-width: 1024px)" srcset="https://static.rerun.io/point3d_random/7e94e1806d2c381943748abbb3bedb68d564de24/1024w.png">
      <source media="(max-width: 1200px)" srcset="https://static.rerun.io/point3d_random/7e94e1806d2c381943748abbb3bedb68d564de24/1200w.png">
      <img src="https://static.rerun.io/point3d_random/7e94e1806d2c381943748abbb3bedb68d564de24/full.png" width="640">
    </picture>
    </center>
    """

    def __attrs_clear__(self) -> None:
        if False:
            i = 10
            return i + 15
        'Convenience method for calling `__attrs_init__` with all `None`s.'
        self.__attrs_init__(positions=None, radii=None, colors=None, labels=None, class_ids=None, keypoint_ids=None, instance_keys=None)

    @classmethod
    def _clear(cls) -> Points3D:
        if False:
            return 10
        'Produce an empty Points3D, bypassing `__init__`.'
        inst = cls.__new__(cls)
        inst.__attrs_clear__()
        return inst
    positions: components.Position3DBatch = field(metadata={'component': 'required'}, converter=components.Position3DBatch._required)
    radii: components.RadiusBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.RadiusBatch._optional)
    colors: components.ColorBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.ColorBatch._optional)
    labels: components.TextBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.TextBatch._optional)
    class_ids: components.ClassIdBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.ClassIdBatch._optional)
    keypoint_ids: components.KeypointIdBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.KeypointIdBatch._optional)
    instance_keys: components.InstanceKeyBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.InstanceKeyBatch._optional)
    __str__ = Archetype.__str__
    __repr__ = Archetype.__repr__