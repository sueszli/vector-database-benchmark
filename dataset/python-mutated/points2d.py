from __future__ import annotations
from attrs import define, field
from .. import components
from .._baseclasses import Archetype
from .points2d_ext import Points2DExt
__all__ = ['Points2D']

@define(str=False, repr=False, init=False)
class Points2D(Points2DExt, Archetype):
    """
    **Archetype**: A 2D point cloud with positions and optional colors, radii, labels, etc.

    Example
    -------
    ### Randomly distributed 2D points with varying color and radius:
    ```python
    import rerun as rr
    from numpy.random import default_rng

    rr.init("rerun_example_points2d", spawn=True)
    rng = default_rng(12345)

    positions = rng.uniform(-3, 3, size=[10, 2])
    colors = rng.uniform(0, 255, size=[10, 4])
    radii = rng.uniform(0, 1, size=[10])

    rr.log("random", rr.Points2D(positions, colors=colors, radii=radii))

    # Log an extra rect to set the view bounds
    rr.log("bounds", rr.Boxes2D(half_sizes=[4, 3]))
    ```
    <center>
    <picture>
      <source media="(max-width: 480px)" srcset="https://static.rerun.io/point2d_random/8e8ac75373677bd72bd3f56a15e44fcab309a168/480w.png">
      <source media="(max-width: 768px)" srcset="https://static.rerun.io/point2d_random/8e8ac75373677bd72bd3f56a15e44fcab309a168/768w.png">
      <source media="(max-width: 1024px)" srcset="https://static.rerun.io/point2d_random/8e8ac75373677bd72bd3f56a15e44fcab309a168/1024w.png">
      <source media="(max-width: 1200px)" srcset="https://static.rerun.io/point2d_random/8e8ac75373677bd72bd3f56a15e44fcab309a168/1200w.png">
      <img src="https://static.rerun.io/point2d_random/8e8ac75373677bd72bd3f56a15e44fcab309a168/full.png" width="640">
    </picture>
    </center>
    """

    def __attrs_clear__(self) -> None:
        if False:
            while True:
                i = 10
        'Convenience method for calling `__attrs_init__` with all `None`s.'
        self.__attrs_init__(positions=None, radii=None, colors=None, labels=None, draw_order=None, class_ids=None, keypoint_ids=None, instance_keys=None)

    @classmethod
    def _clear(cls) -> Points2D:
        if False:
            i = 10
            return i + 15
        'Produce an empty Points2D, bypassing `__init__`.'
        inst = cls.__new__(cls)
        inst.__attrs_clear__()
        return inst
    positions: components.Position2DBatch = field(metadata={'component': 'required'}, converter=components.Position2DBatch._required)
    radii: components.RadiusBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.RadiusBatch._optional)
    colors: components.ColorBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.ColorBatch._optional)
    labels: components.TextBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.TextBatch._optional)
    draw_order: components.DrawOrderBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.DrawOrderBatch._optional)
    class_ids: components.ClassIdBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.ClassIdBatch._optional)
    keypoint_ids: components.KeypointIdBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.KeypointIdBatch._optional)
    instance_keys: components.InstanceKeyBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.InstanceKeyBatch._optional)
    __str__ = Archetype.__str__
    __repr__ = Archetype.__repr__