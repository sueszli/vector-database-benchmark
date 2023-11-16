from __future__ import annotations
from attrs import define, field
from .. import components
from .._baseclasses import Archetype
from .boxes2d_ext import Boxes2DExt
__all__ = ['Boxes2D']

@define(str=False, repr=False, init=False)
class Boxes2D(Boxes2DExt, Archetype):
    """
    **Archetype**: 2D boxes with half-extents and optional center, rotations, rotations, colors etc.

    Example
    -------
    ### Simple 2D boxes:
    ```python
    import rerun as rr

    rr.init("rerun_example_box2d", spawn=True)

    rr.log("simple", rr.Boxes2D(mins=[-1, -1], sizes=[2, 2]))

    # Log an extra rect to set the view bounds
    rr.log("bounds", rr.Boxes2D(sizes=[4.0, 3.0]))
    ```
    <center>
    <picture>
      <source media="(max-width: 480px)" srcset="https://static.rerun.io/box2d_simple/ac4424f3cf747382867649610cbd749c45b2020b/480w.png">
      <source media="(max-width: 768px)" srcset="https://static.rerun.io/box2d_simple/ac4424f3cf747382867649610cbd749c45b2020b/768w.png">
      <source media="(max-width: 1024px)" srcset="https://static.rerun.io/box2d_simple/ac4424f3cf747382867649610cbd749c45b2020b/1024w.png">
      <source media="(max-width: 1200px)" srcset="https://static.rerun.io/box2d_simple/ac4424f3cf747382867649610cbd749c45b2020b/1200w.png">
      <img src="https://static.rerun.io/box2d_simple/ac4424f3cf747382867649610cbd749c45b2020b/full.png" width="640">
    </picture>
    </center>
    """

    def __attrs_clear__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Convenience method for calling `__attrs_init__` with all `None`s.'
        self.__attrs_init__(half_sizes=None, centers=None, colors=None, radii=None, labels=None, draw_order=None, class_ids=None, instance_keys=None)

    @classmethod
    def _clear(cls) -> Boxes2D:
        if False:
            for i in range(10):
                print('nop')
        'Produce an empty Boxes2D, bypassing `__init__`.'
        inst = cls.__new__(cls)
        inst.__attrs_clear__()
        return inst
    half_sizes: components.HalfSizes2DBatch = field(metadata={'component': 'required'}, converter=components.HalfSizes2DBatch._required)
    centers: components.Position2DBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.Position2DBatch._optional)
    colors: components.ColorBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.ColorBatch._optional)
    radii: components.RadiusBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.RadiusBatch._optional)
    labels: components.TextBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.TextBatch._optional)
    draw_order: components.DrawOrderBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.DrawOrderBatch._optional)
    class_ids: components.ClassIdBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.ClassIdBatch._optional)
    instance_keys: components.InstanceKeyBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.InstanceKeyBatch._optional)
    __str__ = Archetype.__str__
    __repr__ = Archetype.__repr__