from __future__ import annotations
from attrs import define, field
from .. import components
from .._baseclasses import Archetype
from .boxes3d_ext import Boxes3DExt
__all__ = ['Boxes3D']

@define(str=False, repr=False, init=False)
class Boxes3D(Boxes3DExt, Archetype):
    """
    **Archetype**: 3D boxes with half-extents and optional center, rotations, rotations, colors etc.

    Example
    -------
    ### Batch of 3D boxes:
    ```python
    import rerun as rr
    from rerun.datatypes import Angle, Quaternion, Rotation3D, RotationAxisAngle

    rr.init("rerun_example_box3d_batch", spawn=True)

    rr.log(
        "batch",
        rr.Boxes3D(
            centers=[[2, 0, 0], [-2, 0, 0], [0, 0, 2]],
            half_sizes=[[2.0, 2.0, 1.0], [1.0, 1.0, 0.5], [2.0, 0.5, 1.0]],
            rotations=[
                Rotation3D.identity(),
                Quaternion(xyzw=[0.0, 0.0, 0.382683, 0.923880]),  # 45 degrees around Z
                RotationAxisAngle(axis=[0, 1, 0], angle=Angle(deg=30)),
            ],
            radii=0.025,
            colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
            labels=["red", "green", "blue"],
        ),
    )
    ```
    <center>
    <picture>
      <source media="(max-width: 480px)" srcset="https://static.rerun.io/box3d_batch/6d3e453c3a0201ae42bbae9de941198513535f1d/480w.png">
      <source media="(max-width: 768px)" srcset="https://static.rerun.io/box3d_batch/6d3e453c3a0201ae42bbae9de941198513535f1d/768w.png">
      <source media="(max-width: 1024px)" srcset="https://static.rerun.io/box3d_batch/6d3e453c3a0201ae42bbae9de941198513535f1d/1024w.png">
      <source media="(max-width: 1200px)" srcset="https://static.rerun.io/box3d_batch/6d3e453c3a0201ae42bbae9de941198513535f1d/1200w.png">
      <img src="https://static.rerun.io/box3d_batch/6d3e453c3a0201ae42bbae9de941198513535f1d/full.png" width="640">
    </picture>
    </center>
    """

    def __attrs_clear__(self) -> None:
        if False:
            while True:
                i = 10
        'Convenience method for calling `__attrs_init__` with all `None`s.'
        self.__attrs_init__(half_sizes=None, centers=None, rotations=None, colors=None, radii=None, labels=None, class_ids=None, instance_keys=None)

    @classmethod
    def _clear(cls) -> Boxes3D:
        if False:
            i = 10
            return i + 15
        'Produce an empty Boxes3D, bypassing `__init__`.'
        inst = cls.__new__(cls)
        inst.__attrs_clear__()
        return inst
    half_sizes: components.HalfSizes3DBatch = field(metadata={'component': 'required'}, converter=components.HalfSizes3DBatch._required)
    centers: components.Position3DBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.Position3DBatch._optional)
    rotations: components.Rotation3DBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.Rotation3DBatch._optional)
    colors: components.ColorBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.ColorBatch._optional)
    radii: components.RadiusBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.RadiusBatch._optional)
    labels: components.TextBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.TextBatch._optional)
    class_ids: components.ClassIdBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.ClassIdBatch._optional)
    instance_keys: components.InstanceKeyBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.InstanceKeyBatch._optional)
    __str__ = Archetype.__str__
    __repr__ = Archetype.__repr__