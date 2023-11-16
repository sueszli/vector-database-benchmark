from __future__ import annotations
from attrs import define, field
from .. import components
from .._baseclasses import Archetype
from .arrows3d_ext import Arrows3DExt
__all__ = ['Arrows3D']

@define(str=False, repr=False, init=False)
class Arrows3D(Arrows3DExt, Archetype):
    """
    **Archetype**: 3D arrows with optional colors, radii, labels, etc.

    Example
    -------
    ### Simple batch of 3D Arrows:
    ```python
    from math import tau

    import numpy as np
    import rerun as rr

    rr.init("rerun_example_arrow3d", spawn=True)

    lengths = np.log2(np.arange(0, 100) + 1)
    angles = np.arange(start=0, stop=tau, step=tau * 0.01)
    origins = np.zeros((100, 3))
    vectors = np.column_stack([np.sin(angles) * lengths, np.zeros(100), np.cos(angles) * lengths])
    colors = [[1.0 - c, c, 0.5, 0.5] for c in angles / tau]

    rr.log("arrows", rr.Arrows3D(origins=origins, vectors=vectors, colors=colors))
    ```
    <center>
    <picture>
      <source media="(max-width: 480px)" srcset="https://static.rerun.io/arrow3d_simple/55e2f794a520bbf7527d7b828b0264732146c5d0/480w.png">
      <source media="(max-width: 768px)" srcset="https://static.rerun.io/arrow3d_simple/55e2f794a520bbf7527d7b828b0264732146c5d0/768w.png">
      <source media="(max-width: 1024px)" srcset="https://static.rerun.io/arrow3d_simple/55e2f794a520bbf7527d7b828b0264732146c5d0/1024w.png">
      <source media="(max-width: 1200px)" srcset="https://static.rerun.io/arrow3d_simple/55e2f794a520bbf7527d7b828b0264732146c5d0/1200w.png">
      <img src="https://static.rerun.io/arrow3d_simple/55e2f794a520bbf7527d7b828b0264732146c5d0/full.png" width="640">
    </picture>
    </center>
    """

    def __attrs_clear__(self) -> None:
        if False:
            return 10
        'Convenience method for calling `__attrs_init__` with all `None`s.'
        self.__attrs_init__(vectors=None, origins=None, radii=None, colors=None, labels=None, class_ids=None, instance_keys=None)

    @classmethod
    def _clear(cls) -> Arrows3D:
        if False:
            for i in range(10):
                print('nop')
        'Produce an empty Arrows3D, bypassing `__init__`.'
        inst = cls.__new__(cls)
        inst.__attrs_clear__()
        return inst
    vectors: components.Vector3DBatch = field(metadata={'component': 'required'}, converter=components.Vector3DBatch._required)
    origins: components.Position3DBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.Position3DBatch._optional)
    radii: components.RadiusBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.RadiusBatch._optional)
    colors: components.ColorBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.ColorBatch._optional)
    labels: components.TextBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.TextBatch._optional)
    class_ids: components.ClassIdBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.ClassIdBatch._optional)
    instance_keys: components.InstanceKeyBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.InstanceKeyBatch._optional)
    __str__ = Archetype.__str__
    __repr__ = Archetype.__repr__