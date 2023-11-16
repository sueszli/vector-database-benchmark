from __future__ import annotations
from typing import Any
from attrs import define, field
from .. import components, datatypes
from .._baseclasses import Archetype
from ..error_utils import catch_and_log_exceptions
__all__ = ['LineStrips3D']

@define(str=False, repr=False, init=False)
class LineStrips3D(Archetype):
    """
    **Archetype**: 3D line strips with positions and optional colors, radii, labels, etc.

    Example
    -------
    ### Many strips:
    ```python
    import rerun as rr

    rr.init("rerun_example_line_strip3d", spawn=True)

    rr.log(
        "strips",
        rr.LineStrips3D(
            [
                [
                    [0, 0, 2],
                    [1, 0, 2],
                    [1, 1, 2],
                    [0, 1, 2],
                ],
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                ],
            ],
            colors=[[255, 0, 0], [0, 255, 0]],
            radii=[0.025, 0.005],
            labels=["one strip here", "and one strip there"],
        ),
    )
    ```
    <center>
    <picture>
      <source media="(max-width: 480px)" srcset="https://static.rerun.io/line_strip3d_batch/102e5ec5271475657fbc76b469267e4ec8e84337/480w.png">
      <source media="(max-width: 768px)" srcset="https://static.rerun.io/line_strip3d_batch/102e5ec5271475657fbc76b469267e4ec8e84337/768w.png">
      <source media="(max-width: 1024px)" srcset="https://static.rerun.io/line_strip3d_batch/102e5ec5271475657fbc76b469267e4ec8e84337/1024w.png">
      <source media="(max-width: 1200px)" srcset="https://static.rerun.io/line_strip3d_batch/102e5ec5271475657fbc76b469267e4ec8e84337/1200w.png">
      <img src="https://static.rerun.io/line_strip3d_batch/102e5ec5271475657fbc76b469267e4ec8e84337/full.png" width="640">
    </picture>
    </center>
    """

    def __init__(self: Any, strips: components.LineStrip3DArrayLike, *, radii: components.RadiusArrayLike | None=None, colors: datatypes.Rgba32ArrayLike | None=None, labels: datatypes.Utf8ArrayLike | None=None, class_ids: datatypes.ClassIdArrayLike | None=None, instance_keys: components.InstanceKeyArrayLike | None=None):
        if False:
            while True:
                i = 10
        '\n        Create a new instance of the LineStrips3D archetype.\n\n        Parameters\n        ----------\n        strips:\n            All the actual 3D line strips that make up the batch.\n        radii:\n            Optional radii for the line strips.\n        colors:\n            Optional colors for the line strips.\n        labels:\n            Optional text labels for the line strips.\n        class_ids:\n            Optional `ClassId`s for the lines.\n\n            The class ID provides colors and labels if not specified explicitly.\n        instance_keys:\n            Unique identifiers for each individual line strip in the batch.\n        '
        with catch_and_log_exceptions(context=self.__class__.__name__):
            self.__attrs_init__(strips=strips, radii=radii, colors=colors, labels=labels, class_ids=class_ids, instance_keys=instance_keys)
            return
        self.__attrs_clear__()

    def __attrs_clear__(self) -> None:
        if False:
            i = 10
            return i + 15
        'Convenience method for calling `__attrs_init__` with all `None`s.'
        self.__attrs_init__(strips=None, radii=None, colors=None, labels=None, class_ids=None, instance_keys=None)

    @classmethod
    def _clear(cls) -> LineStrips3D:
        if False:
            while True:
                i = 10
        'Produce an empty LineStrips3D, bypassing `__init__`.'
        inst = cls.__new__(cls)
        inst.__attrs_clear__()
        return inst
    strips: components.LineStrip3DBatch = field(metadata={'component': 'required'}, converter=components.LineStrip3DBatch._required)
    radii: components.RadiusBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.RadiusBatch._optional)
    colors: components.ColorBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.ColorBatch._optional)
    labels: components.TextBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.TextBatch._optional)
    class_ids: components.ClassIdBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.ClassIdBatch._optional)
    instance_keys: components.InstanceKeyBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.InstanceKeyBatch._optional)
    __str__ = Archetype.__str__
    __repr__ = Archetype.__repr__