from __future__ import annotations
from typing import Any
from attrs import define, field
from .. import components, datatypes
from .._baseclasses import Archetype
from ..error_utils import catch_and_log_exceptions
from .segmentation_image_ext import SegmentationImageExt
__all__ = ['SegmentationImage']

@define(str=False, repr=False, init=False)
class SegmentationImage(SegmentationImageExt, Archetype):
    """
    **Archetype**: An image made up of integer class-ids.

    The shape of the `TensorData` must be mappable to an `HxW` tensor.
    Each pixel corresponds to a depth value in units specified by meter.

    Leading and trailing unit-dimensions are ignored, so that
    `1x640x480x1` is treated as a `640x480` image.

    See also [`AnnotationContext`][rerun.archetypes.AnnotationContext] to associate each class with a color and a label.

    Example
    -------
    ### Simple segmentation image:
    ```python
    import numpy as np
    import rerun as rr

    # Create a segmentation image
    image = np.zeros((8, 12), dtype=np.uint8)
    image[0:4, 0:6] = 1
    image[4:8, 6:12] = 2

    rr.init("rerun_example_segmentation_image", spawn=True)

    # Assign a label and color to each class
    rr.log("/", rr.AnnotationContext([(1, "red", (255, 0, 0)), (2, "green", (0, 255, 0))]), timeless=True)

    rr.log("image", rr.SegmentationImage(image))
    ```
    <center>
    <picture>
      <source media="(max-width: 480px)" srcset="https://static.rerun.io/segmentation_image_simple/eb49e0b8cb870c75a69e2a47a2d202e5353115f6/480w.png">
      <source media="(max-width: 768px)" srcset="https://static.rerun.io/segmentation_image_simple/eb49e0b8cb870c75a69e2a47a2d202e5353115f6/768w.png">
      <source media="(max-width: 1024px)" srcset="https://static.rerun.io/segmentation_image_simple/eb49e0b8cb870c75a69e2a47a2d202e5353115f6/1024w.png">
      <source media="(max-width: 1200px)" srcset="https://static.rerun.io/segmentation_image_simple/eb49e0b8cb870c75a69e2a47a2d202e5353115f6/1200w.png">
      <img src="https://static.rerun.io/segmentation_image_simple/eb49e0b8cb870c75a69e2a47a2d202e5353115f6/full.png" width="640">
    </picture>
    </center>
    """

    def __init__(self: Any, data: datatypes.TensorDataLike, *, draw_order: components.DrawOrderLike | None=None):
        if False:
            i = 10
            return i + 15
        '\n        Create a new instance of the SegmentationImage archetype.\n\n        Parameters\n        ----------\n        data:\n            The image data. Should always be a rank-2 tensor.\n        draw_order:\n            An optional floating point value that specifies the 2D drawing order.\n\n            Objects with higher values are drawn on top of those with lower values.\n        '
        with catch_and_log_exceptions(context=self.__class__.__name__):
            self.__attrs_init__(data=data, draw_order=draw_order)
            return
        self.__attrs_clear__()

    def __attrs_clear__(self) -> None:
        if False:
            return 10
        'Convenience method for calling `__attrs_init__` with all `None`s.'
        self.__attrs_init__(data=None, draw_order=None)

    @classmethod
    def _clear(cls) -> SegmentationImage:
        if False:
            return 10
        'Produce an empty SegmentationImage, bypassing `__init__`.'
        inst = cls.__new__(cls)
        inst.__attrs_clear__()
        return inst
    data: components.TensorDataBatch = field(metadata={'component': 'required'}, converter=SegmentationImageExt.data__field_converter_override)
    draw_order: components.DrawOrderBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.DrawOrderBatch._optional)
    __str__ = Archetype.__str__
    __repr__ = Archetype.__repr__