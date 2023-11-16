from __future__ import annotations
from typing import Any
from .. import components, datatypes
from ..error_utils import catch_and_log_exceptions

class Arrows3DExt:
    """Extension for [Arrows3D][rerun.archetypes.Arrows3D]."""

    def __init__(self: Any, *, vectors: datatypes.Vec3DArrayLike, origins: datatypes.Vec3DArrayLike | None=None, radii: components.RadiusArrayLike | None=None, colors: datatypes.Rgba32ArrayLike | None=None, labels: datatypes.Utf8ArrayLike | None=None, class_ids: datatypes.ClassIdArrayLike | None=None, instance_keys: components.InstanceKeyArrayLike | None=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Create a new instance of the Arrows3D archetype.\n\n        Parameters\n        ----------\n        vectors:\n            All the vectors for each arrow in the batch.\n        origins:\n            All the origin points for each arrow in the batch.\n\n            If no origins are set, (0, 0, 0) is used as the origin for each arrow.\n        radii:\n            Optional radii for the arrows.\n\n            The shaft is rendered as a line with `radius = 0.5 * radius`.\n            The tip is rendered with `height = 2.0 * radius` and `radius = 1.0 * radius`.\n        colors:\n            Optional colors for the points.\n        labels:\n            Optional text labels for the arrows.\n        class_ids:\n            Optional class Ids for the points.\n\n            The class ID provides colors and labels if not specified explicitly.\n        instance_keys:\n            Unique identifiers for each individual point in the batch.\n        '
        with catch_and_log_exceptions(context=self.__class__.__name__):
            self.__attrs_init__(vectors=vectors, origins=origins, radii=radii, colors=colors, labels=labels, class_ids=class_ids, instance_keys=instance_keys)
            return
        self.__attrs_clear__()