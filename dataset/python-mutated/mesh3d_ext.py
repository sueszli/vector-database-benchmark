from __future__ import annotations
from typing import Any
import numpy.typing as npt
from .. import components, datatypes
from ..error_utils import catch_and_log_exceptions

class Mesh3DExt:
    """Extension for [Mesh3D][rerun.archetypes.Mesh3D]."""

    def __init__(self: Any, *, vertex_positions: datatypes.Vec3DArrayLike, indices: npt.ArrayLike | None=None, mesh_properties: datatypes.MeshPropertiesLike | None=None, vertex_normals: datatypes.Vec3DArrayLike | None=None, vertex_colors: datatypes.Rgba32ArrayLike | None=None, mesh_material: datatypes.MaterialLike | None=None, class_ids: datatypes.ClassIdArrayLike | None=None, instance_keys: components.InstanceKeyArrayLike | None=None):
        if False:
            while True:
                i = 10
        "\n        Create a new instance of the Mesh3D archetype.\n\n        Parameters\n        ----------\n        vertex_positions:\n            The positions of each vertex.\n            If no `indices` are specified, then each triplet of positions is interpreted as a triangle.\n        indices:\n            If specified, a flattened array of indices that describe the mesh's triangles,\n            i.e. its length must be divisible by 3.\n            Mutually exclusive with `mesh_properties`.\n        mesh_properties:\n            Optional properties for the mesh as a whole (including indexed drawing).\n            Mutually exclusive with `indices`.\n        vertex_normals:\n            An optional normal for each vertex.\n            If specified, this must have as many elements as `vertex_positions`.\n        vertex_colors:\n            An optional color for each vertex.\n        mesh_material:\n            Optional material properties for the mesh as a whole.\n        class_ids:\n            Optional class Ids for the vertices.\n            The class ID provides colors and labels if not specified explicitly.\n        instance_keys:\n            Unique identifiers for each individual vertex in the mesh.\n        "
        with catch_and_log_exceptions(context=self.__class__.__name__):
            if indices is not None:
                if mesh_properties is not None:
                    raise ValueError('indices and mesh_properties are mutually exclusive')
                mesh_properties = datatypes.MeshProperties(indices=indices)
            self.__attrs_init__(vertex_positions=vertex_positions, mesh_properties=mesh_properties, vertex_normals=vertex_normals, vertex_colors=vertex_colors, mesh_material=mesh_material, class_ids=class_ids, instance_keys=instance_keys)
            return
        self.__attrs_clear__()