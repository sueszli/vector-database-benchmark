from __future__ import annotations
import itertools
from typing import Any, Optional, cast
import rerun as rr
from rerun.components import InstanceKeyArrayLike, MaterialBatch, MeshPropertiesBatch, Position3DBatch, Vector3DBatch
from rerun.datatypes import ClassIdArrayLike, Material, MaterialLike, MeshProperties, MeshPropertiesLike, Rgba32ArrayLike, Vec3DArrayLike
from .common_arrays import class_ids_arrays, class_ids_expected, colors_arrays, colors_expected, instance_keys_arrays, instance_keys_expected, none_empty_or_value, vec3ds_arrays, vec3ds_expected
mesh_properties_objects: list[MeshPropertiesLike | None] = [None, MeshProperties(indices=[1, 2, 3, 4, 5, 6])]

def mesh_properties_expected(obj: Any) -> Any:
    if False:
        i = 10
        return i + 15
    expected = none_empty_or_value(obj, MeshProperties(indices=[1, 2, 3, 4, 5, 6]))
    return MeshPropertiesBatch._optional(expected)
mesh_materials: list[MaterialLike | None] = [None, Material(albedo_factor=2852126924)]

def mesh_material_expected(obj: Any) -> Any:
    if False:
        i = 10
        return i + 15
    expected = none_empty_or_value(obj, Material(albedo_factor=2852126924))
    return MaterialBatch._optional(expected)

def test_mesh3d() -> None:
    if False:
        return 10
    vertex_positions_arrays = vec3ds_arrays
    vertex_normals_arrays = vec3ds_arrays
    vertex_colors_arrays = colors_arrays
    all_arrays = itertools.zip_longest(vertex_positions_arrays, vertex_normals_arrays, vertex_colors_arrays, mesh_properties_objects, mesh_materials, class_ids_arrays, instance_keys_arrays)
    for (vertex_positions, vertex_normals, vertex_colors, mesh_properties, mesh_material, class_ids, instance_keys) in all_arrays:
        vertex_positions = vertex_positions if vertex_positions is not None else vertex_positions_arrays[-1]
        vertex_positions = cast(Vec3DArrayLike, vertex_positions)
        vertex_normals = cast(Optional[Vec3DArrayLike], vertex_normals)
        vertex_colors = cast(Optional[Rgba32ArrayLike], vertex_colors)
        mesh_properties = cast(Optional[MeshPropertiesLike], mesh_properties)
        mesh_material = cast(Optional[MaterialLike], mesh_material)
        class_ids = cast(Optional[ClassIdArrayLike], class_ids)
        instance_keys = cast(Optional[InstanceKeyArrayLike], instance_keys)
        print(f'E: rr.Mesh3D(\n    vertex_normals={vertex_positions}\n    vertex_normals={vertex_normals}\n    vertex_colors={vertex_colors}\n    mesh_properties={mesh_properties_objects}\n    mesh_material={mesh_material}\n    class_ids={class_ids}\n    instance_keys={instance_keys}\n)')
        arch = rr.Mesh3D(vertex_positions=vertex_positions, vertex_normals=vertex_normals, vertex_colors=vertex_colors, mesh_properties=mesh_properties, mesh_material=mesh_material, class_ids=class_ids, instance_keys=instance_keys)
        print(f'A: {arch}\n')
        assert arch.vertex_positions == vec3ds_expected(vertex_positions, Position3DBatch)
        assert arch.vertex_normals == vec3ds_expected(vertex_normals, Vector3DBatch)
        assert arch.vertex_colors == colors_expected(vertex_colors)
        assert arch.mesh_properties == mesh_properties_expected(mesh_properties)
        assert arch.mesh_material == mesh_material_expected(mesh_material)
        assert arch.class_ids == class_ids_expected(class_ids)
        assert arch.instance_keys == instance_keys_expected(instance_keys)

def test_nullable_albedo_factor() -> None:
    if False:
        return 10
    assert len(MaterialBatch([Material(albedo_factor=[204, 0, 204, 255]), Material()])) == 2

def test_nullable_indices() -> None:
    if False:
        for i in range(10):
            print('nop')
    assert len(MeshPropertiesBatch([MeshProperties(indices=[1, 2, 3, 4, 5, 6]), MeshProperties()])) == 2

def test_indices_parameter() -> None:
    if False:
        print('Hello World!')
    assert rr.Mesh3D(vertex_positions=[(0, 0, 0)] * 3, indices=[0, 1, 2]) == rr.Mesh3D(vertex_positions=[(0, 0, 0)] * 3, mesh_properties=MeshProperties(indices=[0, 1, 2]))
if __name__ == '__main__':
    test_mesh3d()