from __future__ import annotations
import itertools
from typing import Optional, cast
import rerun as rr
from rerun.components import InstanceKeyArrayLike, Position3DBatch, RadiusArrayLike, Vector3DBatch
from rerun.datatypes import ClassIdArrayLike, Rgba32ArrayLike, Utf8ArrayLike, Vec3DArrayLike
from .common_arrays import class_ids_arrays, class_ids_expected, colors_arrays, colors_expected, instance_keys_arrays, instance_keys_expected, labels_arrays, labels_expected, radii_arrays, radii_expected, vec3ds_arrays, vec3ds_expected

def test_arrows3d() -> None:
    if False:
        while True:
            i = 10
    vectors_arrays = vec3ds_arrays
    origins_arrays = vec3ds_arrays
    all_arrays = itertools.zip_longest(vectors_arrays, origins_arrays, radii_arrays, colors_arrays, labels_arrays, class_ids_arrays, instance_keys_arrays)
    for (vectors, origins, radii, colors, labels, class_ids, instance_keys) in all_arrays:
        vectors = vectors if vectors is not None else vectors_arrays[-1]
        origins = origins if origins is not None else origins_arrays[-1]
        vectors = cast(Vec3DArrayLike, vectors)
        origins = cast(Optional[Vec3DArrayLike], origins)
        radii = cast(Optional[RadiusArrayLike], radii)
        colors = cast(Optional[Rgba32ArrayLike], colors)
        labels = cast(Optional[Utf8ArrayLike], labels)
        class_ids = cast(Optional[ClassIdArrayLike], class_ids)
        instance_keys = cast(Optional[InstanceKeyArrayLike], instance_keys)
        print(f'E: rr.Arrows3D(\n    vectors={vectors}\n    origins={origins!r}\n    radii={radii!r}\n    colors={colors!r}\n    labels={labels!r}\n    class_ids={class_ids!r}\n    instance_keys={instance_keys!r}\n)')
        arch = rr.Arrows3D(vectors=vectors, origins=origins, radii=radii, colors=colors, labels=labels, class_ids=class_ids, instance_keys=instance_keys)
        print(f'A: {arch}\n')
        assert arch.vectors == vec3ds_expected(vectors, Vector3DBatch)
        assert arch.origins == vec3ds_expected(origins, Position3DBatch)
        assert arch.radii == radii_expected(radii)
        assert arch.colors == colors_expected(colors)
        assert arch.labels == labels_expected(labels)
        assert arch.class_ids == class_ids_expected(class_ids)
        assert arch.instance_keys == instance_keys_expected(instance_keys)
if __name__ == '__main__':
    test_arrows3d()