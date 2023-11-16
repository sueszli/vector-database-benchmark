from __future__ import annotations
from attrs import define, field
from .. import components
from .._baseclasses import Archetype
from .asset3d_ext import Asset3DExt
__all__ = ['Asset3D']

@define(str=False, repr=False, init=False)
class Asset3D(Asset3DExt, Archetype):
    """
    **Archetype**: A prepacked 3D asset (`.gltf`, `.glb`, `.obj`, etc.).

    See also [`Mesh3D`][rerun.archetypes.Mesh3D].

    Example
    -------
    ### Simple 3D asset:
    ```python
    import sys

    import rerun as rr

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_asset.[gltf|glb|obj]>")
        sys.exit(1)

    rr.init("rerun_example_asset3d_simple", spawn=True)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)  # Set an up-axis
    rr.log("world/asset", rr.Asset3D(path=sys.argv[1]))
    ```
    <center>
    <picture>
      <source media="(max-width: 480px)" srcset="https://static.rerun.io/asset3d_simple/af238578188d3fd0de3e330212120e2842a8ddb2/480w.png">
      <source media="(max-width: 768px)" srcset="https://static.rerun.io/asset3d_simple/af238578188d3fd0de3e330212120e2842a8ddb2/768w.png">
      <source media="(max-width: 1024px)" srcset="https://static.rerun.io/asset3d_simple/af238578188d3fd0de3e330212120e2842a8ddb2/1024w.png">
      <source media="(max-width: 1200px)" srcset="https://static.rerun.io/asset3d_simple/af238578188d3fd0de3e330212120e2842a8ddb2/1200w.png">
      <img src="https://static.rerun.io/asset3d_simple/af238578188d3fd0de3e330212120e2842a8ddb2/full.png" width="640">
    </picture>
    </center>
    """

    def __attrs_clear__(self) -> None:
        if False:
            return 10
        'Convenience method for calling `__attrs_init__` with all `None`s.'
        self.__attrs_init__(blob=None, media_type=None, transform=None)

    @classmethod
    def _clear(cls) -> Asset3D:
        if False:
            print('Hello World!')
        'Produce an empty Asset3D, bypassing `__init__`.'
        inst = cls.__new__(cls)
        inst.__attrs_clear__()
        return inst
    blob: components.BlobBatch = field(metadata={'component': 'required'}, converter=components.BlobBatch._required)
    media_type: components.MediaTypeBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.MediaTypeBatch._optional)
    transform: components.OutOfTreeTransform3DBatch | None = field(metadata={'component': 'optional'}, default=None, converter=components.OutOfTreeTransform3DBatch._optional)
    __str__ = Archetype.__str__
    __repr__ = Archetype.__repr__