from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from . import Mat3x3Like, Vec3DLike

class TranslationAndMat3x3Ext:
    """Extension for [TranslationAndMat3x3][rerun.datatypes.TranslationAndMat3x3]."""

    def __init__(self: Any, translation: Vec3DLike | None=None, mat3x3: Mat3x3Like | None=None, *, from_parent: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new instance of the TranslationAndMat3x3 datatype.\n\n        Parameters\n        ----------\n        translation:\n             3D translation, applied after the matrix.\n        mat3x3:\n             3x3 matrix for scale, rotation & shear.\n        from_parent:\n             If true, the transform maps from the parent space to the space where the transform was logged.\n             Otherwise, the transform maps from the space to its parent.\n        '
        self.__attrs_init__(translation=translation, mat3x3=mat3x3, from_parent=from_parent)