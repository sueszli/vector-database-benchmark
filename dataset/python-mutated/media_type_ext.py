from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from . import MediaType

class MediaTypeExt:
    """Extension for [MediaType][rerun.components.MediaType]."""
    TEXT: MediaType = None
    'Plain text: `text/plain`.'
    MARKDOWN: MediaType = None
    '\n    Markdown: `text/markdown`.\n\n    <https://www.iana.org/assignments/media-types/text/markdown>\n    '
    GLB: MediaType = None
    '\n    Binary [`glTF`](https://en.wikipedia.org/wiki/GlTF): `model/gltf-binary`.\n\n    <https://www.iana.org/assignments/media-types/model/gltf-binary>\n    '
    GLTF: MediaType = None
    '\n    [`glTF`](https://en.wikipedia.org/wiki/GlTF): `model/gltf+json`.\n\n    <https://www.iana.org/assignments/media-types/model/gltf+json>\n    '
    OBJ: MediaType = None
    '\n    [Wavefront .obj](https://en.wikipedia.org/wiki/Wavefront_.obj_file): `model/obj`.\n\n    <https://www.iana.org/assignments/media-types/model/obj>\n    '

    @staticmethod
    def deferred_patch_class(cls: Any) -> None:
        if False:
            i = 10
            return i + 15
        cls.TEXT = cls('text/plain')
        cls.MARKDOWN = cls('text/markdown')
        cls.GLB = cls('model/gltf-binary')
        cls.GLTF = cls('model/gltf+json')
        cls.OBJ = cls('model/obj')