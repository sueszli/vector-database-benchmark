"""
Export requests for media metadata.
"""
from __future__ import annotations
import typing
from ....util.observer import Observer
from .formats.sprite_metadata import SpriteMetadata
from .formats.texture_metadata import TextureMetadata
if typing.TYPE_CHECKING:
    from openage.util.observer import Observable
    from openage.convert.entity_object.export.formats.sprite_metadata import LayerMode

class MetadataExport(Observer):
    """
    A class for exporting metadata from another format. MetadataExports are
    observers so they can receive data from media conversion.
    """

    def __init__(self, targetdir: str, target_filename: str):
        if False:
            for i in range(10):
                print('nop')
        self.targetdir = targetdir
        self.filename = target_filename

    def update(self, observable: Observable, message=None):
        if False:
            return 10
        return NotImplementedError('Interface does not implement update()')

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'MetadataExport<{type(self)}>'

class SpriteMetadataExport(MetadataExport):
    """
    Export requests for sprite definition files.
    """

    def __init__(self, targetdir, target_filename):
        if False:
            print('Hello World!')
        super().__init__(targetdir, target_filename)
        self.graphics_metadata: dict[int, tuple] = {}
        self.subtex_count: dict[str, int] = {}

    def add_graphics_metadata(self, img_filename: str, tex_filename: str, layer_mode: LayerMode, layer_pos: int, frame_rate: float, replay_delay: float, frame_count: int, angle_count: int, mirror_mode: int, start_angle: int=0):
        if False:
            i = 10
            return i + 15
        '\n        Add metadata from the GenieGraphic object.\n\n        :param tex_filename: Filename of the .texture file.\n        :param start_angle: Angle used for the first frame in the .texture file.\n        '
        self.graphics_metadata[img_filename] = (tex_filename, layer_mode, layer_pos, frame_rate, replay_delay, frame_count, angle_count, mirror_mode, start_angle)

    def dump(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a human-readable string that can be written to a file.\n        '
        sprite_file = SpriteMetadata(self.targetdir, self.filename)
        tex_index = 0
        for (img_filename, metadata) in self.graphics_metadata.items():
            tex_filename = metadata[0]
            sprite_file.add_texture(tex_index, tex_filename)
            sprite_file.add_layer(tex_index, *metadata[1:5])
            frame_count = metadata[5]
            angle_count = metadata[6]
            mirror_mode = metadata[7]
            start_angle = metadata[8]
            if angle_count == 0:
                angle_count = 1
            degree = 0
            if start_angle and angle_count > 1:
                degree = start_angle % 360
            degree_step = 360 / angle_count
            for angle_index in range(angle_count):
                mirror_from = None
                if mirror_mode:
                    if degree > 180:
                        mirrored_angle = (angle_index - angle_count) * -1
                        mirror_from = (start_angle + int(mirrored_angle * degree_step)) % 360
                sprite_file.add_angle(int(degree), mirror_from)
                if not mirror_from:
                    for frame_idx in range(frame_count):
                        subtex_index = frame_idx + angle_index * frame_count
                        if subtex_index >= self.subtex_count[img_filename]:
                            break
                        sprite_file.add_frame(frame_idx, int(degree), tex_index, tex_index, subtex_index)
                degree = (degree + degree_step) % 360
            tex_index += 1
        return sprite_file.dump()

    def update(self, observable, message=None):
        if False:
            print('Hello World!')
        '\n        Receive metdata from the graphics file export.\n\n        :param message: A dict with frame metadata from the exported PNG file.\n        :type message: dict\n        '
        if message:
            for (tex_filename, metadata) in message.items():
                self.subtex_count[tex_filename] = len(metadata['subtex_metadata'])

class TextureMetadataExport(MetadataExport):
    """
    Export requests for texture definition files.
    """

    def __init__(self, targetdir, target_filename):
        if False:
            i = 10
            return i + 15
        super().__init__(targetdir, target_filename)
        self.imagefile = None
        self.size = None
        self.pxformat = 'rgba8'
        self.cbits = True
        self.subtex_metadata = []

    def add_imagefile(self, img_filename):
        if False:
            return 10
        '\n        Add metadata from the GenieGraphic object.\n\n        :param img_filename: Filename of the exported PNG file.\n        '
        self.imagefile = img_filename

    def dump(self):
        if False:
            i = 10
            return i + 15
        '\n        Creates a human-readable string that can be written to a file.\n        '
        texture_file = TextureMetadata(self.targetdir, self.filename)
        texture_file.set_imagefile(self.imagefile)
        texture_file.set_size(self.size[0], self.size[1])
        texture_file.set_pxformat(self.pxformat, self.cbits)
        for subtex_metadata in self.subtex_metadata:
            texture_file.add_subtex(*subtex_metadata.values())
        return texture_file.dump()

    def update(self, observable: Observable, message: dict=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Receive metdata from the graphics file export.\n\n        :param message: A dict with texture metadata from the exported PNG file.\n        :type message: dict\n        '
        if message:
            texture_metadata = message[self.imagefile]
            self.size = texture_metadata['size']
            self.subtex_metadata = texture_metadata['subtex_metadata']