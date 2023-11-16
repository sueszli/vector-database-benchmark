""" Routines for texture generation etc """
from __future__ import annotations
import typing
from PIL import Image
import numpy
from ....log import spam
from ...value_object.read.media.blendomatic import BlendingMode
from ...value_object.read.media.hardcoded.terrain_tile_size import TILE_HALFSIZE
from ...value_object.read.genie_structure import GenieStructure
if typing.TYPE_CHECKING:
    from openage.convert.value_object.read.media.colortable import ColorTable
    from openage.convert.service.export.interface.cutter import InterfaceCutter
    from openage.convert.value_object.read.media.slp import SLP, SLPFrame
    from openage.convert.value_object.read.media.smp import SMP, SMPLayer
    from openage.convert.value_object.read.media.smx import SMX, SMXLayer
    from openage.convert.value_object.read.media.sld import SLD, SLDLayer

class TextureImage:
    """
    represents a image created from a (r,g,b,a) matrix.
    """

    def __init__(self, picture_data: typing.Union[Image.Image, numpy.ndarray], hotspot: tuple[int, int]=None):
        if False:
            return 10
        if isinstance(picture_data, Image.Image):
            if picture_data.mode != 'RGBA':
                picture_data = picture_data.convert('RGBA')
            picture_data = numpy.array(picture_data)
        if not isinstance(picture_data, numpy.ndarray):
            raise ValueError("Texture image must be created from PIL Image or numpy array, not '%s'" % type(picture_data))
        self.width: int = picture_data.shape[1]
        self.height: int = picture_data.shape[0]
        spam('creating TextureImage with size %d x %d', self.width, self.height)
        if hotspot is None:
            self.hotspot = (0, 0)
        else:
            self.hotspot = hotspot
        self.data = picture_data

    def get_pil_image(self) -> Image.Image:
        if False:
            i = 10
            return i + 15
        return Image.fromarray(self.data)

    def get_data(self) -> numpy.ndarray:
        if False:
            for i in range(10):
                print('nop')
        return self.data

class Texture(GenieStructure):
    image_format = 'png'
    name_struct = 'subtexture'
    name_struct_file = 'texture'
    struct_description = "one sprite, as part of a texture atlas.\n\nthis struct stores information about positions and sizes\nof sprites included in the 'big texture'."

    def __init__(self, input_data: typing.Union[SLP, SMP, SMX, SLD, BlendingMode], palettes: dict[int, ColorTable]=None, custom_cutter: InterfaceCutter=None, layer: int=0):
        if False:
            while True:
                i = 10
        super().__init__()
        self.best_compr: tuple = None
        self.best_packer_hints: tuple = None
        self.image_data: TextureImage = None
        self.image_metadata: list[dict[str, int]] = {}
        spam('creating Texture from %s', repr(input_data))
        from ...value_object.read.media.slp import SLP
        from ...value_object.read.media.smp import SMP
        from ...value_object.read.media.smx import SMX
        from ...value_object.read.media.sld import SLD
        self.frames = []
        if isinstance(input_data, (SLP, SMP, SMX)):
            input_frames = input_data.get_frames(layer)
            for frame in input_frames:
                palette_number = frame.get_palette_number()
                if palette_number is None:
                    main_palette = None
                else:
                    main_palette = palettes[palette_number].array
                for subtex in self._to_subtextures(frame, main_palette, custom_cutter):
                    self.frames.append(subtex)
        elif isinstance(input_data, SLD):
            input_frames = input_data.get_frames(layer)
            if layer == 0 and len(input_frames) == 0:
                input_frames = input_data.get_frames(layer=1)
            for frame in input_frames:
                subtex = TextureImage(frame.get_picture_data(), hotspot=frame.get_hotspot())
                self.frames.append(subtex)
        elif isinstance(input_data, BlendingMode):
            self.frames = [TextureImage(tile.get_picture_data(), hotspot=(0, TILE_HALFSIZE['y'])) for tile in input_data.alphamasks]
        else:
            raise TypeError('cannot create Texture from unknown source type: %s' % type(input_data))

    def _to_subtextures(self, frame: typing.Union[SLPFrame, SMPLayer, SMXLayer], main_palette: ColorTable, custom_cutter: InterfaceCutter=None):
        if False:
            print('Hello World!')
        '\n        convert slp to subtexture or subtextures, using a palette.\n        '
        subtex = TextureImage(frame.get_picture_data(main_palette), hotspot=frame.get_hotspot())
        if custom_cutter:
            return custom_cutter.cut(subtex)
        else:
            return [subtex]

    def get_metadata(self) -> list[dict[str, int]]:
        if False:
            i = 10
            return i + 15
        '\n        Get the image metadata information.\n        '
        return self.image_metadata

    def get_cache_params(self) -> tuple[tuple, tuple]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the parameters used for packing and saving the texture.\n            - Packing hints (sprite index, (xpos, ypos) in the final texture)\n            - PNG compression parameters (compression level + deflate params)\n        '
        return (self.best_packer_hints, self.best_compr)

    @classmethod
    def get_data_format_members(cls, game_version) -> tuple:
        if False:
            i = 10
            return i + 15
        '\n        Return the members in this struct.\n        '
        data_format = ((True, 'x', None, 'int32_t'), (True, 'y', None, 'int32_t'), (True, 'w', None, 'int32_t'), (True, 'h', None, 'int32_t'), (True, 'cx', None, 'int32_t'), (True, 'cy', None, 'int32_t'))
        return data_format