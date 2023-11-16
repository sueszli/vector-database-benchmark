"""
Blendmask definition file.
"""
from __future__ import annotations
import typing
from ..data_definition import DataDefinition
FORMAT_VERSION = '1'

class BlendmaskMetadata(DataDefinition):
    """
    Collects blendmask metadata and can format it
    as a .blmask custom format
    """

    def __init__(self, targetdir: str, filename: str):
        if False:
            i = 10
            return i + 15
        super().__init__(targetdir, filename)
        self.image_files: dict[int, dict[str, typing.Any]] = {}
        self.scalefactor = 1.0
        self.masks: dict[int, dict[str, int]] = {}

    def add_image(self, img_id: int, filename: str) -> None:
        if False:
            return 10
        '\n        Add an image and the relative file name.\n\n        :param img_id: Image identifier.\n        :type img_id: int\n        :param filename: Path to the image file.\n        :type filename: str\n        '
        self.image_files[img_id] = {'image_id': img_id, 'filename': filename}

    def add_mask(self, directions: int, img_id: int, xpos: int, ypos: int, xsize: int, ysize: int) -> None:
        if False:
            while True:
                i = 10
        '\n        Add a mask for directions.\n\n        :param directions: Directions bitfield value.\n        :type directions: int\n        :param img_id: ID of the image used by this mask.\n        :type img_id: int\n        :param xpos: X position of the mask on the image canvas.\n        :type xpos: int\n        :param ypos: Y position of the mask on the image canvas.\n        :type ypos: int\n        :param xsize: Width of the mask.\n        :type xsize: int\n        :param ysize: Height of the mask.\n        :type ysize: int\n        '
        self.masks[directions] = {'directions': directions, 'img_id': img_id, 'xpos': xpos, 'ypos': ypos, 'xsize': xsize, 'ysize': ysize}

    def set_scalefactor(self, factor: typing.Union[int, float]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the scale factor of the animation.\n\n        :param factor: Factor by which sprite images are scaled down at default zoom level.\n        :type factor: float\n        '
        self.scalefactor = float(factor)

    def dump(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        output_str = ''
        output_str += '# openage blendmask definition file\n\n'
        output_str += f'version {FORMAT_VERSION}\n\n'
        for image in self.image_files.values():
            output_str += f"imagefile {image['image_id']} {image['filename']}\n"
        output_str += '\n'
        output_str += f'scalefactor {self.scalefactor}\n\n'
        for mask in self.masks.values():
            output_str += f"mask {' '.join((str(param) for param in mask.values()))}\n"
        return output_str

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'BlendmaskMetadata<{self.filename}>'