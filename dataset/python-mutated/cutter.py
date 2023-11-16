"""
Cutting some user interface assets into subtextures.
"""
from __future__ import annotations
import typing
from ....entity_object.export.texture import TextureImage
from ....value_object.read.media.hardcoded.interface import TOP_STRIP_PATTERN_CORNERS, TOP_STRIP_PATTERN_SEARCH_AREA_CORNERS, MID_STRIP_PATTERN_CORNERS, MID_STRIP_PATTERN_SEARCH_AREA_CORNERS, KNOWN_SUBTEX_CORNER_COORDS, INGAME_HUD_BACKGROUNDS, INGAME_HUD_BACKGROUNDS_SET
from .visgrep import visgrep, crop_array
if typing.TYPE_CHECKING:
    from numpy import ndarray

class InterfaceCutter:
    """
    Cuts interface textures into repeatable parts.
    """

    def __init__(self, idx: int):
        if False:
            i = 10
            return i + 15
        self.idx = idx

    def cut(self, image: TextureImage) -> TextureImage:
        if False:
            i = 10
            return i + 15
        '\n        Create subtextures by searching for patterns at hardcoded positions.\n        '
        if not isinstance(image, TextureImage):
            raise ValueError(f"we can only cut TextureImage, not '{type(image)}'")
        if is_ingame_hud_background(self.idx):
            img_data = image.get_data()
            yield self.cut_strip(img_data, TOP_STRIP_PATTERN_CORNERS, TOP_STRIP_PATTERN_SEARCH_AREA_CORNERS)
            yield self.cut_strip(img_data, MID_STRIP_PATTERN_CORNERS, MID_STRIP_PATTERN_SEARCH_AREA_CORNERS)
            for coords in KNOWN_SUBTEX_CORNER_COORDS:
                yield TextureImage(crop_array(img_data, coords))
        else:
            yield image

    def cut_strip(self, img_array: ndarray, pattern_corners: tuple[int, int, int, int], search_area_corners: tuple[int, int, int, int]) -> TextureImage:
        if False:
            i = 10
            return i + 15
        '\n        Finds a horizontally tilable piece of the strip (ex. the top of the HUD).\n\n        ||----///////////-------------///////////-------------///////////-------------///////////||\n                  ^      pattern_corners     ^                    ^  where it is found last  ^\n                  ^           this piece is tileable              ^\n\n        so, cut out a subtexture:\n                  ///////-------------///////////-------------////\n        '
        search_area = crop_array(img_array, search_area_corners)
        pattern = crop_array(img_array, pattern_corners)
        matches = visgrep(search_area, pattern, 100000)
        if len(matches) < 2:
            raise RuntimeError(f'visgrep failed to find repeating pattern in id={self.idx})\n')
        return TextureImage(crop_array(img_array, (pattern_corners[0], pattern_corners[1], search_area_corners[0] + matches[-1].point[0], pattern_corners[3])))

def ingame_hud_background_index(idx: int):
    if False:
        while True:
            i = 10
    '\n    Index in the hardcoded list of the known ingame hud backgrounds to match the civ.\n    '
    return INGAME_HUD_BACKGROUNDS.index(int(idx))

def is_ingame_hud_background(idx: int):
    if False:
        return 10
    '\n    True if in the hardcoded list of the known ingame hud backgrounds.\n    '
    return int(idx) in INGAME_HUD_BACKGROUNDS_SET