"""
Sprite definition file.
"""
from __future__ import annotations
import typing
from enum import Enum
from ..data_definition import DataDefinition
FORMAT_VERSION = '2'

class LayerMode(Enum):
    """
    Possible values for the mode of a layer.
    """
    OFF = 'off'
    ONCE = 'once'
    LOOP = 'loop'

class SpriteMetadata(DataDefinition):
    """
    Collects sprite metadata and can format it
    as a .sprite custom format
    """

    def __init__(self, targetdir: str, filename: str):
        if False:
            i = 10
            return i + 15
        super().__init__(targetdir, filename)
        self.texture_files: dict[int, dict[str, typing.Any]] = {}
        self.scalefactor = 1.0
        self.layers: dict[int, dict[str, typing.Any]] = {}
        self.angles: dict[int, dict[str, int]] = {}
        self.frames: list[dict[str, int]] = []

    def add_texture(self, texture_id: int, filename: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Add a texture and the relative file name.\n\n        :param texture_id: Texture identifier.\n        :type texture_id: int\n        :param filename: Path to the image file.\n        :type filename: str\n        '
        self.texture_files[texture_id] = {'texture_id': texture_id, 'filename': filename}

    def add_layer(self, layer_id: int, mode: LayerMode=None, position: int=None, time_per_frame: float=None, replay_delay: float=None) -> None:
        if False:
            while True:
                i = 10
        '\n        Define a layer for the rendered sprite.\n\n        :param layer_id: Layer identifier.\n        :type layer_id: int\n        :param mode: Animation mode (off, once, loop).\n        :type mode: LayerMode\n        :param position: Layer position.\n        :type position: int\n        :param time_per_frame: Time spent on each frame.\n        :type time_per_frame: float\n        :param replay_delay: Time delay before replaying the animation.\n        :type replay_delay: float\n        '
        self.layers[layer_id] = {'layer_id': layer_id, 'mode': mode, 'position': position, 'time_per_frame': time_per_frame, 'replay_delay': replay_delay}

    def add_angle(self, degree: int, mirror_from: int=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Specifies an angle that frames can get assigned to.\n\n        :param degree: Angle identifier expressed in degrees.\n        :type degree: int\n        :param mirror_from: Other angle to copy frames from, if any.\n        :type mirror_from: int\n        '
        self.angles[degree] = {'degree': degree, 'mirror_from': mirror_from}

    def add_frame(self, frame_idx: int, angle: int, layer_id: int, texture_id: int, subtex_id: int) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Add frame with all its spacial information.\n\n        :param frame_idx: Index of the frame in the animation for the specified angle.\n        :type frame_idx: int\n        :param angle: Angle to which the frame belongs, in degrees.\n        :type angle: int\n        :param layer_id: ID of the layer to which the frame belongs.\n        :type layer_id: int\n        :param texture_id: ID of the texture used by this frame.\n        :type texture_id: int\n        :param subtex_id: ID of the subtexture from the texture used by this frame.\n        :type subtex_id: int\n        '
        self.frames.append({'frame_idx': frame_idx, 'angle': angle, 'layer_id': layer_id, 'texture_id': texture_id, 'subtex_id': subtex_id})

    def set_scalefactor(self, factor: typing.Union[int, float]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Set the scale factor of the animation.\n\n        :param factor: Factor by which sprite images are scaled down at default zoom level.\n        :type factor: float\n        '
        self.scalefactor = float(factor)

    def dump(self) -> str:
        if False:
            return 10
        output_str = ''
        output_str += '# openage sprite definition file\n\n'
        output_str += f'version {FORMAT_VERSION}\n\n'
        for texture in self.texture_files.values():
            output_str += f'''texture {texture['texture_id']} "{texture['filename']}"\n'''
        output_str += '\n'
        output_str += f'scalefactor {self.scalefactor}\n\n'
        for layer in self.layers.values():
            output_str += f"layer {layer['layer_id']}"
            if layer['mode']:
                output_str += f" mode={layer['mode'].value}"
            if layer['position']:
                output_str += f" position={layer['position']}"
            if layer['time_per_frame']:
                output_str += f" time_per_frame={layer['time_per_frame']}"
            if layer['replay_delay']:
                output_str += f" replay_delay={layer['replay_delay']}"
            output_str += '\n'
        output_str += '\n'
        for angle in self.angles.values():
            output_str += f"angle {angle['degree']}"
            if angle['mirror_from']:
                output_str += f" mirror_from={angle['mirror_from']}"
            output_str += '\n'
        output_str += '\n'
        for frame in self.frames:
            output_str += f"frame {' '.join((str(param) for param in frame.values()))}\n"
        return output_str

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'SpriteMetadata<{self.filename}>'