from typing import Optional
from UM.Scene.SceneNodeDecorator import SceneNodeDecorator
from cura.LayerData import LayerData

class LayerDataDecorator(SceneNodeDecorator):
    """Simple decorator to indicate a scene node holds layer data."""

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self._layer_data = None

    def getLayerData(self) -> Optional['LayerData']:
        if False:
            i = 10
            return i + 15
        return self._layer_data

    def setLayerData(self, layer_data: LayerData) -> None:
        if False:
            i = 10
            return i + 15
        self._layer_data = layer_data

    def __deepcopy__(self, memo) -> 'LayerDataDecorator':
        if False:
            while True:
                i = 10
        copied_decorator = LayerDataDecorator()
        copied_decorator._layer_data = self._layer_data
        return copied_decorator