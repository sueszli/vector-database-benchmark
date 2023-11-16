from UM.Mesh.MeshData import MeshData

class LayerData(MeshData):
    """Class to holds the layer mesh and information about the layers.

    Immutable, use :py:class:`cura.LayerDataBuilder.LayerDataBuilder` to create one of these.
    """

    def __init__(self, vertices=None, normals=None, indices=None, colors=None, uvs=None, file_name=None, center_position=None, layers=None, element_counts=None, attributes=None):
        if False:
            print('Hello World!')
        super().__init__(vertices=vertices, normals=normals, indices=indices, colors=colors, uvs=uvs, file_name=file_name, center_position=center_position, attributes=attributes)
        self._layers = layers
        self._element_counts = element_counts

    def getLayer(self, layer):
        if False:
            while True:
                i = 10
        if layer in self._layers:
            return self._layers[layer]
        return None

    def getLayers(self):
        if False:
            for i in range(10):
                print('nop')
        return self._layers

    def getElementCounts(self):
        if False:
            i = 10
            return i + 15
        return self._element_counts