from typing import TYPE_CHECKING
from Charon.VirtualFile import VirtualFile
from UM.Mesh.MeshReader import MeshReader
from UM.MimeTypeDatabase import MimeType, MimeTypeDatabase
from UM.PluginRegistry import PluginRegistry
if TYPE_CHECKING:
    from cura.Scene.CuraSceneNode import CuraSceneNode

class UFPReader(MeshReader):

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        MimeTypeDatabase.addMimeType(MimeType(name='application/x-ufp', comment='UltiMaker Format Package', suffixes=['ufp']))
        self._supported_extensions = ['.ufp']

    def _read(self, file_name: str) -> 'CuraSceneNode':
        if False:
            for i in range(10):
                print('nop')
        archive = VirtualFile()
        archive.open(file_name)
        gcode_data = archive.getData('/3D/model.gcode')
        gcode_stream = gcode_data['/3D/model.gcode'].decode('utf-8')
        gcode_reader = PluginRegistry.getInstance().getPluginObject('GCodeReader')
        gcode_reader.preReadFromStream(gcode_stream)
        return gcode_reader.readFromStream(gcode_stream, file_name)