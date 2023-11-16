import gzip
from UM.Mesh.MeshReader import MeshReader
from UM.MimeTypeDatabase import MimeTypeDatabase, MimeType
from UM.PluginRegistry import PluginRegistry

class GCodeGzReader(MeshReader):
    """A file reader that reads gzipped g-code.

    If you're zipping g-code, you might as well use gzip!
    """

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        MimeTypeDatabase.addMimeType(MimeType(name='application/x-cura-compressed-gcode-file', comment='Cura Compressed G-code File', suffixes=['gcode.gz']))
        self._supported_extensions = ['.gcode.gz']

    def _read(self, file_name):
        if False:
            while True:
                i = 10
        with open(file_name, 'rb') as file:
            file_data = file.read()
        uncompressed_gcode = gzip.decompress(file_data).decode('utf-8')
        PluginRegistry.getInstance().getPluginObject('GCodeReader').preReadFromStream(uncompressed_gcode)
        result = PluginRegistry.getInstance().getPluginObject('GCodeReader').readFromStream(uncompressed_gcode, file_name)
        return result