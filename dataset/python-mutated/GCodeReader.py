from typing import Optional, Union, List, TYPE_CHECKING
from UM.FileHandler.FileReader import FileReader
from UM.Mesh.MeshReader import MeshReader
from UM.i18n import i18nCatalog
from UM.Application import Application
from UM.MimeTypeDatabase import MimeTypeDatabase, MimeType
catalog = i18nCatalog('cura')
from .FlavorParser import FlavorParser
from . import MarlinFlavorParser, RepRapFlavorParser
if TYPE_CHECKING:
    from UM.Scene.SceneNode import SceneNode
    from cura.Scene.CuraSceneNode import CuraSceneNode

class GCodeReader(MeshReader):
    _flavor_default = 'Marlin'
    _flavor_keyword = ';FLAVOR:'
    _flavor_readers_dict = {'RepRap': RepRapFlavorParser.RepRapFlavorParser(), 'Marlin': MarlinFlavorParser.MarlinFlavorParser()}

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__()
        MimeTypeDatabase.addMimeType(MimeType(name='application/x-cura-gcode-file', comment='Cura G-code File', suffixes=['gcode']))
        self._supported_extensions = ['.gcode', '.g']
        self._flavor_reader = None
        Application.getInstance().getPreferences().addPreference('gcodereader/show_caution', True)

    def preReadFromStream(self, stream, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        for line in stream.split('\n'):
            if line[:len(self._flavor_keyword)] == self._flavor_keyword:
                try:
                    self._flavor_reader = self._flavor_readers_dict[line[len(self._flavor_keyword):].rstrip()]
                    return FileReader.PreReadResult.accepted
                except:
                    break
        self._flavor_reader = self._flavor_readers_dict[self._flavor_default]
        return FileReader.PreReadResult.accepted

    def preRead(self, file_name, *args, **kwargs):
        if False:
            return 10
        with open(file_name, 'r', encoding='utf-8') as file:
            file_data = file.read()
        return self.preReadFromStream(file_data, args, kwargs)

    def readFromStream(self, stream: str, filename: str) -> Optional['CuraSceneNode']:
        if False:
            i = 10
            return i + 15
        if self._flavor_reader is None:
            return None
        return self._flavor_reader.processGCodeStream(stream, filename)

    def _read(self, file_name: str) -> Union['SceneNode', List['SceneNode']]:
        if False:
            i = 10
            return i + 15
        with open(file_name, 'r', encoding='utf-8') as file:
            file_data = file.read()
        result = []
        node = self.readFromStream(file_data, file_name)
        if node is not None:
            result.append(node)
        return result