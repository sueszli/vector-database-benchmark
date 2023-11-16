import gzip
from io import StringIO, BufferedIOBase
from typing import cast, List
from UM.Logger import Logger
from UM.Mesh.MeshWriter import MeshWriter
from UM.PluginRegistry import PluginRegistry
from UM.Scene.SceneNode import SceneNode
from UM.i18n import i18nCatalog
catalog = i18nCatalog('cura')

class GCodeGzWriter(MeshWriter):
    """A file writer that writes gzipped g-code.

    If you're zipping g-code, you might as well use gzip!
    """

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__(add_to_recent_files=False)

    def write(self, stream: BufferedIOBase, nodes: List[SceneNode], mode=MeshWriter.OutputMode.BinaryMode) -> bool:
        if False:
            i = 10
            return i + 15
        'Writes the gzipped g-code to a stream.\n\n        Note that even though the function accepts a collection of nodes, the\n        entire scene is always written to the file since it is not possible to\n        separate the g-code for just specific nodes.\n\n        :param stream: The stream to write the gzipped g-code to.\n        :param nodes: This is ignored.\n        :param mode: Additional information on what type of stream to use. This\n            must always be binary mode.\n        :return: Whether the write was successful.\n        '
        if mode != MeshWriter.OutputMode.BinaryMode:
            Logger.log('e', 'GCodeGzWriter does not support text mode.')
            self.setInformation(catalog.i18nc('@error:not supported', 'GCodeGzWriter does not support text mode.'))
            return False
        gcode_textio = StringIO()
        gcode_writer = cast(MeshWriter, PluginRegistry.getInstance().getPluginObject('GCodeWriter'))
        success = gcode_writer.write(gcode_textio, None)
        if not success:
            self.setInformation(gcode_writer.getInformation())
            return False
        result = gzip.compress(gcode_textio.getvalue().encode('utf-8'))
        stream.write(result)
        return True