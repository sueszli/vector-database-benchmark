import io
from typing import Optional, Dict, Union, List, cast
from UM.FileHandler.FileHandler import FileHandler
from UM.FileHandler.FileWriter import FileWriter
from UM.Logger import Logger
from UM.OutputDevice import OutputDeviceError
from UM.Scene.SceneNode import SceneNode
from UM.Version import Version
from UM.i18n import i18nCatalog
from cura.CuraApplication import CuraApplication
I18N_CATALOG = i18nCatalog('cura')

class MeshFormatHandler:
    """This class is responsible for choosing the formats used by the connected clusters."""

    def __init__(self, file_handler: Optional[FileHandler], firmware_version: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._file_handler = file_handler or CuraApplication.getInstance().getMeshFileHandler()
        self._preferred_format = self._getPreferredFormat(firmware_version)
        self._writer = self._getWriter(self.mime_type) if self._preferred_format else None

    @property
    def is_valid(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return bool(self._writer)

    @property
    def preferred_format(self) -> Dict[str, Union[str, int, bool]]:
        if False:
            for i in range(10):
                print('nop')
        'Chooses the preferred file format.\n\n        :return: A dict with the file format details, with the following keys:\n        {id: str, extension: str, description: str, mime_type: str, mode: int, hide_in_file_dialog: bool}\n        '
        return self._preferred_format

    @property
    def writer(self) -> Optional[FileWriter]:
        if False:
            while True:
                i = 10
        'Gets the file writer for the given file handler and mime type.\n\n        :return: A file writer.\n        '
        return self._writer

    @property
    def mime_type(self) -> str:
        if False:
            return 10
        return cast(str, self._preferred_format['mime_type'])

    @property
    def file_mode(self) -> int:
        if False:
            i = 10
            return i + 15
        'Gets the file mode (FileWriter.OutputMode.TextMode or FileWriter.OutputMode.BinaryMode)'
        return cast(int, self._preferred_format['mode'])

    @property
    def file_extension(self) -> str:
        if False:
            while True:
                i = 10
        'Gets the file extension'
        return cast(str, self._preferred_format['extension'])

    def createStream(self) -> Union[io.BytesIO, io.StringIO]:
        if False:
            return 10
        'Creates the right kind of stream based on the preferred format.'
        if self.file_mode == FileWriter.OutputMode.TextMode:
            return io.StringIO()
        else:
            return io.BytesIO()

    def getBytes(self, nodes: List[SceneNode]) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        'Writes the mesh and returns its value.'
        if self.writer is None:
            raise ValueError('There is no writer for the mesh format handler.')
        stream = self.createStream()
        self.writer.write(stream, nodes)
        value = stream.getvalue()
        if isinstance(value, str):
            value = value.encode()
        return value

    def _getPreferredFormat(self, firmware_version: str) -> Dict[str, Union[str, int, bool]]:
        if False:
            for i in range(10):
                print('nop')
        'Chooses the preferred file format for the given file handler.\n\n        :param firmware_version: The version of the firmware.\n        :return: A dict with the file format details.\n        '
        application = CuraApplication.getInstance()
        file_formats = self._file_handler.getSupportedFileTypesWrite()
        global_stack = application.getGlobalContainerStack()
        if not global_stack:
            Logger.log('e', 'Missing global stack!')
            return {}
        machine_file_formats = global_stack.getMetaDataEntry('file_formats').split(';')
        machine_file_formats = [file_type.strip() for file_type in machine_file_formats]
        if 'application/x-ufp' not in machine_file_formats and Version(firmware_version) >= Version('4.4'):
            machine_file_formats = ['application/x-ufp'] + machine_file_formats
        elif 'application/x-makerbot' not in machine_file_formats and Version(firmware_version >= Version('2.700')):
            machine_file_formats = ['application/x-makerbot'] + machine_file_formats
        format_by_mimetype = {f['mime_type']: f for f in file_formats}
        file_formats = [format_by_mimetype[mimetype] for mimetype in machine_file_formats]
        if len(file_formats) == 0:
            Logger.log('e', 'There are no file formats available to write with!')
            raise OutputDeviceError.WriteRequestFailedError(I18N_CATALOG.i18nc('@info:status', 'There are no file formats available to write with!'))
        return file_formats[0]

    def _getWriter(self, mime_type: str) -> Optional[FileWriter]:
        if False:
            print('Hello World!')
        'Gets the file writer for the given file handler and mime type.\n\n        :param mime_type: The mine type.\n        :return: A file writer.\n        '
        return self._file_handler.getWriterByMimeType(mime_type)