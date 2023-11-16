import io
from typing import List, Optional, Union
from UM.FileHandler.FileHandler import FileHandler
from UM.FileHandler.FileWriter import FileWriter
from UM.FileHandler.WriteFileJob import WriteFileJob
from UM.Logger import Logger
from UM.MimeTypeDatabase import MimeTypeDatabase
from UM.OutputDevice import OutputDeviceError
from UM.Scene.SceneNode import SceneNode

class ExportFileJob(WriteFileJob):
    """Job that exports the build plate to the correct file format for the Digital Factory Library project."""

    def __init__(self, file_handler: FileHandler, nodes: List[SceneNode], job_name: str, extension: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        file_types = file_handler.getSupportedFileTypesWrite()
        if len(file_types) == 0:
            Logger.log('e', 'There are no file types available to write with!')
            raise OutputDeviceError.WriteRequestFailedError('There are no file types available to write with!')
        mode = None
        file_writer = None
        for file_type in file_types:
            if file_type['extension'] == extension:
                file_writer = file_handler.getWriter(file_type['id'])
                mode = file_type.get('mode')
        super().__init__(file_writer, self.createStream(mode=mode), nodes, mode)
        self.setFileName('{}.{}'.format(job_name, extension))

    def getOutput(self) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        'Get the job result as bytes as that is what we need to upload to the Digital Factory Library.'
        output = self.getStream().getvalue()
        if isinstance(output, str):
            output = output.encode('utf-8')
        return output

    def getMimeType(self) -> str:
        if False:
            i = 10
            return i + 15
        'Get the mime type of the selected export file type.'
        return MimeTypeDatabase.getMimeTypeForFile(self.getFileName()).name

    @staticmethod
    def createStream(mode) -> Union[io.BytesIO, io.StringIO]:
        if False:
            for i in range(10):
                print('nop')
        'Creates the right kind of stream based on the preferred format.'
        if mode == FileWriter.OutputMode.TextMode:
            return io.StringIO()
        else:
            return io.BytesIO()