"""Binary data provider."""
import typing as t
from pathlib import Path
from mimesis.enums import AudioFile, CompressedFile, DocumentFile, ImageFile, VideoFile
from mimesis.providers.base import BaseProvider
__all__ = ['BinaryFile']

class BinaryFile(BaseProvider):
    """Class for generating binary data"""

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        if False:
            return 10
        'Initialize attributes.\n\n        :param locale: Current locale.\n        :param seed: Seed.\n        '
        super().__init__(*args, **kwargs)
        self._data_dir = Path(__file__).parent.parent.joinpath('data', 'bin')
        self._sample_name: t.Final[str] = 'sample'

    class Meta:
        name = 'binaryfile'

    def _read_file(self, *, file_type: t.Union[AudioFile, CompressedFile, DocumentFile, ImageFile, VideoFile]) -> bytes:
        if False:
            return 10
        file_type = self.validate_enum(file_type, file_type.__class__)
        file_path = self._data_dir.joinpath(f'{self._sample_name}.{file_type}')
        with open(file_path, 'rb') as file:
            return file.read()

    def video(self, *, file_type: VideoFile=VideoFile.MP4) -> bytes:
        if False:
            while True:
                i = 10
        'Generates video file of given format and returns it as bytes.\n\n        .. note:: This method accepts keyword-only arguments.\n\n        :param file_type: File extension.\n        :return: File as a sequence of bytes.\n        '
        return self._read_file(file_type=file_type)

    def audio(self, *, file_type: AudioFile=AudioFile.MP3) -> bytes:
        if False:
            i = 10
            return i + 15
        'Generates audio file of given format and returns it as bytes.\n\n        .. note:: This method accepts keyword-only arguments.\n\n        :param file_type: File extension.\n        :return: File as a sequence of bytes.\n        '
        return self._read_file(file_type=file_type)

    def document(self, *, file_type: DocumentFile=DocumentFile.PDF) -> bytes:
        if False:
            i = 10
            return i + 15
        'Generates document of given format and returns it as bytes.\n\n        .. note:: This method accepts keyword-only arguments.\n\n        :param file_type: File extension.\n        :return: File as a sequence of bytes.\n        '
        return self._read_file(file_type=file_type)

    def image(self, *, file_type: ImageFile=ImageFile.PNG) -> bytes:
        if False:
            return 10
        'Generates image of given format and returns it as bytes.\n\n        .. note:: This method accepts keyword-only arguments.\n\n        :param file_type: File extension.\n        :return: File as a sequence of bytes.\n        '
        return self._read_file(file_type=file_type)

    def compressed(self, *, file_type: CompressedFile=CompressedFile.ZIP) -> bytes:
        if False:
            print('Hello World!')
        'Generates compressed file of given format and returns it as bytes.\n\n        .. note:: This method accepts keyword-only arguments.\n\n        :param file_type: File extension.\n        :return: File as a sequence of bytes.\n        '
        return self._read_file(file_type=file_type)