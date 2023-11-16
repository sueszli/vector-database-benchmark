"""
Specifies a request for a media resource that should be
converted and exported into a modpack.
"""
from __future__ import annotations
import typing
from ....util.observer import Observable
if typing.TYPE_CHECKING:
    from openage.convert.value_object.read.media_types import MediaType

class MediaExportRequest(Observable):
    """
    Generic superclass for export requests.
    """
    __slots__ = ('media_type', 'targetdir', 'source_filename', 'target_filename')

    def __init__(self, media_type: MediaType, targetdir: str, source_filename: str, target_filename: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a request for a media file.\n\n        :param media_type: Media type of the requested  source file.\n        :type media_type: MediaType\n        :param targetdir: Relative path to the export directory.\n        :type targetdir: str\n        :param source_filename: Filename of the source file.\n        :type source_filename: str\n        :param target_filename: Filename of the resulting file.\n        :type target_filename: str\n        '
        super().__init__()
        self.media_type = media_type
        self.targetdir = targetdir
        self.source_filename = source_filename
        self.target_filename = target_filename

    def get_type(self) -> MediaType:
        if False:
            i = 10
            return i + 15
        '\n        Return the media type.\n        '
        return self.media_type

    def set_source_filename(self, filename: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the filename for the source file.\n\n        :param filename: Filename of the source file.\n        :type filename: str\n        '
        if not isinstance(filename, str):
            raise ValueError(f'str expected as source filename, not {type(filename)}')
        self.source_filename = filename

    def set_target_filename(self, filename: str) -> None:
        if False:
            print('Hello World!')
        '\n        Sets the filename for the target file.\n\n        :param filename: Filename of the resulting file.\n        :type filename: str\n        '
        if not isinstance(filename, str):
            raise ValueError(f'str expected as target filename, not {type(filename)}')
        self.target_filename = filename

    def set_targetdir(self, targetdir: str) -> None:
        if False:
            return 10
        '\n        Sets the target directory for the file.\n\n        :param targetdir: Relative path to the export directory.\n        :type targetdir: str\n        '
        if not isinstance(targetdir, str):
            raise ValueError(f'str expected as targetdir, not {type(targetdir)}')
        self.targetdir = targetdir