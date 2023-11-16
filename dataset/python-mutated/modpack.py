"""
Defines a modpack that can be exported.
"""
from __future__ import annotations
from ..export.data_definition import DataDefinition
from ..export.formats.modpack_info import ModpackInfo
from ..export.formats.modpack_manifest import ManifestFile
from ..export.media_export_request import MediaExportRequest
from ..export.metadata_export import MetadataExport

class Modpack:
    """
    A collection of data and media files.
    """

    def __init__(self, name: str):
        if False:
            print('Hello World!')
        self.name = name
        self.info = ModpackInfo('', 'modpack.toml')
        self.manifest = ManifestFile('', 'manifest.toml')
        self.data_export_files: list[DataDefinition] = []
        self.media_export_files: list[MediaExportRequest] = {}
        self.metadata_files: list[MetadataExport] = []

    def add_data_export(self, export_file: DataDefinition) -> None:
        if False:
            while True:
                i = 10
        '\n        Add a data file to the modpack for exporting.\n        '
        if not isinstance(export_file, DataDefinition):
            raise TypeError(f'{repr(self)}: export file must be of type DataDefinition not {type(export_file)}')
        self.data_export_files.append(export_file)

    def add_media_export(self, export_request: MediaExportRequest) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Add a media export request to the modpack.\n        '
        if not isinstance(export_request, MediaExportRequest):
            raise TypeError(f'{repr(self)}: export file must be of type MediaExportRequest not {type(export_request)}')
        if export_request.get_type() in self.media_export_files:
            self.media_export_files[export_request.get_type()].append(export_request)
        else:
            self.media_export_files[export_request.get_type()] = [export_request]

    def add_metadata_export(self, export_file: MetadataExport) -> None:
        if False:
            while True:
                i = 10
        '\n        Add a metadata file to the modpack for exporting.\n        '
        if not isinstance(export_file, MetadataExport):
            raise TypeError(f'{repr(self)}: export file must be of type MetadataExport not {type(export_file)}')
        self.metadata_files.append(export_file)

    def get_info(self) -> ModpackInfo:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the modpack definition file.\n        '
        return self.info

    def get_data_files(self) -> list[DataDefinition]:
        if False:
            while True:
                i = 10
        '\n        Returns the data files for exporting.\n        '
        return self.data_export_files

    def get_media_files(self) -> list[MediaExportRequest]:
        if False:
            return 10
        '\n        Returns the media requests for exporting.\n        '
        return self.media_export_files

    def get_metadata_files(self) -> list[MetadataExport]:
        if False:
            i = 10
            return i + 15
        '\n        Returns the metadata exports.\n        '
        return self.metadata_files