"""
Exports data formats from a modpack to files.
"""
from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from openage.util.fslike.directory import Directory
    from openage.convert.entity_object.export.data_definition import DataDefinition

class DataExporter:
    """
    Writes the contents of a created modpack into a targetdir.
    """

    @staticmethod
    def export(data_files: list[DataDefinition], exportdir: Directory) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Exports data files.\n\n        :param data_files: Data definitions for data files.\n        :param exportdir: Directory the resulting file(s) will be exported to. Target subfolder\n                          and target filename should be stored in the export request.\n        :type exportdir: Directory\n        :type data_files: list\n        '
        for data_file in data_files:
            output_dir = exportdir.joinpath(data_file.targetdir)
            output_content = data_file.dump()
            with output_dir[data_file.filename].open('wb') as outfile:
                outfile.write(output_content.encode('utf-8'))