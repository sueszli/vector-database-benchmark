import logging
from io import BytesIO, IOBase
from typing import Any, Dict, Iterable, List, Mapping, Optional
from airbyte_cdk.sources.file_based.config.file_based_stream_config import FileBasedStreamConfig
from airbyte_cdk.sources.file_based.config.unstructured_format import UnstructuredFormat
from airbyte_cdk.sources.file_based.exceptions import FileBasedSourceError, RecordParseError
from airbyte_cdk.sources.file_based.file_based_stream_reader import AbstractFileBasedStreamReader, FileReadMode
from airbyte_cdk.sources.file_based.file_types.file_type_parser import FileTypeParser
from airbyte_cdk.sources.file_based.remote_file import RemoteFile
from airbyte_cdk.sources.file_based.schema_helpers import SchemaType
from unstructured.documents.elements import Formula, ListItem, Title
from unstructured.file_utils.filetype import STR_TO_FILETYPE, FileType, detect_filetype
unstructured_partition_pdf = None
unstructured_partition_docx = None
unstructured_partition_pptx = None
unstructured_optional_decode = None

def _import_unstructured() -> None:
    if False:
        i = 10
        return i + 15
    'Dynamically imported as needed, due to slow import speed.'
    global unstructured_partition_pdf
    global unstructured_partition_docx
    global unstructured_partition_pptx
    global unstructured_optional_decode
    from unstructured.partition.docx import partition_docx
    from unstructured.partition.md import optional_decode
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.pptx import partition_pptx
    unstructured_partition_pdf = partition_pdf
    unstructured_partition_docx = partition_docx
    unstructured_partition_pptx = partition_pptx
    unstructured_optional_decode = optional_decode

class UnstructuredParser(FileTypeParser):

    @property
    def parser_max_n_files_for_schema_inference(self) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        '\n        Just check one file as the schema is static\n        '
        return 1

    @property
    def parser_max_n_files_for_parsability(self) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Do not check any files for parsability because it might be an expensive operation and doesn't give much confidence whether the sync will succeed.\n        "
        return 0

    async def infer_schema(self, config: FileBasedStreamConfig, file: RemoteFile, stream_reader: AbstractFileBasedStreamReader, logger: logging.Logger) -> SchemaType:
        format = _extract_format(config)
        with stream_reader.open_file(file, self.file_read_mode, None, logger) as file_handle:
            filetype = self._get_filetype(file_handle, file)
            if filetype not in self._supported_file_types():
                self._handle_unprocessable_file(file, format, logger)
            return {'content': {'type': 'string'}, 'document_key': {'type': 'string'}}

    def parse_records(self, config: FileBasedStreamConfig, file: RemoteFile, stream_reader: AbstractFileBasedStreamReader, logger: logging.Logger, discovered_schema: Optional[Mapping[str, SchemaType]]) -> Iterable[Dict[str, Any]]:
        if False:
            i = 10
            return i + 15
        format = _extract_format(config)
        with stream_reader.open_file(file, self.file_read_mode, None, logger) as file_handle:
            markdown = self._read_file(file_handle, file, format, logger)
            if markdown is not None:
                yield {'content': markdown, 'document_key': file.uri}

    def _read_file(self, file_handle: IOBase, remote_file: RemoteFile, format: UnstructuredFormat, logger: logging.Logger) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        _import_unstructured()
        if not unstructured_partition_pdf or not unstructured_partition_docx or (not unstructured_partition_pptx) or (not unstructured_optional_decode):
            raise Exception('unstructured library is not available')
        filetype = self._get_filetype(file_handle, remote_file)
        if filetype == FileType.MD:
            file_content: bytes = file_handle.read()
            decoded_content: str = unstructured_optional_decode(file_content)
            return decoded_content
        if filetype not in self._supported_file_types():
            self._handle_unprocessable_file(remote_file, format, logger)
            return None
        file: Any = file_handle
        if filetype == FileType.PDF:
            file_handle.seek(0)
            file = BytesIO(file_handle.read())
            file_handle.seek(0)
            elements = unstructured_partition_pdf(file=file)
        elif filetype == FileType.DOCX:
            elements = unstructured_partition_docx(file=file)
        elif filetype == FileType.PPTX:
            elements = unstructured_partition_pptx(file=file)
        return self._render_markdown(elements)

    def _handle_unprocessable_file(self, remote_file: RemoteFile, format: UnstructuredFormat, logger: logging.Logger) -> None:
        if False:
            print('Hello World!')
        if format.skip_unprocessable_file_types:
            logger.warn(f'File {remote_file.uri} cannot be parsed. Skipping it.')
        else:
            raise RecordParseError(FileBasedSourceError.ERROR_PARSING_RECORD, filename=remote_file.uri)

    def _get_filetype(self, file: IOBase, remote_file: RemoteFile) -> Optional[FileType]:
        if False:
            i = 10
            return i + 15
        '\n        Detect the file type based on the file name and the file content.\n\n        There are three strategies to determine the file type:\n        1. Use the mime type if available (only some sources support it)\n        2. Use the file name if available\n        3. Use the file content\n        '
        if remote_file.mime_type and remote_file.mime_type in STR_TO_FILETYPE:
            return STR_TO_FILETYPE[remote_file.mime_type]
        if hasattr(file, 'name'):
            file.name = None
        file_type = detect_filetype(filename=remote_file.uri)
        if file_type is not None and (not file_type == FileType.UNK):
            return file_type
        type_based_on_content = detect_filetype(file=file)
        file.seek(0)
        return type_based_on_content

    def _supported_file_types(self) -> List[Any]:
        if False:
            return 10
        return [FileType.MD, FileType.PDF, FileType.DOCX, FileType.PPTX]

    def _render_markdown(self, elements: List[Any]) -> str:
        if False:
            while True:
                i = 10
        return '\n\n'.join((self._convert_to_markdown(el) for el in elements))

    def _convert_to_markdown(self, el: Any) -> str:
        if False:
            i = 10
            return i + 15
        if isinstance(el, Title):
            heading_str = '#' * (el.metadata.category_depth or 1)
            return f'{heading_str} {el.text}'
        elif isinstance(el, ListItem):
            return f'- {el.text}'
        elif isinstance(el, Formula):
            return f'```\n{el.text}\n```'
        else:
            return str(el.text) if hasattr(el, 'text') else ''

    @property
    def file_read_mode(self) -> FileReadMode:
        if False:
            for i in range(10):
                print('nop')
        return FileReadMode.READ_BINARY

def _extract_format(config: FileBasedStreamConfig) -> UnstructuredFormat:
    if False:
        i = 10
        return i + 15
    config_format = config.format
    if not isinstance(config_format, UnstructuredFormat):
        raise ValueError(f'Invalid format config: {config_format}')
    return config_format