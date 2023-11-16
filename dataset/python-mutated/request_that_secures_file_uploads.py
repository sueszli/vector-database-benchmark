from io import BytesIO
from typing import IO, Optional
from flask import wrappers
from secure_tempfile import SecureTemporaryFile
from werkzeug.formparser import FormDataParser

class RequestThatSecuresFileUploads(wrappers.Request):

    def _secure_file_stream(self, total_content_length: Optional[int], content_type: Optional[str], filename: Optional[str]=None, content_length: Optional[int]=None) -> IO[bytes]:
        if False:
            for i in range(10):
                print('nop')
        'Storage class for data streamed in from requests.\n\n        If the data is relatively small (512KB), just store it in\n        memory. Otherwise, use the SecureTemporaryFile class to buffer\n        it on disk, encrypted with an ephemeral key to mitigate\n        forensic recovery of the plaintext.\n\n        '
        if total_content_length is None or total_content_length > 1024 * 512:
            return SecureTemporaryFile('/tmp')
        return BytesIO()

    def make_form_data_parser(self) -> FormDataParser:
        if False:
            return 10
        return self.form_data_parser_class(self._secure_file_stream, self.charset, self.encoding_errors, self.max_form_memory_size, self.max_content_length, self.parameter_storage_class)