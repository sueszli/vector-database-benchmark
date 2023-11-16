"""
Download mode implementation.

"""
import mimetypes
import os
import re
from mailbox import Message
from time import monotonic
from typing import IO, Optional, Tuple
from urllib.parse import urlsplit
import requests
from .models import HTTPResponse, OutputOptions
from .output.streams import RawStream
from .context import Environment
PARTIAL_CONTENT = 206

class ContentRangeError(ValueError):
    pass

def parse_content_range(content_range: str, resumed_from: int) -> int:
    if False:
        return 10
    '\n    Parse and validate Content-Range header.\n\n    <https://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html>\n\n    :param content_range: the value of a Content-Range response header\n                          eg. "bytes 21010-47021/47022"\n    :param resumed_from: first byte pos. from the Range request header\n    :return: total size of the response body when fully downloaded.\n\n    '
    if content_range is None:
        raise ContentRangeError('Missing Content-Range')
    pattern = '^bytes (?P<first_byte_pos>\\d+)-(?P<last_byte_pos>\\d+)/(\\*|(?P<instance_length>\\d+))$'
    match = re.match(pattern, content_range)
    if not match:
        raise ContentRangeError(f'Invalid Content-Range format {content_range!r}')
    content_range_dict = match.groupdict()
    first_byte_pos = int(content_range_dict['first_byte_pos'])
    last_byte_pos = int(content_range_dict['last_byte_pos'])
    instance_length = int(content_range_dict['instance_length']) if content_range_dict['instance_length'] else None
    if first_byte_pos > last_byte_pos or (instance_length is not None and instance_length <= last_byte_pos):
        raise ContentRangeError(f'Invalid Content-Range returned: {content_range!r}')
    if first_byte_pos != resumed_from or (instance_length is not None and last_byte_pos + 1 != instance_length):
        raise ContentRangeError(f'Unexpected Content-Range returned ({content_range!r}) for the requested Range ("bytes={resumed_from}-")')
    return last_byte_pos + 1

def filename_from_content_disposition(content_disposition: str) -> Optional[str]:
    if False:
        return 10
    '\n    Extract and validate filename from a Content-Disposition header.\n\n    :param content_disposition: Content-Disposition value\n    :return: the filename if present and valid, otherwise `None`\n\n    '
    msg = Message(f'Content-Disposition: {content_disposition}')
    filename = msg.get_filename()
    if filename:
        filename = os.path.basename(filename).lstrip('.').strip()
        if filename:
            return filename

def filename_from_url(url: str, content_type: Optional[str]) -> str:
    if False:
        for i in range(10):
            print('nop')
    fn = urlsplit(url).path.rstrip('/')
    fn = os.path.basename(fn) if fn else 'index'
    if '.' not in fn and content_type:
        content_type = content_type.split(';')[0]
        if content_type == 'text/plain':
            ext = '.txt'
        else:
            ext = mimetypes.guess_extension(content_type)
        if ext == '.htm':
            ext = '.html'
        if ext:
            fn += ext
    return fn

def trim_filename(filename: str, max_len: int) -> str:
    if False:
        while True:
            i = 10
    if len(filename) > max_len:
        trim_by = len(filename) - max_len
        (name, ext) = os.path.splitext(filename)
        if trim_by >= len(name):
            filename = filename[:-trim_by]
        else:
            filename = name[:-trim_by] + ext
    return filename

def get_filename_max_length(directory: str) -> int:
    if False:
        i = 10
        return i + 15
    max_len = 255
    if hasattr(os, 'pathconf') and 'PC_NAME_MAX' in os.pathconf_names:
        max_len = os.pathconf(directory, 'PC_NAME_MAX')
    return max_len

def trim_filename_if_needed(filename: str, directory='.', extra=0) -> str:
    if False:
        return 10
    max_len = get_filename_max_length(directory) - extra
    if len(filename) > max_len:
        filename = trim_filename(filename, max_len)
    return filename

def get_unique_filename(filename: str, exists=os.path.exists) -> str:
    if False:
        while True:
            i = 10
    attempt = 0
    while True:
        suffix = f'-{attempt}' if attempt > 0 else ''
        try_filename = trim_filename_if_needed(filename, extra=len(suffix))
        try_filename += suffix
        if not exists(try_filename):
            return try_filename
        attempt += 1

class Downloader:

    def __init__(self, env: Environment, output_file: IO=None, resume: bool=False):
        if False:
            return 10
        '\n        :param resume: Should the download resume if partial download\n                       already exists.\n\n        :param output_file: The file to store response body in. If not\n                            provided, it will be guessed from the response.\n\n        :param progress_file: Where to report download progress.\n\n        '
        self.finished = False
        self.status = DownloadStatus(env=env)
        self._output_file = output_file
        self._resume = resume
        self._resumed_from = 0

    def pre_request(self, request_headers: dict):
        if False:
            print('Hello World!')
        'Called just before the HTTP request is sent.\n\n        Might alter `request_headers`.\n\n        '
        request_headers['Accept-Encoding'] = 'identity'
        if self._resume:
            bytes_have = os.path.getsize(self._output_file.name)
            if bytes_have:
                request_headers['Range'] = f'bytes={bytes_have}-'
                self._resumed_from = bytes_have

    def start(self, initial_url: str, final_response: requests.Response) -> Tuple[RawStream, IO]:
        if False:
            return 10
        '\n        Initiate and return a stream for `response` body  with progress\n        callback attached. Can be called only once.\n\n        :param initial_url: The original requested URL\n        :param final_response: Initiated response object with headers already fetched\n\n        :return: RawStream, output_file\n\n        '
        assert not self.status.time_started
        try:
            total_size = int(final_response.headers['Content-Length'])
        except (KeyError, ValueError, TypeError):
            total_size = None
        if not self._output_file:
            self._output_file = self._get_output_file_from_response(initial_url=initial_url, final_response=final_response)
        elif self._resume and final_response.status_code == PARTIAL_CONTENT:
            total_size = parse_content_range(final_response.headers.get('Content-Range'), self._resumed_from)
        else:
            self._resumed_from = 0
            try:
                self._output_file.seek(0)
                self._output_file.truncate()
            except OSError:
                pass
        output_options = OutputOptions.from_message(final_response, headers=False, body=True)
        stream = RawStream(msg=HTTPResponse(final_response), output_options=output_options, on_body_chunk_downloaded=self.chunk_downloaded)
        self.status.started(output_file=self._output_file, resumed_from=self._resumed_from, total_size=total_size)
        return (stream, self._output_file)

    def finish(self):
        if False:
            print('Hello World!')
        assert not self.finished
        self.finished = True
        self.status.finished()

    def failed(self):
        if False:
            while True:
                i = 10
        self.status.terminate()

    @property
    def interrupted(self) -> bool:
        if False:
            while True:
                i = 10
        return self.finished and self.status.total_size and (self.status.total_size != self.status.downloaded)

    def chunk_downloaded(self, chunk: bytes):
        if False:
            print('Hello World!')
        '\n        A download progress callback.\n\n        :param chunk: A chunk of response body data that has just\n                      been downloaded and written to the output.\n\n        '
        self.status.chunk_downloaded(len(chunk))

    @staticmethod
    def _get_output_file_from_response(initial_url: str, final_response: requests.Response) -> IO:
        if False:
            while True:
                i = 10
        filename = None
        if 'Content-Disposition' in final_response.headers:
            filename = filename_from_content_disposition(final_response.headers['Content-Disposition'])
        if not filename:
            filename = filename_from_url(url=initial_url, content_type=final_response.headers.get('Content-Type'))
        unique_filename = get_unique_filename(filename)
        return open(unique_filename, buffering=0, mode='a+b')

class DownloadStatus:
    """Holds details about the download status."""

    def __init__(self, env):
        if False:
            while True:
                i = 10
        self.env = env
        self.downloaded = 0
        self.total_size = None
        self.resumed_from = 0
        self.time_started = None
        self.time_finished = None

    def started(self, output_file, resumed_from=0, total_size=None):
        if False:
            for i in range(10):
                print('nop')
        assert self.time_started is None
        self.total_size = total_size
        self.downloaded = self.resumed_from = resumed_from
        self.time_started = monotonic()
        self.start_display(output_file=output_file)

    def start_display(self, output_file):
        if False:
            while True:
                i = 10
        from httpie.output.ui.rich_progress import DummyDisplay, StatusDisplay, ProgressDisplay
        message = f'Downloading to {output_file.name}'
        if self.env.show_displays:
            if self.total_size is None:
                self.display = StatusDisplay(self.env)
            else:
                self.display = ProgressDisplay(self.env)
        else:
            self.display = DummyDisplay(self.env)
        self.display.start(total=self.total_size, at=self.downloaded, description=message)

    def chunk_downloaded(self, size):
        if False:
            print('Hello World!')
        assert self.time_finished is None
        self.downloaded += size
        self.display.update(size)

    @property
    def has_finished(self):
        if False:
            while True:
                i = 10
        return self.time_finished is not None

    @property
    def time_spent(self):
        if False:
            print('Hello World!')
        if self.time_started is not None and self.time_finished is not None:
            return self.time_finished - self.time_started
        else:
            return None

    def finished(self):
        if False:
            print('Hello World!')
        assert self.time_started is not None
        assert self.time_finished is None
        self.time_finished = monotonic()
        if hasattr(self, 'display'):
            self.display.stop(self.time_spent)

    def terminate(self):
        if False:
            print('Hello World!')
        if hasattr(self, 'display'):
            self.display.stop(self.time_spent)