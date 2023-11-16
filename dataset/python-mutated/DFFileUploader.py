from PyQt6.QtNetwork import QNetworkRequest, QNetworkReply
from typing import Callable, Any, cast, Optional, Union
from UM.Logger import Logger
from UM.TaskManagement.HttpRequestManager import HttpRequestManager
from .DFLibraryFileUploadResponse import DFLibraryFileUploadResponse
from .DFPrintJobUploadResponse import DFPrintJobUploadResponse

class DFFileUploader:
    """Class responsible for uploading meshes to the the digital factory library in separate requests."""
    MAX_RETRIES = 10
    RETRY_HTTP_CODES = {500, 502, 503, 504}

    def __init__(self, http: HttpRequestManager, df_file: Union[DFLibraryFileUploadResponse, DFPrintJobUploadResponse], data: bytes, on_finished: Callable[[str], Any], on_success: Callable[[str], Any], on_progress: Callable[[str, int], Any], on_error: Callable[[str, 'QNetworkReply', 'QNetworkReply.NetworkError'], Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Creates a mesh upload object.\n\n        :param http: The network access manager that will handle the HTTP requests.\n        :param df_file: The file response that was received by the Digital Factory after registering the upload.\n        :param data: The mesh bytes to be uploaded.\n        :param on_finished: The method to be called when done.\n        :param on_success: The method to be called when the upload is successful.\n        :param on_progress: The method to be called when the progress changes (receives a percentage 0-100).\n        :param on_error: The method to be called when an error occurs.\n        '
        self._http: HttpRequestManager = http
        self._df_file: Union[DFLibraryFileUploadResponse, DFPrintJobUploadResponse] = df_file
        self._file_name = ''
        if isinstance(self._df_file, DFLibraryFileUploadResponse):
            self._file_name = self._df_file.file_name
        elif isinstance(self._df_file, DFPrintJobUploadResponse):
            if self._df_file.job_name is not None:
                self._file_name = self._df_file.job_name
            else:
                self._file_name = ''
        else:
            raise TypeError('Incorrect input type')
        self._data: bytes = data
        self._on_finished = on_finished
        self._on_success = on_success
        self._on_progress = on_progress
        self._on_error = on_error
        self._retries = 0
        self._finished = False

    def start(self) -> None:
        if False:
            i = 10
            return i + 15
        'Starts uploading the mesh.'
        if self._finished:
            self._retries = 0
            self._finished = False
        self._upload()

    def stop(self):
        if False:
            return 10
        'Stops uploading the mesh, marking it as finished.'
        Logger.log('i', 'Finished uploading')
        self._finished = True
        self._on_finished(self._file_name)

    def _upload(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Uploads the file to the Digital Factory Library project\n        '
        if self._finished:
            raise ValueError('The upload is already finished')
        if isinstance(self._df_file, DFLibraryFileUploadResponse):
            Logger.log('i', "Uploading Cura project file '{file_name}' via link '{upload_url}'".format(file_name=self._df_file.file_name, upload_url=self._df_file.upload_url))
        elif isinstance(self._df_file, DFPrintJobUploadResponse):
            Logger.log('i', "Uploading Cura print file '{file_name}' via link '{upload_url}'".format(file_name=self._df_file.job_name, upload_url=self._df_file.upload_url))
        self._http.put(url=cast(str, self._df_file.upload_url), headers_dict={'Content-Type': cast(str, self._df_file.content_type)}, data=self._data, callback=self._onUploadFinished, error_callback=self._onUploadError, upload_progress_callback=self._onUploadProgressChanged)

    def _onUploadProgressChanged(self, bytes_sent: int, bytes_total: int) -> None:
        if False:
            while True:
                i = 10
        'Handles an update to the upload progress\n\n        :param bytes_sent: The amount of bytes sent in the current request.\n        :param bytes_total: The amount of bytes to send in the current request.\n        '
        Logger.debug('Cloud upload progress %s / %s', bytes_sent, bytes_total)
        if bytes_total:
            self._on_progress(self._file_name, int(bytes_sent / len(self._data) * 100))

    def _onUploadError(self, reply: QNetworkReply, error: QNetworkReply.NetworkError) -> None:
        if False:
            while True:
                i = 10
        'Handles an error uploading.'
        body = bytes(reply.peek(reply.bytesAvailable())).decode()
        Logger.log('e', 'Received error while uploading: %s', body)
        self._on_error(self._file_name, reply, error)
        self.stop()

    def _onUploadFinished(self, reply: QNetworkReply) -> None:
        if False:
            print('Hello World!')
        '\n        Checks whether a chunk of data was uploaded successfully, starting the next chunk if needed.\n        '
        Logger.log('i', 'Finished callback %s %s', reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute), reply.url().toString())
        status_code = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        if not status_code:
            Logger.log('e', 'Reply contained no status code.')
            self._onUploadError(reply, None)
            return
        if self._retries < self.MAX_RETRIES and status_code in self.RETRY_HTTP_CODES:
            self._retries += 1
            Logger.log('i', 'Retrying %s/%s request %s', self._retries, self.MAX_RETRIES, reply.url().toString())
            try:
                self._upload()
            except ValueError:
                pass
            return
        if status_code > 308:
            self._onUploadError(reply, None)
            return
        Logger.log('d', 'status_code: %s, Headers: %s, body: %s', status_code, [bytes(header).decode() for header in reply.rawHeaderList()], bytes(reply.readAll()).decode())
        self._on_success(self._file_name)
        self.stop()