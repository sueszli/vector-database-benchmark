from PyQt6.QtNetwork import QNetworkRequest, QNetworkReply
from typing import Callable, Any, Tuple, cast, Dict, Optional
from UM.Logger import Logger
from UM.TaskManagement.HttpRequestManager import HttpRequestManager
from ..Models.Http.CloudPrintJobResponse import CloudPrintJobResponse

class ToolPathUploader:
    """Class responsible for uploading meshes to the cloud in separate requests."""
    MAX_RETRIES = 10
    RETRY_HTTP_CODES = {500, 502, 503, 504}

    def __init__(self, http: HttpRequestManager, print_job: CloudPrintJobResponse, data: bytes, on_finished: Callable[[], Any], on_progress: Callable[[int], Any], on_error: Callable[[], Any]) -> None:
        if False:
            return 10
        'Creates a mesh upload object.\n\n        :param manager: The network access manager that will handle the HTTP requests.\n        :param print_job: The print job response that was returned by the cloud after registering the upload.\n        :param data: The mesh bytes to be uploaded.\n        :param on_finished: The method to be called when done.\n        :param on_progress: The method to be called when the progress changes (receives a percentage 0-100).\n        :param on_error: The method to be called when an error occurs.\n        '
        self._http = http
        self._print_job = print_job
        self._data = data
        self._on_finished = on_finished
        self._on_progress = on_progress
        self._on_error = on_error
        self._retries = 0
        self._finished = False

    @property
    def printJob(self):
        if False:
            return 10
        'Returns the print job for which this object was created.'
        return self._print_job

    def start(self) -> None:
        if False:
            print('Hello World!')
        'Starts uploading the mesh.'
        if self._finished:
            self._retries = 0
            self._finished = False
        self._upload()

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        'Stops uploading the mesh, marking it as finished.'
        Logger.log('i', 'Finished uploading')
        self._finished = True
        self._on_finished()

    def _upload(self) -> None:
        if False:
            return 10
        '\n        Uploads the print job to the cloud printer.\n        '
        if self._finished:
            raise ValueError('The upload is already finished')
        Logger.log('i', 'Uploading print to {upload_url}'.format(upload_url=self._print_job.upload_url))
        self._http.put(url=cast(str, self._print_job.upload_url), headers_dict={'Content-Type': cast(str, self._print_job.content_type)}, data=self._data, callback=self._finishedCallback, error_callback=self._errorCallback, upload_progress_callback=self._progressCallback)

    def _progressCallback(self, bytes_sent: int, bytes_total: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Handles an update to the upload progress\n\n        :param bytes_sent: The amount of bytes sent in the current request.\n        :param bytes_total: The amount of bytes to send in the current request.\n        '
        Logger.debug('Cloud upload progress %s / %s', bytes_sent, bytes_total)
        if bytes_total:
            self._on_progress(int(bytes_sent / len(self._data) * 100))

    def _errorCallback(self, reply: QNetworkReply, error: QNetworkReply.NetworkError) -> None:
        if False:
            print('Hello World!')
        'Handles an error uploading.'
        body = bytes(reply.readAll()).decode()
        Logger.log('e', 'Received error while uploading: %s', body)
        self.stop()
        self._on_error()

    def _finishedCallback(self, reply: QNetworkReply) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Checks whether a chunk of data was uploaded successfully, starting the next chunk if needed.'
        Logger.log('i', 'Finished callback %s %s', reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute), reply.url().toString())
        status_code = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        if not status_code:
            Logger.log('e', 'Reply contained no status code.')
            self._errorCallback(reply, None)
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
            self._errorCallback(reply, None)
            return
        Logger.log('d', 'status_code: %s, Headers: %s, body: %s', status_code, [bytes(header).decode() for header in reply.rawHeaderList()], bytes(reply.readAll()).decode())
        self.stop()