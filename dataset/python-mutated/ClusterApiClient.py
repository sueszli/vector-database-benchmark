import json
from json import JSONDecodeError
from typing import Callable, List, Optional, Dict, Union, Any, Type, cast, TypeVar, Tuple
from PyQt6.QtCore import QUrl
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from UM.Logger import Logger
from ..Models.BaseModel import BaseModel
from ..Models.Http.ClusterPrintJobStatus import ClusterPrintJobStatus
from ..Models.Http.ClusterPrinterStatus import ClusterPrinterStatus
from ..Models.Http.PrinterSystemStatus import PrinterSystemStatus
from ..Models.Http.ClusterMaterial import ClusterMaterial
ClusterApiClientModel = TypeVar('ClusterApiClientModel', bound=BaseModel)
'The generic type variable used to document the methods below.'

class ClusterApiClient:
    """The ClusterApiClient is responsible for all network calls to local network clusters."""
    PRINTER_API_PREFIX = '/api/v1'
    CLUSTER_API_PREFIX = '/cluster-api/v1'
    _anti_gc_callbacks = []

    def __init__(self, address: str, on_error: Callable) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initializes a new cluster API client.\n\n        :param address: The network address of the cluster to call.\n        :param on_error: The callback to be called whenever we receive errors from the server.\n        '
        super().__init__()
        self._manager = QNetworkAccessManager()
        self._address = address
        self._on_error = on_error

    def getSystem(self, on_finished: Callable) -> None:
        if False:
            return 10
        'Get printer system information.\n\n        :param on_finished: The callback in case the response is successful.\n        '
        url = '{}/system'.format(self.PRINTER_API_PREFIX)
        reply = self._manager.get(self._createEmptyRequest(url))
        self._addCallback(reply, on_finished, PrinterSystemStatus)

    def getMaterials(self, on_finished: Callable[[List[ClusterMaterial]], Any]) -> None:
        if False:
            print('Hello World!')
        'Get the installed materials on the printer.\n\n        :param on_finished: The callback in case the response is successful.\n        '
        url = '{}/materials'.format(self.CLUSTER_API_PREFIX)
        reply = self._manager.get(self._createEmptyRequest(url))
        self._addCallback(reply, on_finished, ClusterMaterial)

    def getPrinters(self, on_finished: Callable[[List[ClusterPrinterStatus]], Any]) -> None:
        if False:
            i = 10
            return i + 15
        'Get the printers in the cluster.\n\n        :param on_finished: The callback in case the response is successful.\n        '
        url = '{}/printers'.format(self.CLUSTER_API_PREFIX)
        reply = self._manager.get(self._createEmptyRequest(url))
        self._addCallback(reply, on_finished, ClusterPrinterStatus)

    def getPrintJobs(self, on_finished: Callable[[List[ClusterPrintJobStatus]], Any]) -> None:
        if False:
            return 10
        'Get the print jobs in the cluster.\n\n        :param on_finished: The callback in case the response is successful.\n        '
        url = '{}/print_jobs'.format(self.CLUSTER_API_PREFIX)
        reply = self._manager.get(self._createEmptyRequest(url))
        self._addCallback(reply, on_finished, ClusterPrintJobStatus)

    def movePrintJobToTop(self, print_job_uuid: str) -> None:
        if False:
            return 10
        'Move a print job to the top of the queue.'
        url = '{}/print_jobs/{}/action/move'.format(self.CLUSTER_API_PREFIX, print_job_uuid)
        self._manager.post(self._createEmptyRequest(url), json.dumps({'to_position': 0, 'list': 'queued'}).encode())

    def forcePrintJob(self, print_job_uuid: str) -> None:
        if False:
            return 10
        'Override print job configuration and force it to be printed.'
        url = '{}/print_jobs/{}'.format(self.CLUSTER_API_PREFIX, print_job_uuid)
        self._manager.put(self._createEmptyRequest(url), json.dumps({'force': True}).encode())

    def deletePrintJob(self, print_job_uuid: str) -> None:
        if False:
            return 10
        'Delete a print job from the queue.'
        url = '{}/print_jobs/{}'.format(self.CLUSTER_API_PREFIX, print_job_uuid)
        self._manager.deleteResource(self._createEmptyRequest(url))

    def setPrintJobState(self, print_job_uuid: str, state: str) -> None:
        if False:
            while True:
                i = 10
        'Set the state of a print job.'
        url = '{}/print_jobs/{}/action'.format(self.CLUSTER_API_PREFIX, print_job_uuid)
        action = 'print' if state == 'resume' else state
        self._manager.put(self._createEmptyRequest(url), json.dumps({'action': action}).encode())

    def getPrintJobPreviewImage(self, print_job_uuid: str, on_finished: Callable) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Get the preview image data of a print job.'
        url = '{}/print_jobs/{}/preview_image'.format(self.CLUSTER_API_PREFIX, print_job_uuid)
        reply = self._manager.get(self._createEmptyRequest(url))
        self._addCallback(reply, on_finished)

    def _createEmptyRequest(self, path: str, content_type: Optional[str]='application/json') -> QNetworkRequest:
        if False:
            print('Hello World!')
        'We override _createEmptyRequest in order to add the user credentials.\n\n        :param url: The URL to request\n        :param content_type: The type of the body contents.\n        '
        url = QUrl('http://' + self._address + path)
        request = QNetworkRequest(url)
        if content_type:
            request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, content_type)
        return request

    @staticmethod
    def _parseReply(reply: QNetworkReply) -> Tuple[int, Dict[str, Any]]:
        if False:
            print('Hello World!')
        'Parses the given JSON network reply into a status code and a dictionary, handling unexpected errors as well.\n\n        :param reply: The reply from the server.\n        :return: A tuple with a status code and a dictionary.\n        '
        status_code = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        try:
            response = bytes(reply.readAll()).decode()
            return (status_code, json.loads(response))
        except (UnicodeDecodeError, JSONDecodeError, ValueError) as err:
            Logger.logException('e', 'Could not parse the cluster response: %s', err)
            return (status_code, {'errors': [err]})

    def _parseModels(self, response: Dict[str, Any], on_finished: Union[Callable[[ClusterApiClientModel], Any], Callable[[List[ClusterApiClientModel]], Any]], model_class: Type[ClusterApiClientModel]) -> None:
        if False:
            i = 10
            return i + 15
        'Parses the given models and calls the correct callback depending on the result.\n\n        :param response: The response from the server, after being converted to a dict.\n        :param on_finished: The callback in case the response is successful.\n        :param model_class: The type of the model to convert the response to. It may either be a single record or a list.\n        '
        try:
            if isinstance(response, list):
                results = [model_class(**c) for c in response]
                on_finished_list = cast(Callable[[List[ClusterApiClientModel]], Any], on_finished)
                on_finished_list(results)
            else:
                result = model_class(**response)
                on_finished_item = cast(Callable[[ClusterApiClientModel], Any], on_finished)
                on_finished_item(result)
        except (JSONDecodeError, TypeError, ValueError):
            Logger.log('e', 'Could not parse response from network: %s', str(response))

    def _addCallback(self, reply: QNetworkReply, on_finished: Union[Callable[[ClusterApiClientModel], Any], Callable[[List[ClusterApiClientModel]], Any]], model: Type[ClusterApiClientModel]=None) -> None:
        if False:
            i = 10
            return i + 15
        "Creates a callback function so that it includes the parsing of the response into the correct model.\n\n        The callback is added to the 'finished' signal of the reply.\n        :param reply: The reply that should be listened to.\n        :param on_finished: The callback in case the response is successful.\n        "

        def parse() -> None:
            if False:
                return 10
            try:
                self._anti_gc_callbacks.remove(parse)
            except ValueError:
                return
            if reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute) is None:
                return
            if reply.error() != QNetworkReply.NetworkError.NoError:
                self._on_error(reply.errorString())
                return
            if not model:
                on_finished(reply.readAll())
                return
            (status_code, response) = self._parseReply(reply)
            self._parseModels(response, on_finished, model)
        self._anti_gc_callbacks.append(parse)
        reply.finished.connect(parse)