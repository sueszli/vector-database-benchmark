from copy import deepcopy
from typing import Any, TYPE_CHECKING
from azure.core.rest import HttpRequest, HttpResponse
from azure.mgmt.core import ARMPipelineClient
from . import models
from ._configuration import AppComplianceAutomationToolForMicrosoft365Configuration
from ._serialization import Deserializer, Serializer
from .operations import Operations, ReportOperations, ReportsOperations, SnapshotOperations, SnapshotsOperations
if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class AppComplianceAutomationToolForMicrosoft365:
    """App Compliance Automation Tool for Microsoft 365 API spec.

    :ivar operations: Operations operations
    :vartype operations: azure.mgmt.appcomplianceautomation.operations.Operations
    :ivar reports: ReportsOperations operations
    :vartype reports: azure.mgmt.appcomplianceautomation.operations.ReportsOperations
    :ivar report: ReportOperations operations
    :vartype report: azure.mgmt.appcomplianceautomation.operations.ReportOperations
    :ivar snapshots: SnapshotsOperations operations
    :vartype snapshots: azure.mgmt.appcomplianceautomation.operations.SnapshotsOperations
    :ivar snapshot: SnapshotOperations operations
    :vartype snapshot: azure.mgmt.appcomplianceautomation.operations.SnapshotOperations
    :param credential: Credential needed for the client to connect to Azure. Required.
    :type credential: ~azure.core.credentials.TokenCredential
    :param base_url: Service URL. Default value is "https://management.azure.com".
    :type base_url: str
    :keyword api_version: Api Version. Default value is "2022-11-16-preview". Note that overriding
     this default value may result in unsupported behavior.
    :paramtype api_version: str
    :keyword int polling_interval: Default waiting time between two polls for LRO operations if no
     Retry-After header is present.
    """

    def __init__(self, credential: 'TokenCredential', base_url: str='https://management.azure.com', **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._config = AppComplianceAutomationToolForMicrosoft365Configuration(credential=credential, **kwargs)
        self._client = ARMPipelineClient(base_url=base_url, config=self._config, **kwargs)
        client_models = {k: v for (k, v) in models.__dict__.items() if isinstance(v, type)}
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)
        self._serialize.client_side_validation = False
        self.operations = Operations(self._client, self._config, self._serialize, self._deserialize)
        self.reports = ReportsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.report = ReportOperations(self._client, self._config, self._serialize, self._deserialize)
        self.snapshots = SnapshotsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.snapshot = SnapshotOperations(self._client, self._config, self._serialize, self._deserialize)

    def _send_request(self, request: HttpRequest, **kwargs: Any) -> HttpResponse:
        if False:
            print('Hello World!')
        'Runs the network request through the client\'s chained policies.\n\n        >>> from azure.core.rest import HttpRequest\n        >>> request = HttpRequest("GET", "https://www.example.org/")\n        <HttpRequest [GET], url: \'https://www.example.org/\'>\n        >>> response = client._send_request(request)\n        <HttpResponse: 200 OK>\n\n        For more information on this code flow, see https://aka.ms/azsdk/dpcodegen/python/send_request\n\n        :param request: The network request you want to make. Required.\n        :type request: ~azure.core.rest.HttpRequest\n        :keyword bool stream: Whether the response payload will be streamed. Defaults to False.\n        :return: The response of your network call. Does not do error handling on your response.\n        :rtype: ~azure.core.rest.HttpResponse\n        '
        request_copy = deepcopy(request)
        request_copy.url = self._client.format_url(request_copy.url)
        return self._client.send_request(request_copy, **kwargs)

    def close(self):
        if False:
            while True:
                i = 10
        self._client.close()

    def __enter__(self):
        if False:
            return 10
        self._client.__enter__()
        return self

    def __exit__(self, *exc_details):
        if False:
            print('Hello World!')
        self._client.__exit__(*exc_details)