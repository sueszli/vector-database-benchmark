from copy import deepcopy
from typing import Any, Awaitable, Optional, TYPE_CHECKING
from azure.core.rest import AsyncHttpResponse, HttpRequest
from azure.mgmt.core import AsyncARMPipelineClient
from .. import models as _models
from .._serialization import Deserializer, Serializer
from ._configuration import BillingBenefitsRPConfiguration
from .operations import BillingBenefitsRPOperationsMixin, Operations, ReservationOrderAliasOperations, SavingsPlanOperations, SavingsPlanOrderAliasOperations, SavingsPlanOrderOperations
if TYPE_CHECKING:
    from azure.core.credentials_async import AsyncTokenCredential

class BillingBenefitsRP(BillingBenefitsRPOperationsMixin):
    """Azure Benefits RP let users create and manage benefits like savings plan.

    :ivar operations: Operations operations
    :vartype operations: azure.mgmt.billingbenefits.aio.operations.Operations
    :ivar savings_plan_order_alias: SavingsPlanOrderAliasOperations operations
    :vartype savings_plan_order_alias:
     azure.mgmt.billingbenefits.aio.operations.SavingsPlanOrderAliasOperations
    :ivar savings_plan_order: SavingsPlanOrderOperations operations
    :vartype savings_plan_order:
     azure.mgmt.billingbenefits.aio.operations.SavingsPlanOrderOperations
    :ivar savings_plan: SavingsPlanOperations operations
    :vartype savings_plan: azure.mgmt.billingbenefits.aio.operations.SavingsPlanOperations
    :ivar reservation_order_alias: ReservationOrderAliasOperations operations
    :vartype reservation_order_alias:
     azure.mgmt.billingbenefits.aio.operations.ReservationOrderAliasOperations
    :param credential: Credential needed for the client to connect to Azure. Required.
    :type credential: ~azure.core.credentials_async.AsyncTokenCredential
    :param expand: May be used to expand the detail information of some properties. Default value
     is None.
    :type expand: str
    :param base_url: Service URL. Default value is "https://management.azure.com".
    :type base_url: str
    :keyword api_version: Api Version. Default value is "2022-11-01". Note that overriding this
     default value may result in unsupported behavior.
    :paramtype api_version: str
    :keyword int polling_interval: Default waiting time between two polls for LRO operations if no
     Retry-After header is present.
    """

    def __init__(self, credential: 'AsyncTokenCredential', expand: Optional[str]=None, base_url: str='https://management.azure.com', **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        self._config = BillingBenefitsRPConfiguration(credential=credential, expand=expand, **kwargs)
        self._client = AsyncARMPipelineClient(base_url=base_url, config=self._config, **kwargs)
        client_models = {k: v for (k, v) in _models.__dict__.items() if isinstance(v, type)}
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)
        self._serialize.client_side_validation = False
        self.operations = Operations(self._client, self._config, self._serialize, self._deserialize)
        self.savings_plan_order_alias = SavingsPlanOrderAliasOperations(self._client, self._config, self._serialize, self._deserialize)
        self.savings_plan_order = SavingsPlanOrderOperations(self._client, self._config, self._serialize, self._deserialize)
        self.savings_plan = SavingsPlanOperations(self._client, self._config, self._serialize, self._deserialize)
        self.reservation_order_alias = ReservationOrderAliasOperations(self._client, self._config, self._serialize, self._deserialize)

    def _send_request(self, request: HttpRequest, **kwargs: Any) -> Awaitable[AsyncHttpResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Runs the network request through the client\'s chained policies.\n\n        >>> from azure.core.rest import HttpRequest\n        >>> request = HttpRequest("GET", "https://www.example.org/")\n        <HttpRequest [GET], url: \'https://www.example.org/\'>\n        >>> response = await client._send_request(request)\n        <AsyncHttpResponse: 200 OK>\n\n        For more information on this code flow, see https://aka.ms/azsdk/dpcodegen/python/send_request\n\n        :param request: The network request you want to make. Required.\n        :type request: ~azure.core.rest.HttpRequest\n        :keyword bool stream: Whether the response payload will be streamed. Defaults to False.\n        :return: The response of your network call. Does not do error handling on your response.\n        :rtype: ~azure.core.rest.AsyncHttpResponse\n        '
        request_copy = deepcopy(request)
        request_copy.url = self._client.format_url(request_copy.url)
        return self._client.send_request(request_copy, **kwargs)

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> 'BillingBenefitsRP':
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc_details) -> None:
        await self._client.__aexit__(*exc_details)