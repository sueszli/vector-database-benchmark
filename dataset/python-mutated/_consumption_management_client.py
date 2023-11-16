from copy import deepcopy
from typing import Any, TYPE_CHECKING
from azure.core.rest import HttpRequest, HttpResponse
from azure.mgmt.core import ARMPipelineClient
from . import models as _models
from ._configuration import ConsumptionManagementClientConfiguration
from ._serialization import Deserializer, Serializer
from .operations import AggregatedCostOperations, BalancesOperations, BudgetsOperations, ChargesOperations, CreditsOperations, EventsOperations, LotsOperations, MarketplacesOperations, Operations, PriceSheetOperations, ReservationRecommendationDetailsOperations, ReservationRecommendationsOperations, ReservationTransactionsOperations, ReservationsDetailsOperations, ReservationsSummariesOperations, TagsOperations, UsageDetailsOperations
if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class ConsumptionManagementClient:
    """Consumption management client provides access to consumption resources for Azure Enterprise
    Subscriptions.

    :ivar usage_details: UsageDetailsOperations operations
    :vartype usage_details: azure.mgmt.consumption.operations.UsageDetailsOperations
    :ivar marketplaces: MarketplacesOperations operations
    :vartype marketplaces: azure.mgmt.consumption.operations.MarketplacesOperations
    :ivar budgets: BudgetsOperations operations
    :vartype budgets: azure.mgmt.consumption.operations.BudgetsOperations
    :ivar tags: TagsOperations operations
    :vartype tags: azure.mgmt.consumption.operations.TagsOperations
    :ivar charges: ChargesOperations operations
    :vartype charges: azure.mgmt.consumption.operations.ChargesOperations
    :ivar balances: BalancesOperations operations
    :vartype balances: azure.mgmt.consumption.operations.BalancesOperations
    :ivar reservations_summaries: ReservationsSummariesOperations operations
    :vartype reservations_summaries:
     azure.mgmt.consumption.operations.ReservationsSummariesOperations
    :ivar reservations_details: ReservationsDetailsOperations operations
    :vartype reservations_details: azure.mgmt.consumption.operations.ReservationsDetailsOperations
    :ivar reservation_recommendations: ReservationRecommendationsOperations operations
    :vartype reservation_recommendations:
     azure.mgmt.consumption.operations.ReservationRecommendationsOperations
    :ivar reservation_recommendation_details: ReservationRecommendationDetailsOperations operations
    :vartype reservation_recommendation_details:
     azure.mgmt.consumption.operations.ReservationRecommendationDetailsOperations
    :ivar reservation_transactions: ReservationTransactionsOperations operations
    :vartype reservation_transactions:
     azure.mgmt.consumption.operations.ReservationTransactionsOperations
    :ivar price_sheet: PriceSheetOperations operations
    :vartype price_sheet: azure.mgmt.consumption.operations.PriceSheetOperations
    :ivar operations: Operations operations
    :vartype operations: azure.mgmt.consumption.operations.Operations
    :ivar aggregated_cost: AggregatedCostOperations operations
    :vartype aggregated_cost: azure.mgmt.consumption.operations.AggregatedCostOperations
    :ivar events: EventsOperations operations
    :vartype events: azure.mgmt.consumption.operations.EventsOperations
    :ivar lots: LotsOperations operations
    :vartype lots: azure.mgmt.consumption.operations.LotsOperations
    :ivar credits: CreditsOperations operations
    :vartype credits: azure.mgmt.consumption.operations.CreditsOperations
    :param credential: Credential needed for the client to connect to Azure. Required.
    :type credential: ~azure.core.credentials.TokenCredential
    :param subscription_id: Azure Subscription ID. Required.
    :type subscription_id: str
    :param base_url: Service URL. Default value is "https://management.azure.com".
    :type base_url: str
    :keyword api_version: Api Version. Default value is "2021-10-01". Note that overriding this
     default value may result in unsupported behavior.
    :paramtype api_version: str
    """

    def __init__(self, credential: 'TokenCredential', subscription_id: str, base_url: str='https://management.azure.com', **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        self._config = ConsumptionManagementClientConfiguration(credential=credential, subscription_id=subscription_id, **kwargs)
        self._client = ARMPipelineClient(base_url=base_url, config=self._config, **kwargs)
        client_models = {k: v for (k, v) in _models.__dict__.items() if isinstance(v, type)}
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)
        self._serialize.client_side_validation = False
        self.usage_details = UsageDetailsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.marketplaces = MarketplacesOperations(self._client, self._config, self._serialize, self._deserialize)
        self.budgets = BudgetsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.tags = TagsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.charges = ChargesOperations(self._client, self._config, self._serialize, self._deserialize)
        self.balances = BalancesOperations(self._client, self._config, self._serialize, self._deserialize)
        self.reservations_summaries = ReservationsSummariesOperations(self._client, self._config, self._serialize, self._deserialize)
        self.reservations_details = ReservationsDetailsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.reservation_recommendations = ReservationRecommendationsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.reservation_recommendation_details = ReservationRecommendationDetailsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.reservation_transactions = ReservationTransactionsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.price_sheet = PriceSheetOperations(self._client, self._config, self._serialize, self._deserialize)
        self.operations = Operations(self._client, self._config, self._serialize, self._deserialize)
        self.aggregated_cost = AggregatedCostOperations(self._client, self._config, self._serialize, self._deserialize)
        self.events = EventsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.lots = LotsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.credits = CreditsOperations(self._client, self._config, self._serialize, self._deserialize)

    def _send_request(self, request: HttpRequest, **kwargs: Any) -> HttpResponse:
        if False:
            while True:
                i = 10
        'Runs the network request through the client\'s chained policies.\n\n        >>> from azure.core.rest import HttpRequest\n        >>> request = HttpRequest("GET", "https://www.example.org/")\n        <HttpRequest [GET], url: \'https://www.example.org/\'>\n        >>> response = client._send_request(request)\n        <HttpResponse: 200 OK>\n\n        For more information on this code flow, see https://aka.ms/azsdk/dpcodegen/python/send_request\n\n        :param request: The network request you want to make. Required.\n        :type request: ~azure.core.rest.HttpRequest\n        :keyword bool stream: Whether the response payload will be streamed. Defaults to False.\n        :return: The response of your network call. Does not do error handling on your response.\n        :rtype: ~azure.core.rest.HttpResponse\n        '
        request_copy = deepcopy(request)
        request_copy.url = self._client.format_url(request_copy.url)
        return self._client.send_request(request_copy, **kwargs)

    def close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._client.close()

    def __enter__(self) -> 'ConsumptionManagementClient':
        if False:
            for i in range(10):
                print('nop')
        self._client.__enter__()
        return self

    def __exit__(self, *exc_details) -> None:
        if False:
            while True:
                i = 10
        self._client.__exit__(*exc_details)