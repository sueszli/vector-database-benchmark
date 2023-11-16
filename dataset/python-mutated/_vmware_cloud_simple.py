from copy import deepcopy
from typing import Any, TYPE_CHECKING
from azure.core.rest import HttpRequest, HttpResponse
from azure.mgmt.core import ARMPipelineClient
from . import models
from ._configuration import VMwareCloudSimpleConfiguration
from ._serialization import Deserializer, Serializer
from .operations import CustomizationPoliciesOperations, DedicatedCloudNodesOperations, DedicatedCloudServicesOperations, Operations, PrivateCloudsOperations, ResourcePoolsOperations, SkusAvailabilityOperations, UsagesOperations, VirtualMachineTemplatesOperations, VirtualMachinesOperations, VirtualNetworksOperations
if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class VMwareCloudSimple:
    """Description of the new service.

    :ivar operations: Operations operations
    :vartype operations: azure.mgmt.vmwarecloudsimple.operations.Operations
    :ivar dedicated_cloud_nodes: DedicatedCloudNodesOperations operations
    :vartype dedicated_cloud_nodes:
     azure.mgmt.vmwarecloudsimple.operations.DedicatedCloudNodesOperations
    :ivar dedicated_cloud_services: DedicatedCloudServicesOperations operations
    :vartype dedicated_cloud_services:
     azure.mgmt.vmwarecloudsimple.operations.DedicatedCloudServicesOperations
    :ivar skus_availability: SkusAvailabilityOperations operations
    :vartype skus_availability: azure.mgmt.vmwarecloudsimple.operations.SkusAvailabilityOperations
    :ivar private_clouds: PrivateCloudsOperations operations
    :vartype private_clouds: azure.mgmt.vmwarecloudsimple.operations.PrivateCloudsOperations
    :ivar customization_policies: CustomizationPoliciesOperations operations
    :vartype customization_policies:
     azure.mgmt.vmwarecloudsimple.operations.CustomizationPoliciesOperations
    :ivar resource_pools: ResourcePoolsOperations operations
    :vartype resource_pools: azure.mgmt.vmwarecloudsimple.operations.ResourcePoolsOperations
    :ivar virtual_machine_templates: VirtualMachineTemplatesOperations operations
    :vartype virtual_machine_templates:
     azure.mgmt.vmwarecloudsimple.operations.VirtualMachineTemplatesOperations
    :ivar virtual_networks: VirtualNetworksOperations operations
    :vartype virtual_networks: azure.mgmt.vmwarecloudsimple.operations.VirtualNetworksOperations
    :ivar usages: UsagesOperations operations
    :vartype usages: azure.mgmt.vmwarecloudsimple.operations.UsagesOperations
    :ivar virtual_machines: VirtualMachinesOperations operations
    :vartype virtual_machines: azure.mgmt.vmwarecloudsimple.operations.VirtualMachinesOperations
    :param credential: Credential needed for the client to connect to Azure. Required.
    :type credential: ~azure.core.credentials.TokenCredential
    :param subscription_id: The subscription ID. Required.
    :type subscription_id: str
    :param base_url: Service URL. Default value is "https://management.azure.com".
    :type base_url: str
    :keyword api_version: Api Version. Default value is "2019-04-01". Note that overriding this
     default value may result in unsupported behavior.
    :paramtype api_version: str
    :keyword int polling_interval: Default waiting time between two polls for LRO operations if no
     Retry-After header is present.
    """

    def __init__(self, credential: 'TokenCredential', subscription_id: str, base_url: str='https://management.azure.com', **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._config = VMwareCloudSimpleConfiguration(credential=credential, subscription_id=subscription_id, **kwargs)
        self._client = ARMPipelineClient(base_url=base_url, config=self._config, **kwargs)
        client_models = {k: v for (k, v) in models.__dict__.items() if isinstance(v, type)}
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)
        self._serialize.client_side_validation = False
        self.operations = Operations(self._client, self._config, self._serialize, self._deserialize)
        self.dedicated_cloud_nodes = DedicatedCloudNodesOperations(self._client, self._config, self._serialize, self._deserialize)
        self.dedicated_cloud_services = DedicatedCloudServicesOperations(self._client, self._config, self._serialize, self._deserialize)
        self.skus_availability = SkusAvailabilityOperations(self._client, self._config, self._serialize, self._deserialize)
        self.private_clouds = PrivateCloudsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.customization_policies = CustomizationPoliciesOperations(self._client, self._config, self._serialize, self._deserialize)
        self.resource_pools = ResourcePoolsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.virtual_machine_templates = VirtualMachineTemplatesOperations(self._client, self._config, self._serialize, self._deserialize)
        self.virtual_networks = VirtualNetworksOperations(self._client, self._config, self._serialize, self._deserialize)
        self.usages = UsagesOperations(self._client, self._config, self._serialize, self._deserialize)
        self.virtual_machines = VirtualMachinesOperations(self._client, self._config, self._serialize, self._deserialize)

    def _send_request(self, request: HttpRequest, **kwargs: Any) -> HttpResponse:
        if False:
            for i in range(10):
                print('nop')
        'Runs the network request through the client\'s chained policies.\n\n        >>> from azure.core.rest import HttpRequest\n        >>> request = HttpRequest("GET", "https://www.example.org/")\n        <HttpRequest [GET], url: \'https://www.example.org/\'>\n        >>> response = client._send_request(request)\n        <HttpResponse: 200 OK>\n\n        For more information on this code flow, see https://aka.ms/azsdk/dpcodegen/python/send_request\n\n        :param request: The network request you want to make. Required.\n        :type request: ~azure.core.rest.HttpRequest\n        :keyword bool stream: Whether the response payload will be streamed. Defaults to False.\n        :return: The response of your network call. Does not do error handling on your response.\n        :rtype: ~azure.core.rest.HttpResponse\n        '
        request_copy = deepcopy(request)
        request_copy.url = self._client.format_url(request_copy.url)
        return self._client.send_request(request_copy, **kwargs)

    def close(self):
        if False:
            i = 10
            return i + 15
        self._client.close()

    def __enter__(self):
        if False:
            while True:
                i = 10
        self._client.__enter__()
        return self

    def __exit__(self, *exc_details):
        if False:
            while True:
                i = 10
        self._client.__exit__(*exc_details)