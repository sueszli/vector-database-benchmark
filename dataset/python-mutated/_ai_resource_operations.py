from typing import Any, Iterable
from azure.core.tracing.decorator import distributed_trace
from azure.ai.resources.constants._common import DEFAULT_OPEN_AI_CONNECTION_NAME
from azure.ai.resources.entities import AIResource
from azure.ai.ml import MLClient
from azure.ai.ml.constants._common import Scope
from azure.ai.ml.entities._workspace_hub._constants import ENDPOINT_AI_SERVICE_KIND
from azure.core.polling import LROPoller
from azure.ai.resources._telemetry import ActivityType, monitor_with_activity, monitor_with_telemetry_mixin, OpsLogger
ops_logger = OpsLogger(__name__)
(logger, module_logger) = (ops_logger.package_logger, ops_logger.module_logger)

class AIResourceOperations:
    """AIResourceOperations.

    You should not instantiate this class directly. Instead, you should
    create an MLClient instance that instantiates it for you and
    attaches it as an attribute.
    """

    def __init__(self, ml_client: MLClient, **kwargs: Any):
        if False:
            for i in range(10):
                print('nop')
        self._ml_client = ml_client
        ops_logger.update_info(kwargs)

    @distributed_trace
    @monitor_with_activity(logger, 'AIResource.Get', ActivityType.PUBLICAPI)
    def get(self, *, name: str, **kwargs) -> AIResource:
        if False:
            while True:
                i = 10
        'Get an AI resource by name.\n\n        :keyword name: Name of the AI resource.\n        :paramtype name: str\n\n        :return: The AI resource with the provided name.\n        :rtype: AIResource\n        '
        workspace_hub = self._ml_client._workspace_hubs.get(name=name, **kwargs)
        resource = AIResource._from_v2_workspace_hub(workspace_hub)
        return resource

    @distributed_trace
    @monitor_with_activity(logger, 'AIResource.List', ActivityType.PUBLICAPI)
    def list(self, *, scope: str=Scope.RESOURCE_GROUP) -> Iterable[AIResource]:
        if False:
            return 10
        'List all AI resource assets in a project.\n\n        :keyword scope: The scope of the listing. Can be either "resource_group" or "subscription", and defaults to "resource_group".\n        :paramtype scope: str\n\n        :return: An iterator like instance of AI resource objects\n        :rtype: Iterable[AIResource]\n        '
        return [AIResource._from_v2_workspace_hub(wh) for wh in self._ml_client._workspace_hubs.list(scope=scope)]

    @distributed_trace
    @monitor_with_activity(logger, 'AIResource.BeginCreate', ActivityType.PUBLICAPI)
    def begin_create(self, *, ai_resource: AIResource, update_dependent_resources: bool=False, endpoint_resource_id: str=None, endpoint_kind: str=ENDPOINT_AI_SERVICE_KIND, **kwargs) -> LROPoller[AIResource]:
        if False:
            for i in range(10):
                print('nop')
        'Create a new AI resource.\n\n        :keyword ai_resource: Resource definition\n            or object which can be translated to a AI resource.\n        :paramtype ai_resource: ~azure.ai.resources.entities.AIResource\n        :keyword update_dependent_resources: Whether to update dependent resources. Defaults to False.\n        :paramtype update_dependent_resources: boolean\n        :keyword endpoint_resource_id: The UID of an AI service or Open AI resource.\n            The created hub will automatically create several\n            endpoints connecting to this resource, and creates its own otherwise.\n            If an Open AI resource ID is provided, then only a single Open AI\n            endpoint will be created. If set, then endpoint_resource_id should also be\n            set unless its default value is applicable.\n        :paramtype endpoint_resource_id: str\n        :keyword endpoint_kind: What kind of endpoint resource is being provided\n            by the endpoint_resource_id field. Defaults to "AIServices". The only other valid\n            input is "OpenAI".\n        :paramtype endpoint_kind: str\n        :return: An instance of LROPoller that returns the created AI resource.\n        :rtype: ~azure.core.polling.LROPoller[~azure.ai.resources.entities.AIResource]\n        '
        return self._ml_client.workspace_hubs.begin_create(workspace_hub=ai_resource._workspace_hub, update_dependent_resources=update_dependent_resources, endpoint_resource_id=endpoint_resource_id, endpoint_kind=endpoint_kind, cls=lambda hub: AIResource._from_v2_workspace_hub(hub), **kwargs)

    @distributed_trace
    @monitor_with_activity(logger, 'AIResource.BeginUpdate', ActivityType.PUBLICAPI)
    def begin_update(self, *, ai_resource: AIResource, update_dependent_resources: bool=False, **kwargs) -> LROPoller[AIResource]:
        if False:
            return 10
        'Update the name, description, tags, PNA, manageNetworkSettings, or encryption of a Resource\n\n        :keyword ai_resource: AI resource definition.\n        :paramtype ai_resource: ~azure.ai.resources.entities.AIResource\n        :keyword update_dependent_resources: Whether to update dependent resources. Defaults to False.\n        :paramtype update_dependent_resources: boolean\n        :return: An instance of LROPoller that returns the updated AI resource.\n        :rtype: ~azure.core.polling.LROPoller[~azure.ai.resources.entities.AIResource]\n        '
        return self._ml_client.workspace_hubs.begin_update(workspace_hub=ai_resource._workspace_hub, update_dependent_resources=update_dependent_resources, cls=lambda hub: AIResource._from_v2_workspace_hub(hub), **kwargs)

    @distributed_trace
    @monitor_with_activity(logger, 'AIResource.BeginDelete', ActivityType.PUBLICAPI)
    def begin_delete(self, *, name: str, delete_dependent_resources: bool, permanently_delete: bool=False, **kwargs) -> LROPoller[None]:
        if False:
            print('Hello World!')
        'Delete an AI resource.\n\n        :keyword name: Name of the Resource\n        :paramtype name: str\n        :keyword delete_dependent_resources: Whether to delete dependent resources associated with the AI resource.\n        :paramtype delete_dependent_resources: bool\n        :keyword permanently_delete: AI resource are soft-deleted by default to allow recovery of data.\n            Set this flag to true to override the soft-delete behavior and permanently delete your AI resource.\n        :paramtype permanently_delete: bool\n        :return: A poller to track the operation status.\n        :rtype: ~azure.core.polling.LROPoller[None]\n        '
        return self._ml_client.workspace_hubs.begin_delete(name=name, delete_dependent_resources=delete_dependent_resources, permanently_delete=permanently_delete, **kwargs)