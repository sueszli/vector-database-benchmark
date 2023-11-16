from typing import Any, Dict, Iterable, Optional
from azure.core.tracing.decorator import distributed_trace
from azure.ai.resources.entities.project import Project
from azure.ai.ml import MLClient
from azure.ai.ml._restclient.v2023_06_01_preview import AzureMachineLearningWorkspaces as ServiceClient062023Preview
from azure.ai.ml.constants._common import Scope
from azure.ai.ml.entities import Workspace
from azure.core.polling import LROPoller
from azure.ai.resources._telemetry import ActivityType, monitor_with_activity, monitor_with_telemetry_mixin, OpsLogger
ops_logger = OpsLogger(__name__)
(logger, module_logger) = (ops_logger.package_logger, ops_logger.module_logger)

class ProjectOperations:
    """ProjectOperations.

    You should not instantiate this class directly. Instead, you should
    create an MLClient instance that instantiates it for you and
    attaches it as an attribute.
    """

    def __init__(self, resource_group_name: str, ml_client: MLClient, service_client: ServiceClient062023Preview, **kwargs: Any):
        if False:
            while True:
                i = 10
        self._ml_client = ml_client
        self._service_client = service_client
        self._resource_group_name = resource_group_name
        ops_logger.update_info(kwargs)

    @distributed_trace
    @monitor_with_activity(logger, 'Project.Get', ActivityType.PUBLICAPI)
    def get(self, *, name: Optional[str]=None, **kwargs: Dict) -> Project:
        if False:
            while True:
                i = 10
        'Get a project by name.\n\n        :keyword name: Name of the project.\n        :paramtype name: str\n        :return: The project with the provided name.\n        :rtype: Project\n        '
        workspace = self._ml_client._workspaces.get(name=name, **kwargs)
        project = Project._from_v2_workspace(workspace)
        return project

    @distributed_trace
    @monitor_with_activity(logger, 'Project.List', ActivityType.PUBLICAPI)
    def list(self, *, scope: str=Scope.RESOURCE_GROUP) -> Iterable[Project]:
        if False:
            i = 10
            return i + 15
        'List all projects that the user has access to in the current resource group or subscription.\n\n        :keyword scope: The scope of the listing. Can be either "resource_group" or "subscription", and defaults to "resource_group".\n        :paramtype scope: str\n        :return: An iterator like instance of Project objects\n        :rtype: Iterable[Project]\n        '
        workspaces = []
        if scope == Scope.SUBSCRIPTION:
            workspaces = self._service_client.workspaces.list_by_subscription(kind='project', cls=lambda objs: [Workspace._from_rest_object(obj) for obj in objs])
        else:
            workspaces = self._service_client.workspaces.list_by_resource_group(kind='project', resource_group_name=self._resource_group_name, cls=lambda objs: [Workspace._from_rest_object(obj) for obj in objs])
        return [Project._from_v2_workspace(ws) for ws in workspaces]

    @distributed_trace
    @monitor_with_activity(logger, 'Project.BeginCreate', ActivityType.PUBLICAPI)
    def begin_create(self, *, project: Project, update_dependent_resources: bool=False, **kwargs) -> LROPoller[Project]:
        if False:
            return 10
        'Create a new project. Returns the project if it already exists.\n\n        :keyword project: Project definition.\n        :paramtype project: ~azure.ai.resources.entities.project\n        :keyword update_dependent_resources: Whether to update dependent resources\n        :paramtype update_dependent_resources: boolean\n        :return: An instance of LROPoller that returns a project.\n        :rtype: ~azure.core.polling.LROPoller[~azure.ai.resources.entities.project]\n        '
        return self._ml_client.workspaces.begin_create(workspace=project._workspace, update_dependent_resources=update_dependent_resources, cls=lambda workspace: Project._from_v2_workspace(workspace=workspace), **kwargs)

    @distributed_trace
    @monitor_with_activity(logger, 'Project.BeginUpdate', ActivityType.PUBLICAPI)
    def begin_update(self, *, project: Project, update_dependent_resources: bool=False, **kwargs) -> LROPoller[Project]:
        if False:
            print('Hello World!')
        'Update a project.\n\n        :keyword project: Project definition.\n        :paramtype project: ~azure.ai.resources.entities.project\n        :keyword update_dependent_resources: Whether to update dependent resources\n        :paramtype update_dependent_resources: boolean\n        :return: An instance of LROPoller that returns a project.\n        :rtype: ~azure.core.polling.LROPoller[~azure.ai.resources.entities.project]\n        '
        return self._ml_client.workspaces.begin_update(workspace=project._workspace, update_dependent_resources=update_dependent_resources, cls=lambda workspace: Project._from_v2_workspace(workspace=workspace), **kwargs)

    @distributed_trace
    @monitor_with_activity(logger, 'Project.BeginDelete', ActivityType.PUBLICAPI)
    def begin_delete(self, *, name: str, delete_dependent_resources: bool, permanently_delete: bool=False):
        if False:
            while True:
                i = 10
        'Delete a project.\n\n        :keyword name: Name of the project\n        :paramtype name: str\n        :keyword delete_dependent_resources: Whether to delete resources associated with the project,\n            i.e., container registry, storage account, key vault, and application insights.\n            The default is False. Set to True to delete these resources.\n        :paramtype delete_dependent_resources: bool\n        :keyword permanently_delete: Project are soft-deleted by default to allow recovery of project data.\n            Set this flag to true to override the soft-delete behavior and permanently delete your project.\n        :paramtype permanently_delete: bool\n        :return: A poller to track the operation status.\n        :rtype: ~azure.core.polling.LROPoller[None]\n        '
        return self._ml_client.workspaces.begin_delete(name=name, delete_dependent_resources=delete_dependent_resources, permanently_delete=permanently_delete)