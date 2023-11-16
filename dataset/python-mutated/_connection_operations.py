from typing import Any, Iterable, Optional
from azure.core.tracing.decorator import distributed_trace
from azure.ai.resources._project_scope import OperationScope
from azure.ai.resources.constants._common import DEFAULT_OPEN_AI_CONNECTION_NAME
from azure.ai.resources.entities import BaseConnection
from azure.ai.ml import MLClient
from azure.ai.resources._telemetry import ActivityType, monitor_with_activity, monitor_with_telemetry_mixin, OpsLogger
ops_logger = OpsLogger(__name__)
(logger, module_logger) = (ops_logger.package_logger, ops_logger.module_logger)

class ConnectionOperations:
    """ConnectionOperations.

    You should not instantiate this class directly. Instead, you should
    create an MLClient instance that instantiates it for you and
    attaches it as an attribute.
    """

    def __init__(self, ml_client: MLClient, **kwargs: Any):
        if False:
            i = 10
            return i + 15
        self._ml_client = ml_client
        ops_logger.update_info(kwargs)

    @distributed_trace
    @monitor_with_activity(logger, 'Connection.List', ActivityType.PUBLICAPI)
    def list(self, connection_type: Optional[str]=None) -> Iterable[BaseConnection]:
        if False:
            while True:
                i = 10
        'List all connection assets in a project.\n\n        :param connection_type: If set, return only connections of the specified type.\n        :type connection_type: str\n\n        :return: An iterator like instance of connection objects\n        :rtype: Iterable[Connection]\n        '
        return [BaseConnection._from_v2_workspace_connection(conn) for conn in self._ml_client._workspace_connections.list(connection_type=connection_type)]

    @distributed_trace
    @monitor_with_activity(logger, 'Connection.Get', ActivityType.PUBLICAPI)
    def get(self, name: str, **kwargs) -> BaseConnection:
        if False:
            for i in range(10):
                print('nop')
        'Get a connection by name.\n\n        :param name: Name of the connection.\n        :type name: str\n\n        :return: The connection with the provided name.\n        :rtype: Connection\n        '
        workspace_connection = self._ml_client._workspace_connections.get(name=name, **kwargs)
        connection = BaseConnection._from_v2_workspace_connection(workspace_connection)
        if not connection.credentials.key:
            list_secrets_response = self._ml_client.connections._operation.list_secrets(connection_name=name, resource_group_name=self._ml_client.resource_group_name, workspace_name=self._ml_client.workspace_name)
            connection.credentials.key = list_secrets_response.properties.credentials.key
        return connection

    @distributed_trace
    @monitor_with_activity(logger, 'Connection.CreateOrUpdate', ActivityType.PUBLICAPI)
    def create_or_update(self, connection: BaseConnection, **kwargs) -> BaseConnection:
        if False:
            i = 10
            return i + 15
        'Create or update a connection.\n\n        :param connection: Connection definition\n            or object which can be translated to a connection.\n        :type connection: Connection\n        :return: Created or updated connection.\n        :rtype: Connection\n        '
        response = self._ml_client._workspace_connections.create_or_update(workspace_connection=connection._workspace_connection, **kwargs)
        return BaseConnection._from_v2_workspace_connection(response)

    @distributed_trace
    @monitor_with_activity(logger, 'Connection.Delete', ActivityType.PUBLICAPI)
    def delete(self, name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Delete the connection.\n\n        :param name: Name of the connection to delete.\n        :type name: str\n        '
        return self._ml_client._workspace_connections.delete(name=name)