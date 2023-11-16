import abc
from typing import List
import airbyte_api_client
import octavia_cli.list.formatting as formatting
from airbyte_api_client.api import connection_api, destination_api, destination_definition_api, source_api, source_definition_api
from airbyte_api_client.model.workspace_id_request_body import WorkspaceIdRequestBody

class BaseListing(abc.ABC):
    COMMON_LIST_FUNCTION_KWARGS = {'_check_return_type': False}

    @property
    @abc.abstractmethod
    def api(self):
        if False:
            i = 10
            return i + 15
        pass

    @property
    @abc.abstractmethod
    def fields_to_display(self) -> List[str]:
        if False:
            print('Hello World!')
        pass

    @property
    @abc.abstractmethod
    def list_field_in_response(self) -> str:
        if False:
            print('Hello World!')
        pass

    @property
    @abc.abstractmethod
    def list_function_name(self) -> str:
        if False:
            print('Hello World!')
        pass

    @property
    def _list_fn(self):
        if False:
            return 10
        return getattr(self.api, self.list_function_name)

    @property
    def list_function_kwargs(self) -> dict:
        if False:
            return 10
        return {}

    def __init__(self, api_client: airbyte_api_client.ApiClient):
        if False:
            for i in range(10):
                print('nop')
        self.api_instance = self.api(api_client)

    def _parse_response(self, api_response) -> List[List[str]]:
        if False:
            for i in range(10):
                print('nop')
        items = [[item[field] for field in self.fields_to_display] for item in api_response[self.list_field_in_response]]
        return items

    def get_listing(self) -> List[List[str]]:
        if False:
            print('Hello World!')
        api_response = self._list_fn(self.api_instance, **self.list_function_kwargs, **self.COMMON_LIST_FUNCTION_KWARGS)
        return self._parse_response(api_response)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        items = [formatting.format_column_names(self.fields_to_display)] + self.get_listing()
        return formatting.display_as_table(items)

class SourceConnectorsDefinitions(BaseListing):
    api = source_definition_api.SourceDefinitionApi
    fields_to_display = ['name', 'dockerRepository', 'dockerImageTag', 'sourceDefinitionId']
    list_field_in_response = 'source_definitions'
    list_function_name = 'list_source_definitions'

class DestinationConnectorsDefinitions(BaseListing):
    api = destination_definition_api.DestinationDefinitionApi
    fields_to_display = ['name', 'dockerRepository', 'dockerImageTag', 'destinationDefinitionId']
    list_field_in_response = 'destination_definitions'
    list_function_name = 'list_destination_definitions'

class WorkspaceListing(BaseListing, abc.ABC):

    def __init__(self, api_client: airbyte_api_client.ApiClient, workspace_id: str):
        if False:
            for i in range(10):
                print('nop')
        self.workspace_id = workspace_id
        super().__init__(api_client)

    @property
    def list_function_kwargs(self) -> dict:
        if False:
            i = 10
            return i + 15
        return {'workspace_id_request_body': WorkspaceIdRequestBody(workspace_id=self.workspace_id)}

class Sources(WorkspaceListing):
    api = source_api.SourceApi
    fields_to_display = ['name', 'sourceName', 'sourceId']
    list_field_in_response = 'sources'
    list_function_name = 'list_sources_for_workspace'

class Destinations(WorkspaceListing):
    api = destination_api.DestinationApi
    fields_to_display = ['name', 'destinationName', 'destinationId']
    list_field_in_response = 'destinations'
    list_function_name = 'list_destinations_for_workspace'

class Connections(WorkspaceListing):
    api = connection_api.ConnectionApi
    fields_to_display = ['name', 'connectionId', 'status', 'sourceId', 'destinationId']
    list_field_in_response = 'connections'
    list_function_name = 'list_connections_for_workspace'