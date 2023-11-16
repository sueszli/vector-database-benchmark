""" To isolate Schemas API calls """
import json
import logging
from json import JSONDecodeError
from botocore.exceptions import ClientError, EndpointConnectionError
from samcli.commands.exceptions import SchemasApiException
from samcli.commands.local.cli_common.user_exceptions import NotAvailableInRegion, ResourceNotFound
from samcli.lib.schemas.schemas_constants import DEFAULT_EVENT_DETAIL_TYPE, DEFAULT_EVENT_SOURCE
from samcli.lib.schemas.schemas_directory_hierarchy_builder import get_package_hierarchy, sanitize_name
SCHEMAS_NOT_AVAILABLE_IN_REGION_ERROR = 'EventBridge Schemas are not available in provided region. Please check AWS doc for Schemas supported regions.'
LOG = logging.getLogger(__name__)

class SchemasApiCaller:

    def __init__(self, schemas_client):
        if False:
            for i in range(10):
                print('nop')
        self._schemas_client = schemas_client

    def list_registries(self, next_token=None, limit=10):
        if False:
            while True:
                i = 10
        '\n        Calls schemas service to get list of schema registries.\n\n        Parameters\n        ----------\n        next_token:\n            Continuation token\n        limit:\n            Number of items tro fetch\n\n        Returns\n        -------\n        List of registries available\n        '
        if limit is None:
            limit = 10
        registries = []
        try:
            paginator = self._schemas_client.get_paginator('list_registries')
            page_iterator = paginator.paginate(PaginationConfig={'StartingToken': next_token, 'MaxItems': limit, 'PageSize': limit})
            page = None
            for page in page_iterator:
                for registry in page['Registries']:
                    registries.append(registry['RegistryName'])
            if not registries:
                raise ResourceNotFound('No Registries found. This should not be possible, please raise an issue.')
            next_token = page.get('NextToken', None)
            return {'registries': registries, 'next_token': next_token}
        except EndpointConnectionError as ex:
            raise NotAvailableInRegion(SCHEMAS_NOT_AVAILABLE_IN_REGION_ERROR) from ex

    def list_schemas(self, registry_name, next_token=None, limit=10):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calls schemas service to get list of schemas for given registry.\n\n        Parameters\n        ----------\n        registry_name:\n            Name of the registry\n        next_token:\n            Continuation token\n        limit:\n            Number of items to fetch\n\n        Returns\n        -------\n        List of Schemas available for given registry\n        '
        schemas = []
        try:
            paginator = self._schemas_client.get_paginator('list_schemas')
            page_iterator = paginator.paginate(RegistryName=registry_name, PaginationConfig={'StartingToken': next_token, 'MaxItems': limit, 'PageSize': limit})
            page = None
            for page in page_iterator:
                for schema in page['Schemas']:
                    schemas.append(schema['SchemaName'])
            if not schemas:
                raise ResourceNotFound('No Schemas found for registry %s' % registry_name)
            next_token = page.get('NextToken', None)
            return {'schemas': schemas, 'next_token': next_token}
        except EndpointConnectionError as ex:
            raise NotAvailableInRegion(SCHEMAS_NOT_AVAILABLE_IN_REGION_ERROR) from ex

    def list_schema_versions(self, registry_name, schema_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calls schemas service to list all schema versions.\n\n        Parameters\n        ----------\n        registry_name:\n            Registry name\n        schema_name:\n            Schema Name\n\n        Returns\n        -------\n        List of Schema versions\n        '
        versions = []
        next_token = None
        try:
            while True:
                paginator = self._schemas_client.get_paginator('list_schema_versions')
                page_iterator = paginator.paginate(RegistryName=registry_name, SchemaName=schema_name, PaginationConfig={'StartingToken': next_token})
                page = None
                for page in page_iterator:
                    for version in page['SchemaVersions']:
                        versions.append(version['SchemaVersion'])
                next_token = page.get('NextToken')
                if next_token is None:
                    break
        except EndpointConnectionError as ex:
            raise NotAvailableInRegion(SCHEMAS_NOT_AVAILABLE_IN_REGION_ERROR) from ex
        versions.sort(key=int)
        return versions

    def get_latest_schema_version(self, registry_name, schema_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calls schemas service to get schema latest version.\n\n        Parameters\n        ----------\n        registry_name:\n            Registry name\n        schema_name:\n            Schema Name\n\n        Returns\n        -------\n        Latest Schema version\n        '
        versions = self.list_schema_versions(registry_name, schema_name)
        return versions[-1]

    def get_schema_metadata(self, registry_name, schema_name):
        if False:
            return 10
        '\n        Calls schemas service to get schema metadata.\n\n        Parameters\n        ----------\n        registry_name:\n            Registry Name\n        schema_name:\n            Schema Name\n\n        Returns\n        -------\n        Schema metadata\n        '
        try:
            describe_schema_response = self._schemas_client.describe_schema(RegistryName=registry_name, SchemaName=schema_name)
        except EndpointConnectionError as ex:
            raise NotAvailableInRegion(SCHEMAS_NOT_AVAILABLE_IN_REGION_ERROR) from ex
        try:
            content = json.loads(describe_schema_response['Content'])
            schemas = content['components']['schemas']
            event_source = DEFAULT_EVENT_SOURCE
            event_source_detail_type = DEFAULT_EVENT_DETAIL_TYPE
            schema_root_name = sanitize_name(list(schemas.keys())[0])
            schemas_package_hierarchy = get_package_hierarchy(schema_name)
            if schemas.get('AWSEvent') is not None:
                aws_event = schemas.get('AWSEvent')
                if aws_event.get('x-amazon-events-source') is not None:
                    event_source = aws_event.get('x-amazon-events-source')
                if aws_event.get('x-amazon-events-detail-type') is not None:
                    event_source_detail_type = aws_event.get('x-amazon-events-detail-type')
                possible_root_schema_name = aws_event['properties']['detail']['$ref']
                schema_root_name = sanitize_name(possible_root_schema_name[len('#/components/schemas/'):])
            return {'event_source': event_source, 'event_source_detail_type': event_source_detail_type, 'schema_root_name': schema_root_name, 'schemas_package_hierarchy': schemas_package_hierarchy}
        except JSONDecodeError as ex:
            raise SchemasApiException('Parse error reading the content from Schemas response. This should not be possible, please raise an issue.') from ex

    def download_source_code_binding(self, runtime, registry_name, schema_name, schema_version, download_location):
        if False:
            return 10
        '\n        Calls schemas service to download code binding for given schema in download_location.\n\n        Parameters\n        ----------\n        runtime:\n            Code binding runtime e.g: Java, Python, Go\n        registry_name:\n            Registry Name\n        schema_name:\n            Schema Name\n        schema_version:\n            Schema version for which code binding needs to be downloaded\n        download_location:\n            Location at which code binding should be downloaded\n        '
        try:
            response = self._schemas_client.get_code_binding_source(Language=runtime, RegistryName=registry_name, SchemaName=schema_name, SchemaVersion=schema_version)
        except EndpointConnectionError as ex:
            raise NotAvailableInRegion(SCHEMAS_NOT_AVAILABLE_IN_REGION_ERROR) from ex
        for data in response['Body']:
            download_location.write(data)

    def put_code_binding(self, runtime, registry_name, schema_name, schema_version):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calls schemas service to generate code binding for given schema.\n\n        Parameters\n        ----------\n        runtime:\n            Code binding runtime e.g: Java, Python, Go\n        registry_name:\n            Registry Name\n        schema_name:\n            Schema Name\n        schema_version:\n            Schema version for which code binding needs to be generated\n        '
        try:
            self._schemas_client.put_code_binding(Language=runtime, RegistryName=registry_name, SchemaName=schema_name, SchemaVersion=schema_version)
        except EndpointConnectionError as ex:
            raise NotAvailableInRegion('EventBridge Schemas are not available in provided region. Please check AWS doc for Schemas supported regions.') from ex
        except ClientError as e:
            if e.response['Error']['Code'] != 'ConflictException':
                raise e

    def poll_for_code_binding_status(self, schemas_runtime, registry_name, schema_name, schema_version):
        if False:
            while True:
                i = 10
        '\n        Calls schemas service and wait for code binding to be generated.\n\n        Parameters\n        ----------\n        schemas_runtime:\n            Code binding runtime e.g: Java, Python, Go\n        registry_name:\n            Registry Name\n        schema_name:\n            Schema Name\n        schema_version:\n            Schema version\n        '
        try:
            waiter = self._schemas_client.get_waiter('code_binding_exists')
            waiter.wait(Language=schemas_runtime, RegistryName=registry_name, SchemaName=schema_name, SchemaVersion=schema_version)
        except EndpointConnectionError as ex:
            raise NotAvailableInRegion(SCHEMAS_NOT_AVAILABLE_IN_REGION_ERROR) from ex

    def discover_schema(self, event_data: str, schema_type: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a schema based on an event using the DiscoverSchema API\n\n        Parameters\n        ----------\n        event_data:\n            A JSON test event as a string\n        schema_type:\n            Type of the schema to generate ("OpenApi3" or "JSONSchemaDraft4")\n        Returns\n        -------\n        Generated schema JSON as a string\n        '
        try:
            LOG.debug("Discover schema from contents: '%s'.", event_data)
            schema = self._schemas_client.get_discovered_schema(Events=[event_data], Type=schema_type)
            return str(schema['Content'])
        except EndpointConnectionError as ex:
            LOG.error('Failure calling get_discovered_schema')
            raise NotAvailableInRegion(SCHEMAS_NOT_AVAILABLE_IN_REGION_ERROR) from ex

    def create_schema(self, schema: str, registry_name: str, schema_name: str, schema_type: str):
        if False:
            return 10
        '\n        Creates a new schema in the specified registry\n\n        Parameters\n        ----------\n        schema:\n            Contents for the schema to be created\n        registry_name:\n            The registry the schema will be created in\n        schema_name:\n            The name for the new created schema\n        schema_type:\n            Type of the schema to generate ("OpenApi3" or "JSONSchemaDraft4")\n        '
        try:
            LOG.debug('Creating schema %s on registry %s.', schema_name, registry_name)
            self._schemas_client.create_schema(Content=schema, RegistryName=registry_name, SchemaName=schema_name, Type=schema_type)
            return True
        except EndpointConnectionError as ex:
            LOG.error('Failure calling create_schema in registry %s for schema %s', registry_name, schema_name)
            raise NotAvailableInRegion(SCHEMAS_NOT_AVAILABLE_IN_REGION_ERROR) from ex

    def update_schema(self, schema: str, registry_name: str, schema_name: str, schema_type: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Updates an existing schema\n\n        Parameters\n        ----------\n        schema:\n            Contents for the updated schema\n        registry_name:\n            The registry of the schema that will be updated\n        schema_name:\n            The name of the schema to be updated\n        schema_type:\n            Type of the schema to generate ("OpenApi3" or "JSONSchemaDraft4")\n        '
        try:
            LOG.debug('Updating schema %s on registry %s.', schema_name, registry_name)
            self._schemas_client.update_schema(Content=schema, RegistryName=registry_name, SchemaName=schema_name, Type=schema_type)
            return True
        except ClientError as ex:
            error_message: str = ex.response.get('Message', '')
            if ex.response.get('Code') == 'Conflict' and 'No change since the previous' in error_message:
                LOG.debug('No changes to the schema from the previous version')
                return True
            raise ex
        except EndpointConnectionError as ex:
            LOG.error('Failure calling update_schema in registry %s for schema %s', registry_name, schema_name)
            raise NotAvailableInRegion(SCHEMAS_NOT_AVAILABLE_IN_REGION_ERROR) from ex

    def get_schema(self, registry_name: str, schema_name: str) -> str:
        if False:
            return 10
        '\n        Gets a schema from the registry\n\n        Parameters\n        ----------\n        registry_name:\n            The registry of the schema that will be updated\n        schema_name:\n            The name of the schema to be updated\n        Returns\n        -------\n        A schema dict\n        '
        try:
            LOG.debug('Describing schema %s on registry %s.', schema_name, registry_name)
            schema = self._schemas_client.describe_schema(RegistryName=registry_name, SchemaName=schema_name)
            return str(schema['Content'])
        except ClientError as ex:
            if ex.response.get('Error', {}).get('Code') != 'NotFoundException':
                LOG.error('%s error calling describe_schema in registry %s for schema %s', ex.response.get('Error', {}).get('Code'), registry_name, schema_name)
                raise ex
            LOG.debug("Schema %s doesn't exist", schema_name)
            return ''
        except EndpointConnectionError as ex:
            LOG.error('Failure calling describe_schema in registry %s for schema %s', registry_name, schema_name)
            raise NotAvailableInRegion(SCHEMAS_NOT_AVAILABLE_IN_REGION_ERROR) from ex

    def check_registry_exists(self, registry_name: str) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Gets a registry with the specified name\n\n        Parameters\n        ----------\n        registry_name:\n            The name of the registry to fetch\n        Returns\n        -------\n        The specified registry, or None if it does not exist\n        '
        try:
            LOG.debug('Describing registry %s.', registry_name)
            self._schemas_client.describe_registry(RegistryName=registry_name)
            return True
        except ClientError as ex:
            if ex.response.get('Error', {}).get('Code') != 'NotFoundException':
                LOG.error('%s error calling describe_registry in registry %s', ex.response.get('Error', {}).get('Code'), registry_name)
                raise ex
            LOG.debug("Registry %s doesn't exist", registry_name)
        except EndpointConnectionError as ex:
            LOG.error('Failure calling describe_registry in registry %s', registry_name)
            raise NotAvailableInRegion(SCHEMAS_NOT_AVAILABLE_IN_REGION_ERROR) from ex
        return False

    def create_registry(self, registry_name: str):
        if False:
            while True:
                i = 10
        '\n        Creates a new registry with the specified name\n\n        Parameters\n        ----------\n        registry_name:\n            The name of the registry to be created\n        '
        try:
            LOG.debug('Creating registry %s.', registry_name)
            self._schemas_client.create_registry(RegistryName=registry_name)
            return True
        except ClientError as ex:
            if ex.response.get('Error', {}).get('Code') != 'ConflictException':
                LOG.error('%s error calling create_registry for registry %s', ex.response.get('Error', {}).get('Code'), registry_name)
                raise ex
            LOG.debug('Registry %s already exists', registry_name)
        except EndpointConnectionError as ex:
            LOG.error('Failure calling create_registry for registry %s', registry_name)
            raise NotAvailableInRegion(SCHEMAS_NOT_AVAILABLE_IN_REGION_ERROR) from ex
        return False

    def delete_schema(self, registry_name, schema_name) -> bool:
        if False:
            print('Hello World!')
        '\n        Deletes a schema from the EBSR\n\n        Parameters\n        ----------\n        registry_name:\n            The registry that contains the schema that will be deleted\n        schema_name:\n            The name of the schema to be deleted\n        '
        try:
            LOG.debug('Deleting schema %s on registry %s.', schema_name, registry_name)
            self._schemas_client.delete_schema(RegistryName=registry_name, SchemaName=schema_name)
            return True
        except ClientError as ex:
            if ex.response.get('Error', {}).get('Code') != 'NotFoundException':
                LOG.error('%s error when calling delete_delete schema with %s schema in %s registry', ex.response.get('Error', {}).get('Code'), schema_name, registry_name)
                raise ex
            LOG.debug("Schema %s doesn't exist so it couldn't be deleted", schema_name)
        except EndpointConnectionError as ex:
            LOG.error('Failure calling delete_schema for schema %s in registry %s', schema_name, registry_name)
            raise NotAvailableInRegion(SCHEMAS_NOT_AVAILABLE_IN_REGION_ERROR) from ex
        return False

    def delete_version(self, registry_name, schema_name, schema_version: str):
        if False:
            print('Hello World!')
        '\n        Delete a version of a schema\n\n        Parameters\n        ----------\n        registry_name:\n            The registry that contains the schema\n        schema_name:\n            The name of the schema\n        schema_version:\n            Version to be deleted\n        '
        try:
            LOG.debug('Deleting version %s of schema %s on registry %s.', schema_version, schema_name, registry_name)
            self._schemas_client.delete_schema_version(RegistryName=registry_name, SchemaName=schema_name, SchemaVersion=schema_version)
            return True
        except ClientError as ex:
            if ex.response.get('Error', {}).get('Code') != 'NotFoundException':
                raise ex
            LOG.debug("Schema version %s of %s doesn't exist so it couldn't be deleted", schema_version, schema_name)
        except EndpointConnectionError as ex:
            LOG.error('Error when calling limit_versions')
            raise NotAvailableInRegion(SCHEMAS_NOT_AVAILABLE_IN_REGION_ERROR) from ex
        return False