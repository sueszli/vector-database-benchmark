from typing import TYPE_CHECKING
import warnings
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, map_error
from azure.core.paging import ItemPaged
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpRequest, HttpResponse
from azure.core.polling import LROPoller, NoPolling, PollingMethod
from azure.mgmt.core.exceptions import ARMErrorFormat
from azure.mgmt.core.polling.arm_polling import ARMPolling
from .. import models as _models
if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Generic, Iterable, Optional, TypeVar, Union
    T = TypeVar('T')
    ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]

class CommunicationServiceOperations(object):
    """CommunicationServiceOperations operations.

    You should not instantiate this class directly. Instead, you should create a Client instance that
    instantiates it for you and attaches it as an attribute.

    :ivar models: Alias to model classes used in this operation group.
    :type models: ~communication_service_management_client.models
    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """
    models = _models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            while True:
                i = 10
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self._config = config

    def check_name_availability(self, name_availability_parameters=None, **kwargs):
        if False:
            while True:
                i = 10
        'Check Name Availability.\n\n        Checks that the CommunicationService name is valid and is not already in use.\n\n        :param name_availability_parameters: Parameters supplied to the operation.\n        :type name_availability_parameters: ~communication_service_management_client.models.NameAvailabilityParameters\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: NameAvailability, or the result of cls(response)\n        :rtype: ~communication_service_management_client.models.NameAvailability\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2020-08-20'
        content_type = kwargs.pop('content_type', 'application/json')
        accept = 'application/json'
        url = self.check_name_availability.metadata['url']
        path_format_arguments = {'subscriptionId': self._serialize.url('self._config.subscription_id', self._config.subscription_id, 'str', min_length=1)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        body_content_kwargs = {}
        if name_availability_parameters is not None:
            body_content = self._serialize.body(name_availability_parameters, 'NameAvailabilityParameters')
        else:
            body_content = None
        body_content_kwargs['content'] = body_content
        request = self._client.post(url, query_parameters, header_parameters, **body_content_kwargs)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize(_models.ErrorResponse, response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('NameAvailability', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    check_name_availability.metadata = {'url': '/subscriptions/{subscriptionId}/providers/Microsoft.Communication/checkNameAvailability'}

    def link_notification_hub(self, resource_group_name, communication_service_name, link_notification_hub_parameters=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Link Notification Hub.\n\n        Links an Azure Notification Hub to this communication service.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n        :type resource_group_name: str\n        :param communication_service_name: The name of the CommunicationService resource.\n        :type communication_service_name: str\n        :param link_notification_hub_parameters: Parameters supplied to the operation.\n        :type link_notification_hub_parameters: ~communication_service_management_client.models.LinkNotificationHubParameters\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: LinkedNotificationHub, or the result of cls(response)\n        :rtype: ~communication_service_management_client.models.LinkedNotificationHub\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2020-08-20'
        content_type = kwargs.pop('content_type', 'application/json')
        accept = 'application/json'
        url = self.link_notification_hub.metadata['url']
        path_format_arguments = {'subscriptionId': self._serialize.url('self._config.subscription_id', self._config.subscription_id, 'str', min_length=1), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'communicationServiceName': self._serialize.url('communication_service_name', communication_service_name, 'str', max_length=63, min_length=1, pattern='^[-\\w]+$')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        body_content_kwargs = {}
        if link_notification_hub_parameters is not None:
            body_content = self._serialize.body(link_notification_hub_parameters, 'LinkNotificationHubParameters')
        else:
            body_content = None
        body_content_kwargs['content'] = body_content
        request = self._client.post(url, query_parameters, header_parameters, **body_content_kwargs)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize(_models.ErrorResponse, response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('LinkedNotificationHub', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    link_notification_hub.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Communication/communicationServices/{communicationServiceName}/linkNotificationHub'}

    def list_by_subscription(self, **kwargs):
        if False:
            return 10
        'List By Subscription.\n\n        Handles requests to list all resources in a subscription.\n\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either CommunicationServiceResourceList or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~communication_service_management_client.models.CommunicationServiceResourceList]\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2020-08-20'
        accept = 'application/json'

        def prepare_request(next_link=None):
            if False:
                while True:
                    i = 10
            header_parameters = {}
            header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
            if not next_link:
                url = self.list_by_subscription.metadata['url']
                path_format_arguments = {'subscriptionId': self._serialize.url('self._config.subscription_id', self._config.subscription_id, 'str', min_length=1)}
                url = self._client.format_url(url, **path_format_arguments)
                query_parameters = {}
                query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
                request = self._client.get(url, query_parameters, header_parameters)
            else:
                url = next_link
                query_parameters = {}
                request = self._client.get(url, query_parameters, header_parameters)
            return request

        def extract_data(pipeline_response):
            if False:
                return 10
            deserialized = self._deserialize('CommunicationServiceResourceList', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            request = prepare_request(next_link)
            pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                error = self._deserialize(_models.ErrorResponse, response)
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_by_subscription.metadata = {'url': '/subscriptions/{subscriptionId}/providers/Microsoft.Communication/communicationServices'}

    def list_by_resource_group(self, resource_group_name, **kwargs):
        if False:
            i = 10
            return i + 15
        'List By Resource Group.\n\n        Handles requests to list all resources in a resource group.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n        :type resource_group_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either CommunicationServiceResourceList or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~communication_service_management_client.models.CommunicationServiceResourceList]\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2020-08-20'
        accept = 'application/json'

        def prepare_request(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            header_parameters = {}
            header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
            if not next_link:
                url = self.list_by_resource_group.metadata['url']
                path_format_arguments = {'subscriptionId': self._serialize.url('self._config.subscription_id', self._config.subscription_id, 'str', min_length=1), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$')}
                url = self._client.format_url(url, **path_format_arguments)
                query_parameters = {}
                query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
                request = self._client.get(url, query_parameters, header_parameters)
            else:
                url = next_link
                query_parameters = {}
                request = self._client.get(url, query_parameters, header_parameters)
            return request

        def extract_data(pipeline_response):
            if False:
                for i in range(10):
                    print('nop')
            deserialized = self._deserialize('CommunicationServiceResourceList', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                i = 10
                return i + 15
            request = prepare_request(next_link)
            pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                error = self._deserialize(_models.ErrorResponse, response)
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return ItemPaged(get_next, extract_data)
    list_by_resource_group.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Communication/communicationServices'}

    def update(self, resource_group_name, communication_service_name, parameters=None, **kwargs):
        if False:
            print('Hello World!')
        'Update.\n\n        Operation to update an existing CommunicationService.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n        :type resource_group_name: str\n        :param communication_service_name: The name of the CommunicationService resource.\n        :type communication_service_name: str\n        :param parameters: Parameters for the update operation.\n        :type parameters: ~communication_service_management_client.models.CommunicationServiceResource\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: CommunicationServiceResource, or the result of cls(response)\n        :rtype: ~communication_service_management_client.models.CommunicationServiceResource\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2020-08-20'
        content_type = kwargs.pop('content_type', 'application/json')
        accept = 'application/json'
        url = self.update.metadata['url']
        path_format_arguments = {'subscriptionId': self._serialize.url('self._config.subscription_id', self._config.subscription_id, 'str', min_length=1), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'communicationServiceName': self._serialize.url('communication_service_name', communication_service_name, 'str', max_length=63, min_length=1, pattern='^[-\\w]+$')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        body_content_kwargs = {}
        if parameters is not None:
            body_content = self._serialize.body(parameters, 'CommunicationServiceResource')
        else:
            body_content = None
        body_content_kwargs['content'] = body_content
        request = self._client.patch(url, query_parameters, header_parameters, **body_content_kwargs)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize(_models.ErrorResponse, response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('CommunicationServiceResource', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    update.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Communication/communicationServices/{communicationServiceName}'}

    def get(self, resource_group_name, communication_service_name, **kwargs):
        if False:
            return 10
        'Get.\n\n        Get the CommunicationService and its properties.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n        :type resource_group_name: str\n        :param communication_service_name: The name of the CommunicationService resource.\n        :type communication_service_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: CommunicationServiceResource, or the result of cls(response)\n        :rtype: ~communication_service_management_client.models.CommunicationServiceResource\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2020-08-20'
        accept = 'application/json'
        url = self.get.metadata['url']
        path_format_arguments = {'subscriptionId': self._serialize.url('self._config.subscription_id', self._config.subscription_id, 'str', min_length=1), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'communicationServiceName': self._serialize.url('communication_service_name', communication_service_name, 'str', max_length=63, min_length=1, pattern='^[-\\w]+$')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        request = self._client.get(url, query_parameters, header_parameters)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize(_models.ErrorResponse, response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('CommunicationServiceResource', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Communication/communicationServices/{communicationServiceName}'}

    def _create_or_update_initial(self, resource_group_name, communication_service_name, parameters=None, **kwargs):
        if False:
            return 10
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2020-08-20'
        content_type = kwargs.pop('content_type', 'application/json')
        accept = 'application/json'
        url = self._create_or_update_initial.metadata['url']
        path_format_arguments = {'subscriptionId': self._serialize.url('self._config.subscription_id', self._config.subscription_id, 'str', min_length=1), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'communicationServiceName': self._serialize.url('communication_service_name', communication_service_name, 'str', max_length=63, min_length=1, pattern='^[-\\w]+$')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        body_content_kwargs = {}
        if parameters is not None:
            body_content = self._serialize.body(parameters, 'CommunicationServiceResource')
        else:
            body_content = None
        body_content_kwargs['content'] = body_content
        request = self._client.put(url, query_parameters, header_parameters, **body_content_kwargs)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize(_models.ErrorResponse, response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        response_headers = {}
        if response.status_code == 200:
            deserialized = self._deserialize('CommunicationServiceResource', pipeline_response)
        if response.status_code == 201:
            response_headers['Azure-AsyncOperation'] = self._deserialize('str', response.headers.get('Azure-AsyncOperation'))
            deserialized = self._deserialize('CommunicationServiceResource', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized
    _create_or_update_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Communication/communicationServices/{communicationServiceName}'}

    def begin_create_or_update(self, resource_group_name, communication_service_name, parameters=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Create Or Update.\n\n        Create a new CommunicationService or update an existing CommunicationService.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n        :type resource_group_name: str\n        :param communication_service_name: The name of the CommunicationService resource.\n        :type communication_service_name: str\n        :param parameters: Parameters for the create or update operation.\n        :type parameters: ~communication_service_management_client.models.CommunicationServiceResource\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: True for ARMPolling, False for no polling, or a\n         polling object for personal polling strategy\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no Retry-After header is present.\n        :return: An instance of LROPoller that returns either CommunicationServiceResource or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~communication_service_management_client.models.CommunicationServiceResource]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        polling = kwargs.pop('polling', True)
        cls = kwargs.pop('cls', None)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_or_update_initial(resource_group_name=resource_group_name, communication_service_name=communication_service_name, parameters=parameters, cls=lambda x, y, z: x, **kwargs)
        kwargs.pop('error_map', None)
        kwargs.pop('content_type', None)

        def get_long_running_output(pipeline_response):
            if False:
                i = 10
                return i + 15
            deserialized = self._deserialize('CommunicationServiceResource', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, {})
            return deserialized
        path_format_arguments = {'subscriptionId': self._serialize.url('self._config.subscription_id', self._config.subscription_id, 'str', min_length=1), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'communicationServiceName': self._serialize.url('communication_service_name', communication_service_name, 'str', max_length=63, min_length=1, pattern='^[-\\w]+$')}
        if polling is True:
            polling_method = ARMPolling(lro_delay, lro_options={'final-state-via': 'azure-async-operation'}, path_format_arguments=path_format_arguments, **kwargs)
        elif polling is False:
            polling_method = NoPolling()
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        else:
            return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_create_or_update.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Communication/communicationServices/{communicationServiceName}'}

    def _delete_initial(self, resource_group_name, communication_service_name, **kwargs):
        if False:
            i = 10
            return i + 15
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2020-08-20'
        accept = 'application/json'
        url = self._delete_initial.metadata['url']
        path_format_arguments = {'subscriptionId': self._serialize.url('self._config.subscription_id', self._config.subscription_id, 'str', min_length=1), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'communicationServiceName': self._serialize.url('communication_service_name', communication_service_name, 'str', max_length=63, min_length=1, pattern='^[-\\w]+$')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        request = self._client.delete(url, query_parameters, header_parameters)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 202, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize(_models.ErrorResponse, response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        response_headers = {}
        if response.status_code == 202:
            response_headers['location'] = self._deserialize('str', response.headers.get('location'))
        if cls:
            return cls(pipeline_response, None, response_headers)
    _delete_initial.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Communication/communicationServices/{communicationServiceName}'}

    def begin_delete(self, resource_group_name, communication_service_name, **kwargs):
        if False:
            return 10
        'Delete.\n\n        Operation to delete a CommunicationService.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n        :type resource_group_name: str\n        :param communication_service_name: The name of the CommunicationService resource.\n        :type communication_service_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: True for ARMPolling, False for no polling, or a\n         polling object for personal polling strategy\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no Retry-After header is present.\n        :return: An instance of LROPoller that returns either None or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[None]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        polling = kwargs.pop('polling', True)
        cls = kwargs.pop('cls', None)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._delete_initial(resource_group_name=resource_group_name, communication_service_name=communication_service_name, cls=lambda x, y, z: x, **kwargs)
        kwargs.pop('error_map', None)
        kwargs.pop('content_type', None)

        def get_long_running_output(pipeline_response):
            if False:
                for i in range(10):
                    print('nop')
            if cls:
                return cls(pipeline_response, None, {})
        path_format_arguments = {'subscriptionId': self._serialize.url('self._config.subscription_id', self._config.subscription_id, 'str', min_length=1), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'communicationServiceName': self._serialize.url('communication_service_name', communication_service_name, 'str', max_length=63, min_length=1, pattern='^[-\\w]+$')}
        if polling is True:
            polling_method = ARMPolling(lro_delay, lro_options={'final-state-via': 'location'}, path_format_arguments=path_format_arguments, **kwargs)
        elif polling is False:
            polling_method = NoPolling()
        else:
            polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        else:
            return LROPoller(self._client, raw_result, get_long_running_output, polling_method)
    begin_delete.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Communication/communicationServices/{communicationServiceName}'}

    def list_keys(self, resource_group_name, communication_service_name, **kwargs):
        if False:
            while True:
                i = 10
        'List Keys.\n\n        Get the access keys of the CommunicationService resource.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n        :type resource_group_name: str\n        :param communication_service_name: The name of the CommunicationService resource.\n        :type communication_service_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: CommunicationServiceKeys, or the result of cls(response)\n        :rtype: ~communication_service_management_client.models.CommunicationServiceKeys\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2020-08-20'
        accept = 'application/json'
        url = self.list_keys.metadata['url']
        path_format_arguments = {'subscriptionId': self._serialize.url('self._config.subscription_id', self._config.subscription_id, 'str', min_length=1), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'communicationServiceName': self._serialize.url('communication_service_name', communication_service_name, 'str', max_length=63, min_length=1, pattern='^[-\\w]+$')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        request = self._client.post(url, query_parameters, header_parameters)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize(_models.ErrorResponse, response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('CommunicationServiceKeys', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    list_keys.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Communication/communicationServices/{communicationServiceName}/listKeys'}

    def regenerate_key(self, resource_group_name, communication_service_name, parameters, **kwargs):
        if False:
            i = 10
            return i + 15
        'Regenerate Key.\n\n        Regenerate CommunicationService access key. PrimaryKey and SecondaryKey cannot be regenerated\n        at the same time.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n        :type resource_group_name: str\n        :param communication_service_name: The name of the CommunicationService resource.\n        :type communication_service_name: str\n        :param parameters: Parameter that describes the Regenerate Key Operation.\n        :type parameters: ~communication_service_management_client.models.RegenerateKeyParameters\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: CommunicationServiceKeys, or the result of cls(response)\n        :rtype: ~communication_service_management_client.models.CommunicationServiceKeys\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2020-08-20'
        content_type = kwargs.pop('content_type', 'application/json')
        accept = 'application/json'
        url = self.regenerate_key.metadata['url']
        path_format_arguments = {'subscriptionId': self._serialize.url('self._config.subscription_id', self._config.subscription_id, 'str', min_length=1), 'resourceGroupName': self._serialize.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1, pattern='^[-\\w\\._\\(\\)]+$'), 'communicationServiceName': self._serialize.url('communication_service_name', communication_service_name, 'str', max_length=63, min_length=1, pattern='^[-\\w]+$')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        body_content_kwargs = {}
        body_content = self._serialize.body(parameters, 'RegenerateKeyParameters')
        body_content_kwargs['content'] = body_content
        request = self._client.post(url, query_parameters, header_parameters, **body_content_kwargs)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize(_models.ErrorResponse, response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('CommunicationServiceKeys', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    regenerate_key.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Communication/communicationServices/{communicationServiceName}/regenerateKey'}