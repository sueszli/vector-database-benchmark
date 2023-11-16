from msrest.pipeline import ClientRawResponse
from .. import models

class EventsOperations(object):
    """EventsOperations operations.

    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """
    models = models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            return 10
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self.config = config

    def get_by_type(self, app_id, event_type, timespan=None, filter=None, search=None, orderby=None, select=None, skip=None, top=None, format=None, count=None, apply=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Execute OData query.\n\n        Executes an OData query for events.\n\n        :param app_id: ID of the application. This is Application ID from the\n         API Access settings blade in the Azure portal.\n        :type app_id: str\n        :param event_type: The type of events to query; either a standard\n         event type (`traces`, `customEvents`, `pageViews`, `requests`,\n         `dependencies`, `exceptions`, `availabilityResults`) or `$all` to\n         query across all event types. Possible values include: \'$all\',\n         \'traces\', \'customEvents\', \'pageViews\', \'browserTimings\', \'requests\',\n         \'dependencies\', \'exceptions\', \'availabilityResults\',\n         \'performanceCounters\', \'customMetrics\'\n        :type event_type: str or ~azure.applicationinsights.models.EventType\n        :param timespan: Optional. The timespan over which to retrieve events.\n         This is an ISO8601 time period value.  This timespan is applied in\n         addition to any that are specified in the Odata expression.\n        :type timespan: str\n        :param filter: An expression used to filter the returned events\n        :type filter: str\n        :param search: A free-text search expression to match for whether a\n         particular event should be returned\n        :type search: str\n        :param orderby: A comma-separated list of properties with \\"asc\\"\n         (the default) or \\"desc\\" to control the order of returned events\n        :type orderby: str\n        :param select: Limits the properties to just those requested on each\n         returned event\n        :type select: str\n        :param skip: The number of items to skip over before returning events\n        :type skip: int\n        :param top: The number of events to return\n        :type top: int\n        :param format: Format for the returned events\n        :type format: str\n        :param count: Request a count of matching items included with the\n         returned events\n        :type count: bool\n        :param apply: An expression used for aggregation over returned events\n        :type apply: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: EventsResults or ClientRawResponse if raw=true\n        :rtype: ~azure.applicationinsights.models.EventsResults or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.applicationinsights.models.ErrorResponseException>`\n        '
        url = self.get_by_type.metadata['url']
        path_format_arguments = {'appId': self._serialize.url('app_id', app_id, 'str'), 'eventType': self._serialize.url('event_type', event_type, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if timespan is not None:
            query_parameters['timespan'] = self._serialize.query('timespan', timespan, 'str')
        if filter is not None:
            query_parameters['$filter'] = self._serialize.query('filter', filter, 'str')
        if search is not None:
            query_parameters['$search'] = self._serialize.query('search', search, 'str')
        if orderby is not None:
            query_parameters['$orderby'] = self._serialize.query('orderby', orderby, 'str')
        if select is not None:
            query_parameters['$select'] = self._serialize.query('select', select, 'str')
        if skip is not None:
            query_parameters['$skip'] = self._serialize.query('skip', skip, 'int')
        if top is not None:
            query_parameters['$top'] = self._serialize.query('top', top, 'int')
        if format is not None:
            query_parameters['$format'] = self._serialize.query('format', format, 'str')
        if count is not None:
            query_parameters['$count'] = self._serialize.query('count', count, 'bool')
        if apply is not None:
            query_parameters['$apply'] = self._serialize.query('apply', apply, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('EventsResults', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_by_type.metadata = {'url': '/apps/{appId}/events/{eventType}'}

    def get(self, app_id, event_type, event_id, timespan=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        "Get an event.\n\n        Gets the data for a single event.\n\n        :param app_id: ID of the application. This is Application ID from the\n         API Access settings blade in the Azure portal.\n        :type app_id: str\n        :param event_type: The type of events to query; either a standard\n         event type (`traces`, `customEvents`, `pageViews`, `requests`,\n         `dependencies`, `exceptions`, `availabilityResults`) or `$all` to\n         query across all event types. Possible values include: '$all',\n         'traces', 'customEvents', 'pageViews', 'browserTimings', 'requests',\n         'dependencies', 'exceptions', 'availabilityResults',\n         'performanceCounters', 'customMetrics'\n        :type event_type: str or ~azure.applicationinsights.models.EventType\n        :param event_id: ID of event.\n        :type event_id: str\n        :param timespan: Optional. The timespan over which to retrieve events.\n         This is an ISO8601 time period value.  This timespan is applied in\n         addition to any that are specified in the Odata expression.\n        :type timespan: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: EventsResults or ClientRawResponse if raw=true\n        :rtype: ~azure.applicationinsights.models.EventsResults or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.applicationinsights.models.ErrorResponseException>`\n        "
        url = self.get.metadata['url']
        path_format_arguments = {'appId': self._serialize.url('app_id', app_id, 'str'), 'eventType': self._serialize.url('event_type', event_type, 'str'), 'eventId': self._serialize.url('event_id', event_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if timespan is not None:
            query_parameters['timespan'] = self._serialize.query('timespan', timespan, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('EventsResults', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get.metadata = {'url': '/apps/{appId}/events/{eventType}/{eventId}'}

    def get_odata_metadata(self, app_id, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Get OData metadata.\n\n        Gets OData EDMX metadata describing the event data model.\n\n        :param app_id: ID of the application. This is Application ID from the\n         API Access settings blade in the Azure portal.\n        :type app_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: object or ClientRawResponse if raw=true\n        :rtype: object or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.applicationinsights.models.ErrorResponseException>`\n        '
        url = self.get_odata_metadata.metadata['url']
        path_format_arguments = {'appId': self._serialize.url('app_id', app_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/xml;charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('object', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_odata_metadata.metadata = {'url': '/apps/{appId}/events/$metadata'}