from msrest.pipeline import ClientRawResponse
from .. import models

class MetricsOperations(object):
    """MetricsOperations operations.

    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """
    models = models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            for i in range(10):
                print('nop')
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self.config = config

    def get(self, app_id, metric_id, timespan=None, interval=None, aggregation=None, segment=None, top=None, orderby=None, filter=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Retrieve metric data.\n\n        Gets metric values for a single metric.\n\n        :param app_id: ID of the application. This is Application ID from the\n         API Access settings blade in the Azure portal.\n        :type app_id: str\n        :param metric_id: ID of the metric. This is either a standard AI\n         metric, or an application-specific custom metric. Possible values\n         include: \'requests/count\', \'requests/duration\', \'requests/failed\',\n         \'users/count\', \'users/authenticated\', \'pageViews/count\',\n         \'pageViews/duration\', \'client/processingDuration\',\n         \'client/receiveDuration\', \'client/networkDuration\',\n         \'client/sendDuration\', \'client/totalDuration\', \'dependencies/count\',\n         \'dependencies/failed\', \'dependencies/duration\', \'exceptions/count\',\n         \'exceptions/browser\', \'exceptions/server\', \'sessions/count\',\n         \'performanceCounters/requestExecutionTime\',\n         \'performanceCounters/requestsPerSecond\',\n         \'performanceCounters/requestsInQueue\',\n         \'performanceCounters/memoryAvailableBytes\',\n         \'performanceCounters/exceptionsPerSecond\',\n         \'performanceCounters/processCpuPercentage\',\n         \'performanceCounters/processIOBytesPerSecond\',\n         \'performanceCounters/processPrivateBytes\',\n         \'performanceCounters/processorCpuPercentage\',\n         \'availabilityResults/availabilityPercentage\',\n         \'availabilityResults/duration\', \'billing/telemetryCount\',\n         \'customEvents/count\'\n        :type metric_id: str or ~azure.applicationinsights.models.MetricId\n        :param timespan: The timespan over which to retrieve metric values.\n         This is an ISO8601 time period value. If timespan is omitted, a\n         default time range of `PT12H` ("last 12 hours") is used. The actual\n         timespan that is queried may be adjusted by the server based. In all\n         cases, the actual time span used for the query is included in the\n         response.\n        :type timespan: str\n        :param interval: The time interval to use when retrieving metric\n         values. This is an ISO8601 duration. If interval is omitted, the\n         metric value is aggregated across the entire timespan. If interval is\n         supplied, the server may adjust the interval to a more appropriate\n         size based on the timespan used for the query. In all cases, the\n         actual interval used for the query is included in the response.\n        :type interval: timedelta\n        :param aggregation: The aggregation to use when computing the metric\n         values. To retrieve more than one aggregation at a time, separate them\n         with a comma. If no aggregation is specified, then the default\n         aggregation for the metric is used.\n        :type aggregation: list[str or\n         ~azure.applicationinsights.models.MetricsAggregation]\n        :param segment: The name of the dimension to segment the metric values\n         by. This dimension must be applicable to the metric you are\n         retrieving. To segment by more than one dimension at a time, separate\n         them with a comma (,). In this case, the metric data will be segmented\n         in the order the dimensions are listed in the parameter.\n        :type segment: list[str or\n         ~azure.applicationinsights.models.MetricsSegment]\n        :param top: The number of segments to return.  This value is only\n         valid when segment is specified.\n        :type top: int\n        :param orderby: The aggregation function and direction to sort the\n         segments by.  This value is only valid when segment is specified.\n        :type orderby: str\n        :param filter: An expression used to filter the results.  This value\n         should be a valid OData filter expression where the keys of each\n         clause should be applicable dimensions for the metric you are\n         retrieving.\n        :type filter: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: MetricsResult or ClientRawResponse if raw=true\n        :rtype: ~azure.applicationinsights.models.MetricsResult or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.applicationinsights.models.ErrorResponseException>`\n        '
        url = self.get.metadata['url']
        path_format_arguments = {'appId': self._serialize.url('app_id', app_id, 'str'), 'metricId': self._serialize.url('metric_id', metric_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if timespan is not None:
            query_parameters['timespan'] = self._serialize.query('timespan', timespan, 'str')
        if interval is not None:
            query_parameters['interval'] = self._serialize.query('interval', interval, 'duration')
        if aggregation is not None:
            query_parameters['aggregation'] = self._serialize.query('aggregation', aggregation, '[MetricsAggregation]', div=',', min_items=1)
        if segment is not None:
            query_parameters['segment'] = self._serialize.query('segment', segment, '[str]', div=',', min_items=1)
        if top is not None:
            query_parameters['top'] = self._serialize.query('top', top, 'int')
        if orderby is not None:
            query_parameters['orderby'] = self._serialize.query('orderby', orderby, 'str')
        if filter is not None:
            query_parameters['filter'] = self._serialize.query('filter', filter, 'str')
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
            deserialized = self._deserialize('MetricsResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get.metadata = {'url': '/apps/{appId}/metrics/{metricId}'}

    def get_multiple(self, app_id, body, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Retrieve metric data.\n\n        Gets metric values for multiple metrics.\n\n        :param app_id: ID of the application. This is Application ID from the\n         API Access settings blade in the Azure portal.\n        :type app_id: str\n        :param body: The batched metrics query.\n        :type body:\n         list[~azure.applicationinsights.models.MetricsPostBodySchema]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype: list[~azure.applicationinsights.models.MetricsResultsItem] or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.applicationinsights.models.ErrorResponseException>`\n        '
        url = self.get_multiple.metadata['url']
        path_format_arguments = {'appId': self._serialize.url('app_id', app_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(body, '[MetricsPostBodySchema]')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[MetricsResultsItem]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_multiple.metadata = {'url': '/apps/{appId}/metrics'}

    def get_metadata(self, app_id, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Retrieve metric metatadata.\n\n        Gets metadata describing the available metrics.\n\n        :param app_id: ID of the application. This is Application ID from the\n         API Access settings blade in the Azure portal.\n        :type app_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: object or ClientRawResponse if raw=true\n        :rtype: object or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.applicationinsights.models.ErrorResponseException>`\n        '
        url = self.get_metadata.metadata['url']
        path_format_arguments = {'appId': self._serialize.url('app_id', app_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
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
            deserialized = self._deserialize('object', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_metadata.metadata = {'url': '/apps/{appId}/metrics/metadata'}