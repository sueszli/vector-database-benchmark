from msrest.pipeline import ClientRawResponse
from .. import models

class QueryOperations(object):
    """QueryOperations operations.

    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """
    models = models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            i = 10
            return i + 15
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self.config = config

    def execute(self, app_id, body, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Execute an Analytics query.\n\n        Executes an Analytics query for data.\n\n        `Here <https://dev.applicationinsights.io/documentation/Using-the-API/Query>`_\n        is an example for using POST with an Analytics query.\n\n        :param app_id: ID of the application. This is Application ID from the\n         API Access settings blade in the Azure portal.\n        :type app_id: str\n        :param body: The Analytics query. Learn more about the `Analytics\n         query syntax <https://azure.microsoft.com/documentation/articles/app-insights-analytics-reference/>`_.\n        :type body: ~azure.applicationinsights.models.QueryBody\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: QueryResults or ClientRawResponse if raw=true\n        :rtype: ~azure.applicationinsights.models.QueryResults or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.applicationinsights.models.ErrorResponseException>`\n        '
        url = self.execute.metadata['url']
        path_format_arguments = {'appId': self._serialize.url('app_id', app_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(body, 'QueryBody')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('QueryResults', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    execute.metadata = {'url': '/apps/{appId}/query'}