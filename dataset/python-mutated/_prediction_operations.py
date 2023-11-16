from msrest.pipeline import ClientRawResponse
from .. import models

class PredictionOperations(object):
    """PredictionOperations operations.

    You should not instantiate directly this class, but create a Client instance that will create it for you and attach it as attribute.

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

    def get_version_prediction(self, app_id, version_id, prediction_request, verbose=None, show_all_intents=None, log=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Gets the predictions for an application version.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param version_id: The application version ID.\n        :type version_id: str\n        :param prediction_request: The prediction request parameters.\n        :type prediction_request:\n         ~azure.cognitiveservices.language.luis.runtime.models.PredictionRequest\n        :param verbose: Indicates whether to get extra metadata for the\n         entities predictions or not.\n        :type verbose: bool\n        :param show_all_intents: Indicates whether to return all the intents\n         in the response or just the top intent.\n        :type show_all_intents: bool\n        :param log: Indicates whether to log the endpoint query or not.\n        :type log: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: PredictionResponse or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.runtime.models.PredictionResponse\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorException<azure.cognitiveservices.language.luis.runtime.models.ErrorException>`\n        '
        url = self.get_version_prediction.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'versionId': self._serialize.url('version_id', version_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if verbose is not None:
            query_parameters['verbose'] = self._serialize.query('verbose', verbose, 'bool')
        if show_all_intents is not None:
            query_parameters['show-all-intents'] = self._serialize.query('show_all_intents', show_all_intents, 'bool')
        if log is not None:
            query_parameters['log'] = self._serialize.query('log', log, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(prediction_request, 'PredictionRequest')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('PredictionResponse', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_version_prediction.metadata = {'url': '/apps/{appId}/versions/{versionId}/predict'}

    def get_slot_prediction(self, app_id, slot_name, prediction_request, verbose=None, show_all_intents=None, log=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Gets the predictions for an application slot.\n\n        :param app_id: The application ID.\n        :type app_id: str\n        :param slot_name: The application slot name.\n        :type slot_name: str\n        :param prediction_request: The prediction request parameters.\n        :type prediction_request:\n         ~azure.cognitiveservices.language.luis.runtime.models.PredictionRequest\n        :param verbose: Indicates whether to get extra metadata for the\n         entities predictions or not.\n        :type verbose: bool\n        :param show_all_intents: Indicates whether to return all the intents\n         in the response or just the top intent.\n        :type show_all_intents: bool\n        :param log: Indicates whether to log the endpoint query or not.\n        :type log: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: PredictionResponse or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.language.luis.runtime.models.PredictionResponse\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorException<azure.cognitiveservices.language.luis.runtime.models.ErrorException>`\n        '
        url = self.get_slot_prediction.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'appId': self._serialize.url('app_id', app_id, 'str'), 'slotName': self._serialize.url('slot_name', slot_name, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if verbose is not None:
            query_parameters['verbose'] = self._serialize.query('verbose', verbose, 'bool')
        if show_all_intents is not None:
            query_parameters['show-all-intents'] = self._serialize.query('show_all_intents', show_all_intents, 'bool')
        if log is not None:
            query_parameters['log'] = self._serialize.query('log', log, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(prediction_request, 'PredictionRequest')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('PredictionResponse', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_slot_prediction.metadata = {'url': '/apps/{appId}/slots/{slotName}/predict'}