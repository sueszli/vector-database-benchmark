from msrest.pipeline import ClientRawResponse
from .. import models

class AnomalyDetectorClientOperationsMixin(object):

    def entire_detect(self, body, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Detect anomalies for the entire series in batch.\n\n        This operation generates a model using an entire series, each point is\n        detected with the same model. With this method, points before and after\n        a certain point are used to determine whether it is an anomaly. The\n        entire detection can give user an overall status of the time series.\n\n        :param body: Time series points and period if needed. Advanced model\n         parameters can also be set in the request.\n        :type body: ~azure.cognitiveservices.anomalydetector.models.Request\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: EntireDetectResponse or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.anomalydetector.models.EntireDetectResponse\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.anomalydetector.models.APIErrorException>`\n        '
        url = self.entire_detect.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(body, 'Request')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('EntireDetectResponse', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    entire_detect.metadata = {'url': '/timeseries/entire/detect'}

    def last_detect(self, body, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Detect anomaly status of the latest point in time series.\n\n        This operation generates a model using points before the latest one.\n        With this method, only historical points are used to determine whether\n        the target point is an anomaly. The latest point detecting operation\n        matches the scenario of real-time monitoring of business metrics.\n\n        :param body: Time series points and period if needed. Advanced model\n         parameters can also be set in the request.\n        :type body: ~azure.cognitiveservices.anomalydetector.models.Request\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: LastDetectResponse or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.anomalydetector.models.LastDetectResponse or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.anomalydetector.models.APIErrorException>`\n        '
        url = self.last_detect.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(body, 'Request')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('LastDetectResponse', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    last_detect.metadata = {'url': '/timeseries/last/detect'}

    def change_point_detect(self, body, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Detect change point for the entire series.\n\n        Evaluate change point score of every series point.\n\n        :param body: Time series points and granularity is needed. Advanced\n         model parameters can also be set in the request if needed.\n        :type body:\n         ~azure.cognitiveservices.anomalydetector.models.ChangePointDetectRequest\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ChangePointDetectResponse or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.anomalydetector.models.ChangePointDetectResponse\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.anomalydetector.models.APIErrorException>`\n        '
        url = self.change_point_detect.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(body, 'ChangePointDetectRequest')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ChangePointDetectResponse', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    change_point_detect.metadata = {'url': '/timeseries/changePoint/detect'}