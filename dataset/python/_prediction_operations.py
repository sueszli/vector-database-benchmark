# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is
# regenerated.
# --------------------------------------------------------------------------

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

        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer

        self.config = config

    def get_version_prediction(
            self, app_id, version_id, prediction_request, verbose=None, show_all_intents=None, log=None, custom_headers=None, raw=False, **operation_config):
        """Gets the predictions for an application version.

        :param app_id: The application ID.
        :type app_id: str
        :param version_id: The application version ID.
        :type version_id: str
        :param prediction_request: The prediction request parameters.
        :type prediction_request:
         ~azure.cognitiveservices.language.luis.runtime.models.PredictionRequest
        :param verbose: Indicates whether to get extra metadata for the
         entities predictions or not.
        :type verbose: bool
        :param show_all_intents: Indicates whether to return all the intents
         in the response or just the top intent.
        :type show_all_intents: bool
        :param log: Indicates whether to log the endpoint query or not.
        :type log: bool
        :param dict custom_headers: headers that will be added to the request
        :param bool raw: returns the direct response alongside the
         deserialized response
        :param operation_config: :ref:`Operation configuration
         overrides<msrest:optionsforoperations>`.
        :return: PredictionResponse or ClientRawResponse if raw=true
        :rtype:
         ~azure.cognitiveservices.language.luis.runtime.models.PredictionResponse
         or ~msrest.pipeline.ClientRawResponse
        :raises:
         :class:`ErrorException<azure.cognitiveservices.language.luis.runtime.models.ErrorException>`
        """
        # Construct URL
        url = self.get_version_prediction.metadata['url']
        path_format_arguments = {
            'Endpoint': self._serialize.url("self.config.endpoint", self.config.endpoint, 'str', skip_quote=True),
            'appId': self._serialize.url("app_id", app_id, 'str'),
            'versionId': self._serialize.url("version_id", version_id, 'str')
        }
        url = self._client.format_url(url, **path_format_arguments)

        # Construct parameters
        query_parameters = {}
        if verbose is not None:
            query_parameters['verbose'] = self._serialize.query("verbose", verbose, 'bool')
        if show_all_intents is not None:
            query_parameters['show-all-intents'] = self._serialize.query("show_all_intents", show_all_intents, 'bool')
        if log is not None:
            query_parameters['log'] = self._serialize.query("log", log, 'bool')

        # Construct headers
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)

        # Construct body
        body_content = self._serialize.body(prediction_request, 'PredictionRequest')

        # Construct and send request
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

    def get_slot_prediction(
            self, app_id, slot_name, prediction_request, verbose=None, show_all_intents=None, log=None, custom_headers=None, raw=False, **operation_config):
        """Gets the predictions for an application slot.

        :param app_id: The application ID.
        :type app_id: str
        :param slot_name: The application slot name.
        :type slot_name: str
        :param prediction_request: The prediction request parameters.
        :type prediction_request:
         ~azure.cognitiveservices.language.luis.runtime.models.PredictionRequest
        :param verbose: Indicates whether to get extra metadata for the
         entities predictions or not.
        :type verbose: bool
        :param show_all_intents: Indicates whether to return all the intents
         in the response or just the top intent.
        :type show_all_intents: bool
        :param log: Indicates whether to log the endpoint query or not.
        :type log: bool
        :param dict custom_headers: headers that will be added to the request
        :param bool raw: returns the direct response alongside the
         deserialized response
        :param operation_config: :ref:`Operation configuration
         overrides<msrest:optionsforoperations>`.
        :return: PredictionResponse or ClientRawResponse if raw=true
        :rtype:
         ~azure.cognitiveservices.language.luis.runtime.models.PredictionResponse
         or ~msrest.pipeline.ClientRawResponse
        :raises:
         :class:`ErrorException<azure.cognitiveservices.language.luis.runtime.models.ErrorException>`
        """
        # Construct URL
        url = self.get_slot_prediction.metadata['url']
        path_format_arguments = {
            'Endpoint': self._serialize.url("self.config.endpoint", self.config.endpoint, 'str', skip_quote=True),
            'appId': self._serialize.url("app_id", app_id, 'str'),
            'slotName': self._serialize.url("slot_name", slot_name, 'str')
        }
        url = self._client.format_url(url, **path_format_arguments)

        # Construct parameters
        query_parameters = {}
        if verbose is not None:
            query_parameters['verbose'] = self._serialize.query("verbose", verbose, 'bool')
        if show_all_intents is not None:
            query_parameters['show-all-intents'] = self._serialize.query("show_all_intents", show_all_intents, 'bool')
        if log is not None:
            query_parameters['log'] = self._serialize.query("log", log, 'bool')

        # Construct headers
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)

        # Construct body
        body_content = self._serialize.body(prediction_request, 'PredictionRequest')

        # Construct and send request
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
