from msrest.pipeline import ClientRawResponse
from .. import models

class KnowledgebaseOperations(object):
    """KnowledgebaseOperations operations.

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

    def list_all(self, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Gets all knowledgebases for a user.\n\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: KnowledgebasesDTO or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.knowledge.qnamaker.models.KnowledgebasesDTO\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.knowledge.qnamaker.models.ErrorResponseException>`\n        '
        url = self.list_all.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
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
            deserialized = self._deserialize('KnowledgebasesDTO', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    list_all.metadata = {'url': '/knowledgebases'}

    def get_details(self, kb_id, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Gets details of a specific knowledgebase.\n\n        :param kb_id: Knowledgebase id.\n        :type kb_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: KnowledgebaseDTO or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.knowledge.qnamaker.models.KnowledgebaseDTO or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.knowledge.qnamaker.models.ErrorResponseException>`\n        '
        url = self.get_details.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'kbId': self._serialize.url('kb_id', kb_id, 'str')}
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
            deserialized = self._deserialize('KnowledgebaseDTO', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_details.metadata = {'url': '/knowledgebases/{kbId}'}

    def delete(self, kb_id, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Deletes the knowledgebase and all its data.\n\n        :param kb_id: Knowledgebase id.\n        :type kb_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.knowledge.qnamaker.models.ErrorResponseException>`\n        '
        url = self.delete.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'kbId': self._serialize.url('kb_id', kb_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.ErrorResponseException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    delete.metadata = {'url': '/knowledgebases/{kbId}'}

    def publish(self, kb_id, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Publishes all changes in test index of a knowledgebase to its prod\n        index.\n\n        :param kb_id: Knowledgebase id.\n        :type kb_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.knowledge.qnamaker.models.ErrorResponseException>`\n        '
        url = self.publish.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'kbId': self._serialize.url('kb_id', kb_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.post(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.ErrorResponseException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    publish.metadata = {'url': '/knowledgebases/{kbId}'}

    def replace(self, kb_id, qn_alist, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Replace knowledgebase contents.\n\n        :param kb_id: Knowledgebase id.\n        :type kb_id: str\n        :param qn_alist: List of Q-A (QnADTO) to be added to the\n         knowledgebase. Q-A Ids are assigned by the service and should be\n         omitted.\n        :type qn_alist:\n         list[~azure.cognitiveservices.knowledge.qnamaker.models.QnADTO]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.knowledge.qnamaker.models.ErrorResponseException>`\n        '
        replace_kb = models.ReplaceKbDTO(qn_alist=qn_alist)
        url = self.replace.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'kbId': self._serialize.url('kb_id', kb_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(replace_kb, 'ReplaceKbDTO')
        request = self._client.put(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.ErrorResponseException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    replace.metadata = {'url': '/knowledgebases/{kbId}'}

    def update(self, kb_id, update_kb, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Asynchronous operation to modify a knowledgebase.\n\n        :param kb_id: Knowledgebase id.\n        :type kb_id: str\n        :param update_kb: Post body of the request.\n        :type update_kb:\n         ~azure.cognitiveservices.knowledge.qnamaker.models.UpdateKbOperationDTO\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Operation or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.knowledge.qnamaker.models.Operation\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.knowledge.qnamaker.models.ErrorResponseException>`\n        '
        url = self.update.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'kbId': self._serialize.url('kb_id', kb_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(update_kb, 'UpdateKbOperationDTO')
        request = self._client.patch(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [202]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        header_dict = {}
        if response.status_code == 202:
            deserialized = self._deserialize('Operation', response)
            header_dict = {'Location': 'str'}
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            client_raw_response.add_headers(header_dict)
            return client_raw_response
        return deserialized
    update.metadata = {'url': '/knowledgebases/{kbId}'}

    def create(self, create_kb_payload, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Asynchronous operation to create a new knowledgebase.\n\n        :param create_kb_payload: Post body of the request.\n        :type create_kb_payload:\n         ~azure.cognitiveservices.knowledge.qnamaker.models.CreateKbDTO\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Operation or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.knowledge.qnamaker.models.Operation\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.knowledge.qnamaker.models.ErrorResponseException>`\n        '
        url = self.create.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(create_kb_payload, 'CreateKbDTO')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [202]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 202:
            deserialized = self._deserialize('Operation', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    create.metadata = {'url': '/knowledgebases/create'}

    def download(self, kb_id, environment, source=None, changed_since=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        "Download the knowledgebase.\n\n        :param kb_id: Knowledgebase id.\n        :type kb_id: str\n        :param environment: Specifies whether environment is Test or Prod.\n         Possible values include: 'Prod', 'Test'\n        :type environment: str or\n         ~azure.cognitiveservices.knowledge.qnamaker.models.EnvironmentType\n        :param source: The source property filter to apply.\n        :type source: str\n        :param changed_since: The last changed status property filter to\n         apply.\n        :type changed_since: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: QnADocumentsDTO or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.knowledge.qnamaker.models.QnADocumentsDTO or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.knowledge.qnamaker.models.ErrorResponseException>`\n        "
        url = self.download.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'kbId': self._serialize.url('kb_id', kb_id, 'str'), 'environment': self._serialize.url('environment', environment, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if source is not None:
            query_parameters['source'] = self._serialize.query('source', source, 'str')
        if changed_since is not None:
            query_parameters['changedSince'] = self._serialize.query('changed_since', changed_since, 'str')
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
            deserialized = self._deserialize('QnADocumentsDTO', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    download.metadata = {'url': '/knowledgebases/{kbId}/{environment}/qna'}

    def generate_answer(self, kb_id, generate_answer_payload, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'GenerateAnswer call to query knowledgebase (QnA Maker Managed).\n\n        :param kb_id: Knowledgebase id.\n        :type kb_id: str\n        :param generate_answer_payload: Post body of the request.\n        :type generate_answer_payload:\n         ~azure.cognitiveservices.knowledge.qnamaker.models.QueryDTO\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: QnASearchResultList or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.knowledge.qnamaker.models.QnASearchResultList\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.knowledge.qnamaker.models.ErrorResponseException>`\n        '
        url = self.generate_answer.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'kbId': self._serialize.url('kb_id', kb_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(generate_answer_payload, 'QueryDTO')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.ErrorResponseException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('QnASearchResultList', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    generate_answer.metadata = {'url': '/knowledgebases/{kbId}/generateAnswer'}

    def train(self, kb_id, feedback_records=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Train call to add suggestions to knowledgebase (QnAMaker Managed).\n\n        :param kb_id: Knowledgebase id.\n        :type kb_id: str\n        :param feedback_records: List of feedback records.\n        :type feedback_records:\n         list[~azure.cognitiveservices.knowledge.qnamaker.models.FeedbackRecordDTO]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`ErrorResponseException<azure.cognitiveservices.knowledge.qnamaker.models.ErrorResponseException>`\n        '
        train_payload = models.FeedbackRecordsDTO(feedback_records=feedback_records)
        url = self.train.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'kbId': self._serialize.url('kb_id', kb_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(train_payload, 'FeedbackRecordsDTO')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.ErrorResponseException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    train.metadata = {'url': '/knowledgebases/{kbId}/train'}