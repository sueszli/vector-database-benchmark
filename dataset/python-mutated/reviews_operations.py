from msrest.pipeline import ClientRawResponse
from .. import models

class ReviewsOperations(object):
    """ReviewsOperations operations.

    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    :ivar content_type: The content type. Constant value: "text/plain".
    """
    models = models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            while True:
                i = 10
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self.config = config
        self.content_type = 'text/plain'

    def get_review(self, team_name, review_id, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Returns review details for the review Id passed.\n\n        :param team_name: Your Team Name.\n        :type team_name: str\n        :param review_id: Id of the review.\n        :type review_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Review or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.contentmoderator.models.Review\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.get_review.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'teamName': self._serialize.url('team_name', team_name, 'str'), 'reviewId': self._serialize.url('review_id', review_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Review', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_review.metadata = {'url': '/contentmoderator/review/v1.0/teams/{teamName}/reviews/{reviewId}'}

    def get_job_details(self, team_name, job_id, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Get the Job Details for a Job Id.\n\n        :param team_name: Your Team Name.\n        :type team_name: str\n        :param job_id: Id of the job.\n        :type job_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Job or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.contentmoderator.models.Job or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.get_job_details.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'teamName': self._serialize.url('team_name', team_name, 'str'), 'JobId': self._serialize.url('job_id', job_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Job', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_job_details.metadata = {'url': '/contentmoderator/review/v1.0/teams/{teamName}/jobs/{JobId}'}

    def create_reviews(self, url_content_type, team_name, create_review_body, sub_team=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'The reviews created would show up for Reviewers on your team. As\n        Reviewers complete reviewing, results of the Review would be POSTED\n        (i.e. HTTP POST) on the specified CallBackEndpoint.\n        <h3>CallBack Schemas </h3>\n        <h4>Review Completion CallBack Sample</h4>\n        <p>\n        {<br/>\n        "ReviewId": "<Review Id>",<br/>\n        "ModifiedOn": "2016-10-11T22:36:32.9934851Z",<br/>\n        "ModifiedBy": "<Name of the Reviewer>",<br/>\n        "CallBackType": "Review",<br/>\n        "ContentId": "<The ContentId that was specified input>",<br/>\n        "Metadata": {<br/>\n        "adultscore": "0.xxx",<br/>\n        "a": "False",<br/>\n        "racyscore": "0.xxx",<br/>\n        "r": "True"<br/>\n        },<br/>\n        "ReviewerResultTags": {<br/>\n        "a": "False",<br/>\n        "r": "True"<br/>\n        }<br/>\n        }<br/>\n        </p>.\n\n        :param url_content_type: The content type.\n        :type url_content_type: str\n        :param team_name: Your team name.\n        :type team_name: str\n        :param create_review_body: Body for create reviews API\n        :type create_review_body:\n         list[~azure.cognitiveservices.vision.contentmoderator.models.CreateReviewBodyItem]\n        :param sub_team: SubTeam of your team, you want to assign the created\n         review to.\n        :type sub_team: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype: list[str] or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.create_reviews.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'teamName': self._serialize.url('team_name', team_name, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if sub_team is not None:
            query_parameters['subTeam'] = self._serialize.query('sub_team', sub_team, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        header_parameters['UrlContentType'] = self._serialize.header('url_content_type', url_content_type, 'str')
        body_content = self._serialize.body(create_review_body, '[CreateReviewBodyItem]')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[str]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    create_reviews.metadata = {'url': '/contentmoderator/review/v1.0/teams/{teamName}/reviews'}

    def create_job(self, team_name, content_type, content_id, workflow_name, job_content_type, content_value, call_back_endpoint=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'A job Id will be returned for the content posted on this endpoint.\n        Once the content is evaluated against the Workflow provided the review\n        will be created or ignored based on the workflow expression.\n        <h3>CallBack Schemas </h3>\n        <p>\n        <h4>Job Completion CallBack Sample</h4><br/>\n        {<br/>\n        "JobId": "<Job Id>,<br/>\n        "ReviewId": "<Review Id, if the Job resulted in a Review to be\n        created>",<br/>\n        "WorkFlowId": "default",<br/>\n        "Status": "<This will be one of Complete, InProgress, Error>",<br/>\n        "ContentType": "Image",<br/>\n        "ContentId": "<This is the ContentId that was specified on\n        input>",<br/>\n        "CallBackType": "Job",<br/>\n        "Metadata": {<br/>\n        "adultscore": "0.xxx",<br/>\n        "a": "False",<br/>\n        "racyscore": "0.xxx",<br/>\n        "r": "True"<br/>\n        }<br/>\n        }<br/>\n        </p>\n        <p>\n        <h4>Review Completion CallBack Sample</h4><br/>\n        {\n        "ReviewId": "<Review Id>",<br/>\n        "ModifiedOn": "2016-10-11T22:36:32.9934851Z",<br/>\n        "ModifiedBy": "<Name of the Reviewer>",<br/>\n        "CallBackType": "Review",<br/>\n        "ContentId": "<The ContentId that was specified input>",<br/>\n        "Metadata": {<br/>\n        "adultscore": "0.xxx",\n        "a": "False",<br/>\n        "racyscore": "0.xxx",<br/>\n        "r": "True"<br/>\n        },<br/>\n        "ReviewerResultTags": {<br/>\n        "a": "False",<br/>\n        "r": "True"<br/>\n        }<br/>\n        }<br/>\n        </p>.\n\n        :param team_name: Your team name.\n        :type team_name: str\n        :param content_type: Image, Text or Video. Possible values include:\n         \'Image\', \'Text\', \'Video\'\n        :type content_type: str\n        :param content_id: Id/Name to identify the content submitted.\n        :type content_id: str\n        :param workflow_name: Workflow Name that you want to invoke.\n        :type workflow_name: str\n        :param job_content_type: The content type. Possible values include:\n         \'application/json\', \'image/jpeg\'\n        :type job_content_type: str\n        :param content_value: Content to evaluate for a job.\n        :type content_value: str\n        :param call_back_endpoint: Callback endpoint for posting the create\n         job result.\n        :type call_back_endpoint: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: JobId or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.contentmoderator.models.JobId\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        content = models.Content(content_value=content_value)
        url = self.create_job.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'teamName': self._serialize.url('team_name', team_name, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['ContentType'] = self._serialize.query('content_type', content_type, 'str')
        query_parameters['ContentId'] = self._serialize.query('content_id', content_id, 'str')
        query_parameters['WorkflowName'] = self._serialize.query('workflow_name', workflow_name, 'str')
        if call_back_endpoint is not None:
            query_parameters['CallBackEndpoint'] = self._serialize.query('call_back_endpoint', call_back_endpoint, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        header_parameters['Content-Type'] = self._serialize.header('job_content_type', job_content_type, 'str')
        body_content = self._serialize.body(content, 'Content')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('JobId', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    create_job.metadata = {'url': '/contentmoderator/review/v1.0/teams/{teamName}/jobs'}

    def add_video_frame(self, team_name, review_id, timescale=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'The reviews created would show up for Reviewers on your team. As\n        Reviewers complete reviewing, results of the Review would be POSTED\n        (i.e. HTTP POST) on the specified CallBackEndpoint.\n        <h3>CallBack Schemas </h3>\n        <h4>Review Completion CallBack Sample</h4>\n        <p>\n        {<br/>\n        "ReviewId": "<Review Id>",<br/>\n        "ModifiedOn": "2016-10-11T22:36:32.9934851Z",<br/>\n        "ModifiedBy": "<Name of the Reviewer>",<br/>\n        "CallBackType": "Review",<br/>\n        "ContentId": "<The ContentId that was specified input>",<br/>\n        "Metadata": {<br/>\n        "adultscore": "0.xxx",<br/>\n        "a": "False",<br/>\n        "racyscore": "0.xxx",<br/>\n        "r": "True"<br/>\n        },<br/>\n        "ReviewerResultTags": {<br/>\n        "a": "False",<br/>\n        "r": "True"<br/>\n        }<br/>\n        }<br/>\n        </p>.\n\n        :param team_name: Your team name.\n        :type team_name: str\n        :param review_id: Id of the review.\n        :type review_id: str\n        :param timescale: Timescale of the video you are adding frames to.\n        :type timescale: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.add_video_frame.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'teamName': self._serialize.url('team_name', team_name, 'str'), 'reviewId': self._serialize.url('review_id', review_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if timescale is not None:
            query_parameters['timescale'] = self._serialize.query('timescale', timescale, 'int')
        header_parameters = {}
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.post(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    add_video_frame.metadata = {'url': '/contentmoderator/review/v1.0/teams/{teamName}/reviews/{reviewId}/frames'}

    def get_video_frames(self, team_name, review_id, start_seed=None, no_of_records=None, filter=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'The reviews created would show up for Reviewers on your team. As\n        Reviewers complete reviewing, results of the Review would be POSTED\n        (i.e. HTTP POST) on the specified CallBackEndpoint.\n        <h3>CallBack Schemas </h3>\n        <h4>Review Completion CallBack Sample</h4>\n        <p>\n        {<br/>\n        "ReviewId": "<Review Id>",<br/>\n        "ModifiedOn": "2016-10-11T22:36:32.9934851Z",<br/>\n        "ModifiedBy": "<Name of the Reviewer>",<br/>\n        "CallBackType": "Review",<br/>\n        "ContentId": "<The ContentId that was specified input>",<br/>\n        "Metadata": {<br/>\n        "adultscore": "0.xxx",<br/>\n        "a": "False",<br/>\n        "racyscore": "0.xxx",<br/>\n        "r": "True"<br/>\n        },<br/>\n        "ReviewerResultTags": {<br/>\n        "a": "False",<br/>\n        "r": "True"<br/>\n        }<br/>\n        }<br/>\n        </p>.\n\n        :param team_name: Your team name.\n        :type team_name: str\n        :param review_id: Id of the review.\n        :type review_id: str\n        :param start_seed: Time stamp of the frame from where you want to\n         start fetching the frames.\n        :type start_seed: int\n        :param no_of_records: Number of frames to fetch.\n        :type no_of_records: int\n        :param filter: Get frames filtered by tags.\n        :type filter: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Frames or ClientRawResponse if raw=true\n        :rtype: ~azure.cognitiveservices.vision.contentmoderator.models.Frames\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.get_video_frames.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'teamName': self._serialize.url('team_name', team_name, 'str'), 'reviewId': self._serialize.url('review_id', review_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if start_seed is not None:
            query_parameters['startSeed'] = self._serialize.query('start_seed', start_seed, 'int')
        if no_of_records is not None:
            query_parameters['noOfRecords'] = self._serialize.query('no_of_records', no_of_records, 'int')
        if filter is not None:
            query_parameters['filter'] = self._serialize.query('filter', filter, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Frames', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_video_frames.metadata = {'url': '/contentmoderator/review/v1.0/teams/{teamName}/reviews/{reviewId}/frames'}

    def publish_video_review(self, team_name, review_id, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Publish video review to make it available for review.\n\n        :param team_name: Your team name.\n        :type team_name: str\n        :param review_id: Id of the review.\n        :type review_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.publish_video_review.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'teamName': self._serialize.url('team_name', team_name, 'str'), 'reviewId': self._serialize.url('review_id', review_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.post(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.APIErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    publish_video_review.metadata = {'url': '/contentmoderator/review/v1.0/teams/{teamName}/reviews/{reviewId}/publish'}

    def add_video_transcript_moderation_result(self, content_type, team_name, review_id, transcript_moderation_body, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'This API adds a transcript screen text result file for a video review.\n        Transcript screen text result file is a result of Screen Text API . In\n        order to generate transcript screen text result file , a transcript\n        file has to be screened for profanity using Screen Text API.\n\n        :param content_type: The content type.\n        :type content_type: str\n        :param team_name: Your team name.\n        :type team_name: str\n        :param review_id: Id of the review.\n        :type review_id: str\n        :param transcript_moderation_body: Body for add video transcript\n         moderation result API\n        :type transcript_moderation_body:\n         list[~azure.cognitiveservices.vision.contentmoderator.models.TranscriptModerationBodyItem]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.add_video_transcript_moderation_result.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'teamName': self._serialize.url('team_name', team_name, 'str'), 'reviewId': self._serialize.url('review_id', review_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        body_content = self._serialize.body(transcript_moderation_body, '[TranscriptModerationBodyItem]')
        request = self._client.put(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.APIErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    add_video_transcript_moderation_result.metadata = {'url': '/contentmoderator/review/v1.0/teams/{teamName}/reviews/{reviewId}/transcriptmoderationresult'}

    def add_video_transcript(self, team_name, review_id, vt_tfile, custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            i = 10
            return i + 15
        'This API adds a transcript file (text version of all the words spoken\n        in a video) to a video review. The file should be a valid WebVTT\n        format.\n\n        :param team_name: Your team name.\n        :type team_name: str\n        :param review_id: Id of the review.\n        :type review_id: str\n        :param vt_tfile: Transcript file of the video.\n        :type vt_tfile: Generator\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.add_video_transcript.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'teamName': self._serialize.url('team_name', team_name, 'str'), 'reviewId': self._serialize.url('review_id', review_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Content-Type'] = 'text/plain'
        if custom_headers:
            header_parameters.update(custom_headers)
        header_parameters['Content-Type'] = self._serialize.header('self.content_type', self.content_type, 'str')
        body_content = self._client.stream_upload(vt_tfile, callback)
        request = self._client.put(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.APIErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    add_video_transcript.metadata = {'url': '/contentmoderator/review/v1.0/teams/{teamName}/reviews/{reviewId}/transcript'}

    def create_video_reviews(self, content_type, team_name, create_video_reviews_body, sub_team=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'The reviews created would show up for Reviewers on your team. As\n        Reviewers complete reviewing, results of the Review would be POSTED\n        (i.e. HTTP POST) on the specified CallBackEndpoint.\n        <h3>CallBack Schemas </h3>\n        <h4>Review Completion CallBack Sample</h4>\n        <p>\n        {<br/>\n        "ReviewId": "<Review Id>",<br/>\n        "ModifiedOn": "2016-10-11T22:36:32.9934851Z",<br/>\n        "ModifiedBy": "<Name of the Reviewer>",<br/>\n        "CallBackType": "Review",<br/>\n        "ContentId": "<The ContentId that was specified input>",<br/>\n        "Metadata": {<br/>\n        "adultscore": "0.xxx",<br/>\n        "a": "False",<br/>\n        "racyscore": "0.xxx",<br/>\n        "r": "True"<br/>\n        },<br/>\n        "ReviewerResultTags": {<br/>\n        "a": "False",<br/>\n        "r": "True"<br/>\n        }<br/>\n        }<br/>\n        </p>.\n\n        :param content_type: The content type.\n        :type content_type: str\n        :param team_name: Your team name.\n        :type team_name: str\n        :param create_video_reviews_body: Body for create reviews API\n        :type create_video_reviews_body:\n         list[~azure.cognitiveservices.vision.contentmoderator.models.CreateVideoReviewsBodyItem]\n        :param sub_team: SubTeam of your team, you want to assign the created\n         review to.\n        :type sub_team: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype: list[str] or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.create_video_reviews.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'teamName': self._serialize.url('team_name', team_name, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if sub_team is not None:
            query_parameters['subTeam'] = self._serialize.query('sub_team', sub_team, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        body_content = self._serialize.body(create_video_reviews_body, '[CreateVideoReviewsBodyItem]')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.APIErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[str]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    create_video_reviews.metadata = {'url': '/contentmoderator/review/v1.0/teams/{teamName}/reviews'}

    def add_video_frame_url(self, content_type, team_name, review_id, video_frame_body, timescale=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Use this method to add frames for a video review.Timescale: This\n        parameter is a factor which is used to convert the timestamp on a frame\n        into milliseconds. Timescale is provided in the output of the Content\n        Moderator video media processor on the Azure Media Services\n        platform.Timescale in the Video Moderation output is Ticks/Second.\n\n        :param content_type: The content type.\n        :type content_type: str\n        :param team_name: Your team name.\n        :type team_name: str\n        :param review_id: Id of the review.\n        :type review_id: str\n        :param video_frame_body: Body for add video frames API\n        :type video_frame_body:\n         list[~azure.cognitiveservices.vision.contentmoderator.models.VideoFrameBodyItem]\n        :param timescale: Timescale of the video.\n        :type timescale: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.add_video_frame_url.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'teamName': self._serialize.url('team_name', team_name, 'str'), 'reviewId': self._serialize.url('review_id', review_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if timescale is not None:
            query_parameters['timescale'] = self._serialize.query('timescale', timescale, 'int')
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        body_content = self._serialize.body(video_frame_body, '[VideoFrameBodyItem]')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.APIErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    add_video_frame_url.metadata = {'url': '/contentmoderator/review/v1.0/teams/{teamName}/reviews/{reviewId}/frames'}

    def add_video_frame_stream(self, content_type, team_name, review_id, frame_image_zip, frame_metadata, timescale=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Use this method to add frames for a video review.Timescale: This\n        parameter is a factor which is used to convert the timestamp on a frame\n        into milliseconds. Timescale is provided in the output of the Content\n        Moderator video media processor on the Azure Media Services\n        platform.Timescale in the Video Moderation output is Ticks/Second.\n\n        :param content_type: The content type.\n        :type content_type: str\n        :param team_name: Your team name.\n        :type team_name: str\n        :param review_id: Id of the review.\n        :type review_id: str\n        :param frame_image_zip: Zip file containing frame images.\n        :type frame_image_zip: Generator\n        :param frame_metadata: Metadata of the frame.\n        :type frame_metadata: str\n        :param timescale: Timescale of the video .\n        :type timescale: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`APIErrorException<azure.cognitiveservices.vision.contentmoderator.models.APIErrorException>`\n        '
        url = self.add_video_frame_stream.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'teamName': self._serialize.url('team_name', team_name, 'str'), 'reviewId': self._serialize.url('review_id', review_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if timescale is not None:
            query_parameters['timescale'] = self._serialize.query('timescale', timescale, 'int')
        header_parameters = {}
        header_parameters['Content-Type'] = 'multipart/form-data'
        if custom_headers:
            header_parameters.update(custom_headers)
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        form_data_content = {'frameImageZip': frame_image_zip, 'frameMetadata': frame_metadata}
        request = self._client.post(url, query_parameters, header_parameters, form_content=form_data_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.APIErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    add_video_frame_stream.metadata = {'url': '/contentmoderator/review/v1.0/teams/{teamName}/reviews/{reviewId}/frames'}