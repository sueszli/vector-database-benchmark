from msrest.pipeline import ClientRawResponse
from msrest.exceptions import HttpOperationError
from .. import models

class CustomVisionTrainingClientOperationsMixin(object):

    def get_domains(self, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Get a list of the available domains.\n\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.vision.customvision.training.models.Domain]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_domains.metadata['url']
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
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[Domain]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_domains.metadata = {'url': '/domains'}

    def get_domain(self, domain_id, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Get information about a specific domain.\n\n        :param domain_id: The id of the domain to get information about.\n        :type domain_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Domain or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.Domain or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_domain.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'domainId': self._serialize.url('domain_id', domain_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Domain', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_domain.metadata = {'url': '/domains/{domainId}'}

    def get_projects(self, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Get your projects.\n\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.vision.customvision.training.models.Project]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_projects.metadata['url']
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
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[Project]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_projects.metadata = {'url': '/projects'}

    def create_project(self, name, description=None, domain_id=None, classification_type=None, target_export_platforms=None, export_model_container_uri=None, notification_queue_uri=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Create a project.\n\n        :param name: Name of the project.\n        :type name: str\n        :param description: The description of the project.\n        :type description: str\n        :param domain_id: The id of the domain to use for this project.\n         Defaults to General.\n        :type domain_id: str\n        :param classification_type: The type of classifier to create for this\n         project. Possible values include: \'Multiclass\', \'Multilabel\'\n        :type classification_type: str\n        :param target_export_platforms: List of platforms the trained model is\n         intending exporting to.\n        :type target_export_platforms: list[str]\n        :param export_model_container_uri: The uri to the Azure Storage\n         container that will be used to store exported models.\n        :type export_model_container_uri: str\n        :param notification_queue_uri: The uri to the Azure Storage queue that\n         will be used to send project-related notifications. See <a\n         href="https://go.microsoft.com/fwlink/?linkid=2144149">Storage\n         notifications</a> documentation for setup and message format.\n        :type notification_queue_uri: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Project or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.Project\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        options = None
        if export_model_container_uri is not None or notification_queue_uri is not None:
            options = models.CreateProjectOptions(export_model_container_uri=export_model_container_uri, notification_queue_uri=notification_queue_uri)
        url = self.create_project.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['name'] = self._serialize.query('name', name, 'str')
        if description is not None:
            query_parameters['description'] = self._serialize.query('description', description, 'str')
        if domain_id is not None:
            query_parameters['domainId'] = self._serialize.query('domain_id', domain_id, 'str')
        if classification_type is not None:
            query_parameters['classificationType'] = self._serialize.query('classification_type', classification_type, 'str')
        if target_export_platforms is not None:
            query_parameters['targetExportPlatforms'] = self._serialize.query('target_export_platforms', target_export_platforms, '[str]', div=',')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        if options is not None:
            body_content = self._serialize.body(options, 'CreateProjectOptions')
        else:
            body_content = None
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Project', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    create_project.metadata = {'url': '/projects'}

    def get_project(self, project_id, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Get a specific project.\n\n        :param project_id: The id of the project to get.\n        :type project_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Project or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.Project\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_project.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Project', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_project.metadata = {'url': '/projects/{projectId}'}

    def delete_project(self, project_id, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Delete a specific project.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.delete_project.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    delete_project.metadata = {'url': '/projects/{projectId}'}

    def update_project(self, project_id, updated_project, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Update a specific project.\n\n        :param project_id: The id of the project to update.\n        :type project_id: str\n        :param updated_project: The updated project model.\n        :type updated_project:\n         ~azure.cognitiveservices.vision.customvision.training.models.Project\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Project or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.Project\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.update_project.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(updated_project, 'Project')
        request = self._client.patch(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Project', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    update_project.metadata = {'url': '/projects/{projectId}'}

    def get_artifact(self, project_id, path, custom_headers=None, raw=False, callback=None, **operation_config):
        if False:
            return 10
        'Get artifact content from blob storage, based on artifact relative path\n        in the blob.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param path: The relative path for artifact.\n        :type path: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param callback: When specified, will be called with each chunk of\n         data that is streamed. The callback should take two arguments, the\n         bytes of the current chunk of data and the response object. If the\n         data is uploading, response will be None.\n        :type callback: Callable[Bytes, response=None]\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: object or ClientRawResponse if raw=true\n        :rtype: Generator or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`HttpOperationError<msrest.exceptions.HttpOperationError>`\n        '
        url = self.get_artifact.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['path'] = self._serialize.query('path', path, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/octet-stream'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=True, **operation_config)
        if response.status_code not in [200]:
            raise HttpOperationError(self._deserialize, response)
        deserialized = self._client.stream_download(response, callback)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_artifact.metadata = {'url': '/projects/{projectId}/artifacts'}

    def export_project(self, project_id, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Exports a project.\n\n        :param project_id: The project id of the project to export.\n        :type project_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ProjectExport or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.ProjectExport\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.export_project.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ProjectExport', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    export_project.metadata = {'url': '/projects/{projectId}/export'}

    def get_images(self, project_id, iteration_id=None, tag_ids=None, tagging_status=None, filter=None, order_by=None, take=50, skip=0, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Get images for a given project iteration or workspace.\n\n        This API supports batching and range selection. By default it will only\n        return first 50 images matching images.\n        Use the {take} and {skip} parameters to control how many images to\n        return in a given batch.\n        The filtering is on an and/or relationship. For example, if the\n        provided tag ids are for the "Dog" and\n        "Cat" tags, then only images tagged with Dog and/or Cat will be\n        returned.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param iteration_id: The iteration id. Defaults to workspace.\n        :type iteration_id: str\n        :param tag_ids: A list of tags ids to filter the images. Defaults to\n         all tagged images when null. Limited to 20.\n        :type tag_ids: list[str]\n        :param tagging_status: The tagging status filter. It can be \'All\',\n         \'Tagged\', or \'Untagged\'. Defaults to \'All\'. Possible values include:\n         \'All\', \'Tagged\', \'Untagged\'\n        :type tagging_status: str\n        :param filter: An expression to filter the images against image\n         metadata. Only images where the expression evaluates to true are\n         included in the response.\n         The expression supports eq (Equal), ne (Not equal), and (Logical and),\n         or (Logical or) operators.\n         Here is an example, metadata=key1 eq \'value1\' and key2 ne \'value2\'.\n        :type filter: str\n        :param order_by: The ordering. Defaults to newest. Possible values\n         include: \'Newest\', \'Oldest\'\n        :type order_by: str\n        :param take: Maximum number of images to return. Defaults to 50,\n         limited to 256.\n        :type take: int\n        :param skip: Number of images to skip before beginning the image\n         batch. Defaults to 0.\n        :type skip: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.vision.customvision.training.models.Image]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_images.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if iteration_id is not None:
            query_parameters['iterationId'] = self._serialize.query('iteration_id', iteration_id, 'str')
        if tag_ids is not None:
            query_parameters['tagIds'] = self._serialize.query('tag_ids', tag_ids, '[str]', div=',', max_items=20, min_items=0)
        if tagging_status is not None:
            query_parameters['taggingStatus'] = self._serialize.query('tagging_status', tagging_status, 'str')
        if filter is not None:
            query_parameters['$filter'] = self._serialize.query('filter', filter, 'str')
        if order_by is not None:
            query_parameters['orderBy'] = self._serialize.query('order_by', order_by, 'str')
        if take is not None:
            query_parameters['take'] = self._serialize.query('take', take, 'int', maximum=256, minimum=0)
        if skip is not None:
            query_parameters['skip'] = self._serialize.query('skip', skip, 'int')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[Image]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_images.metadata = {'url': '/projects/{projectId}/images'}

    def create_images_from_data(self, project_id, image_data, tag_ids=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Add the provided images to the set of training images.\n\n        This API accepts body content as multipart/form-data and\n        application/octet-stream. When using multipart\n        multiple image files can be sent at once, with a maximum of 64 files.\n        If all images are successful created, 200(OK) status code will be\n        returned.\n        Otherwise, 207 (Multi-Status) status code will be returned and detail\n        status for each image will be listed in the response payload.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param image_data: Binary image data. Supported formats are JPEG, GIF,\n         PNG, and BMP. Supports images up to 6MB.\n        :type image_data: Generator\n        :param tag_ids: The tags ids with which to tag each image. Limited to\n         20.\n        :type tag_ids: list[str]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImageCreateSummary or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.ImageCreateSummary\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.create_images_from_data.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if tag_ids is not None:
            query_parameters['tagIds'] = self._serialize.query('tag_ids', tag_ids, '[str]', div=',', max_items=20, min_items=0)
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'multipart/form-data'
        if custom_headers:
            header_parameters.update(custom_headers)
        form_data_content = {'imageData': image_data}
        request = self._client.post(url, query_parameters, header_parameters, form_content=form_data_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200, 207]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImageCreateSummary', response)
        if response.status_code == 207:
            deserialized = self._deserialize('ImageCreateSummary', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    create_images_from_data.metadata = {'url': '/projects/{projectId}/images'}

    def delete_images(self, project_id, image_ids=None, all_images=None, all_iterations=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Delete images from the set of training images.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param image_ids: Ids of the images to be deleted. Limited to 256\n         images per batch.\n        :type image_ids: list[str]\n        :param all_images: Flag to specify delete all images, specify this\n         flag or a list of images. Using this flag will return a 202 response\n         to indicate the images are being deleted.\n        :type all_images: bool\n        :param all_iterations: Removes these images from all iterations, not\n         just the current workspace. Using this flag will return a 202 response\n         to indicate the images are being deleted.\n        :type all_iterations: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.delete_images.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if image_ids is not None:
            query_parameters['imageIds'] = self._serialize.query('image_ids', image_ids, '[str]', div=',', max_items=256, min_items=0)
        if all_images is not None:
            query_parameters['allImages'] = self._serialize.query('all_images', all_images, 'bool')
        if all_iterations is not None:
            query_parameters['allIterations'] = self._serialize.query('all_iterations', all_iterations, 'bool')
        header_parameters = {}
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [202, 204]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    delete_images.metadata = {'url': '/projects/{projectId}/images'}

    def get_image_region_proposals(self, project_id, image_id, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Get region proposals for an image. Returns empty array if no proposals\n        are found.\n\n        This API will get region proposals for an image along with confidences\n        for the region. It returns an empty array if no proposals are found.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param image_id: The image id.\n        :type image_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImageRegionProposal or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.ImageRegionProposal\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_image_region_proposals.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'imageId': self._serialize.url('image_id', image_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.post(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImageRegionProposal', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_image_region_proposals.metadata = {'url': '/projects/{projectId}/images/{imageId}/regionproposals'}

    def get_image_count(self, project_id, iteration_id=None, tagging_status=None, filter=None, tag_ids=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Get the number of images.\n\n        The filtering is on an and/or relationship. For example, if the\n        provided tag ids are for the "Dog" and\n        "Cat" tags, then only images tagged with Dog and/or Cat will be\n        returned.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param iteration_id: The iteration id. Defaults to workspace.\n        :type iteration_id: str\n        :param tagging_status: The tagging status filter. It can be \'All\',\n         \'Tagged\', or \'Untagged\'. Defaults to \'All\'. Possible values include:\n         \'All\', \'Tagged\', \'Untagged\'\n        :type tagging_status: str\n        :param filter: An expression to filter the images against image\n         metadata. Only images where the expression evaluates to true are\n         included in the response.\n         The expression supports eq (Equal), ne (Not equal), and (Logical and),\n         or (Logical or) operators.\n         Here is an example, metadata=key1 eq \'value1\' and key2 ne \'value2\'.\n        :type filter: str\n        :param tag_ids: A list of tags ids to filter the images to count.\n         Defaults to all tags when null.\n        :type tag_ids: list[str]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: int or ClientRawResponse if raw=true\n        :rtype: int or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_image_count.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if iteration_id is not None:
            query_parameters['iterationId'] = self._serialize.query('iteration_id', iteration_id, 'str')
        if tagging_status is not None:
            query_parameters['taggingStatus'] = self._serialize.query('tagging_status', tagging_status, 'str')
        if filter is not None:
            query_parameters['$filter'] = self._serialize.query('filter', filter, 'str')
        if tag_ids is not None:
            query_parameters['tagIds'] = self._serialize.query('tag_ids', tag_ids, '[str]', div=',')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('int', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_image_count.metadata = {'url': '/projects/{projectId}/images/count'}

    def create_images_from_files(self, project_id, batch, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Add the provided batch of images to the set of training images.\n\n        This API accepts a batch of files, and optionally tags, to create\n        images. There is a limit of 64 images and 20 tags.\n        If all images are successful created, 200(OK) status code will be\n        returned.\n        Otherwise, 207 (Multi-Status) status code will be returned and detail\n        status for each image will be listed in the response payload.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param batch: The batch of image files to add. Limited to 64 images\n         and 20 tags per batch.\n        :type batch:\n         ~azure.cognitiveservices.vision.customvision.training.models.ImageFileCreateBatch\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImageCreateSummary or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.ImageCreateSummary\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.create_images_from_files.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(batch, 'ImageFileCreateBatch')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200, 207]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImageCreateSummary', response)
        if response.status_code == 207:
            deserialized = self._deserialize('ImageCreateSummary', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    create_images_from_files.metadata = {'url': '/projects/{projectId}/images/files'}

    def get_images_by_ids(self, project_id, image_ids=None, iteration_id=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Get images by id for a given project iteration.\n\n        This API will return a set of Images for the specified tags and\n        optionally iteration. If no iteration is specified the\n        current workspace is used.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param image_ids: The list of image ids to retrieve. Limited to 256.\n        :type image_ids: list[str]\n        :param iteration_id: The iteration id. Defaults to workspace.\n        :type iteration_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.vision.customvision.training.models.Image]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_images_by_ids.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if image_ids is not None:
            query_parameters['imageIds'] = self._serialize.query('image_ids', image_ids, '[str]', div=',', max_items=256, min_items=0)
        if iteration_id is not None:
            query_parameters['iterationId'] = self._serialize.query('iteration_id', iteration_id, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[Image]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_images_by_ids.metadata = {'url': '/projects/{projectId}/images/id'}

    def update_image_metadata(self, project_id, image_ids, metadata, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Update metadata of images.\n\n        This API accepts a batch of image Ids, and metadata, to update images.\n        There is a limit of 64 images.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param image_ids: The list of image ids to update. Limited to 64.\n        :type image_ids: list[str]\n        :param metadata: The metadata to be updated to the specified images.\n         Limited to 10 key-value pairs per image. The length of key is limited\n         to 128. The length of value is limited to 256.\n        :type metadata: dict[str, str]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImageMetadataUpdateSummary or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.ImageMetadataUpdateSummary\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.update_image_metadata.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['imageIds'] = self._serialize.query('image_ids', image_ids, '[str]', div=',', max_items=256, min_items=0)
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(metadata, '{str}')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200, 207]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImageMetadataUpdateSummary', response)
        if response.status_code == 207:
            deserialized = self._deserialize('ImageMetadataUpdateSummary', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    update_image_metadata.metadata = {'url': '/projects/{projectId}/images/metadata'}

    def create_images_from_predictions(self, project_id, batch, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Add the specified predicted images to the set of training images.\n\n        This API creates a batch of images from predicted images specified.\n        There is a limit of 64 images and 20 tags.\n        If all images are successful created, 200(OK) status code will be\n        returned.\n        Otherwise, 207 (Multi-Status) status code will be returned and detail\n        status for each image will be listed in the response payload.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param batch: Image, tag ids, and metadata. Limited to 64 images and\n         20 tags per batch.\n        :type batch:\n         ~azure.cognitiveservices.vision.customvision.training.models.ImageIdCreateBatch\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImageCreateSummary or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.ImageCreateSummary\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.create_images_from_predictions.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(batch, 'ImageIdCreateBatch')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200, 207]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImageCreateSummary', response)
        if response.status_code == 207:
            deserialized = self._deserialize('ImageCreateSummary', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    create_images_from_predictions.metadata = {'url': '/projects/{projectId}/images/predictions'}

    def create_image_regions(self, project_id, regions=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Create a set of image regions.\n\n        This API accepts a batch of image regions, and optionally tags, to\n        update existing images with region information.\n        There is a limit of 64 entries in the batch.\n        If all regions are successful created, 200(OK) status code will be\n        returned.\n        Otherwise, 207 (Multi-Status) status code will be returned and detail\n        status for each region will be listed in the response payload.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param regions:\n        :type regions:\n         list[~azure.cognitiveservices.vision.customvision.training.models.ImageRegionCreateEntry]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImageRegionCreateSummary or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.ImageRegionCreateSummary\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        batch = models.ImageRegionCreateBatch(regions=regions)
        url = self.create_image_regions.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(batch, 'ImageRegionCreateBatch')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200, 207]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImageRegionCreateSummary', response)
        if response.status_code == 207:
            deserialized = self._deserialize('ImageRegionCreateSummary', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    create_image_regions.metadata = {'url': '/projects/{projectId}/images/regions'}

    def delete_image_regions(self, project_id, region_ids, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Delete a set of image regions.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param region_ids: Regions to delete. Limited to 64.\n        :type region_ids: list[str]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.delete_image_regions.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['regionIds'] = self._serialize.query('region_ids', region_ids, '[str]', div=',', max_items=64, min_items=0)
        header_parameters = {}
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    delete_image_regions.metadata = {'url': '/projects/{projectId}/images/regions'}

    def query_suggested_images(self, project_id, iteration_id, query, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Get untagged images whose suggested tags match given tags. Returns\n        empty array if no images are found.\n\n        This API will fetch untagged images filtered by suggested tags Ids. It\n        returns an empty array if no images are found.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param iteration_id: IterationId to use for the suggested tags and\n         regions.\n        :type iteration_id: str\n        :param query: Contains properties we need to query suggested images.\n        :type query:\n         ~azure.cognitiveservices.vision.customvision.training.models.SuggestedTagAndRegionQueryToken\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: SuggestedTagAndRegionQuery or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.SuggestedTagAndRegionQuery\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.query_suggested_images.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['iterationId'] = self._serialize.query('iteration_id', iteration_id, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(query, 'SuggestedTagAndRegionQueryToken')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('SuggestedTagAndRegionQuery', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    query_suggested_images.metadata = {'url': '/projects/{projectId}/images/suggested'}

    def query_suggested_image_count(self, project_id, iteration_id, tag_ids=None, threshold=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Get count of images whose suggested tags match given tags and their\n        probabilities are greater than or equal to the given threshold. Returns\n        count as 0 if none found.\n\n        This API takes in tagIds to get count of untagged images per suggested\n        tags for a given threshold.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param iteration_id: IterationId to use for the suggested tags and\n         regions.\n        :type iteration_id: str\n        :param tag_ids: Existing TagIds in project to get suggested tags count\n         for.\n        :type tag_ids: list[str]\n        :param threshold: Confidence threshold to filter suggested tags on.\n        :type threshold: float\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: dict or ClientRawResponse if raw=true\n        :rtype: dict[str, int] or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        query = models.TagFilter(tag_ids=tag_ids, threshold=threshold)
        url = self.query_suggested_image_count.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['iterationId'] = self._serialize.query('iteration_id', iteration_id, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(query, 'TagFilter')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('{int}', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    query_suggested_image_count.metadata = {'url': '/projects/{projectId}/images/suggested/count'}

    def get_tagged_images(self, project_id, iteration_id=None, tag_ids=None, order_by=None, take=50, skip=0, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Get tagged images for a given project iteration.\n\n        This API supports batching and range selection. By default it will only\n        return first 50 images matching images.\n        Use the {take} and {skip} parameters to control how many images to\n        return in a given batch.\n        The filtering is on an and/or relationship. For example, if the\n        provided tag ids are for the "Dog" and\n        "Cat" tags, then only images tagged with Dog and/or Cat will be\n        returned.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param iteration_id: The iteration id. Defaults to workspace.\n        :type iteration_id: str\n        :param tag_ids: A list of tags ids to filter the images. Defaults to\n         all tagged images when null. Limited to 20.\n        :type tag_ids: list[str]\n        :param order_by: The ordering. Defaults to newest. Possible values\n         include: \'Newest\', \'Oldest\'\n        :type order_by: str\n        :param take: Maximum number of images to return. Defaults to 50,\n         limited to 256.\n        :type take: int\n        :param skip: Number of images to skip before beginning the image\n         batch. Defaults to 0.\n        :type skip: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.vision.customvision.training.models.Image]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_tagged_images.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if iteration_id is not None:
            query_parameters['iterationId'] = self._serialize.query('iteration_id', iteration_id, 'str')
        if tag_ids is not None:
            query_parameters['tagIds'] = self._serialize.query('tag_ids', tag_ids, '[str]', div=',', max_items=20, min_items=0)
        if order_by is not None:
            query_parameters['orderBy'] = self._serialize.query('order_by', order_by, 'str')
        if take is not None:
            query_parameters['take'] = self._serialize.query('take', take, 'int', maximum=256, minimum=0)
        if skip is not None:
            query_parameters['skip'] = self._serialize.query('skip', skip, 'int')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[Image]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_tagged_images.metadata = {'url': '/projects/{projectId}/images/tagged'}

    def get_tagged_image_count(self, project_id, iteration_id=None, tag_ids=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Gets the number of images tagged with the provided {tagIds}.\n\n        The filtering is on an and/or relationship. For example, if the\n        provided tag ids are for the "Dog" and\n        "Cat" tags, then only images tagged with Dog and/or Cat will be\n        returned.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param iteration_id: The iteration id. Defaults to workspace.\n        :type iteration_id: str\n        :param tag_ids: A list of tags ids to filter the images to count.\n         Defaults to all tags when null.\n        :type tag_ids: list[str]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: int or ClientRawResponse if raw=true\n        :rtype: int or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_tagged_image_count.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if iteration_id is not None:
            query_parameters['iterationId'] = self._serialize.query('iteration_id', iteration_id, 'str')
        if tag_ids is not None:
            query_parameters['tagIds'] = self._serialize.query('tag_ids', tag_ids, '[str]', div=',')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('int', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_tagged_image_count.metadata = {'url': '/projects/{projectId}/images/tagged/count'}

    def create_image_tags(self, project_id, tags=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Associate a set of images with a set of tags.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param tags: Image Tag entries to include in this batch.\n        :type tags:\n         list[~azure.cognitiveservices.vision.customvision.training.models.ImageTagCreateEntry]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImageTagCreateSummary or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.ImageTagCreateSummary\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        batch = models.ImageTagCreateBatch(tags=tags)
        url = self.create_image_tags.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(batch, 'ImageTagCreateBatch')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImageTagCreateSummary', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    create_image_tags.metadata = {'url': '/projects/{projectId}/images/tags'}

    def delete_image_tags(self, project_id, image_ids, tag_ids, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Remove a set of tags from a set of images.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param image_ids: Image ids. Limited to 64 images.\n        :type image_ids: list[str]\n        :param tag_ids: Tags to be deleted from the specified images. Limited\n         to 20 tags.\n        :type tag_ids: list[str]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.delete_image_tags.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['imageIds'] = self._serialize.query('image_ids', image_ids, '[str]', div=',', max_items=64, min_items=0)
        query_parameters['tagIds'] = self._serialize.query('tag_ids', tag_ids, '[str]', div=',', max_items=20, min_items=0)
        header_parameters = {}
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    delete_image_tags.metadata = {'url': '/projects/{projectId}/images/tags'}

    def get_untagged_images(self, project_id, iteration_id=None, order_by=None, take=50, skip=0, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        "Get untagged images for a given project iteration.\n\n        This API supports batching and range selection. By default it will only\n        return first 50 images matching images.\n        Use the {take} and {skip} parameters to control how many images to\n        return in a given batch.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param iteration_id: The iteration id. Defaults to workspace.\n        :type iteration_id: str\n        :param order_by: The ordering. Defaults to newest. Possible values\n         include: 'Newest', 'Oldest'\n        :type order_by: str\n        :param take: Maximum number of images to return. Defaults to 50,\n         limited to 256.\n        :type take: int\n        :param skip: Number of images to skip before beginning the image\n         batch. Defaults to 0.\n        :type skip: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.vision.customvision.training.models.Image]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        "
        url = self.get_untagged_images.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if iteration_id is not None:
            query_parameters['iterationId'] = self._serialize.query('iteration_id', iteration_id, 'str')
        if order_by is not None:
            query_parameters['orderBy'] = self._serialize.query('order_by', order_by, 'str')
        if take is not None:
            query_parameters['take'] = self._serialize.query('take', take, 'int', maximum=256, minimum=0)
        if skip is not None:
            query_parameters['skip'] = self._serialize.query('skip', skip, 'int')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[Image]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_untagged_images.metadata = {'url': '/projects/{projectId}/images/untagged'}

    def get_untagged_image_count(self, project_id, iteration_id=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Gets the number of untagged images.\n\n        This API returns the images which have no tags for a given project and\n        optionally an iteration. If no iteration is specified the\n        current workspace is used.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param iteration_id: The iteration id. Defaults to workspace.\n        :type iteration_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: int or ClientRawResponse if raw=true\n        :rtype: int or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_untagged_image_count.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if iteration_id is not None:
            query_parameters['iterationId'] = self._serialize.query('iteration_id', iteration_id, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('int', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_untagged_image_count.metadata = {'url': '/projects/{projectId}/images/untagged/count'}

    def create_images_from_urls(self, project_id, batch, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Add the provided images urls to the set of training images.\n\n        This API accepts a batch of urls, and optionally tags, to create\n        images. There is a limit of 64 images and 20 tags.\n        If all images are successful created, 200(OK) status code will be\n        returned.\n        Otherwise, 207 (Multi-Status) status code will be returned and detail\n        status for each image will be listed in the response payload.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param batch: Image urls, tag ids, and metadata. Limited to 64 images\n         and 20 tags per batch.\n        :type batch:\n         ~azure.cognitiveservices.vision.customvision.training.models.ImageUrlCreateBatch\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImageCreateSummary or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.ImageCreateSummary\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.create_images_from_urls.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(batch, 'ImageUrlCreateBatch')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200, 207]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImageCreateSummary', response)
        if response.status_code == 207:
            deserialized = self._deserialize('ImageCreateSummary', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    create_images_from_urls.metadata = {'url': '/projects/{projectId}/images/urls'}

    def get_iterations(self, project_id, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Get iterations for the project.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.vision.customvision.training.models.Iteration]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_iterations.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[Iteration]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_iterations.metadata = {'url': '/projects/{projectId}/iterations'}

    def get_iteration(self, project_id, iteration_id, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Get a specific iteration.\n\n        :param project_id: The id of the project the iteration belongs to.\n        :type project_id: str\n        :param iteration_id: The id of the iteration to get.\n        :type iteration_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Iteration or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.Iteration\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_iteration.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'iterationId': self._serialize.url('iteration_id', iteration_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Iteration', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_iteration.metadata = {'url': '/projects/{projectId}/iterations/{iterationId}'}

    def delete_iteration(self, project_id, iteration_id, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Delete a specific iteration of a project.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param iteration_id: The iteration id.\n        :type iteration_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.delete_iteration.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'iterationId': self._serialize.url('iteration_id', iteration_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    delete_iteration.metadata = {'url': '/projects/{projectId}/iterations/{iterationId}'}

    def update_iteration(self, project_id, iteration_id, name, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Update a specific iteration.\n\n        :param project_id: Project id.\n        :type project_id: str\n        :param iteration_id: Iteration id.\n        :type iteration_id: str\n        :param name: Gets or sets the name of the iteration.\n        :type name: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Iteration or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.Iteration\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        updated_iteration = models.Iteration(name=name)
        url = self.update_iteration.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'iterationId': self._serialize.url('iteration_id', iteration_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(updated_iteration, 'Iteration')
        request = self._client.patch(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Iteration', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    update_iteration.metadata = {'url': '/projects/{projectId}/iterations/{iterationId}'}

    def get_exports(self, project_id, iteration_id, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Get the list of exports for a specific iteration.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param iteration_id: The iteration id.\n        :type iteration_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.vision.customvision.training.models.Export]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_exports.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'iterationId': self._serialize.url('iteration_id', iteration_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[Export]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_exports.metadata = {'url': '/projects/{projectId}/iterations/{iterationId}/export'}

    def export_iteration(self, project_id, iteration_id, platform, flavor=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        "Export a trained iteration.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param iteration_id: The iteration id.\n        :type iteration_id: str\n        :param platform: The target platform. Possible values include:\n         'CoreML', 'TensorFlow', 'DockerFile', 'ONNX', 'VAIDK', 'OpenVino'\n        :type platform: str\n        :param flavor: The flavor of the target platform. Possible values\n         include: 'Linux', 'Windows', 'ONNX10', 'ONNX12', 'ARM',\n         'TensorFlowNormal', 'TensorFlowLite'\n        :type flavor: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Export or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.Export or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        "
        url = self.export_iteration.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'iterationId': self._serialize.url('iteration_id', iteration_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['platform'] = self._serialize.query('platform', platform, 'str')
        if flavor is not None:
            query_parameters['flavor'] = self._serialize.query('flavor', flavor, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.post(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Export', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    export_iteration.metadata = {'url': '/projects/{projectId}/iterations/{iterationId}/export'}

    def get_iteration_performance(self, project_id, iteration_id, threshold=None, overlap_threshold=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Get detailed performance information about an iteration.\n\n        :param project_id: The id of the project the iteration belongs to.\n        :type project_id: str\n        :param iteration_id: The id of the iteration to get.\n        :type iteration_id: str\n        :param threshold: The threshold used to determine true predictions.\n        :type threshold: float\n        :param overlap_threshold: If applicable, the bounding box overlap\n         threshold used to determine true predictions.\n        :type overlap_threshold: float\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: IterationPerformance or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.IterationPerformance\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_iteration_performance.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'iterationId': self._serialize.url('iteration_id', iteration_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if threshold is not None:
            query_parameters['threshold'] = self._serialize.query('threshold', threshold, 'float')
        if overlap_threshold is not None:
            query_parameters['overlapThreshold'] = self._serialize.query('overlap_threshold', overlap_threshold, 'float')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('IterationPerformance', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_iteration_performance.metadata = {'url': '/projects/{projectId}/iterations/{iterationId}/performance'}

    def get_image_performances(self, project_id, iteration_id, tag_ids=None, order_by=None, take=50, skip=0, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Get image with its prediction for a given project iteration.\n\n        This API supports batching and range selection. By default it will only\n        return first 50 images matching images.\n        Use the {take} and {skip} parameters to control how many images to\n        return in a given batch.\n        The filtering is on an and/or relationship. For example, if the\n        provided tag ids are for the "Dog" and\n        "Cat" tags, then only images tagged with Dog and/or Cat will be\n        returned.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param iteration_id: The iteration id. Defaults to workspace.\n        :type iteration_id: str\n        :param tag_ids: A list of tags ids to filter the images. Defaults to\n         all tagged images when null. Limited to 20.\n        :type tag_ids: list[str]\n        :param order_by: The ordering. Defaults to newest. Possible values\n         include: \'Newest\', \'Oldest\'\n        :type order_by: str\n        :param take: Maximum number of images to return. Defaults to 50,\n         limited to 256.\n        :type take: int\n        :param skip: Number of images to skip before beginning the image\n         batch. Defaults to 0.\n        :type skip: int\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.vision.customvision.training.models.ImagePerformance]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_image_performances.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'iterationId': self._serialize.url('iteration_id', iteration_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if tag_ids is not None:
            query_parameters['tagIds'] = self._serialize.query('tag_ids', tag_ids, '[str]', div=',', max_items=20, min_items=0)
        if order_by is not None:
            query_parameters['orderBy'] = self._serialize.query('order_by', order_by, 'str')
        if take is not None:
            query_parameters['take'] = self._serialize.query('take', take, 'int', maximum=256, minimum=0)
        if skip is not None:
            query_parameters['skip'] = self._serialize.query('skip', skip, 'int')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[ImagePerformance]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_image_performances.metadata = {'url': '/projects/{projectId}/iterations/{iterationId}/performance/images'}

    def get_image_performance_count(self, project_id, iteration_id, tag_ids=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Gets the number of images tagged with the provided {tagIds} that have\n        prediction results from\n        training for the provided iteration {iterationId}.\n\n        The filtering is on an and/or relationship. For example, if the\n        provided tag ids are for the "Dog" and\n        "Cat" tags, then only images tagged with Dog and/or Cat will be\n        returned.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param iteration_id: The iteration id. Defaults to workspace.\n        :type iteration_id: str\n        :param tag_ids: A list of tags ids to filter the images to count.\n         Defaults to all tags when null.\n        :type tag_ids: list[str]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: int or ClientRawResponse if raw=true\n        :rtype: int or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_image_performance_count.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'iterationId': self._serialize.url('iteration_id', iteration_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if tag_ids is not None:
            query_parameters['tagIds'] = self._serialize.query('tag_ids', tag_ids, '[str]', div=',')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('int', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_image_performance_count.metadata = {'url': '/projects/{projectId}/iterations/{iterationId}/performance/images/count'}

    def publish_iteration(self, project_id, iteration_id, publish_name, prediction_id, overwrite=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        'Publish a specific iteration.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param iteration_id: The iteration id.\n        :type iteration_id: str\n        :param publish_name: The name to give the published iteration.\n        :type publish_name: str\n        :param prediction_id: The id of the prediction resource to publish to.\n        :type prediction_id: str\n        :param overwrite: Whether to overwrite the published model with the\n         given name (default: false).\n        :type overwrite: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: bool or ClientRawResponse if raw=true\n        :rtype: bool or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.publish_iteration.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'iterationId': self._serialize.url('iteration_id', iteration_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['publishName'] = self._serialize.query('publish_name', publish_name, 'str')
        query_parameters['predictionId'] = self._serialize.query('prediction_id', prediction_id, 'str')
        if overwrite is not None:
            query_parameters['overwrite'] = self._serialize.query('overwrite', overwrite, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.post(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('bool', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    publish_iteration.metadata = {'url': '/projects/{projectId}/iterations/{iterationId}/publish'}

    def unpublish_iteration(self, project_id, iteration_id, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Unpublish a specific iteration.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param iteration_id: The iteration id.\n        :type iteration_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.unpublish_iteration.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'iterationId': self._serialize.url('iteration_id', iteration_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    unpublish_iteration.metadata = {'url': '/projects/{projectId}/iterations/{iterationId}/publish'}

    def delete_prediction(self, project_id, ids, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Delete a set of predicted images and their associated prediction\n        results.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param ids: The prediction ids. Limited to 64.\n        :type ids: list[str]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.delete_prediction.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['ids'] = self._serialize.query('ids', ids, '[str]', div=',', max_items=64, min_items=0)
        header_parameters = {}
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    delete_prediction.metadata = {'url': '/projects/{projectId}/predictions'}

    def query_predictions(self, project_id, query, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Get images that were sent to your prediction endpoint.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param query: Parameters used to query the predictions. Limited to\n         combining 2 tags.\n        :type query:\n         ~azure.cognitiveservices.vision.customvision.training.models.PredictionQueryToken\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: PredictionQueryResult or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.PredictionQueryResult\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.query_predictions.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(query, 'PredictionQueryToken')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('PredictionQueryResult', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    query_predictions.metadata = {'url': '/projects/{projectId}/predictions/query'}

    def quick_test_image(self, project_id, image_data, iteration_id=None, store=True, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Quick test an image.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param image_data: Binary image data. Supported formats are JPEG, GIF,\n         PNG, and BMP. Supports images up to 6MB.\n        :type image_data: Generator\n        :param iteration_id: Optional. Specifies the id of a particular\n         iteration to evaluate against.\n         The default iteration for the project will be used when not specified.\n        :type iteration_id: str\n        :param store: Optional. Specifies whether or not to store the result\n         of this prediction. The default is true, to store.\n        :type store: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImagePrediction or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.ImagePrediction\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.quick_test_image.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if iteration_id is not None:
            query_parameters['iterationId'] = self._serialize.query('iteration_id', iteration_id, 'str')
        if store is not None:
            query_parameters['store'] = self._serialize.query('store', store, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'multipart/form-data'
        if custom_headers:
            header_parameters.update(custom_headers)
        form_data_content = {'imageData': image_data}
        request = self._client.post(url, query_parameters, header_parameters, form_content=form_data_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImagePrediction', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    quick_test_image.metadata = {'url': '/projects/{projectId}/quicktest/image'}

    def quick_test_image_url(self, project_id, url, iteration_id=None, store=True, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Quick test an image url.\n\n        :param project_id: The project to evaluate against.\n        :type project_id: str\n        :param url: Url of the image.\n        :type url: str\n        :param iteration_id: Optional. Specifies the id of a particular\n         iteration to evaluate against.\n         The default iteration for the project will be used when not specified.\n        :type iteration_id: str\n        :param store: Optional. Specifies whether or not to store the result\n         of this prediction. The default is true, to store.\n        :type store: bool\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: ImagePrediction or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.ImagePrediction\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        image_url = models.ImageUrl(url=url)
        url = self.quick_test_image_url.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if iteration_id is not None:
            query_parameters['iterationId'] = self._serialize.query('iteration_id', iteration_id, 'str')
        if store is not None:
            query_parameters['store'] = self._serialize.query('store', store, 'bool')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(image_url, 'ImageUrl')
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('ImagePrediction', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    quick_test_image_url.metadata = {'url': '/projects/{projectId}/quicktest/url'}

    def get_tags(self, project_id, iteration_id=None, custom_headers=None, raw=False, **operation_config):
        if False:
            return 10
        'Get the tags for a given project and iteration.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param iteration_id: The iteration id. Defaults to workspace.\n        :type iteration_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.vision.customvision.training.models.Tag]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_tags.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if iteration_id is not None:
            query_parameters['iterationId'] = self._serialize.query('iteration_id', iteration_id, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[Tag]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_tags.metadata = {'url': '/projects/{projectId}/tags'}

    def create_tag(self, project_id, name, description=None, type=None, custom_headers=None, raw=False, **operation_config):
        if False:
            i = 10
            return i + 15
        "Create a tag for the project.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param name: The tag name.\n        :type name: str\n        :param description: Optional description for the tag.\n        :type description: str\n        :param type: Optional type for the tag. Possible values include:\n         'Regular', 'Negative', 'GeneralProduct'\n        :type type: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Tag or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.Tag or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        "
        url = self.create_tag.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['name'] = self._serialize.query('name', name, 'str')
        if description is not None:
            query_parameters['description'] = self._serialize.query('description', description, 'str')
        if type is not None:
            query_parameters['type'] = self._serialize.query('type', type, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.post(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Tag', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    create_tag.metadata = {'url': '/projects/{projectId}/tags'}

    def get_tag(self, project_id, tag_id, iteration_id=None, custom_headers=None, raw=False, **operation_config):
        if False:
            while True:
                i = 10
        'Get information about a specific tag.\n\n        :param project_id: The project this tag belongs to.\n        :type project_id: str\n        :param tag_id: The tag id.\n        :type tag_id: str\n        :param iteration_id: The iteration to retrieve this tag from.\n         Optional, defaults to current training set.\n        :type iteration_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Tag or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.Tag or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.get_tag.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'tagId': self._serialize.url('tag_id', tag_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if iteration_id is not None:
            query_parameters['iterationId'] = self._serialize.query('iteration_id', iteration_id, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.get(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Tag', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    get_tag.metadata = {'url': '/projects/{projectId}/tags/{tagId}'}

    def delete_tag(self, project_id, tag_id, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Delete a tag from the project.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param tag_id: Id of the tag to be deleted.\n        :type tag_id: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: None or ClientRawResponse if raw=true\n        :rtype: None or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.delete_tag.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'tagId': self._serialize.url('tag_id', tag_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.delete(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [204]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        if raw:
            client_raw_response = ClientRawResponse(None, response)
            return client_raw_response
    delete_tag.metadata = {'url': '/projects/{projectId}/tags/{tagId}'}

    def update_tag(self, project_id, tag_id, updated_tag, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Update a tag.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param tag_id: The id of the target tag.\n        :type tag_id: str\n        :param updated_tag: The updated tag model.\n        :type updated_tag:\n         ~azure.cognitiveservices.vision.customvision.training.models.Tag\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Tag or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.Tag or\n         ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.update_tag.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str'), 'tagId': self._serialize.url('tag_id', tag_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        body_content = self._serialize.body(updated_tag, 'Tag')
        request = self._client.patch(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Tag', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    update_tag.metadata = {'url': '/projects/{projectId}/tags/{tagId}'}

    def suggest_tags_and_regions(self, project_id, iteration_id, image_ids, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        'Suggest tags and regions for an array/batch of untagged images. Returns\n        empty array if no tags are found.\n\n        This API will get suggested tags and regions for an array/batch of\n        untagged images along with confidences for the tags. It returns an\n        empty array if no tags are found.\n        There is a limit of 64 images in the batch.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param iteration_id: IterationId to use for tag and region suggestion.\n        :type iteration_id: str\n        :param image_ids: Array of image ids tag suggestion are needed for.\n         Use GetUntaggedImages API to get imageIds.\n        :type image_ids: list[str]\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: list or ClientRawResponse if raw=true\n        :rtype:\n         list[~azure.cognitiveservices.vision.customvision.training.models.SuggestedTagAndRegion]\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.suggest_tags_and_regions.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['iterationId'] = self._serialize.query('iteration_id', iteration_id, 'str')
        query_parameters['imageIds'] = self._serialize.query('image_ids', image_ids, '[str]', div=',', max_items=64, min_items=0)
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.post(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('[SuggestedTagAndRegion]', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    suggest_tags_and_regions.metadata = {'url': '/projects/{projectId}/tagsandregions/suggestions'}

    def train_project(self, project_id, training_type=None, reserved_budget_in_hours=0, force_train=False, notification_email_address=None, selected_tags=None, custom_base_model_info=None, custom_headers=None, raw=False, **operation_config):
        if False:
            for i in range(10):
                print('nop')
        "Queues project for training.\n\n        :param project_id: The project id.\n        :type project_id: str\n        :param training_type: The type of training to use to train the project\n         (default: Regular). Possible values include: 'Regular', 'Advanced'\n        :type training_type: str\n        :param reserved_budget_in_hours: The number of hours reserved as\n         budget for training (if applicable).\n        :type reserved_budget_in_hours: int\n        :param force_train: Whether to force train even if dataset and\n         configuration does not change (default: false).\n        :type force_train: bool\n        :param notification_email_address: The email address to send\n         notification to when training finishes (default: null).\n        :type notification_email_address: str\n        :param selected_tags: List of tags selected for this training session,\n         other tags in the project will be ignored.\n        :type selected_tags: list[str]\n        :param custom_base_model_info: Information of the previously trained\n         iteration which provides the base model for current iteration's\n         training.\n        :type custom_base_model_info:\n         ~azure.cognitiveservices.vision.customvision.training.models.CustomBaseModelInfo\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Iteration or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.Iteration\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        "
        training_parameters = None
        if selected_tags is not None or custom_base_model_info is not None:
            training_parameters = models.TrainingParameters(selected_tags=selected_tags, custom_base_model_info=custom_base_model_info)
        url = self.train_project.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True), 'projectId': self._serialize.url('project_id', project_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        if training_type is not None:
            query_parameters['trainingType'] = self._serialize.query('training_type', training_type, 'str')
        if reserved_budget_in_hours is not None:
            query_parameters['reservedBudgetInHours'] = self._serialize.query('reserved_budget_in_hours', reserved_budget_in_hours, 'int')
        if force_train is not None:
            query_parameters['forceTrain'] = self._serialize.query('force_train', force_train, 'bool')
        if notification_email_address is not None:
            query_parameters['notificationEmailAddress'] = self._serialize.query('notification_email_address', notification_email_address, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        if custom_headers:
            header_parameters.update(custom_headers)
        if training_parameters is not None:
            body_content = self._serialize.body(training_parameters, 'TrainingParameters')
        else:
            body_content = None
        request = self._client.post(url, query_parameters, header_parameters, body_content)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Iteration', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    train_project.metadata = {'url': '/projects/{projectId}/train'}

    def import_project(self, token, name=None, custom_headers=None, raw=False, **operation_config):
        if False:
            print('Hello World!')
        'Imports a project.\n\n        :param token: Token generated from the export project call.\n        :type token: str\n        :param name: Optional, name of the project to use instead of\n         auto-generated name.\n        :type name: str\n        :param dict custom_headers: headers that will be added to the request\n        :param bool raw: returns the direct response alongside the\n         deserialized response\n        :param operation_config: :ref:`Operation configuration\n         overrides<msrest:optionsforoperations>`.\n        :return: Project or ClientRawResponse if raw=true\n        :rtype:\n         ~azure.cognitiveservices.vision.customvision.training.models.Project\n         or ~msrest.pipeline.ClientRawResponse\n        :raises:\n         :class:`CustomVisionErrorException<azure.cognitiveservices.vision.customvision.training.models.CustomVisionErrorException>`\n        '
        url = self.import_project.metadata['url']
        path_format_arguments = {'Endpoint': self._serialize.url('self.config.endpoint', self.config.endpoint, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['token'] = self._serialize.query('token', token, 'str')
        if name is not None:
            query_parameters['name'] = self._serialize.query('name', name, 'str')
        header_parameters = {}
        header_parameters['Accept'] = 'application/json'
        if custom_headers:
            header_parameters.update(custom_headers)
        request = self._client.post(url, query_parameters, header_parameters)
        response = self._client.send(request, stream=False, **operation_config)
        if response.status_code not in [200]:
            raise models.CustomVisionErrorException(self._deserialize, response)
        deserialized = None
        if response.status_code == 200:
            deserialized = self._deserialize('Project', response)
        if raw:
            client_raw_response = ClientRawResponse(deserialized, response)
            return client_raw_response
        return deserialized
    import_project.metadata = {'url': '/projects/import'}