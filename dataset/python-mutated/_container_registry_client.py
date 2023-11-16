import functools
import hashlib
import json
from io import BytesIO
from typing import Any, Dict, IO, Optional, overload, Union, cast, Tuple, MutableMapping
from azure.core.credentials import TokenCredential
from azure.core.exceptions import ClientAuthenticationError, ResourceNotFoundError, ResourceExistsError, HttpResponseError, map_error
from azure.core.paging import ItemPaged
from azure.core.pipeline import PipelineResponse
from azure.core.tracing.decorator import distributed_trace
from ._base_client import ContainerRegistryBaseClient
from ._generated.models import AcrErrors
from ._download_stream import DownloadBlobStream
from ._helpers import _compute_digest, _is_tag, _parse_next_link, _validate_digest, _get_blob_size, _get_manifest_size, SUPPORTED_API_VERSIONS, OCI_IMAGE_MANIFEST, SUPPORTED_MANIFEST_MEDIA_TYPES, DEFAULT_AUDIENCE, DEFAULT_CHUNK_SIZE, MAX_MANIFEST_SIZE
from ._models import RepositoryProperties, ArtifactTagProperties, ArtifactManifestProperties, GetManifestResult, DigestValidationError
JSON = MutableMapping[str, Any]

def _return_response(pipeline_response, _, __):
    if False:
        return 10
    return pipeline_response

def _return_response_and_headers(pipeline_response, _, response_headers):
    if False:
        print('Hello World!')
    return (pipeline_response, response_headers)

def _return_response_headers(_, __, response_headers):
    if False:
        for i in range(10):
            print('nop')
    return response_headers

class ContainerRegistryClient(ContainerRegistryBaseClient):

    def __init__(self, endpoint: str, credential: Optional[TokenCredential]=None, *, api_version: Optional[str]=None, audience: str=DEFAULT_AUDIENCE, **kwargs: Any) -> None:
        if False:
            return 10
        'Create a ContainerRegistryClient from an ACR endpoint and a credential.\n\n        :param str endpoint: An ACR endpoint.\n        :param credential: The credential with which to authenticate. This should be None in anonymous access.\n        :type credential: ~azure.core.credentials.TokenCredential or None\n        :keyword api_version: API Version. The default value is "2021-07-01".\n        :paramtype api_version: str\n        :keyword audience: URL to use for credential authentication with AAD. Its value could be\n            "https://management.azure.com", "https://management.chinacloudapi.cn" or\n            "https://management.usgovcloudapi.net". The default value is "https://containerregistry.azure.net".\n        :paramtype audience: str\n        :returns: None\n        :rtype: None\n        :raises ValueError: If the provided api_version keyword-only argument isn\'t supported.\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/sample_hello_world.py\n                :start-after: [START create_registry_client]\n                :end-before: [END create_registry_client]\n                :language: python\n                :dedent: 8\n                :caption: Instantiate an instance of `ContainerRegistryClient`\n        '
        if api_version and api_version not in SUPPORTED_API_VERSIONS:
            supported_versions = '\n'.join(SUPPORTED_API_VERSIONS)
            raise ValueError(f"Unsupported API version '{api_version}'. Please select from:\n{supported_versions}")
        if api_version is not None:
            kwargs['api_version'] = api_version
        defaultScope = [audience + '/.default']
        if not endpoint.startswith('https://') and (not endpoint.startswith('http://')):
            endpoint = 'https://' + endpoint
        self._endpoint = endpoint
        self._credential = credential
        super(ContainerRegistryClient, self).__init__(endpoint=endpoint, credential=credential, credential_scopes=defaultScope, **kwargs)

    def _get_digest_from_tag(self, repository: str, tag: str) -> str:
        if False:
            print('Hello World!')
        tag_props = self.get_tag_properties(repository, tag)
        return tag_props.digest

    @distributed_trace
    def delete_repository(self, repository: str, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Delete a repository. If the repository cannot be found or a response status code of\n        404 is returned an error will not be raised.\n\n        :param str repository: The repository to delete\n        :returns: None\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/sample_hello_world.py\n                :start-after: [START delete_repository]\n                :end-before: [END delete_repository]\n                :language: python\n                :dedent: 8\n                :caption: Delete a repository from the `ContainerRegistryClient`\n        '
        self._client.container_registry.delete_repository(repository, **kwargs)

    @distributed_trace
    def list_repository_names(self, **kwargs) -> ItemPaged[str]:
        if False:
            i = 10
            return i + 15
        'List all repositories\n\n        :keyword results_per_page: Number of repositories to return per page\n        :paramtype results_per_page: int\n        :returns: An iterable of strings\n        :rtype: ~azure.core.paging.ItemPaged[str]\n        :raises: ~azure.core.exceptions.HttpResponseError\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/sample_delete_tags.py\n                :start-after: [START list_repository_names]\n                :end-before: [END list_repository_names]\n                :language: python\n                :dedent: 8\n                :caption: List repositories in a container registry account\n        '
        n = kwargs.pop('results_per_page', None)
        last = kwargs.pop('last', None)
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        accept = 'application/json'

        def prepare_request(next_link=None):
            if False:
                while True:
                    i = 10
            header_parameters: Dict[str, Any] = {}
            header_parameters['Accept'] = self._client._serialize.header('accept', accept, 'str')
            if not next_link:
                url = '/acr/v1/_catalog'
                path_format_arguments = {'url': self._client._serialize.url('self._config.url', self._client._config.url, 'str', skip_quote=True)}
                url = self._client._client.format_url(url, **path_format_arguments)
                query_parameters: Dict[str, Any] = {}
                if last is not None:
                    query_parameters['last'] = self._client._serialize.query('last', last, 'str')
                if n is not None:
                    query_parameters['n'] = self._client._serialize.query('n', n, 'int')
                request = self._client._client.get(url, query_parameters, header_parameters)
            else:
                url = next_link
                query_parameters: Dict[str, Any] = {}
                path_format_arguments = {'url': self._client._serialize.url('self._config.url', self._client._config.url, 'str', skip_quote=True)}
                url = self._client._client.format_url(url, **path_format_arguments)
                request = self._client._client.get(url, query_parameters, header_parameters)
            return request

        def extract_data(pipeline_response):
            if False:
                for i in range(10):
                    print('nop')
            deserialized = self._client._deserialize('Repositories', pipeline_response)
            list_of_elem = deserialized.repositories or []
            if cls:
                list_of_elem = cls(list_of_elem)
            link = None
            if 'Link' in pipeline_response.http_response.headers.keys():
                link = _parse_next_link(pipeline_response.http_response.headers['Link'])
            return (link, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                return 10
            request = prepare_request(next_link)
            pipeline_response = self._client._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                error = self._client._deserialize.failsafe_deserialize(AcrErrors, response)
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, model=error)
            return pipeline_response
        return ItemPaged(get_next, extract_data)

    @distributed_trace
    def get_repository_properties(self, repository: str, **kwargs) -> RepositoryProperties:
        if False:
            return 10
        'Get the properties of a repository\n\n        :param str repository: Name of the repository\n        :rtype: ~azure.containerregistry.RepositoryProperties\n        :return: The properties of a repository\n        :raises: ~azure.core.exceptions.ResourceNotFoundError\n        '
        return RepositoryProperties._from_generated(self._client.container_registry.get_properties(repository, **kwargs))

    @distributed_trace
    def list_manifest_properties(self, repository: str, **kwargs) -> ItemPaged[ArtifactManifestProperties]:
        if False:
            print('Hello World!')
        'List the artifacts for a repository\n\n        :param str repository: Name of the repository\n        :keyword order_by: Query parameter for ordering by time ascending or descending\n        :paramtype order_by: ~azure.containerregistry.ArtifactManifestOrder or str\n        :keyword results_per_page: Number of repositories to return per page\n        :paramtype results_per_page: int\n        :returns: An iterable of :class:`~azure.containerregistry.ArtifactManifestProperties`\n        :rtype: ~azure.core.paging.ItemPaged[~azure.containerregistry.ArtifactManifestProperties]\n        :raises: ~azure.core.exceptions.ResourceNotFoundError\n        '
        name = repository
        last = kwargs.pop('last', None)
        n = kwargs.pop('results_per_page', None)
        orderby = kwargs.pop('order_by', None)
        cls = kwargs.pop('cls', lambda objs: [ArtifactManifestProperties._from_generated(x, repository_name=repository, registry=self._endpoint) for x in objs])
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        accept = 'application/json'

        def prepare_request(next_link=None):
            if False:
                print('Hello World!')
            header_parameters: Dict[str, Any] = {}
            header_parameters['Accept'] = self._client._serialize.header('accept', accept, 'str')
            if not next_link:
                url = '/acr/v1/{name}/_manifests'
                path_format_arguments = {'url': self._client._serialize.url('self._client._config.url', self._client._config.url, 'str', skip_quote=True), 'name': self._client._serialize.url('name', name, 'str')}
                url = self._client._client.format_url(url, **path_format_arguments)
                query_parameters: Dict[str, Any] = {}
                if last is not None:
                    query_parameters['last'] = self._client._serialize.query('last', last, 'str')
                if n is not None:
                    query_parameters['n'] = self._client._serialize.query('n', n, 'int')
                if orderby is not None:
                    query_parameters['orderby'] = self._client._serialize.query('orderby', orderby, 'str')
                request = self._client._client.get(url, query_parameters, header_parameters)
            else:
                url = next_link
                query_parameters: Dict[str, Any] = {}
                path_format_arguments = {'url': self._client._serialize.url('self._client._config.url', self._client._config.url, 'str', skip_quote=True), 'name': self._client._serialize.url('name', name, 'str')}
                url = self._client._client.format_url(url, **path_format_arguments)
                request = self._client._client.get(url, query_parameters, header_parameters)
            return request

        def extract_data(pipeline_response):
            if False:
                print('Hello World!')
            deserialized = self._client._deserialize('AcrManifests', pipeline_response)
            list_of_elem = deserialized.manifests or []
            if cls:
                list_of_elem = cls(list_of_elem)
            link = None
            if 'Link' in pipeline_response.http_response.headers.keys():
                link = _parse_next_link(pipeline_response.http_response.headers['Link'])
            return (link, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            request = prepare_request(next_link)
            pipeline_response = self._client._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                error = self._client._deserialize.failsafe_deserialize(AcrErrors, response)
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, model=error)
            return pipeline_response
        return ItemPaged(get_next, extract_data)

    @distributed_trace
    def delete_tag(self, repository: str, tag: str, **kwargs) -> None:
        if False:
            return 10
        'Delete a tag from a repository. If the tag cannot be found or a response status code of\n        404 is returned an error will not be raised.\n\n        :param str repository: Name of the repository the tag belongs to\n        :param str tag: The tag to be deleted\n        :returns: None\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError\n\n        Example\n\n        .. code-block:: python\n\n            from azure.containerregistry import ContainerRegistryClient\n            from azure.identity import DefaultAzureCredential\n            endpoint = os.environ["CONTAINERREGISTRY_ENDPOINT"]\n            client = ContainerRegistryClient(endpoint, DefaultAzureCredential(), audience="my_audience")\n            for tag in client.list_tag_properties("my_repository"):\n                client.delete_tag("my_repository", tag.name)\n        '
        self._client.container_registry.delete_tag(repository, tag, **kwargs)

    @distributed_trace
    def get_manifest_properties(self, repository: str, tag_or_digest: str, **kwargs) -> ArtifactManifestProperties:
        if False:
            for i in range(10):
                print('nop')
        'Get the properties of a registry artifact\n\n        :param str repository: Name of the repository\n        :param str tag_or_digest: Tag or digest of the manifest\n        :return: The properties of a registry artifact\n        :rtype: ~azure.containerregistry.ArtifactManifestProperties\n        :raises: ~azure.core.exceptions.ResourceNotFoundError\n\n        Example\n\n        .. code-block:: python\n\n            from azure.containerregistry import ContainerRegistryClient\n            from azure.identity import DefaultAzureCredential\n            endpoint = os.environ["CONTAINERREGISTRY_ENDPOINT"]\n            client = ContainerRegistryClient(endpoint, DefaultAzureCredential(), audience="my_audience")\n            for artifact in client.list_manifest_properties("my_repository"):\n                properties = client.get_manifest_properties("my_repository", artifact.digest)\n        '
        if _is_tag(tag_or_digest):
            tag_or_digest = self._get_digest_from_tag(repository, tag_or_digest)
        manifest_properties = self._client.container_registry.get_manifest_properties(repository, tag_or_digest, **kwargs)
        return ArtifactManifestProperties._from_generated(manifest_properties.manifest, repository_name=repository, registry=self._endpoint)

    @distributed_trace
    def get_tag_properties(self, repository: str, tag: str, **kwargs) -> ArtifactTagProperties:
        if False:
            for i in range(10):
                print('nop')
        'Get the properties for a tag\n\n        :param str repository: Name of the repository\n        :param str tag: The tag to get tag properties for\n        :return: The properties for a tag\n        :rtype: ~azure.containerregistry.ArtifactTagProperties\n        :raises: ~azure.core.exceptions.ResourceNotFoundError\n\n        Example\n\n        .. code-block:: python\n\n            from azure.containerregistry import ContainerRegistryClient\n            from azure.identity import DefaultAzureCredential\n            endpoint = os.environ["CONTAINERREGISTRY_ENDPOINT"]\n            client = ContainerRegistryClient(endpoint, DefaultAzureCredential(), audience="my_audience")\n            for tag in client.list_tag_properties("my_repository"):\n                tag_properties = client.get_tag_properties("my_repository", tag.name)\n        '
        tag_properties = self._client.container_registry.get_tag_properties(repository, tag, **kwargs)
        return ArtifactTagProperties._from_generated(tag_properties.tag, repository_name=repository)

    @distributed_trace
    def list_tag_properties(self, repository: str, **kwargs) -> ItemPaged[ArtifactTagProperties]:
        if False:
            i = 10
            return i + 15
        'List the tags for a repository\n\n        :param str repository: Name of the repository\n        :keyword order_by: Query parameter for ordering by time ascending or descending\n        :paramtype order_by: ~azure.containerregistry.ArtifactTagOrder or str\n        :keyword results_per_page: Number of repositories to return per page\n        :paramtype results_per_page: int\n        :returns: An iterable of :class:`~azure.containerregistry.ArtifactTagProperties`\n        :rtype: ~azure.core.paging.ItemPaged[~azure.containerregistry.ArtifactTagProperties]\n        :raises: ~azure.core.exceptions.ResourceNotFoundError\n\n        Example\n\n        .. code-block:: python\n\n            from azure.containerregistry import ContainerRegistryClient\n            from azure.identity import DefaultAzureCredential\n            endpoint = os.environ["CONTAINERREGISTRY_ENDPOINT"]\n            client = ContainerRegistryClient(endpoint, DefaultAzureCredential(), audience="my_audience")\n            for tag in client.list_tag_properties("my_repository"):\n                tag_properties = client.get_tag_properties("my_repository", tag.name)\n        '
        name = repository
        last = kwargs.pop('last', None)
        n = kwargs.pop('results_per_page', None)
        orderby = kwargs.pop('order_by', None)
        digest = kwargs.pop('digest', None)
        cls = kwargs.pop('cls', lambda objs: [ArtifactTagProperties._from_generated(o, repository_name=repository) for o in objs])
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        accept = 'application/json'

        def prepare_request(next_link=None):
            if False:
                for i in range(10):
                    print('nop')
            header_parameters: Dict[str, Any] = {}
            header_parameters['Accept'] = self._client._serialize.header('accept', accept, 'str')
            if not next_link:
                url = '/acr/v1/{name}/_tags'
                path_format_arguments = {'url': self._client._serialize.url('self._config.url', self._client._config.url, 'str', skip_quote=True), 'name': self._client._serialize.url('name', name, 'str')}
                url = self._client._client.format_url(url, **path_format_arguments)
                query_parameters: Dict[str, Any] = {}
                if last is not None:
                    query_parameters['last'] = self._client._serialize.query('last', last, 'str')
                if n is not None:
                    query_parameters['n'] = self._client._serialize.query('n', n, 'int')
                if orderby is not None:
                    query_parameters['orderby'] = self._client._serialize.query('orderby', orderby, 'str')
                if digest is not None:
                    query_parameters['digest'] = self._client._serialize.query('digest', digest, 'str')
                request = self._client._client.get(url, query_parameters, header_parameters)
            else:
                url = next_link
                query_parameters: Dict[str, Any] = {}
                path_format_arguments = {'url': self._client._serialize.url('self._client._config.url', self._client._config.url, 'str', skip_quote=True), 'name': self._client._serialize.url('name', name, 'str')}
                url = self._client._client.format_url(url, **path_format_arguments)
                request = self._client._client.get(url, query_parameters, header_parameters)
            return request

        def extract_data(pipeline_response):
            if False:
                print('Hello World!')
            deserialized = self._client._deserialize('TagList', pipeline_response)
            list_of_elem = deserialized.tag_attribute_bases or []
            if cls:
                list_of_elem = cls(list_of_elem)
            link = None
            if 'Link' in pipeline_response.http_response.headers.keys():
                link = _parse_next_link(pipeline_response.http_response.headers['Link'])
            return (link, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                return 10
            request = prepare_request(next_link)
            pipeline_response = self._client._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                error = self._client._deserialize.failsafe_deserialize(AcrErrors, response)
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, model=error)
            return pipeline_response
        return ItemPaged(get_next, extract_data)

    @overload
    def update_manifest_properties(self, repository: str, tag_or_digest: str, properties: ArtifactManifestProperties, **kwargs: Any) -> ArtifactManifestProperties:
        if False:
            for i in range(10):
                print('nop')
        'Set the permission properties for a manifest.\n\n        The updatable properties include: `can_delete`, `can_list`, `can_read`, and `can_write`.\n\n        :param str repository: Repository the manifest belongs to.\n        :param str tag_or_digest: Tag or digest of the manifest.\n        :param properties: The property\'s values to be set. This is a positional-only\n            parameter. Please provide either this or individual keyword parameters.\n        :type properties: ~azure.containerregistry.ArtifactManifestProperties\n        :rtype: ~azure.containerregistry.ArtifactManifestProperties\n        :raises: ~azure.core.exceptions.ResourceNotFoundError\n\n        Example\n\n        .. code-block:: python\n\n            from azure.containerregistry import ArtifactManifestProperties, ContainerRegistryClient\n            from azure.identity import DefaultAzureCredential\n            endpoint = os.environ["CONTAINERREGISTRY_ENDPOINT"]\n            client = ContainerRegistryClient(endpoint, DefaultAzureCredential(), audience="my_audience")\n            manifest_properties = ArtifactManifestProperties(\n                can_delete=False, can_list=False, can_read=False, can_write=False\n            )\n            for artifact in client.list_manifest_properties("my_repository"):\n                received_properties = client.update_manifest_properties(\n                    "my_repository",\n                    artifact.digest,\n                    manifest_properties,\n                )\n        '

    @overload
    def update_manifest_properties(self, repository: str, tag_or_digest: str, *, can_delete: Optional[bool]=None, can_list: Optional[bool]=None, can_read: Optional[bool]=None, can_write: Optional[bool]=None, **kwargs: Any) -> ArtifactManifestProperties:
        if False:
            return 10
        'Set the permission properties for a manifest.\n\n        The updatable properties include: `can_delete`, `can_list`, `can_read`, and `can_write`.\n\n        :param str repository: Repository the manifest belongs to.\n        :param str tag_or_digest: Tag or digest of the manifest.\n        :keyword bool can_delete: Delete permissions for a manifest.\n        :keyword bool can_list: List permissions for a manifest.\n        :keyword bool can_read: Read permissions for a manifest.\n        :keyword bool can_write: Write permissions for a manifest.\n        :rtype: ~azure.containerregistry.ArtifactManifestProperties\n        :raises: ~azure.core.exceptions.ResourceNotFoundError\n\n        Example\n\n        .. code-block:: python\n\n            from azure.containerregistry import ContainerRegistryClient\n            from azure.identity import DefaultAzureCredential\n            endpoint = os.environ["CONTAINERREGISTRY_ENDPOINT"]\n            client = ContainerRegistryClient(endpoint, DefaultAzureCredential(), audience="my_audience")\n            for artifact in client.list_manifest_properties("my_repository"):\n                received_properties = client.update_manifest_properties(\n                    "my_repository",\n                    artifact.digest,\n                    can_delete=False,\n                    can_list=False,\n                    can_read=False,\n                    can_write=False,\n                )\n        '

    @distributed_trace
    def update_manifest_properties(self, *args: Union[str, ArtifactManifestProperties], **kwargs) -> ArtifactManifestProperties:
        if False:
            return 10
        repository = str(args[0])
        tag_or_digest = str(args[1])
        properties = None
        if len(args) == 3:
            properties = cast(ArtifactManifestProperties, args[2])
        else:
            properties = ArtifactManifestProperties()
        properties.can_delete = kwargs.pop('can_delete', properties.can_delete)
        properties.can_list = kwargs.pop('can_list', properties.can_list)
        properties.can_read = kwargs.pop('can_read', properties.can_read)
        properties.can_write = kwargs.pop('can_write', properties.can_write)
        if _is_tag(tag_or_digest):
            tag_or_digest = self._get_digest_from_tag(repository, tag_or_digest)
        manifest_properties = self._client.container_registry.update_manifest_properties(repository, tag_or_digest, value=properties._to_generated(), **kwargs)
        return ArtifactManifestProperties._from_generated(manifest_properties.manifest, repository_name=repository, registry=self._endpoint)

    @overload
    def update_tag_properties(self, repository: str, tag: str, properties: ArtifactTagProperties, **kwargs: Any) -> ArtifactTagProperties:
        if False:
            print('Hello World!')
        'Set the permission properties for a tag.\n\n        The updatable properties include: `can_delete`, `can_list`, `can_read`, and `can_write`.\n\n        :param str repository: Repository the tag belongs to.\n        :param str tag: Tag to set properties for.\n        :param properties: The property\'s values to be set. This is a positional-only\n            parameter. Please provide either this or individual keyword parameters.\n        :type properties: ~azure.containerregistry.ArtifactTagProperties\n        :rtype: ~azure.containerregistry.ArtifactTagProperties\n        :raises: ~azure.core.exceptions.ResourceNotFoundError\n\n        Example\n\n        .. code-block:: python\n\n            from azure.containerregistry import ArtifactTagProperties, ContainerRegistryClient\n            from azure.identity import DefaultAzureCredential\n            endpoint = os.environ["CONTAINERREGISTRY_ENDPOINT"]\n            client = ContainerRegistryClient(endpoint, DefaultAzureCredential(), audience="my_audience")\n            tag_properties = ArtifactTagProperties(can_delete=False, can_list=False, can_read=False, can_write=False)\n            received = client.update_tag_properties(\n                "my_repository",\n                "latest",\n                tag_properties,\n            )\n        '

    @overload
    def update_tag_properties(self, repository: str, tag: str, *, can_delete: Optional[bool]=None, can_list: Optional[bool]=None, can_read: Optional[bool]=None, can_write: Optional[bool]=None, **kwargs: Any) -> ArtifactTagProperties:
        if False:
            return 10
        'Set the permission properties for a tag.\n\n        The updatable properties include: `can_delete`, `can_list`, `can_read`, and `can_write`.\n\n        :param str repository: Repository the tag belongs to.\n        :param str tag: Tag to set properties for.\n        :keyword bool can_delete: Delete permissions for a tag.\n        :keyword bool can_list: List permissions for a tag.\n        :keyword bool can_read: Read permissions for a tag.\n        :keyword bool can_write: Write permissions for a tag.\n        :rtype: ~azure.containerregistry.ArtifactTagProperties\n        :raises: ~azure.core.exceptions.ResourceNotFoundError\n\n        Example\n\n        .. code-block:: python\n\n            from azure.containerregistry import ContainerRegistryClient\n            from azure.identity import DefaultAzureCredential\n            endpoint = os.environ["CONTAINERREGISTRY_ENDPOINT"]\n            client = ContainerRegistryClient(endpoint, DefaultAzureCredential(), audience="my_audience")\n            received = client.update_tag_properties(\n                "my_repository",\n                "latest",\n                can_delete=False,\n                can_list=False,\n                can_read=False,\n                can_write=False,\n            )\n        '

    @distributed_trace
    def update_tag_properties(self, *args: Union[str, ArtifactTagProperties], **kwargs) -> ArtifactTagProperties:
        if False:
            for i in range(10):
                print('nop')
        repository = str(args[0])
        tag = str(args[1])
        properties = None
        if len(args) == 3:
            properties = cast(ArtifactTagProperties, args[2])
        else:
            properties = ArtifactTagProperties()
        properties.can_delete = kwargs.pop('can_delete', properties.can_delete)
        properties.can_list = kwargs.pop('can_list', properties.can_list)
        properties.can_read = kwargs.pop('can_read', properties.can_read)
        properties.can_write = kwargs.pop('can_write', properties.can_write)
        tag_attributes = self._client.container_registry.update_tag_attributes(repository, tag, value=properties._to_generated(), **kwargs)
        return ArtifactTagProperties._from_generated(tag_attributes.tag, repository_name=repository)

    @overload
    def update_repository_properties(self, repository: str, properties: RepositoryProperties, **kwargs: Any) -> RepositoryProperties:
        if False:
            print('Hello World!')
        'Set the permission properties of a repository.\n\n        The updatable properties include: `can_delete`, `can_list`, `can_read`, and `can_write`.\n\n        :param str repository: Name of the repository.\n        :param properties: Properties to set for the repository. This is a positional-only\n            parameter. Please provide either this or individual keyword parameters.\n        :type properties: ~azure.containerregistry.RepositoryProperties\n        :rtype: ~azure.containerregistry.RepositoryProperties\n        :raises: ~azure.core.exceptions.ResourceNotFoundError\n        '

    @overload
    def update_repository_properties(self, repository: str, *, can_delete: Optional[bool]=None, can_list: Optional[bool]=None, can_read: Optional[bool]=None, can_write: Optional[bool]=None, **kwargs: Any) -> RepositoryProperties:
        if False:
            while True:
                i = 10
        'Set the permission properties of a repository.\n\n        The updatable properties include: `can_delete`, `can_list`, `can_read`, and `can_write`.\n\n        :param str repository: Name of the repository.\n        :keyword bool can_delete: Delete permissions for a repository.\n        :keyword bool can_list: List permissions for a repository.\n        :keyword bool can_read: Read permissions for a repository.\n        :keyword bool can_write: Write permissions for a repository.\n        :rtype: ~azure.containerregistry.RepositoryProperties\n        :raises: ~azure.core.exceptions.ResourceNotFoundError\n        '

    @distributed_trace
    def update_repository_properties(self, *args: Union[str, RepositoryProperties], **kwargs) -> RepositoryProperties:
        if False:
            for i in range(10):
                print('nop')
        repository = str(args[0])
        properties = None
        if len(args) == 2:
            properties = cast(RepositoryProperties, args[1])
        else:
            properties = RepositoryProperties()
        properties.can_delete = kwargs.pop('can_delete', properties.can_delete)
        properties.can_list = kwargs.pop('can_list', properties.can_list)
        properties.can_read = kwargs.pop('can_read', properties.can_read)
        properties.can_write = kwargs.pop('can_write', properties.can_write)
        return RepositoryProperties._from_generated(self._client.container_registry.update_properties(repository, value=properties._to_generated(), **kwargs))

    @distributed_trace
    def set_manifest(self, repository: str, manifest: Union[JSON, IO[bytes]], *, tag: Optional[str]=None, media_type: str=OCI_IMAGE_MANIFEST, **kwargs) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Set a manifest for an artifact.\n\n        :param str repository: Name of the repository\n        :param manifest: The manifest to set. It can be a JSON formatted dict or seekable stream.\n        :type manifest: dict or IO\n        :keyword tag: Tag of the manifest.\n        :paramtype tag: str or None\n        :keyword media_type: The media type of the manifest. If not specified, this value will be set to\n            a default value of "application/vnd.oci.image.manifest.v1+json". Note: the current known media types are:\n            "application/vnd.oci.image.manifest.v1+json", and "application/vnd.docker.distribution.manifest.v2+json".\n        :paramtype media_type: str\n        :returns: The digest of the set manifest, calculated by the registry.\n        :rtype: str\n        :raises ValueError: If the parameter repository or manifest is None.\n        :raises ~azure.containerregistry.DigestValidationError:\n            If the server-computed digest does not match the client-computed digest.\n        '
        try:
            data: IO[bytes]
            if isinstance(manifest, MutableMapping):
                data = BytesIO(json.dumps(manifest).encode())
            else:
                data = manifest
            tag_or_digest = tag
            if tag_or_digest is None:
                tag_or_digest = _compute_digest(data)
            response_headers = self._client.container_registry.create_manifest(name=repository, reference=tag_or_digest, payload=data, content_type=media_type, cls=_return_response_headers, **kwargs)
            digest = response_headers['Docker-Content-Digest']
            if not _validate_digest(data, digest):
                raise DigestValidationError('The server-computed digest does not match the client-computed digest.')
        except Exception as e:
            if repository is None or manifest is None:
                raise ValueError('The parameter repository and manifest cannot be None.') from e
            raise
        return digest

    @distributed_trace
    def get_manifest(self, repository: str, tag_or_digest: str, **kwargs) -> GetManifestResult:
        if False:
            i = 10
            return i + 15
        'Get the manifest for an artifact.\n\n        :param str repository: Name of the repository.\n        :param str tag_or_digest: The tag or digest of the manifest to get.\n            When digest is provided, will use this digest to compare with the one calculated by the response payload.\n            When tag is provided, will use the digest in response headers to compare.\n        :returns: GetManifestResult\n        :rtype: ~azure.containerregistry.GetManifestResult\n        :raises ~azure.containerregistry.DigestValidationError:\n            If the content of retrieved manifest digest does not match the requested digest, or\n            the server-computed digest does not match the client-computed digest when tag is passing.\n        :raises ValueError: If the content-length header is missing or invalid in response, or the manifest size is\n            bigger than 4MB.\n        '
        response = cast(PipelineResponse, self._client.container_registry.get_manifest(name=repository, reference=tag_or_digest, accept=SUPPORTED_MANIFEST_MEDIA_TYPES, cls=_return_response, **kwargs))
        manifest_size = _get_manifest_size(response.http_response.headers)
        if manifest_size > MAX_MANIFEST_SIZE:
            raise ValueError('Manifest size is bigger than max allowed size of 4MB.')
        media_type = response.http_response.headers['Content-Type']
        manifest_bytes = response.http_response.read()
        manifest_json = response.http_response.json()
        manifest_digest = _compute_digest(manifest_bytes)
        if tag_or_digest.startswith('sha256:'):
            if manifest_digest != tag_or_digest:
                raise DigestValidationError('The content of retrieved manifest digest does not match the requested digest.')
        digest = response.http_response.headers['Docker-Content-Digest']
        if manifest_digest != digest:
            raise DigestValidationError('The server-computed digest does not match the client-computed digest.')
        return GetManifestResult(digest=digest, manifest=manifest_json, media_type=media_type)

    @distributed_trace
    def upload_blob(self, repository: str, data: IO[bytes], **kwargs) -> Tuple[str, int]:
        if False:
            for i in range(10):
                print('nop')
        'Upload an artifact blob.\n\n        :param str repository: Name of the repository.\n        :param data: The blob to upload. Note: This must be a seekable stream.\n        :type data: IO\n        :returns: The digest and size in bytes of the uploaded blob.\n        :rtype: Tuple[str, int]\n        :raises ValueError: If the parameter repository or data is None.\n        :raises ~azure.containerregistry.DigestValidationError:\n            If the server-computed digest does not match the client-computed digest.\n        '
        try:
            start_upload_response_headers = cast(Dict[str, str], self._client.container_registry_blob.start_upload(repository, cls=_return_response_headers, **kwargs))
            (digest, location, blob_size) = self._upload_blob_chunk(start_upload_response_headers['Location'], data, **kwargs)
            complete_upload_response_headers = cast(Dict[str, str], self._client.container_registry_blob.complete_upload(digest=digest, next_link=location, cls=_return_response_headers, **kwargs))
            if digest != complete_upload_response_headers['Docker-Content-Digest']:
                raise DigestValidationError('The server-computed digest does not match the client-computed digest.')
        except Exception as e:
            if repository is None or data is None:
                raise ValueError('The parameter repository and data cannot be None.') from e
            raise
        return (complete_upload_response_headers['Docker-Content-Digest'], blob_size)

    def _upload_blob_chunk(self, location: str, data: IO[bytes], **kwargs) -> Tuple[str, str, int]:
        if False:
            for i in range(10):
                print('nop')
        hasher = hashlib.sha256()
        buffer = data.read(DEFAULT_CHUNK_SIZE)
        blob_size = len(buffer)
        while len(buffer) > 0:
            response_headers = cast(Dict[str, str], self._client.container_registry_blob.upload_chunk(location, BytesIO(buffer), cls=_return_response_headers, **kwargs))
            location = response_headers['Location']
            hasher.update(buffer)
            buffer = data.read(DEFAULT_CHUNK_SIZE)
            blob_size += len(buffer)
        return (f'sha256:{hasher.hexdigest()}', location, blob_size)

    @distributed_trace
    def download_blob(self, repository: str, digest: str, **kwargs) -> DownloadBlobStream:
        if False:
            return 10
        'Download a blob that is part of an artifact to a stream.\n\n        :param str repository: Name of the repository.\n        :param str digest: The digest of the blob to download.\n        :returns: DownloadBlobStream\n        :rtype: ~azure.containerregistry.DownloadBlobStream\n        :raises DigestValidationError:\n            If the content of retrieved blob digest does not match the requested digest.\n        :raises ValueError: If the content-range header is missing or invalid in response.\n        '
        end_range = DEFAULT_CHUNK_SIZE - 1
        (first_chunk, headers) = cast(Tuple[PipelineResponse, Dict[str, str]], self._client.container_registry_blob.get_chunk(repository, digest, range_header=f'bytes=0-{end_range}', cls=_return_response_and_headers, **kwargs))
        blob_size = _get_blob_size(headers)
        return DownloadBlobStream(response=first_chunk, digest=digest, get_next=functools.partial(self._client.container_registry_blob.get_chunk, name=repository, digest=digest, cls=_return_response_and_headers, **kwargs), blob_size=blob_size, downloaded=int(headers['Content-Length']), chunk_size=DEFAULT_CHUNK_SIZE)

    @distributed_trace
    def delete_manifest(self, repository: str, tag_or_digest: str, **kwargs) -> None:
        if False:
            print('Hello World!')
        'Delete a manifest. If the manifest cannot be found or a response status code of\n        404 is returned an error will not be raised.\n\n        :param str repository: Name of the repository the manifest belongs to\n        :param str tag_or_digest: Tag or digest of the manifest to be deleted\n        :returns: None\n        :raises: ~azure.core.exceptions.HttpResponseError\n\n        Example\n\n        .. code-block:: python\n\n            from azure.containerregistry import ContainerRegistryClient\n            from azure.identity import DefaultAzureCredential\n            endpoint = os.environ["CONTAINERREGISTRY_ENDPOINT"]\n            client = ContainerRegistryClient(endpoint, DefaultAzureCredential(), audience="my_audience")\n            client.delete_manifest("my_repository", "my_tag_or_digest")\n        '
        if _is_tag(tag_or_digest):
            tag_or_digest = self._get_digest_from_tag(repository, tag_or_digest)
        self._client.container_registry.delete_manifest(repository, tag_or_digest, **kwargs)

    @distributed_trace
    def delete_blob(self, repository: str, digest: str, **kwargs) -> None:
        if False:
            return 10
        'Delete a blob. If the blob cannot be found or a response status code of\n        404 is returned an error will not be raised.\n\n        :param str repository: Name of the repository the manifest belongs to\n        :param str digest: Digest of the blob to be deleted\n        :returns: None\n\n        Example\n\n        .. code-block:: python\n\n            from azure.containerregistry import ContainerRegistryClient\n            from azure.identity import DefaultAzureCredential\n            endpoint = os.environ["CONTAINERREGISTRY_ENDPOINT"]\n            client = ContainerRegistryClient(endpoint, DefaultAzureCredential(), audience="my_audience")\n            client.delete_blob("my_repository", "my_digest")\n        '
        try:
            self._client.container_registry_blob.delete_blob(repository, digest, **kwargs)
        except HttpResponseError as error:
            if error.status_code == 404:
                return
            raise