from copy import deepcopy
from typing import Any, TYPE_CHECKING
from azure.core.rest import HttpRequest, HttpResponse
from azure.mgmt.core import ARMPipelineClient
from . import models as _models
from .._serialization import Deserializer, Serializer
from ._configuration import ComputeManagementClientConfiguration
from .operations import CommunityGalleriesOperations, CommunityGalleryImageVersionsOperations, CommunityGalleryImagesOperations, GalleriesOperations, GalleryApplicationVersionsOperations, GalleryApplicationsOperations, GalleryImageVersionsOperations, GalleryImagesOperations, GallerySharingProfileOperations, SharedGalleriesOperations, SharedGalleryImageVersionsOperations, SharedGalleryImagesOperations
if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class ComputeManagementClient:
    """Compute Client.

    :ivar galleries: GalleriesOperations operations
    :vartype galleries: azure.mgmt.compute.v2022_03_03.operations.GalleriesOperations
    :ivar gallery_images: GalleryImagesOperations operations
    :vartype gallery_images: azure.mgmt.compute.v2022_03_03.operations.GalleryImagesOperations
    :ivar gallery_image_versions: GalleryImageVersionsOperations operations
    :vartype gallery_image_versions:
     azure.mgmt.compute.v2022_03_03.operations.GalleryImageVersionsOperations
    :ivar gallery_applications: GalleryApplicationsOperations operations
    :vartype gallery_applications:
     azure.mgmt.compute.v2022_03_03.operations.GalleryApplicationsOperations
    :ivar gallery_application_versions: GalleryApplicationVersionsOperations operations
    :vartype gallery_application_versions:
     azure.mgmt.compute.v2022_03_03.operations.GalleryApplicationVersionsOperations
    :ivar gallery_sharing_profile: GallerySharingProfileOperations operations
    :vartype gallery_sharing_profile:
     azure.mgmt.compute.v2022_03_03.operations.GallerySharingProfileOperations
    :ivar shared_galleries: SharedGalleriesOperations operations
    :vartype shared_galleries: azure.mgmt.compute.v2022_03_03.operations.SharedGalleriesOperations
    :ivar shared_gallery_images: SharedGalleryImagesOperations operations
    :vartype shared_gallery_images:
     azure.mgmt.compute.v2022_03_03.operations.SharedGalleryImagesOperations
    :ivar shared_gallery_image_versions: SharedGalleryImageVersionsOperations operations
    :vartype shared_gallery_image_versions:
     azure.mgmt.compute.v2022_03_03.operations.SharedGalleryImageVersionsOperations
    :ivar community_galleries: CommunityGalleriesOperations operations
    :vartype community_galleries:
     azure.mgmt.compute.v2022_03_03.operations.CommunityGalleriesOperations
    :ivar community_gallery_images: CommunityGalleryImagesOperations operations
    :vartype community_gallery_images:
     azure.mgmt.compute.v2022_03_03.operations.CommunityGalleryImagesOperations
    :ivar community_gallery_image_versions: CommunityGalleryImageVersionsOperations operations
    :vartype community_gallery_image_versions:
     azure.mgmt.compute.v2022_03_03.operations.CommunityGalleryImageVersionsOperations
    :param credential: Credential needed for the client to connect to Azure. Required.
    :type credential: ~azure.core.credentials.TokenCredential
    :param subscription_id: Subscription credentials which uniquely identify Microsoft Azure
     subscription. The subscription ID forms part of the URI for every service call. Required.
    :type subscription_id: str
    :param base_url: Service URL. Default value is "https://management.azure.com".
    :type base_url: str
    :keyword api_version: Api Version. Default value is "2022-03-03". Note that overriding this
     default value may result in unsupported behavior.
    :paramtype api_version: str
    :keyword int polling_interval: Default waiting time between two polls for LRO operations if no
     Retry-After header is present.
    """

    def __init__(self, credential: 'TokenCredential', subscription_id: str, base_url: str='https://management.azure.com', **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        self._config = ComputeManagementClientConfiguration(credential=credential, subscription_id=subscription_id, **kwargs)
        self._client: ARMPipelineClient = ARMPipelineClient(base_url=base_url, config=self._config, **kwargs)
        client_models = {k: v for (k, v) in _models.__dict__.items() if isinstance(v, type)}
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)
        self._serialize.client_side_validation = False
        self.galleries = GalleriesOperations(self._client, self._config, self._serialize, self._deserialize, '2022-03-03')
        self.gallery_images = GalleryImagesOperations(self._client, self._config, self._serialize, self._deserialize, '2022-03-03')
        self.gallery_image_versions = GalleryImageVersionsOperations(self._client, self._config, self._serialize, self._deserialize, '2022-03-03')
        self.gallery_applications = GalleryApplicationsOperations(self._client, self._config, self._serialize, self._deserialize, '2022-03-03')
        self.gallery_application_versions = GalleryApplicationVersionsOperations(self._client, self._config, self._serialize, self._deserialize, '2022-03-03')
        self.gallery_sharing_profile = GallerySharingProfileOperations(self._client, self._config, self._serialize, self._deserialize, '2022-03-03')
        self.shared_galleries = SharedGalleriesOperations(self._client, self._config, self._serialize, self._deserialize, '2022-03-03')
        self.shared_gallery_images = SharedGalleryImagesOperations(self._client, self._config, self._serialize, self._deserialize, '2022-03-03')
        self.shared_gallery_image_versions = SharedGalleryImageVersionsOperations(self._client, self._config, self._serialize, self._deserialize, '2022-03-03')
        self.community_galleries = CommunityGalleriesOperations(self._client, self._config, self._serialize, self._deserialize, '2022-03-03')
        self.community_gallery_images = CommunityGalleryImagesOperations(self._client, self._config, self._serialize, self._deserialize, '2022-03-03')
        self.community_gallery_image_versions = CommunityGalleryImageVersionsOperations(self._client, self._config, self._serialize, self._deserialize, '2022-03-03')

    def _send_request(self, request: HttpRequest, **kwargs: Any) -> HttpResponse:
        if False:
            i = 10
            return i + 15
        'Runs the network request through the client\'s chained policies.\n\n        >>> from azure.core.rest import HttpRequest\n        >>> request = HttpRequest("GET", "https://www.example.org/")\n        <HttpRequest [GET], url: \'https://www.example.org/\'>\n        >>> response = client._send_request(request)\n        <HttpResponse: 200 OK>\n\n        For more information on this code flow, see https://aka.ms/azsdk/dpcodegen/python/send_request\n\n        :param request: The network request you want to make. Required.\n        :type request: ~azure.core.rest.HttpRequest\n        :keyword bool stream: Whether the response payload will be streamed. Defaults to False.\n        :return: The response of your network call. Does not do error handling on your response.\n        :rtype: ~azure.core.rest.HttpResponse\n        '
        request_copy = deepcopy(request)
        request_copy.url = self._client.format_url(request_copy.url)
        return self._client.send_request(request_copy, **kwargs)

    def close(self) -> None:
        if False:
            print('Hello World!')
        self._client.close()

    def __enter__(self) -> 'ComputeManagementClient':
        if False:
            i = 10
            return i + 15
        self._client.__enter__()
        return self

    def __exit__(self, *exc_details: Any) -> None:
        if False:
            return 10
        self._client.__exit__(*exc_details)