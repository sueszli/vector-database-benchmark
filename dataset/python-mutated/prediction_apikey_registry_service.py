from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto
__protobuf__ = proto.module(package='google.cloud.recommendationengine.v1beta1', manifest={'PredictionApiKeyRegistration', 'CreatePredictionApiKeyRegistrationRequest', 'ListPredictionApiKeyRegistrationsRequest', 'ListPredictionApiKeyRegistrationsResponse', 'DeletePredictionApiKeyRegistrationRequest'})

class PredictionApiKeyRegistration(proto.Message):
    """Registered Api Key.

    Attributes:
        api_key (str):
            The API key.
    """
    api_key: str = proto.Field(proto.STRING, number=1)

class CreatePredictionApiKeyRegistrationRequest(proto.Message):
    """Request message for the ``CreatePredictionApiKeyRegistration``
    method.

    Attributes:
        parent (str):
            Required. The parent resource path.
            ``projects/*/locations/global/catalogs/default_catalog/eventStores/default_event_store``.
        prediction_api_key_registration (google.cloud.recommendationengine_v1beta1.types.PredictionApiKeyRegistration):
            Required. The prediction API key
            registration.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    prediction_api_key_registration: 'PredictionApiKeyRegistration' = proto.Field(proto.MESSAGE, number=2, message='PredictionApiKeyRegistration')

class ListPredictionApiKeyRegistrationsRequest(proto.Message):
    """Request message for the ``ListPredictionApiKeyRegistrations``.

    Attributes:
        parent (str):
            Required. The parent placement resource name such as
            ``projects/1234/locations/global/catalogs/default_catalog/eventStores/default_event_store``
        page_size (int):
            Optional. Maximum number of results to return
            per page. If unset, the service will choose a
            reasonable default.
        page_token (str):
            Optional. The previous
            ``ListPredictionApiKeyRegistration.nextPageToken``.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)

class ListPredictionApiKeyRegistrationsResponse(proto.Message):
    """Response message for the ``ListPredictionApiKeyRegistrations``.

    Attributes:
        prediction_api_key_registrations (MutableSequence[google.cloud.recommendationengine_v1beta1.types.PredictionApiKeyRegistration]):
            The list of registered API keys.
        next_page_token (str):
            If empty, the list is complete. If nonempty, pass the token
            to the next request's
            ``ListPredictionApiKeysRegistrationsRequest.pageToken``.
    """

    @property
    def raw_page(self):
        if False:
            return 10
        return self
    prediction_api_key_registrations: MutableSequence['PredictionApiKeyRegistration'] = proto.RepeatedField(proto.MESSAGE, number=1, message='PredictionApiKeyRegistration')
    next_page_token: str = proto.Field(proto.STRING, number=2)

class DeletePredictionApiKeyRegistrationRequest(proto.Message):
    """Request message for ``DeletePredictionApiKeyRegistration`` method.

    Attributes:
        name (str):
            Required. The API key to unregister including full resource
            path.
            ``projects/*/locations/global/catalogs/default_catalog/eventStores/default_event_store/predictionApiKeyRegistrations/<YOUR_API_KEY>``
    """
    name: str = proto.Field(proto.STRING, number=1)
__all__ = tuple(sorted(__protobuf__.manifest))