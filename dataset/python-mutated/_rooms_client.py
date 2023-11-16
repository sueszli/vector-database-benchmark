from datetime import datetime
from typing import List, Optional, Union, Any
import uuid
from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.core.paging import ItemPaged
from azure.core.tracing.decorator import distributed_trace
from azure.communication.rooms._models import RoomParticipant, CommunicationRoom
from azure.communication.rooms._shared.models import CommunicationIdentifier
from ._generated._client import AzureCommunicationRoomsService
from ._generated._serialization import Serializer
from ._shared.auth_policy_utils import get_authentication_policy
from ._shared.utils import parse_connection_str
from ._version import SDK_MONIKER
from ._api_versions import DEFAULT_VERSION

class RoomsClient(object):
    """A client to interact with the AzureCommunicationService Rooms gateway.

    This client provides operations to manage rooms.

    This client provides operations to manage rooms.

    :param str endpoint:
        The endpoint url for Azure Communication Service resource.
    param Union[TokenCredential, AzureKeyCredential] credential:
        The access key we use to authenticate against the service.
    :keyword api_version: Azure Communication Rooms API version.
        Default value is "2023-10-30-preview".
        Note that overriding this default value may result in unsupported behavior.
    :paramtype api_version: str
    """

    def __init__(self, endpoint: str, credential: Union[TokenCredential, AzureKeyCredential], **kwargs) -> None:
        if False:
            return 10
        try:
            if not endpoint.lower().startswith('http'):
                endpoint = 'https://' + endpoint
        except AttributeError as exc:
            raise ValueError('Account URL must be a string.') from exc
        if not credential:
            raise ValueError('invalid credential from connection string.')
        if endpoint.endswith('/'):
            endpoint = endpoint[:-1]
        self._endpoint = endpoint
        self._api_version = kwargs.pop('api_version', DEFAULT_VERSION)
        self._authentication_policy = get_authentication_policy(endpoint, credential, decode_url=True)
        self._rooms_service_client = AzureCommunicationRoomsService(self._endpoint, api_version=self._api_version, authentication_policy=self._authentication_policy, sdk_moniker=SDK_MONIKER, **kwargs)

    @classmethod
    def from_connection_string(cls, conn_str: str, **kwargs) -> 'RoomsClient':
        if False:
            for i in range(10):
                print('nop')
        'Create RoomsClient from a Connection String.\n\n        :param str conn_str:\n            A connection string to an Azure Communication Service resource.\n        :returns: Instance of RoomsClient.\n        :rtype: ~azure.communication.room.RoomsClient\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/Rooms_sample.py\n                :start-after: [START auth_from_connection_string]\n                :end-before: [END auth_from_connection_string]\n                :language: python\n                :dedent: 8\n                :caption: Creating the RoomsClient from a connection string.\n        '
        (endpoint, access_key) = parse_connection_str(conn_str)
        return cls(endpoint, AzureKeyCredential(access_key), **kwargs)

    @distributed_trace
    def create_room(self, *, valid_from: Optional[datetime]=None, valid_until: Optional[datetime]=None, pstn_dial_out_enabled: bool=False, participants: Optional[List[RoomParticipant]]=None, **kwargs) -> CommunicationRoom:
        if False:
            return 10
        'Create a new room.\n\n        :keyword datetime valid_from: The timestamp from when the room is open for joining. Optional.\n        :keyword datetime valid_until: The timestamp from when the room can no longer be joined. Optional.\n        :keyword bool pstn_dial_out_enabled: Set this flag to true if, at the time of the call,\n        dial out to a PSTN number is enabled in a particular room. Optional.\n        :keyword List[RoomParticipant] participants: Collection of identities invited to the room. Optional.\n        :type participants: List[~azure.communication.rooms.RoomParticipant]\n        :returns: Created room.\n        :rtype: ~azure.communication.rooms.CommunicationRoom\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        create_room_request = {'validFrom': valid_from, 'validUntil': valid_until, 'pstnDialOutEnabled': pstn_dial_out_enabled}
        if participants:
            create_room_request['participants'] = {p.communication_identifier.raw_id: {'role': p.role} for p in participants}
        _SERIALIZER = Serializer()
        repeatability_request_id = str(uuid.uuid1())
        repeatability_first_sent = _SERIALIZER.serialize_data(datetime.utcnow(), 'rfc-1123')
        request_headers = kwargs.pop('headers', {})
        request_headers.update({'Repeatability-Request-Id': repeatability_request_id, 'Repeatability-First-Sent': repeatability_first_sent})
        create_room_response = self._rooms_service_client.rooms.create(create_room_request=create_room_request, headers=request_headers, **kwargs)
        return CommunicationRoom(create_room_response)

    @distributed_trace
    def delete_room(self, room_id: str, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        'Delete room.\n\n        :param room_id: Required. Id of room to be deleted\n        :type room_id: str\n        :returns: None.\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError\n\n        '
        return self._rooms_service_client.rooms.delete(room_id=room_id, **kwargs)

    @distributed_trace
    def update_room(self, *, room_id: str, valid_from: Optional[datetime]=None, valid_until: Optional[datetime]=None, pstn_dial_out_enabled: Optional[bool]=None, **kwargs: Any) -> CommunicationRoom:
        if False:
            return 10
        "Update a valid room's attributes. For any argument that is passed\n        in, the corresponding room property will be replaced with the new value.\n\n        :keyword str room_id: Required. Id of room to be updated\n        :keyword datetime valid_from: The timestamp from when the room is open for joining. Optional.\n        :keyword datetime valid_until: The timestamp from when the room can no longer be joined. Optional.\n        :keyword bool pstn_dial_out_enabled: Set this flag to true if, at the time of the call,\n        dial out to a PSTN number is enabled in a particular room. Optional.\n        :returns: Updated room.\n        :rtype: ~azure.communication.rooms.CommunicationRoom\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n        "
        update_room_request = {'validFrom': valid_from, 'validUntil': valid_until, 'pstnDialOutEnabled': pstn_dial_out_enabled}
        update_room_response = self._rooms_service_client.rooms.update(room_id=room_id, update_room_request=update_room_request, **kwargs)
        return CommunicationRoom(update_room_response)

    @distributed_trace
    def get_room(self, room_id: str, **kwargs) -> CommunicationRoom:
        if False:
            for i in range(10):
                print('nop')
        'Get a valid room\n\n        :param room_id: Required. Id of room to be fetched\n        :type room_id: str\n        :returns: Room with current attributes.\n        :rtype: ~azure.communication.rooms.CommunicationRoom\n        :raises: ~azure.core.exceptions.HttpResponseError\n\n        '
        get_room_response = self._rooms_service_client.rooms.get(room_id=room_id, **kwargs)
        return CommunicationRoom(get_room_response)

    @distributed_trace
    def list_rooms(self, **kwargs) -> ItemPaged[CommunicationRoom]:
        if False:
            print('Hello World!')
        'List all rooms\n\n        :returns: An iterator like instance of CommunicationRoom.\n        :rtype: ~azure.core.paging.ItemPaged[~azure.communication.rooms.CommunicationRoom]\n        :raises: ~azure.core.exceptions.HttpResponseError\n\n        '
        return self._rooms_service_client.rooms.list(cls=lambda rooms: [CommunicationRoom(r) for r in rooms], **kwargs)

    @distributed_trace
    def add_or_update_participants(self, *, room_id: str, participants: List[RoomParticipant], **kwargs) -> None:
        if False:
            while True:
                i = 10
        'Update participants to a room. It looks for the room participants based on their\n        communication identifier and replace the existing participants with the value passed in\n        this API.\n        :keyword str room_id: Required. Id of room to be updated\n        :keyword List[RoomParticipant] participants:\n            Required. Collection of identities invited to be updated\n        :returns: None.\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n        '
        update_participants_request = {'participants': {p.communication_identifier.raw_id: {'role': p.role} for p in participants}}
        self._rooms_service_client.participants.update(room_id=room_id, update_participants_request=update_participants_request, **kwargs)

    @distributed_trace
    def remove_participants(self, *, room_id: str, participants: List[Union[RoomParticipant, CommunicationIdentifier]], **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        'Remove participants from a room\n        :keyword str room_id: Required. Id of room to be updated\n        :keyword List[Union[RoomParticipant, CommunicationIdentifier]] participants:\n            Required. Collection of identities to be removed from the room.\n        :returns: None.\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n        '
        remove_participants_request = {'participants': {}}
        for participant in participants:
            try:
                remove_participants_request['participants'][participant.communication_identifier.raw_id] = None
            except AttributeError:
                remove_participants_request['participants'][participant.raw_id] = None
        self._rooms_service_client.participants.update(room_id=room_id, update_participants_request=remove_participants_request, **kwargs)

    @distributed_trace
    def list_participants(self, room_id: str, **kwargs) -> ItemPaged[RoomParticipant]:
        if False:
            return 10
        'Get participants of a room\n        :param room_id: Required. Id of room whose participants to be fetched.\n        :type room_id: str\n        :returns: An iterator like instance of RoomParticipant.\n        :rtype: ~azure.core.paging.ItemPaged[~azure.communication.rooms.RoomParticipant]\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        return self._rooms_service_client.participants.list(room_id=room_id, cls=lambda objs: [RoomParticipant(x) for x in objs], **kwargs)

    def __enter__(self) -> 'RoomsClient':
        if False:
            while True:
                i = 10
        self._rooms_service_client.__enter__()
        return self

    def __exit__(self, *args: 'Any') -> None:
        if False:
            for i in range(10):
                print('nop')
        self.close()

    def close(self) -> None:
        if False:
            print('Hello World!')
        'Close the :class:\n        `~azure.communication.rooms.RoomsClient` session.\n        '
        self._rooms_service_client.__exit__()