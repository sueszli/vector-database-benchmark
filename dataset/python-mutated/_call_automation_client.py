from typing import List, Union, Optional, TYPE_CHECKING, Iterable, Dict, overload
from urllib.parse import urlparse
import warnings
from azure.core.tracing.decorator import distributed_trace
from ._version import SDK_MONIKER
from ._api_versions import DEFAULT_VERSION
from ._call_connection_client import CallConnectionClient
from ._generated._client import AzureCommunicationCallAutomationService
from ._shared.auth_policy_utils import get_authentication_policy
from ._shared.utils import parse_connection_str
from ._credential.call_automation_auth_policy_utils import get_call_automation_auth_policy
from ._credential.credential_utils import get_custom_enabled, get_custom_url
from ._generated.models import CreateCallRequest, AnswerCallRequest, RedirectCallRequest, RejectCallRequest, StartCallRecordingRequest, CustomContext
from ._models import CallConnectionProperties, RecordingProperties, ChannelAffinity, CallInvite
from ._content_downloader import ContentDownloader
from ._utils import serialize_phone_identifier, serialize_identifier, serialize_communication_user_identifier, build_call_locator, process_repeatability_first_sent
if TYPE_CHECKING:
    from ._models import ServerCallLocator, GroupCallLocator, MediaStreamingConfiguration
    from azure.core.credentials import TokenCredential, AzureKeyCredential
    from ._shared.models import CommunicationIdentifier, CommunicationUserIdentifier, PhoneNumberIdentifier
    from ._generated.models._enums import CallRejectReason, RecordingContent, RecordingChannel, RecordingFormat, RecordingStorage

class CallAutomationClient:
    """A client to interact with the AzureCommunicationService CallAutomation service.
    Call Automation provides developers the ability to build server-based,
    intelligent call workflows, and call recording for voice and PSTN channels.

    :param endpoint: The endpoint of the Azure Communication resource.
    :type endpoint: str
    :param credential: The access key we use to authenticate against the service.
    :type credential: ~azure.core.credentials.TokenCredential
     or ~azure.core.credentials.AzureKeyCredential
    :keyword api_version: Azure Communication Call Automation API version.
    :paramtype api_version: str
    :keyword source: ACS User Identity to be used when the call is created or answered.
     If not provided, service will generate one.
    :paramtype source: ~azure.communication.callautomation.CommunicationUserIdentifier
    """

    def __init__(self, endpoint: str, credential: Union['TokenCredential', 'AzureKeyCredential'], *, api_version: Optional[str]=None, source: Optional['CommunicationUserIdentifier']=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        if not credential:
            raise ValueError('credential can not be None')
        try:
            if not endpoint.lower().startswith('http'):
                endpoint = 'https://' + endpoint
        except AttributeError:
            raise ValueError('Host URL must be a string')
        parsed_url = urlparse(endpoint.rstrip('/'))
        if not parsed_url.netloc:
            raise ValueError(f'Invalid URL: {format(endpoint)}')
        custom_enabled = get_custom_enabled()
        custom_url = get_custom_url()
        if custom_enabled and custom_url is not None:
            self._client = AzureCommunicationCallAutomationService(custom_url, credential, api_version=api_version or DEFAULT_VERSION, authentication_policy=get_call_automation_auth_policy(custom_url, credential, acs_url=endpoint), sdk_moniker=SDK_MONIKER, **kwargs)
        else:
            self._client = AzureCommunicationCallAutomationService(endpoint, credential, api_version=api_version or DEFAULT_VERSION, authentication_policy=get_authentication_policy(endpoint, credential), sdk_moniker=SDK_MONIKER, **kwargs)
        self._call_recording_client = self._client.call_recording
        self._downloader = ContentDownloader(self._call_recording_client)
        self.source = source

    @classmethod
    def from_connection_string(cls, conn_str: str, **kwargs) -> 'CallAutomationClient':
        if False:
            i = 10
            return i + 15
        'Create CallAutomation client from a Connection String.\n\n        :param conn_str: A connection string to an Azure Communication Service resource.\n        :type conn_str: str\n        :return: CallAutomationClient\n        :rtype: ~azure.communication.callautomation.CallAutomationClient\n        '
        (endpoint, access_key) = parse_connection_str(conn_str)
        return cls(endpoint, access_key, **kwargs)

    def get_call_connection(self, call_connection_id: str, **kwargs) -> CallConnectionClient:
        if False:
            i = 10
            return i + 15
        ' Get CallConnectionClient object.\n        Interact with ongoing call with CallConnectionClient.\n\n        :param call_connection_id: CallConnectionId of ongoing call.\n        :type call_connection_id: str\n        :return: CallConnectionClient\n        :rtype: ~azure.communication.callautomation.CallConnectionClient\n        '
        if not call_connection_id:
            raise ValueError('call_connection_id can not be None')
        return CallConnectionClient._from_callautomation_client(callautomation_client=self._client, call_connection_id=call_connection_id, **kwargs)

    @distributed_trace
    def create_call(self, target_participant: Union['CommunicationIdentifier', List['CommunicationIdentifier']], callback_url: str, *, source_caller_id_number: Optional['PhoneNumberIdentifier']=None, source_display_name: Optional[str]=None, sip_headers: Optional[Dict[str, str]]=None, voip_headers: Optional[Dict[str, str]]=None, operation_context: Optional[str]=None, media_streaming_configuration: Optional['MediaStreamingConfiguration']=None, azure_cognitive_services_endpoint_url: Optional[str]=None, **kwargs) -> CallConnectionProperties:
        if False:
            return 10
        "Create a call connection request to a target identity.\n\n        :param target_participant: Call invitee's information.\n        :type target_participant: ~azure.communication.callautomation.CommunicationIdentifier\n         or list[~azure.communication.callautomation.CommunicationIdentifier]\n        :param callback_url: The call back url where callback events are sent.\n        :type callback_url: str\n        :keyword operation_context: Value that can be used to track the call and its associated events.\n        :paramtype operation_context: str or None\n        :keyword source_caller_id_number: The source caller Id, a phone number,\n         that's shown to the PSTN participant being invited.\n         Required only when calling a PSTN callee.\n        :paramtype source_caller_id_number: ~azure.communication.callautomation.PhoneNumberIdentifier or None\n        :keyword source_display_name: Display name of the caller.\n        :paramtype source_display_name: str or None\n        :keyword sip_headers: Sip Headers for PSTN Call\n        :paramtype sip_headers: Dict[str, str] or None\n        :keyword voip_headers: Voip Headers for Voip Call\n        :paramtype voip_headers: Dict[str, str] or None\n        :keyword media_streaming_configuration: Media Streaming Configuration.\n        :paramtype media_streaming_configuration: ~azure.communication.callautomation.MediaStreamingConfiguration\n         or None\n        :keyword azure_cognitive_services_endpoint_url:\n         The identifier of the Cognitive Service resource assigned to this call.\n        :paramtype azure_cognitive_services_endpoint_url: str or None\n        :return: CallConnectionProperties\n        :rtype: ~azure.communication.callautomation.CallConnectionProperties\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        if isinstance(target_participant, CallInvite):
            sip_headers = sip_headers or target_participant.sip_headers
            voip_headers = voip_headers or target_participant.voip_headers
            source_caller_id_number = source_caller_id_number or target_participant.source_caller_id_number
            source_display_name = source_display_name or target_participant.source_display_name
            target_participant = target_participant.target
        user_custom_context = None
        if sip_headers or voip_headers:
            user_custom_context = CustomContext(voip_headers=voip_headers, sip_headers=sip_headers)
        try:
            targets = [serialize_identifier(p) for p in target_participant]
        except TypeError:
            targets = [serialize_identifier(target_participant)]
        media_config = media_streaming_configuration.to_generated() if media_streaming_configuration else None
        create_call_request = CreateCallRequest(targets=targets, callback_uri=callback_url, source_caller_id_number=serialize_phone_identifier(source_caller_id_number), source_display_name=source_display_name, source_identity=serialize_communication_user_identifier(self.source), operation_context=operation_context, media_streaming_configuration=media_config, azure_cognitive_services_endpoint_url=azure_cognitive_services_endpoint_url, custom_context=user_custom_context)
        process_repeatability_first_sent(kwargs)
        result = self._client.create_call(create_call_request=create_call_request, **kwargs)
        return CallConnectionProperties._from_generated(result)

    @distributed_trace
    def create_group_call(self, target_participants: List['CommunicationIdentifier'], callback_url: str, *, source_caller_id_number: Optional['PhoneNumberIdentifier']=None, source_display_name: Optional[str]=None, operation_context: Optional[str]=None, sip_headers: Optional[Dict[str, str]]=None, voip_headers: Optional[Dict[str, str]]=None, **kwargs) -> CallConnectionProperties:
        if False:
            i = 10
            return i + 15
        "Create a call connection request to a list of multiple target identities.\n        This will call all targets simultaneously, and whoever answers the call will join the call.\n\n        :param target_participants: A list of targets.\n        :type target_participants: list[~azure.communication.callautomation.CommunicationIdentifier]\n        :param callback_url: The call back url for receiving events.\n        :type callback_url: str\n        :keyword source_caller_id_number: The source caller Id, a phone number,\n         that's shown to the PSTN participant being invited.\n         Required only when calling a PSTN callee.\n        :paramtype source_caller_id_number: ~azure.communication.callautomation.PhoneNumberIdentifier\n        :keyword source_display_name: Display name of the caller.\n        :paramtype source_display_name: str\n        :keyword operation_context: Value that can be used to track the call and its associated events.\n        :paramtype operation_context: str\n        :keyword sip_headers: Sip Headers for PSTN Call\n        :paramtype sip_headers: Dict[str, str]\n        :keyword voip_headers: Voip Headers for Voip Call\n        :paramtype voip_headers: Dict[str, str]\n        :return: CallConnectionProperties\n        :rtype: ~azure.communication.callautomation.CallConnectionProperties\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        warnings.warn("The method 'create_group_call' is deprecated. Please use 'create_call' instead.", DeprecationWarning)
        return self.create_call(target_participant=target_participants, callback_url=callback_url, source_caller_id_number=source_caller_id_number, source_display_name=source_display_name, sip_headers=sip_headers, voip_headers=voip_headers, operation_context=operation_context, **kwargs)

    @distributed_trace
    def answer_call(self, incoming_call_context: str, callback_url: str, *, media_streaming_configuration: Optional['MediaStreamingConfiguration']=None, azure_cognitive_services_endpoint_url: Optional[str]=None, operation_context: Optional[str]=None, **kwargs) -> CallConnectionProperties:
        if False:
            while True:
                i = 10
        "Answer incoming call with Azure Communication Service's IncomingCall event\n        Retrieving IncomingCall event can be set on Azure Communication Service's Azure Portal.\n\n        :param incoming_call_context: This can be read from body of IncomingCall event.\n         Use this value to answer incoming call.\n        :type incoming_call_context: str\n        :param callback_url: The call back url for receiving events.\n        :type callback_url: str\n        :keyword media_streaming_configuration: Media Streaming Configuration.\n        :paramtype media_streaming_configuration: ~azure.communication.callautomation.MediaStreamingConfiguration\n        :keyword azure_cognitive_services_endpoint_url:\n         The endpoint url of the Azure Cognitive Services resource attached.\n        :paramtype azure_cognitive_services_endpoint_url: str\n        :keyword operation_context: The operation context.\n        :paramtype operation_context: str\n        :return: CallConnectionProperties\n        :rtype: ~azure.communication.callautomation.CallConnectionProperties\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        answer_call_request = AnswerCallRequest(incoming_call_context=incoming_call_context, callback_uri=callback_url, media_streaming_configuration=media_streaming_configuration.to_generated() if media_streaming_configuration else None, azure_cognitive_services_endpoint_url=azure_cognitive_services_endpoint_url, answered_by_identifier=serialize_communication_user_identifier(self.source) if self.source else None, operation_context=operation_context)
        process_repeatability_first_sent(kwargs)
        result = self._client.answer_call(answer_call_request=answer_call_request, **kwargs)
        return CallConnectionProperties._from_generated(result)

    @distributed_trace
    def redirect_call(self, incoming_call_context: str, target_participant: 'CommunicationIdentifier', *, sip_headers: Optional[Dict[str, str]]=None, voip_headers: Optional[Dict[str, str]]=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        'Redirect incoming call to a specific target.\n\n        :param incoming_call_context: This can be read from body of IncomingCall event.\n         Use this value to redirect incoming call.\n        :type incoming_call_context: str\n        :param target_participant: The target identity to redirect the call to.\n        :type target_participant: ~azure.communication.callautomation.CommunicationIdentifier\n        :keyword sip_headers: Sip Headers for PSTN Call\n        :paramtype sip_headers: Dict[str, str] or None\n        :keyword voip_headers: Voip Headers for Voip Call\n        :paramtype voip_headers: Dict[str, str] or None\n        :return: None\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        if isinstance(target_participant, CallInvite):
            sip_headers = sip_headers or target_participant.sip_headers
            voip_headers = voip_headers or target_participant.voip_headers
            target_participant = target_participant.target
        user_custom_context = None
        if sip_headers or voip_headers:
            user_custom_context = CustomContext(voip_headers=voip_headers, sip_headers=sip_headers)
        process_repeatability_first_sent(kwargs)
        redirect_call_request = RedirectCallRequest(incoming_call_context=incoming_call_context, target=serialize_identifier(target_participant), custom_context=user_custom_context)
        self._client.redirect_call(redirect_call_request=redirect_call_request, **kwargs)

    @distributed_trace
    def reject_call(self, incoming_call_context: str, *, call_reject_reason: Optional[Union[str, 'CallRejectReason']]=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        'Reject incoming call.\n\n        :param incoming_call_context: This can be read from body of IncomingCall event.\n         Use this value to reject incoming call.\n        :type incoming_call_context: str\n        :keyword call_reject_reason: The rejection reason.\n        :paramtype call_reject_reason: str or ~azure.communication.callautomation.CallRejectReason\n        :return: None\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseErrorr:\n        '
        reject_call_request = RejectCallRequest(incoming_call_context=incoming_call_context, call_reject_reason=call_reject_reason)
        process_repeatability_first_sent(kwargs)
        self._client.reject_call(reject_call_request=reject_call_request, **kwargs)

    @overload
    def start_recording(self, *, server_call_id: str, recording_state_callback_url: Optional[str]=None, recording_content_type: Optional[Union[str, 'RecordingContent']]=None, recording_channel_type: Optional[Union[str, 'RecordingChannel']]=None, recording_format_type: Optional[Union[str, 'RecordingFormat']]=None, pause_on_start: Optional[bool]=None, audio_channel_participant_ordering: Optional[List['CommunicationIdentifier']]=None, recording_storage_type: Optional[Union[str, 'RecordingStorage']]=None, channel_affinity: Optional[List['ChannelAffinity']]=None, external_storage_location: Optional[str]=None, **kwargs) -> RecordingProperties:
        if False:
            for i in range(10):
                print('nop')
        "Start recording for a ongoing call. Locate the call with call locator.\n\n        :keyword str server_call_id: The server call ID to locate ongoing call.\n        :keyword recording_state_callback_url: The url to send notifications to.\n        :paramtype recording_state_callback_url: str or None\n        :keyword recording_content_type: The content type of call recording.\n        :paramtype recording_content_type: str or ~azure.communication.callautomation.RecordingContent or None\n        :keyword recording_channel_type: The channel type of call recording.\n        :paramtype recording_channel_type: str or ~azure.communication.callautomation.RecordingChannel or None\n        :keyword recording_format_type: The format type of call recording.\n        :paramtype recording_format_type: str or ~azure.communication.callautomation.RecordingFormat or None\n        :keyword pause_on_start: The state of the pause on start option.\n        :paramtype pause_on_start: bool or None\n        :keyword audio_channel_participant_ordering:\n         The sequential order in which audio channels are assigned to participants in the unmixed recording.\n         When 'recordingChannelType' is set to 'unmixed' and `audioChannelParticipantOrdering is not specified,\n         the audio channel to participant mapping will be automatically assigned based on the order in\n         which participant first audio was detected.\n         Channel to participant mapping details can be found in the metadata of the recording.\n        :paramtype audio_channel_participant_ordering:\n         list[~azure.communication.callautomation.CommunicationIdentifier] or None\n        :keyword recording_storage_type: Recording storage mode.\n         ``External`` enables bring your own storage.\n        :paramtype recording_storage_type: str or None\n        :keyword channel_affinity: The channel affinity of call recording\n         When 'recordingChannelType' is set to 'unmixed', if channelAffinity is not specified,\n         'channel' will be automatically assigned.\n         Channel-Participant mapping details can be found in the metadata of the recording.\n        :paramtype channel_affinity: list[~azure.communication.callautomation.ChannelAffinity] or None\n        :keyword external_storage_location: The location where recording is stored,\n         when RecordingStorageType is set to 'BlobStorage'.\n        :paramtype external_storage_location: str or ~azure.communication.callautomation.RecordingStorage or None\n        :return: RecordingProperties\n        :rtype: ~azure.communication.callautomation.RecordingProperties\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "

    @overload
    def start_recording(self, *, group_call_id: str, recording_state_callback_url: Optional[str]=None, recording_content_type: Optional[Union[str, 'RecordingContent']]=None, recording_channel_type: Optional[Union[str, 'RecordingChannel']]=None, recording_format_type: Optional[Union[str, 'RecordingFormat']]=None, pause_on_start: Optional[bool]=None, audio_channel_participant_ordering: Optional[List['CommunicationIdentifier']]=None, recording_storage_type: Optional[Union[str, 'RecordingStorage']]=None, channel_affinity: Optional[List['ChannelAffinity']]=None, external_storage_location: Optional[str]=None, **kwargs) -> RecordingProperties:
        if False:
            i = 10
            return i + 15
        "Start recording for a ongoing call. Locate the call with call locator.\n\n        :keyword str group_call_id: The group call ID to locate ongoing call.\n        :keyword recording_state_callback_url: The url to send notifications to.\n        :paramtype recording_state_callback_url: str or None\n        :keyword recording_content_type: The content type of call recording.\n        :paramtype recording_content_type: str or ~azure.communication.callautomation.RecordingContent or None\n        :keyword recording_channel_type: The channel type of call recording.\n        :paramtype recording_channel_type: str or ~azure.communication.callautomation.RecordingChannel or None\n        :keyword recording_format_type: The format type of call recording.\n        :paramtype recording_format_type: str or ~azure.communication.callautomation.RecordingFormat or None\n        :keyword pause_on_start: The state of the pause on start option.\n        :paramtype pause_on_start: bool or None\n        :keyword audio_channel_participant_ordering:\n         The sequential order in which audio channels are assigned to participants in the unmixed recording.\n         When 'recordingChannelType' is set to 'unmixed' and `audioChannelParticipantOrdering is not specified,\n         the audio channel to participant mapping will be automatically assigned based on the order in\n         which participant first audio was detected.\n         Channel to participant mapping details can be found in the metadata of the recording.\n        :paramtype audio_channel_participant_ordering:\n         list[~azure.communication.callautomation.CommunicationIdentifier] or None\n        :keyword recording_storage_type: Recording storage mode.\n         ``External`` enables bring your own storage.\n        :paramtype recording_storage_type: str or None\n        :keyword channel_affinity: The channel affinity of call recording\n         When 'recordingChannelType' is set to 'unmixed', if channelAffinity is not specified,\n         'channel' will be automatically assigned.\n         Channel-Participant mapping details can be found in the metadata of the recording.\n        :paramtype channel_affinity: list[~azure.communication.callautomation.ChannelAffinity] or None\n        :keyword external_storage_location: The location where recording is stored,\n         when RecordingStorageType is set to 'BlobStorage'.\n        :paramtype external_storage_location: str or ~azure.communication.callautomation.RecordingStorage or None\n        :return: RecordingProperties\n        :rtype: ~azure.communication.callautomation.RecordingProperties\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "

    @distributed_trace
    def start_recording(self, *args: Union['ServerCallLocator', 'GroupCallLocator'], **kwargs) -> RecordingProperties:
        if False:
            print('Hello World!')
        channel_affinity: List[ChannelAffinity] = kwargs.pop('channel_affinity', None) or []
        channel_affinity_internal = [c._to_generated() for c in channel_affinity]
        call_locator = build_call_locator(args, kwargs.pop('call_locator', None), kwargs.pop('server_call_id', None), kwargs.pop('group_call_id', None))
        start_recording_request = StartCallRecordingRequest(call_locator=call_locator, recording_state_callback_uri=kwargs.pop('recording_state_callback_url', None), recording_content_type=kwargs.pop('recording_content_type', None), recording_channel_type=kwargs.pop('recording_channel_type', None), recording_format_type=kwargs.pop('recording_format_type', None), pause_on_start=kwargs.pop('pause_on_start', None), audio_channel_participant_ordering=kwargs.pop('audio_channel_participant_ordering', None), recording_storage_type=kwargs.pop('recording_storage_type', None), external_storage_location=kwargs.pop('external_storage_location', None), channel_affinity=channel_affinity_internal)
        process_repeatability_first_sent(kwargs)
        recording_state_result = self._call_recording_client.start_recording(start_call_recording=start_recording_request, **kwargs)
        return RecordingProperties._from_generated(recording_state_result)

    @distributed_trace
    def stop_recording(self, recording_id: str, **kwargs) -> None:
        if False:
            print('Hello World!')
        'Stop recording the call.\n\n        :param recording_id: The recording id.\n        :type recording_id: str\n        :return: None\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        self._call_recording_client.stop_recording(recording_id=recording_id, **kwargs)

    @distributed_trace
    def pause_recording(self, recording_id: str, **kwargs) -> None:
        if False:
            while True:
                i = 10
        'Pause recording the call.\n\n        :param recording_id: The recording id.\n        :type recording_id: str\n        :return: None\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        self._call_recording_client.pause_recording(recording_id=recording_id, **kwargs)

    @distributed_trace
    def resume_recording(self, recording_id: str, **kwargs) -> None:
        if False:
            print('Hello World!')
        'Resume recording the call.\n\n        :param recording_id: The recording id.\n        :type recording_id: str\n        :return: None\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        self._call_recording_client.resume_recording(recording_id=recording_id, **kwargs)

    @distributed_trace
    def get_recording_properties(self, recording_id: str, **kwargs) -> RecordingProperties:
        if False:
            for i in range(10):
                print('nop')
        'Get call recording properties and its state.\n\n        :param recording_id: The recording id.\n        :type recording_id: str\n        :return: RecordingProperties\n        :rtype: ~azure.communication.callautomation.RecordingProperties\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        recording_state_result = self._call_recording_client.get_recording_properties(recording_id=recording_id, **kwargs)
        return RecordingProperties._from_generated(recording_state_result)

    @distributed_trace
    def download_recording(self, recording_url: str, *, offset: int=None, length: int=None, **kwargs) -> Iterable[bytes]:
        if False:
            print('Hello World!')
        "Download a stream of the call recording.\n\n        :param recording_url: Recording's url to be downloaded\n        :type recording_url: str\n        :keyword offset: If provided, only download the bytes of the content in the specified range.\n         Offset of starting byte.\n        :paramtype offset: int\n        :keyword length: If provided, only download the bytes of the content in the specified range.\n         Length of the bytes to be downloaded.\n        :paramtype length: int\n        :return: Iterable[bytes]\n        :rtype: Iterable[bytes]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        stream = self._downloader.download_streaming(source_location=recording_url, offset=offset, length=length, **kwargs)
        return stream

    @distributed_trace
    def delete_recording(self, recording_url: str, **kwargs) -> None:
        if False:
            print('Hello World!')
        "Delete a call recording from given recording url.\n\n        :param recording_url: Recording's url.\n        :type recording_url: str\n        :return: None\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        self._downloader.delete_recording(recording_location=recording_url, **kwargs)

    def __enter__(self) -> 'CallAutomationClient':
        if False:
            while True:
                i = 10
        self._client.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if False:
            print('Hello World!')
        self.close()

    def close(self) -> None:
        if False:
            print('Hello World!')
        self._client.__exit__()