from typing import TYPE_CHECKING, Optional, List, Union, Dict
from urllib.parse import urlparse
import warnings
from typing_extensions import Literal
from azure.core.paging import ItemPaged
from azure.core.tracing.decorator import distributed_trace
from ._version import SDK_MONIKER
from ._api_versions import DEFAULT_VERSION
from ._utils import serialize_phone_identifier, serialize_identifier, process_repeatability_first_sent
from ._models import CallParticipant, CallConnectionProperties, AddParticipantResult, RemoveParticipantResult, TransferCallResult, MuteParticipantsResult, CallInvite, CancelAddParticipantResult
from ._generated._client import AzureCommunicationCallAutomationService
from ._generated.models import AddParticipantRequest, RemoveParticipantRequest, TransferToParticipantRequest, PlayRequest, RecognizeRequest, ContinuousDtmfRecognitionRequest, SendDtmfRequest, CustomContext, DtmfOptions, SpeechOptions, PlayOptions, RecognizeOptions, MuteParticipantsRequest, CancelAddParticipantRequest, StartHoldMusicRequest, StopHoldMusicRequest
from ._generated.models._enums import RecognizeInputType
from ._shared.auth_policy_utils import get_authentication_policy
from ._shared.utils import parse_connection_str
from ._credential.call_automation_auth_policy_utils import get_call_automation_auth_policy
from ._credential.credential_utils import get_custom_enabled, get_custom_url
if TYPE_CHECKING:
    from ._call_automation_client import CallAutomationClient
    from ._generated.models._enums import DtmfTone
    from ._shared.models import PhoneNumberIdentifier, CommunicationIdentifier
    from ._models import FileSource, TextSource, SsmlSource, Choice
    from azure.core.credentials import TokenCredential, AzureKeyCredential
MediaSources = Union['FileSource', 'TextSource', 'SsmlSource']

class CallConnectionClient:
    """A client to interact with an ongoing call. This client can be used to do mid-call actions,
    such as Transfer and Play Media. Call must be established to perform these actions.

    :param endpoint: The endpoint of the Azure Communication resource.
    :type endpoint: str
    :param credential: The credentials with which to authenticate.
    :type credential: ~azure.core.credentials.TokenCredential
     or ~azure.core.credentials.AzureKeyCredential
    :param call_connection_id: Call Connection ID of ongoing call.
    :type call_connection_id: str
    :keyword api_version: Azure Communication Call Automation API version.
    :paramtype api_version: str
    """

    def __init__(self, endpoint: str, credential: Union['TokenCredential', 'AzureKeyCredential'], call_connection_id: str, *, api_version: Optional[str]=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        call_automation_client = kwargs.get('_callautomation_client', None)
        if call_automation_client is None:
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
        else:
            self._client = call_automation_client
        self._call_connection_id = call_connection_id
        self._call_connection_client = self._client.call_connection
        self._call_media_client = self._client.call_media

    @classmethod
    def from_connection_string(cls, conn_str: str, call_connection_id: str, **kwargs) -> 'CallConnectionClient':
        if False:
            while True:
                i = 10
        'Create CallConnectionClient from a Connection String.\n\n        :param conn_str: A connection string to an Azure Communication Service resource.\n        :type conn_str: str\n        :param call_connection_id: Call Connection Id of ongoing call.\n        :type call_connection_id: str\n        :return: CallConnectionClient\n        :rtype: ~azure.communication.callautomation.CallConnectionClient\n        '
        (endpoint, access_key) = parse_connection_str(conn_str)
        return cls(endpoint, access_key, call_connection_id, **kwargs)

    @classmethod
    def _from_callautomation_client(cls, callautomation_client: 'CallAutomationClient', call_connection_id: str) -> 'CallConnectionClient':
        if False:
            while True:
                i = 10
        'Internal constructor for sharing the pipeline with CallAutomationClient.\n\n        :param callautomation_client: An existing callautomation client.\n        :type callautomation_client: ~azure.communication.callautomation.CallAutomationClient\n        :param call_connection_id: Call Connection Id of ongoing call.\n        :type call_connection_id: str\n        :return: CallConnectionClient\n        :rtype: ~azure.communication.callautomation.CallConnectionClient\n        '
        return cls(None, None, call_connection_id, _callautomation_client=callautomation_client)

    @distributed_trace
    def get_call_properties(self, **kwargs) -> CallConnectionProperties:
        if False:
            return 10
        'Get the latest properties of this call.\n\n        :return: CallConnectionProperties\n        :rtype: ~azure.communication.callautomation.CallConnectionProperties\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        call_properties = self._call_connection_client.get_call(call_connection_id=self._call_connection_id, **kwargs)
        return CallConnectionProperties._from_generated(call_properties)

    @distributed_trace
    def hang_up(self, is_for_everyone: bool, **kwargs) -> None:
        if False:
            print('Hello World!')
        'Hangup this call.\n\n        :param is_for_everyone: Determine if this call should be ended for all participants.\n        :type is_for_everyone: bool\n        :return: None\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        if is_for_everyone:
            process_repeatability_first_sent(kwargs)
            self._call_connection_client.terminate_call(self._call_connection_id, **kwargs)
        else:
            self._call_connection_client.hangup_call(self._call_connection_id, **kwargs)

    @distributed_trace
    def get_participant(self, target_participant: 'CommunicationIdentifier', **kwargs) -> 'CallParticipant':
        if False:
            i = 10
            return i + 15
        'Get details of a participant in this call.\n\n        :param target_participant: The participant to retrieve.\n        :type target_participant: ~azure.communication.callautomation.CommunicationIdentifier\n        :return: CallParticipant\n        :rtype: ~azure.communication.callautomation.CallParticipant\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        participant = self._call_connection_client.get_participant(self._call_connection_id, target_participant.raw_id, **kwargs)
        return CallParticipant._from_generated(participant)

    @distributed_trace
    def list_participants(self, **kwargs) -> ItemPaged[CallParticipant]:
        if False:
            i = 10
            return i + 15
        'List all participants in this call.\n\n        :return: An iterable of CallParticipant\n        :rtype: ~azure.core.paging.ItemPaged[azure.communication.callautomation.CallParticipant]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        return self._call_connection_client.get_participants(self._call_connection_id, cls=lambda participants: [CallParticipant._from_generated(p) for p in participants], **kwargs)

    @distributed_trace
    def transfer_call_to_participant(self, target_participant: 'CommunicationIdentifier', *, sip_headers: Optional[Dict[str, str]]=None, voip_headers: Optional[Dict[str, str]]=None, operation_context: Optional[str]=None, callback_url: Optional[str]=None, transferee: Optional['CommunicationIdentifier']=None, **kwargs) -> TransferCallResult:
        if False:
            for i in range(10):
                print('nop')
        'Transfer this call to another participant.\n\n        :param target_participant: The transfer target.\n        :type target_participant: CommunicationIdentifier\n        :keyword sip_headers: Custom context for PSTN\n        :paramtype sip_headers: dict[str, str]\n        :keyword voip_headers: Custom context for VOIP\n        :paramtype voip_headers: dict[str, str]\n        :keyword operation_context: Value that can be used to track this call and its associated events.\n        :paramtype operation_context: str\n        :keyword callback_url: Url that overrides original callback URI for this request.\n        :paramtype callback_url: str\n        :keyword transferee: Url that overrides original callback URI for this request.\n        :paramtype transferee: ~azure.communication.callautomation.CommunicationIdentifier\n        :return: TransferCallResult\n        :rtype: ~azure.communication.callautomation.TransferCallResult\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        user_custom_context = CustomContext(voip_headers=voip_headers, sip_headers=sip_headers) if sip_headers or voip_headers else None
        request = TransferToParticipantRequest(target_participant=serialize_identifier(target_participant), custom_context=user_custom_context, operation_context=operation_context, callback_uri=callback_url)
        process_repeatability_first_sent(kwargs)
        if transferee:
            request.transferee = serialize_identifier(transferee)
        result = self._call_connection_client.transfer_to_participant(self._call_connection_id, request, **kwargs)
        return TransferCallResult._from_generated(result)

    @distributed_trace
    def add_participant(self, target_participant: 'CommunicationIdentifier', *, invitation_timeout: Optional[int]=None, operation_context: Optional[str]=None, sip_headers: Optional[Dict[str, str]]=None, voip_headers: Optional[Dict[str, str]]=None, source_caller_id_number: Optional['PhoneNumberIdentifier']=None, source_display_name: Optional[str]=None, callback_url: Optional[str]=None, **kwargs) -> AddParticipantResult:
        if False:
            print('Hello World!')
        "Add a participant to this call.\n\n        :param target_participant: The participant being added.\n        :type target_participant: ~azure.communication.callautomation.CommunicationIdentifier\n        :keyword invitation_timeout: Timeout to wait for the invited participant to pickup.\n         The maximum value of this is 180 seconds.\n        :paramtype invitation_timeout: int or None\n        :keyword operation_context: Value that can be used to track this call and its associated events.\n        :paramtype operation_context: str or None\n        :keyword sip_headers: Sip Headers for PSTN Call\n        :paramtype sip_headers: Dict[str, str] or None\n        :keyword voip_headers: Voip Headers for Voip Call\n        :paramtype voip_headers: Dict[str, str] or None\n        :keyword source_caller_id_number: The source caller Id, a phone number,\n         that's shown to the PSTN participant being invited.\n         Required only when calling a PSTN callee.\n        :paramtype source_caller_id_number: ~azure.communication.callautomation.PhoneNumberIdentifier or None\n        :keyword source_display_name: Display name of the caller.\n        :paramtype source_display_name: str or None\n        :keyword callback_url: Url that overrides original callback URI for this request.\n        :paramtype callback_url: str or None\n        :return: AddParticipantResult\n        :rtype: ~azure.communication.callautomation.AddParticipantResult\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        if isinstance(target_participant, CallInvite):
            sip_headers = sip_headers or target_participant.sip_headers
            voip_headers = voip_headers or target_participant.voip_headers
            source_caller_id_number = source_caller_id_number or target_participant.source_caller_id_number
            source_display_name = source_display_name or target_participant.source_display_name
            target_participant = target_participant.target
        user_custom_context = None
        if sip_headers or voip_headers:
            user_custom_context = CustomContext(voip_headers=voip_headers, sip_headers=sip_headers)
        add_participant_request = AddParticipantRequest(participant_to_add=serialize_identifier(target_participant), source_caller_id_number=serialize_phone_identifier(source_caller_id_number), source_display_name=source_display_name, custom_context=user_custom_context, invitation_timeout_in_seconds=invitation_timeout, operation_context=operation_context, callback_uri=callback_url)
        process_repeatability_first_sent(kwargs)
        response = self._call_connection_client.add_participant(self._call_connection_id, add_participant_request, **kwargs)
        return AddParticipantResult._from_generated(response)

    @distributed_trace
    def remove_participant(self, target_participant: 'CommunicationIdentifier', *, operation_context: Optional[str]=None, callback_url: Optional[str]=None, **kwargs) -> RemoveParticipantResult:
        if False:
            while True:
                i = 10
        'Remove a participant from this call.\n\n        :param  target_participant: The participant being removed.\n        :type target_participant: ~azure.communication.callautomation.CommunicationIdentifier\n        :keyword operation_context: Value that can be used to track this call and its associated events.\n        :paramtype operation_context: str\n        :keyword callback_url: Url that overrides original callback URI for this request.\n        :paramtype callback_url: str\n        :return: RemoveParticipantResult\n        :rtype: ~azure.communication.callautomation.RemoveParticipantResult\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        remove_participant_request = RemoveParticipantRequest(participant_to_remove=serialize_identifier(target_participant), operation_context=operation_context, callback_uri=callback_url)
        process_repeatability_first_sent(kwargs)
        response = self._call_connection_client.remove_participant(self._call_connection_id, remove_participant_request, **kwargs)
        return RemoveParticipantResult._from_generated(response)

    @distributed_trace
    def play_media(self, play_source: Union[MediaSources, List[MediaSources]], play_to: Union[Literal['all'], List['CommunicationIdentifier']]='all', *, loop: bool=False, operation_context: Optional[str]=None, callback_url: Optional[str]=None, **kwargs) -> None:
        if False:
            return 10
        "Play media to specific participant(s) in this call.\n\n        :param play_source: A PlaySource representing the source to play.\n        :type play_source: ~azure.communication.callautomation.FileSource or\n         ~azure.communication.callautomation.TextSource or\n         ~azure.communication.callautomation.SsmlSource or\n         list[~azure.communication.callautomation.FileSource or\n          ~azure.communication.callautomation.TextSource or\n          ~azure.communication.callautomation.SsmlSource]\n        :param play_to: The targets to play media to. Default value is 'all', to play media\n         to all participants in the call.\n        :type play_to: list[~azure.communication.callautomation.CommunicationIdentifier]\n        :keyword loop: Whether the media should be repeated until cancelled.\n        :paramtype loop: bool\n        :keyword operation_context: Value that can be used to track this call and its associated events.\n        :paramtype operation_context: str or None\n        :keyword callback_url: Url that overrides original callback URI for this request.\n        :paramtype callback_url: str\n        :return: None\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        play_source_single: Optional[MediaSources] = None
        if isinstance(play_source, list):
            if play_source:
                play_source_single = play_source[0]
        else:
            play_source_single = play_source
        audience = [] if play_to == 'all' else [serialize_identifier(i) for i in play_to]
        play_request = PlayRequest(play_source_info=play_source_single._to_generated(), play_to=audience, play_options=PlayOptions(loop=loop), operation_context=operation_context, callback_uri=callback_url, **kwargs)
        self._call_media_client.play(self._call_connection_id, play_request)

    @distributed_trace
    def play_media_to_all(self, play_source: Union['FileSource', List['FileSource']], *, loop: bool=False, operation_context: Optional[str]=None, callback_url: Optional[str]=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        'Play media to all participants in this call.\n\n        :param play_source: A PlaySource representing the source to play.\n        :type play_source: ~azure.communication.callautomation.FileSource or\n         list[~azure.communication.callautomation.FileSource]\n        :keyword loop: Whether the media should be repeated until cancelled.\n        :paramtype loop: bool\n        :keyword operation_context: Value that can be used to track this call and its associated events.\n        :paramtype operation_context: str or None\n        :keyword callback_url: Url that overrides original callback URI for this request.\n        :paramtype callback_url: str\n        :return: None\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        warnings.warn("The method 'play_media_to_all' is deprecated. Please use 'play_media' instead.", DeprecationWarning)
        self.play_media(play_source=play_source, loop=loop, operation_context=operation_context, callback_url=callback_url, **kwargs)

    @distributed_trace
    def start_recognizing_media(self, input_type: Union[str, 'RecognizeInputType'], target_participant: 'CommunicationIdentifier', *, initial_silence_timeout: Optional[int]=None, play_prompt: Optional[Union[MediaSources, List[MediaSources]]]=None, interrupt_call_media_operation: bool=False, operation_context: Optional[str]=None, interrupt_prompt: bool=False, dtmf_inter_tone_timeout: Optional[int]=None, dtmf_max_tones_to_collect: Optional[str]=None, dtmf_stop_tones: Optional[List[str or 'DtmfTone']]=None, choices: Optional[List['Choice']]=None, end_silence_timeout_in_ms: Optional[int]=None, speech_recognition_model_endpoint_id: Optional[str]=None, callback_url: Optional[str]=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Recognize tones from specific participant in this call.\n\n        :param input_type: Determines the type of the recognition.\n        :type input_type: str or ~azure.communication.callautomation.RecognizeInputType\n        :param target_participant: Target participant of DTMF tone recognition.\n        :type target_participant: ~azure.communication.callautomation.CommunicationIdentifier\n        :keyword initial_silence_timeout: Time to wait for first input after prompt in seconds (if any).\n        :paramtype initial_silence_timeout: int\n        :keyword play_prompt: The source of the audio to be played for recognition.\n        :paramtype play_prompt: ~azure.communication.callautomation.FileSource or\n         ~azure.communication.callautomation.TextSource or\n         ~azure.communication.callautomation.SsmlSource or\n         list[~azure.communication.callautomation.FileSource or\n          ~azure.communication.callautomation.TextSource or\n          ~azure.communication.callautomation.SsmlSource]\n        :keyword interrupt_call_media_operation:\n         If set recognize can barge into other existing queued-up/currently-processing requests.\n        :paramtype interrupt_call_media_operation: bool\n        :keyword operation_context: Value that can be used to track this call and its associated events.\n        :paramtype operation_context: str\n        :keyword interrupt_prompt: Determines if we interrupt the prompt and start recognizing.\n        :paramtype interrupt_prompt: bool\n        :keyword dtmf_inter_tone_timeout: Time to wait between DTMF inputs to stop recognizing.\n        :paramtype dtmf_inter_tone_timeout: int\n        :keyword dtmf_max_tones_to_collect: Maximum number of DTMF tones to be collected.\n        :paramtype dtmf_max_tones_to_collect: int\n        :keyword dtmf_stop_tones: List of tones that will stop recognizing.\n        :paramtype dtmf_stop_tones: list[str or ~azure.communication.callautomation.DtmfTone]\n        :keyword speech_recognition_model_endpoint_id:\n        Endpoint id where the custom speech recognition model was deployed.\n        :paramtype speech_recognition_model_endpoint_id:\n        :keyword callback_url: Url that overrides original callback URI for this request.\n        :paramtype callback_url: str\n        :return: None\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        options = RecognizeOptions(interrupt_prompt=interrupt_prompt, initial_silence_timeout_in_seconds=initial_silence_timeout, target_participant=serialize_identifier(target_participant), speech_recognition_model_endpoint_id=speech_recognition_model_endpoint_id)
        play_source_single: Optional[MediaSources] = None
        if isinstance(play_prompt, list):
            if play_prompt:
                play_source_single = play_prompt[0]
        else:
            play_source_single = play_prompt
        if input_type == RecognizeInputType.DTMF:
            dtmf_options = DtmfOptions(inter_tone_timeout_in_seconds=dtmf_inter_tone_timeout, max_tones_to_collect=dtmf_max_tones_to_collect, stop_tones=dtmf_stop_tones)
            options.dtmf_options = dtmf_options
        elif input_type == RecognizeInputType.SPEECH:
            speech_options = SpeechOptions(end_silence_timeout_in_ms=end_silence_timeout_in_ms)
            options.speech_options = speech_options
        elif input_type == RecognizeInputType.SPEECH_OR_DTMF:
            dtmf_options = DtmfOptions(inter_tone_timeout_in_seconds=dtmf_inter_tone_timeout, max_tones_to_collect=dtmf_max_tones_to_collect, stop_tones=dtmf_stop_tones)
            speech_options = SpeechOptions(end_silence_timeout_in_ms=end_silence_timeout_in_ms)
            options.dtmf_options = dtmf_options
            options.speech_options = speech_options
        elif input_type == RecognizeInputType.CHOICES:
            options.choices = choices
        else:
            raise ValueError(f"Input type '{input_type}' is not supported.")
        recognize_request = RecognizeRequest(recognize_input_type=input_type, play_prompt=play_source_single._to_generated() if play_source_single else None, interrupt_call_media_operation=interrupt_call_media_operation, operation_context=operation_context, recognize_options=options, callback_uri=callback_url)
        self._call_media_client.recognize(self._call_connection_id, recognize_request, **kwargs)

    @distributed_trace
    def cancel_all_media_operations(self, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        'Cancels all the ongoing and queued media operations for this call.\n\n        :return: None\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        self._call_media_client.cancel_all_media_operations(self._call_connection_id, **kwargs)

    @distributed_trace
    def start_continuous_dtmf_recognition(self, target_participant: 'CommunicationIdentifier', *, operation_context: Optional[str]=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Start continuous Dtmf recognition by subscribing to tones.\n\n        :param target_participant: Target participant.\n        :type target_participant: ~azure.communication.callautomation.CommunicationIdentifier\n        :keyword operation_context: The value to identify context of the operation.\n        :paramtype operation_context: str\n        :return: None\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        continuous_dtmf_recognition_request = ContinuousDtmfRecognitionRequest(target_participant=serialize_identifier(target_participant), operation_context=operation_context)
        self._call_media_client.start_continuous_dtmf_recognition(self._call_connection_id, continuous_dtmf_recognition_request, **kwargs)

    @distributed_trace
    def stop_continuous_dtmf_recognition(self, target_participant: 'CommunicationIdentifier', *, operation_context: Optional[str]=None, callback_url: Optional[str]=None, **kwargs) -> None:
        if False:
            return 10
        'Stop continuous Dtmf recognition by unsubscribing to tones.\n\n        :param target_participant: Target participant.\n        :type target_participant: ~azure.communication.callautomation.CommunicationIdentifier\n        :keyword operation_context: The value to identify context of the operation.\n        :paramtype operation_context: str\n        :keyword callback_url: Url that overrides original callback URI for this request.\n        :paramtype callback_url: str\n        :return: None\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        continuous_dtmf_recognition_request = ContinuousDtmfRecognitionRequest(target_participant=serialize_identifier(target_participant), operation_context=operation_context, callback_uri=callback_url)
        self._call_media_client.stop_continuous_dtmf_recognition(self._call_connection_id, continuous_dtmf_recognition_request, **kwargs)

    @distributed_trace
    def send_dtmf(self, tones: List[Union[str, 'DtmfTone']], target_participant: 'CommunicationIdentifier', *, operation_context: Optional[str]=None, callback_url: Optional[str]=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        'Send Dtmf tones to this call.\n\n        :param tones: List of tones to be sent to target participant.\n        :type tones:list[str or ~azure.communication.callautomation.DtmfTone]\n        :param target_participant: Target participant.\n        :type target_participant: ~azure.communication.callautomation.CommunicationIdentifier\n        :keyword operation_context: The value to identify context of the operation.\n        :paramtype operation_context: str\n        :keyword callback_url: Url that overrides original callback URI for this request.\n        :paramtype callback_url: str\n        :return: None\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        send_dtmf_request = SendDtmfRequest(tones=tones, target_participant=serialize_identifier(target_participant), operation_context=operation_context, callback_uri=callback_url)
        self._call_media_client.send_dtmf(self._call_connection_id, send_dtmf_request, **kwargs)

    @distributed_trace
    def mute_participants(self, target_participant: 'CommunicationIdentifier', *, operation_context: Optional[str]=None, **kwargs) -> MuteParticipantsResult:
        if False:
            i = 10
            return i + 15
        'Mute participants from the call using identifier.\n\n        :param target_participant: Participant to be muted from the call. Only ACS Users are supported. Required.\n        :type target_participant: ~azure.communication.callautomation.CommunicationIdentifier\n        :keyword operation_context: Used by customers when calling mid-call actions to correlate the request to the\n         response event.\n        :paramtype operation_context: str\n        :return: MuteParticipantsResult\n        :rtype: ~azure.communication.callautomation.MuteParticipantsResult\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        mute_participants_request = MuteParticipantsRequest(target_participants=[serialize_identifier(target_participant)], operation_context=operation_context)
        process_repeatability_first_sent(kwargs)
        response = self._call_connection_client.mute(self._call_connection_id, mute_participants_request, **kwargs)
        return MuteParticipantsResult._from_generated(response)

    @distributed_trace
    def cancel_add_participant(self, invitation_id: str, *, operation_context: Optional[str]=None, callback_url: Optional[str]=None, **kwargs) -> CancelAddParticipantResult:
        if False:
            print('Hello World!')
        'Cancel add participant request sent out to a participant.\n\n        :param  invitation_id: The invitation ID that was used to add the participant.\n        :type invitation_id: str\n        :keyword operation_context: Value that can be used to track this call and its associated events.\n        :paramtype operation_context: str\n        :keyword callback_url: Url that overrides original callback URI for this request.\n        :paramtype callback_url: str\n        :return: CancelAddParticipantResult\n        :rtype: ~azure.communication.callautomation.CancelAddParticipantResult\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        cancel_add_participant_request = CancelAddParticipantRequest(invitation_id=invitation_id, operation_context=operation_context, callback_uri=callback_url)
        process_repeatability_first_sent(kwargs)
        response = self._call_connection_client.cancel_add_participant(self._call_connection_id, cancel_add_participant_request, **kwargs)
        return CancelAddParticipantResult._from_generated(response)

    @distributed_trace
    def start_hold_music(self, target_participant: 'CommunicationIdentifier', play_source: MediaSources, *, loop: bool=True, operation_context: Optional[str]=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        "Hold participant from call while playing music.\n        :param play_source: A PlaySource representing the source to play.\n        :type play_source: ~azure.communication.callautomation.FileSource or\n         ~azure.communication.callautomation.TextSource or\n         ~azure.communication.callautomation.SsmlSource or\n         list[~azure.communication.callautomation.FileSource or\n          ~azure.communication.callautomation.TextSource or\n          ~azure.communication.callautomation.SsmlSource]\n        :param target_participant: The targets to play media to. Default value is 'all', to play media\n         to all participants in the call.\n        :type target_participant: list[~azure.communication.callautomation.CommunicationIdentifier]\n        :keyword loop: Whether the media should be repeated until stopped.\n        :paramtype loop: bool\n        :keyword operation_context: Value that can be used to track this call and its associated events.\n        :paramtype operation_context: str or None\n        :return: None\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        hold_request = StartHoldMusicRequest(play_source_info=play_source._to_generated(), target_participant=serialize_identifier(target_participant), operation_context=operation_context, loop=loop, **kwargs)
        self._call_media_client.start_hold_music(self._call_connection_id, hold_request)

    @distributed_trace
    def stop_hold_music(self, target_participant: 'CommunicationIdentifier', *, operation_context: Optional[str]=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Remove hold from participant.\n\n        :param target_participant: The targets to play media to. Default value is 'all', to play media\n         to all participants in the call.\n        :type target_participant: list[~azure.communication.callautomation.CommunicationIdentifier]\n        :keyword operation_context: Value that can be used to track this call and its associated events.\n        :paramtype operation_context: str or None\n        :return: None\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        stop_hold_request = StopHoldMusicRequest(target_participant=serialize_identifier(target_participant), operation_context=operation_context, **kwargs)
        self._call_media_client.stop_hold_music(self._call_connection_id, stop_hold_request)