import logging
from typing import TYPE_CHECKING, Awaitable, Callable, Optional
from synapse.api.constants import MAX_DEPTH, EventContentFields, EventTypes, Membership
from synapse.api.errors import Codes, SynapseError
from synapse.api.room_versions import EventFormatVersions, RoomVersion
from synapse.crypto.event_signing import check_event_content_hash
from synapse.crypto.keyring import Keyring
from synapse.events import EventBase, make_event_from_dict
from synapse.events.utils import prune_event, validate_canonicaljson
from synapse.http.servlet import assert_params_in_dict
from synapse.logging.opentracing import log_kv, trace
from synapse.types import JsonDict, get_domain_from_id
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class InvalidEventSignatureError(RuntimeError):
    """Raised when the signature on an event is invalid.

    The stringification of this exception is just the error message without reference
    to the event id. The event id is available as a property.
    """

    def __init__(self, message: str, event_id: str):
        if False:
            i = 10
            return i + 15
        super().__init__(message)
        self.event_id = event_id

class FederationBase:

    def __init__(self, hs: 'HomeServer'):
        if False:
            while True:
                i = 10
        self.hs = hs
        self._is_mine_server_name = hs.is_mine_server_name
        self.keyring = hs.get_keyring()
        self._spam_checker_module_callbacks = hs.get_module_api_callbacks().spam_checker
        self.store = hs.get_datastores().main
        self._clock = hs.get_clock()
        self._storage_controllers = hs.get_storage_controllers()

    @trace
    async def _check_sigs_and_hash(self, room_version: RoomVersion, pdu: EventBase, record_failure_callback: Optional[Callable[[EventBase, str], Awaitable[None]]]=None) -> EventBase:
        """Checks that event is correctly signed by the sending server.

        Also checks the content hash, and redacts the event if there is a mismatch.

        Also runs the event through the spam checker; if it fails, redacts the event
        and flags it as soft-failed.

        Args:
            room_version: The room version of the PDU
            pdu: the event to be checked
            record_failure_callback: A callback to run whenever the given event
                fails signature or hash checks. This includes exceptions
                that would be normally be thrown/raised but also things like
                checking for event tampering where we just return the redacted
                event.

        Returns:
              * the original event if the checks pass
              * a redacted version of the event (if the signature
                matched but the hash did not). In this case a warning will be logged.

        Raises:
          InvalidEventSignatureError if the signature check failed. Nothing
             will be logged in this case.
        """
        try:
            await _check_sigs_on_pdu(self.keyring, room_version, pdu)
        except InvalidEventSignatureError as exc:
            if record_failure_callback:
                await record_failure_callback(pdu, str(exc))
            raise exc
        if not check_event_content_hash(pdu):
            redacted_event = prune_event(pdu)
            if set(redacted_event.keys()) == set(pdu.keys()) and set(redacted_event.content.keys()) == set(pdu.content.keys()):
                logger.debug('Event %s seems to have been redacted; using our redacted copy', pdu.event_id)
                log_kv({'message': 'Event seems to have been redacted; using our redacted copy', 'event_id': pdu.event_id})
            else:
                logger.warning('Event %s content has been tampered, redacting', pdu.event_id)
                log_kv({'message': 'Event content has been tampered, redacting', 'event_id': pdu.event_id})
                if record_failure_callback:
                    await record_failure_callback(pdu, 'Event content has been tampered with')
            return redacted_event
        spam_check = await self._spam_checker_module_callbacks.check_event_for_spam(pdu)
        if spam_check != self._spam_checker_module_callbacks.NOT_SPAM:
            logger.warning('Event contains spam, soft-failing %s', pdu.event_id)
            log_kv({'message': 'Event contains spam, redacting (to save disk space) as well as soft-failing (to stop using the event in prev_events)', 'event_id': pdu.event_id})
            redacted_event = prune_event(pdu)
            redacted_event.internal_metadata.soft_failed = True
            return redacted_event
        return pdu

@trace
async def _check_sigs_on_pdu(keyring: Keyring, room_version: RoomVersion, pdu: EventBase) -> None:
    """Check that the given events are correctly signed

    Args:
        keyring: keyring object to do the checks
        room_version: the room version of the PDUs
        pdus: the events to be checked

    Raises:
        InvalidEventSignatureError if the event wasn't correctly signed.
    """
    sender_domain = get_domain_from_id(pdu.sender)
    if not _is_invite_via_3pid(pdu):
        try:
            await keyring.verify_event_for_server(sender_domain, pdu, pdu.origin_server_ts if room_version.enforce_key_validity else 0)
        except Exception as e:
            raise InvalidEventSignatureError(f'unable to verify signature for sender domain {sender_domain}: {e}', pdu.event_id) from None
    if room_version.event_format == EventFormatVersions.ROOM_V1_V2:
        event_domain = get_domain_from_id(pdu.event_id)
        if event_domain != sender_domain:
            try:
                await keyring.verify_event_for_server(event_domain, pdu, pdu.origin_server_ts if room_version.enforce_key_validity else 0)
            except Exception as e:
                raise InvalidEventSignatureError(f'unable to verify signature for event domain {event_domain}: {e}', pdu.event_id) from None
    if room_version.restricted_join_rule and pdu.type == EventTypes.Member and (pdu.membership == Membership.JOIN) and (EventContentFields.AUTHORISING_USER in pdu.content):
        authorising_server = get_domain_from_id(pdu.content[EventContentFields.AUTHORISING_USER])
        try:
            await keyring.verify_event_for_server(authorising_server, pdu, pdu.origin_server_ts if room_version.enforce_key_validity else 0)
        except Exception as e:
            raise InvalidEventSignatureError(f'unable to verify signature for authorising serve {authorising_server}: {e}', pdu.event_id) from None

def _is_invite_via_3pid(event: EventBase) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return event.type == EventTypes.Member and event.membership == Membership.INVITE and ('third_party_invite' in event.content)

def event_from_pdu_json(pdu_json: JsonDict, room_version: RoomVersion) -> EventBase:
    if False:
        i = 10
        return i + 15
    'Construct an EventBase from an event json received over federation\n\n    Args:\n        pdu_json: pdu as received over federation\n        room_version: The version of the room this event belongs to\n\n    Raises:\n        SynapseError: if the pdu is missing required fields or is otherwise\n            not a valid matrix event\n    '
    assert_params_in_dict(pdu_json, ('type', 'depth'))
    if 'unsigned' in pdu_json:
        _strip_unsigned_values(pdu_json)
    depth = pdu_json['depth']
    if type(depth) is not int:
        raise SynapseError(400, 'Depth %r not an intger' % (depth,), Codes.BAD_JSON)
    if depth < 0:
        raise SynapseError(400, 'Depth too small', Codes.BAD_JSON)
    elif depth > MAX_DEPTH:
        raise SynapseError(400, 'Depth too large', Codes.BAD_JSON)
    if room_version.strict_canonicaljson:
        validate_canonicaljson(pdu_json)
    event = make_event_from_dict(pdu_json, room_version)
    return event

def _strip_unsigned_values(pdu_dict: JsonDict) -> None:
    if False:
        while True:
            i = 10
    '\n    Strip any unsigned values unless specifically allowed, as defined by the whitelist.\n\n    pdu: the json dict to strip values from. Note that the dict is mutated by this\n    function\n    '
    unsigned = pdu_dict['unsigned']
    if not isinstance(unsigned, dict):
        pdu_dict['unsigned'] = {}
    if pdu_dict['type'] == 'm.room.member':
        whitelist = ['knock_room_state', 'invite_room_state', 'age']
    else:
        whitelist = ['age']
    filtered_unsigned = {k: v for (k, v) in unsigned.items() if k in whitelist}
    pdu_dict['unsigned'] = filtered_unsigned