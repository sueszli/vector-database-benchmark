import collections.abc
import hashlib
import logging
from typing import Any, Callable, Dict, Tuple
from canonicaljson import encode_canonical_json
from signedjson.sign import sign_json
from signedjson.types import SigningKey
from unpaddedbase64 import decode_base64, encode_base64
from synapse.api.errors import Codes, SynapseError
from synapse.api.room_versions import RoomVersion
from synapse.events import EventBase
from synapse.events.utils import prune_event, prune_event_dict
from synapse.logging.opentracing import trace
from synapse.types import JsonDict
logger = logging.getLogger(__name__)
Hasher = Callable[[bytes], 'hashlib._Hash']

@trace
def check_event_content_hash(event: EventBase, hash_algorithm: Hasher=hashlib.sha256) -> bool:
    if False:
        i = 10
        return i + 15
    'Check whether the hash for this PDU matches the contents'
    (name, expected_hash) = compute_content_hash(event.get_pdu_json(), hash_algorithm)
    logger.debug('Verifying content hash on %s (expecting: %s)', event.event_id, encode_base64(expected_hash))
    hashes = event.get('hashes')
    if not isinstance(hashes, collections.abc.Mapping):
        raise SynapseError(400, "Malformed 'hashes': %s" % (type(hashes),), Codes.UNAUTHORIZED)
    if name not in hashes:
        raise SynapseError(400, 'Algorithm %s not in hashes %s' % (name, list(hashes)), Codes.UNAUTHORIZED)
    message_hash_base64 = hashes[name]
    try:
        message_hash_bytes = decode_base64(message_hash_base64)
    except Exception:
        raise SynapseError(400, 'Invalid base64: %s' % (message_hash_base64,), Codes.UNAUTHORIZED)
    return message_hash_bytes == expected_hash

def compute_content_hash(event_dict: Dict[str, Any], hash_algorithm: Hasher) -> Tuple[str, bytes]:
    if False:
        i = 10
        return i + 15
    'Compute the content hash of an event, which is the hash of the\n    unredacted event.\n\n    Args:\n        event_dict: The unredacted event as a dict\n        hash_algorithm: A hasher from `hashlib`, e.g. hashlib.sha256, to use\n            to hash the event\n\n    Returns:\n        A tuple of the name of hash and the hash as raw bytes.\n    '
    event_dict = dict(event_dict)
    event_dict.pop('age_ts', None)
    event_dict.pop('unsigned', None)
    event_dict.pop('signatures', None)
    event_dict.pop('hashes', None)
    event_dict.pop('outlier', None)
    event_dict.pop('destinations', None)
    event_json_bytes = encode_canonical_json(event_dict)
    hashed = hash_algorithm(event_json_bytes)
    return (hashed.name, hashed.digest())

def compute_event_reference_hash(event: EventBase, hash_algorithm: Hasher=hashlib.sha256) -> Tuple[str, bytes]:
    if False:
        print('Hello World!')
    'Computes the event reference hash. This is the hash of the redacted\n    event.\n\n    Args:\n        event\n        hash_algorithm: A hasher from `hashlib`, e.g. hashlib.sha256, to use\n            to hash the event\n\n    Returns:\n        A tuple of the name of hash and the hash as raw bytes.\n    '
    tmp_event = prune_event(event)
    event_dict = tmp_event.get_pdu_json()
    event_dict.pop('signatures', None)
    event_dict.pop('age_ts', None)
    event_dict.pop('unsigned', None)
    event_json_bytes = encode_canonical_json(event_dict)
    hashed = hash_algorithm(event_json_bytes)
    return (hashed.name, hashed.digest())

def compute_event_signature(room_version: RoomVersion, event_dict: JsonDict, signature_name: str, signing_key: SigningKey) -> Dict[str, Dict[str, str]]:
    if False:
        return 10
    "Compute the signature of the event for the given name and key.\n\n    Args:\n        room_version: the version of the room that this event is in.\n            (the room version determines the redaction algorithm and hence the\n            json to be signed)\n\n        event_dict: The event as a dict\n\n        signature_name: The name of the entity signing the event\n            (typically the server's hostname).\n\n        signing_key: The key to sign with\n\n    Returns:\n        a dictionary in the same format of an event's signatures field.\n    "
    redact_json = prune_event_dict(room_version, event_dict)
    redact_json.pop('age_ts', None)
    redact_json.pop('unsigned', None)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Signing event: %s', encode_canonical_json(redact_json))
    redact_json = sign_json(redact_json, signature_name, signing_key)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Signed event: %s', encode_canonical_json(redact_json))
    return redact_json['signatures']

def add_hashes_and_signatures(room_version: RoomVersion, event_dict: JsonDict, signature_name: str, signing_key: SigningKey) -> None:
    if False:
        while True:
            i = 10
    "Add content hash and sign the event\n\n    Args:\n        room_version: the version of the room this event is in\n\n        event_dict: The event to add hashes to and sign\n        signature_name: The name of the entity signing the event\n            (typically the server's hostname).\n        signing_key: The key to sign with\n    "
    (name, digest) = compute_content_hash(event_dict, hash_algorithm=hashlib.sha256)
    event_dict.setdefault('hashes', {})[name] = encode_base64(digest)
    event_dict['signatures'] = compute_event_signature(room_version, event_dict, signature_name=signature_name, signing_key=signing_key)