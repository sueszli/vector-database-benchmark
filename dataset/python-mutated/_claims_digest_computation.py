"""Algorithm implementation for computing Azure Confidential Ledger application claims."""
from base64 import b64decode
from hashlib import sha256
import hmac
from typing import Any, Dict, List, cast
from azure.confidentialledger.receipt._claims_models import ApplicationClaim, LedgerEntryClaim, ClaimDigest
LEDGER_ENTRY_CLAIM_TYPE = 'LedgerEntry'
DIGEST_CLAIM_TYPE = 'ClaimDigest'
LEDGER_ENTRY_V1_CLAIM_PROTOCOL = 'LedgerEntryV1'

def compute_claims_digest(application_claims: List[Dict[str, Any]]) -> str:
    if False:
        while True:
            i = 10
    '\n    Compute the claims digest from a list of Azure Confidential Ledger application claims.\n\n    :param application_claims: List of application claims to be verified against the receipt.\n    :type application_claims: List[Dict[str, Any]]\n\n    :return: The claims digest of the application claims.\n    :rtype: str\n    :raises ValueError: If the claims digest computation has failed.\n    '
    _validate_application_claims(application_claims)
    application_claims_obj = []
    for claim_dict in application_claims:
        claim = ApplicationClaim.from_dict(claim_dict)
        application_claims_obj.append(claim)
    return _compute_claims_hexdigest(application_claims_obj)

def _validate_application_claims(application_claims: List[Dict[str, Any]]):
    if False:
        print('Hello World!')
    'Validate the application claims in a write transaction receipt.\n\n    :param list[dict[str, any]] application_claims: List of application claims to be verified against the receipt.\n    '
    assert isinstance(application_claims, list)
    assert len(application_claims) > 0, 'Application claims list cannot be empty'
    for application_claim_object in application_claims:
        assert isinstance(application_claim_object, dict)
        assert 'kind' in application_claim_object
        claim_kind = application_claim_object['kind']
        assert isinstance(claim_kind, str)
        if claim_kind == 'LedgerEntry':
            ledger_entry_claim = application_claim_object.get('ledgerEntry')
            assert isinstance(ledger_entry_claim, dict)
            assert 'collectionId' in ledger_entry_claim
            assert isinstance(ledger_entry_claim['collectionId'], str)
            assert 'contents' in ledger_entry_claim
            assert isinstance(ledger_entry_claim['contents'], str)
            assert 'protocol' in ledger_entry_claim
            assert isinstance(ledger_entry_claim['protocol'], str)
            assert 'secretKey' in ledger_entry_claim
            assert isinstance(ledger_entry_claim['secretKey'], str)
        elif claim_kind == 'ClaimDigest':
            assert 'digest' in application_claim_object
            digest_claim = application_claim_object['digest']
            assert isinstance(digest_claim, dict)
            assert 'value' in digest_claim
            assert isinstance(digest_claim['value'], str)
            assert 'protocol' in digest_claim
            assert isinstance(digest_claim['protocol'], str)
        else:
            assert False, f'Unknown claim kind: {claim_kind}'

def _compute_ledger_entry_v1_claim_digest(ledger_entry_claim: LedgerEntryClaim) -> bytes:
    if False:
        print('Hello World!')
    'Compute the digest of a LedgerEntryV1 claim. It returns the digest in bytes.\n\n    :param LedgerEntryClaim ledger_entry_claim: LedgerEntry claim to be digested.\n    :return: The digest of the LedgerEntry claim.\n    :rtype: bytes\n    '
    secret_key = b64decode(ledger_entry_claim.secretKey, validate=True)
    collection_id_digest = hmac.new(secret_key, ledger_entry_claim.collectionId.encode(), sha256).digest()
    contents_digest = hmac.new(secret_key, ledger_entry_claim.contents.encode(), sha256).digest()
    return sha256(collection_id_digest + contents_digest).digest()

def _compute_ledger_entry_claim_digest(ledger_entry_claim: LedgerEntryClaim) -> bytes:
    if False:
        while True:
            i = 10
    'Compute the digest of a LedgerEntry claim. It returns the digest in bytes.\n\n    :param LedgerEntryClaim ledger_entry_claim: LedgerEntry claim to be digested.\n    :return: The digest of the LedgerEntry claim.\n    :rtype: bytes\n    '
    claim_protocol = ledger_entry_claim.protocol
    if claim_protocol == LEDGER_ENTRY_V1_CLAIM_PROTOCOL:
        ledger_entry_digest = _compute_ledger_entry_v1_claim_digest(ledger_entry_claim)
    else:
        raise ValueError(f'Unsupported claim protocol: {claim_protocol}')
    return sha256(claim_protocol.encode() + ledger_entry_digest).digest()

def _compute_claim_digest_from_object(claim_digest_object: ClaimDigest) -> bytes:
    if False:
        print('Hello World!')
    return sha256(claim_digest_object.protocol.encode() + bytes.fromhex(claim_digest_object.value)).digest()

def _compute_claims_hexdigest(application_claims_list: List[ApplicationClaim]) -> str:
    if False:
        while True:
            i = 10
    'Compute the CCF claims digest from the provided list of application claims objects.\n    It returns the hexdigest of the claims digest.\n\n    :param list[ApplicationClaim] application_claims_list: List of application claims to be digested.\n    :return: The hexdigest of the claims digest.\n    :rtype: str\n    '
    claims_digests_concatenation = b''
    for application_claim_object in application_claims_list:
        claim_kind = application_claim_object.kind
        if claim_kind == LEDGER_ENTRY_CLAIM_TYPE:
            claim_digest = _compute_ledger_entry_claim_digest(cast(LedgerEntryClaim, application_claim_object.ledgerEntry))
        elif claim_kind == DIGEST_CLAIM_TYPE:
            claim_digest = _compute_claim_digest_from_object(cast(ClaimDigest, application_claim_object.digest))
        else:
            raise ValueError(f'Unsupported claim kind: {claim_kind}')
        claims_digests_concatenation += claim_digest
    claims_digests_concatenation = len(application_claims_list).to_bytes(length=4, byteorder='little') + claims_digests_concatenation
    return sha256(claims_digests_concatenation).hexdigest()