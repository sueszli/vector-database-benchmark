"""Tests for models used for receipt application claims verification."""
import pytest
from azure.confidentialledger.receipt._claims_models import ApplicationClaim, LedgerEntryClaim, ClaimDigest
from ._shared.claims_constants import get_test_application_claims_with_claim_digest, get_test_application_claims_with_claim_digest_dict, get_test_application_claims_with_ledger_entry, get_test_application_claims_with_ledger_entry_dict

def test_application_claim_init_with_valid_arguments():
    if False:
        i = 10
        return i + 15
    try:
        ApplicationClaim(kind='test_kind', ledgerEntry=LedgerEntryClaim(protocol='test_protocol', collectionId='test_collection_id', contents='test_contents', secretKey='test_secret_key'), digest=ClaimDigest(protocol='test_protocol', value='test_value'))
    except Exception as e:
        pytest.fail(f'ApplicationClaim __init__ threw an exception with a valid claim {e}')

def test_application_claim_init_with_missing_optional_arguments():
    if False:
        while True:
            i = 10
    try:
        ApplicationClaim(kind='test_kind', ledgerEntry=LedgerEntryClaim(protocol='test_protocol', collectionId='test_collection_id', contents='test_contents', secretKey='test_secret_key'))
    except Exception as e:
        pytest.fail(f'ApplicationClaim __init__ threw an exception with a valid claim {e}')

def test_application_claim_init_throws_exceptions_with_missing_required_fields():
    if False:
        while True:
            i = 10
    with pytest.raises(TypeError, match='missing . required .* argument'):
        ApplicationClaim(ledgerEntry=LedgerEntryClaim(protocol='test_protocol', secretKey='test_secret_key'))

def test_receipt_init_throws_exceptions_with_missing_required_fields_in_subobjects():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(TypeError, match='missing . required .* argument'):
        ApplicationClaim(ledgerEntry=LedgerEntryClaim(collectionId='test_collection_id', secretKey='test_secret_key'))

@pytest.mark.parametrize('input_claims,expected_claims', [[get_test_application_claims_with_ledger_entry_dict(), get_test_application_claims_with_ledger_entry()], [get_test_application_claims_with_claim_digest_dict(), get_test_application_claims_with_claim_digest()]])
def test_claim_creation_from_dict(input_claims, expected_claims):
    if False:
        i = 10
        return i + 15
    for (input_claim_dict, expected_claim) in zip(input_claims, expected_claims):
        claim = ApplicationClaim.from_dict(input_claim_dict)
        assert claim == expected_claim