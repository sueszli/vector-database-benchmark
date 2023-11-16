import os
import deidentify_table_with_multiple_crypto_hash as deid
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_deidentify_table_with_multiple_crypto_hash(capsys: pytest.CaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    table_data = {'header': ['user_id', 'comments'], 'rows': [['user1@example.org', 'my email is user1@example.org and phone is 858-333-2222'], ['abbyabernathy1', 'my userid is abbyabernathy1 and my email is aabernathy@example.com']]}
    deid.deidentify_table_with_multiple_crypto_hash(GCLOUD_PROJECT, table_data, ['EMAIL_ADDRESS', 'PHONE_NUMBER'], 'TRANSIENT-CRYPTO-KEY-1', 'TRANSIENT-CRYPTO-KEY-2', ['user_id'], ['comments'])
    (out, _) = capsys.readouterr()
    assert 'user1@example.org' not in out
    assert '858-555-0222' not in out
    assert 'string_value: "abbyabernathy1"' not in out
    assert 'my userid is abbyabernathy1' in out
    assert 'aabernathy@example.com' not in out