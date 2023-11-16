import os
import pytest
from snippets.cloud_kms_env_aead import init_tink_env_aead

@pytest.fixture(name='kms_uri')
def setup() -> str:
    if False:
        print('Hello World!')
    kms_uri = 'gcp-kms://' + os.environ['CLOUD_KMS_KEY']
    yield kms_uri

def test_cloud_kms_env_aead(kms_uri: str) -> None:
    if False:
        while True:
            i = 10
    credentials = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', None)
    if credentials is None:
        raise Exception('Environment variable GOOGLE_APPLICATION_CREDENTIALS is not set')
    envelope = init_tink_env_aead(kms_uri, credentials)
    assert envelope.key_template == kms_uri