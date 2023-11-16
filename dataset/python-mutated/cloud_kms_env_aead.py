import logging
import tink
from tink import aead
from tink.integration import gcpkms
logger = logging.getLogger(__name__)

def init_tink_env_aead(key_uri: str, credentials: str) -> tink.aead.KmsEnvelopeAead:
    if False:
        i = 10
        return i + 15
    '\n    Initiates the Envelope AEAD object using the KMS credentials.\n    '
    aead.register()
    try:
        gcp_client = gcpkms.GcpKmsClient(key_uri, credentials)
        gcp_aead = gcp_client.get_aead(key_uri)
    except tink.TinkError as e:
        logger.error('Error initializing GCP client: %s', e)
        raise e
    key_template = aead.aead_key_templates.AES256_GCM
    env_aead = aead.KmsEnvelopeAead(key_template, gcp_aead)
    print(f'Created envelope AEAD Primitive using KMS URI: {key_uri}')
    return env_aead