import random
import re
import typing
import uuid
import backoff
import google.auth
import google.cloud.security.privateca_v1 as privateca_v1
from activate_subordinate_ca import activate_subordinate_ca
from conftest import LOCATION
from create_certificate_csr import create_certificate_csr
from create_subordinate_ca import create_subordinate_ca
from revoke_certificate import revoke_certificate
PROJECT = google.auth.default()[1]
COMMON_NAME = 'COMMON_NAME'
ORGANIZATION = 'ORGANIZATION'
CA_DURATION = CERTIFICATE_LIFETIME = 1000000
DOMAIN_NAME = 'domain.com'

def generate_name() -> str:
    if False:
        while True:
            i = 10
    return 'test-' + uuid.uuid4().hex[:10]

def backoff_expo_wrapper():
    if False:
        print('Hello World!')
    for exp in backoff.expo(base=4):
        if exp is None:
            yield None
            continue
        yield (exp * (1 + random.random()))

@backoff.on_exception(backoff_expo_wrapper, Exception, max_tries=3)
def test_subordinate_certificate_authority(certificate_authority, capsys: typing.Any) -> None:
    if False:
        i = 10
        return i + 15
    CSR_CERT_NAME = generate_name()
    SUBORDINATE_CA_NAME = generate_name()
    (CA_POOL_NAME, ROOT_CA_NAME) = certificate_authority
    create_subordinate_ca(PROJECT, LOCATION, CA_POOL_NAME, SUBORDINATE_CA_NAME, COMMON_NAME, ORGANIZATION, DOMAIN_NAME, CA_DURATION)
    ca_service_client = privateca_v1.CertificateAuthorityServiceClient()
    ca_path = ca_service_client.certificate_authority_path(PROJECT, LOCATION, CA_POOL_NAME, SUBORDINATE_CA_NAME)
    response = ca_service_client.fetch_certificate_authority_csr(name=ca_path)
    pem_csr = response.pem_csr
    create_certificate_csr(PROJECT, LOCATION, CA_POOL_NAME, ROOT_CA_NAME, CSR_CERT_NAME, CERTIFICATE_LIFETIME, pem_csr)
    certificate_name = ca_service_client.certificate_path(PROJECT, LOCATION, CA_POOL_NAME, CSR_CERT_NAME)
    pem_certificate = ca_service_client.get_certificate(name=certificate_name).pem_certificate
    activate_subordinate_ca(PROJECT, LOCATION, CA_POOL_NAME, SUBORDINATE_CA_NAME, pem_certificate, ROOT_CA_NAME)
    revoke_certificate(PROJECT, LOCATION, CA_POOL_NAME, CSR_CERT_NAME)
    (out, _) = capsys.readouterr()
    assert re.search(f'Operation result: name: "projects/{PROJECT}/locations/{LOCATION}/caPools/{CA_POOL_NAME}/certificateAuthorities/{SUBORDINATE_CA_NAME}"', out)
    assert 'Certificate created successfully' in out
    assert f'Current state: {privateca_v1.CertificateAuthority.State.STAGED}' in out