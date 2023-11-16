from cryptography.hazmat.primitives import hashes
from devtools_testutils import AzureRecordedTestCase, recorded_by_proxy
from cryptography.hazmat.backends import default_backend
import pytest
from preparers import AllAttestationTypes, AllInstanceTypes
from attestation_preparer import AttestationPreparer
from helpers import pem_from_base64
from azure.security.attestation import AttestationAdministrationClient, AttestationType, AttestationToken, PolicyModification, CertificateModification, AttestationPolicyToken

class TestPolicyGetSet(AzureRecordedTestCase):

    @AttestationPreparer()
    @AllAttestationTypes
    @AllInstanceTypes
    @recorded_by_proxy
    def test_get_policy(self, **kwargs):
        if False:
            print('Hello World!')
        attest_client = self.create_admin_client(kwargs.pop('instance_url'))
        (policy, token) = attest_client.get_policy(kwargs.pop('attestation_type'))
        print('Shared policy: ', policy)
        assert policy.startswith('version') or len(policy) == 0
        print('Token: ', token)

    @AttestationPreparer()
    @AllAttestationTypes
    @recorded_by_proxy
    def test_aad_set_policy_unsecured(self, attestation_aad_url, **kwargs):
        if False:
            return 10
        attestation_policy = u'version=1.0; authorizationrules{=> permit();}; issuancerules{};'
        attestation_type = kwargs.pop('attestation_type')
        attest_client = self.create_admin_client(attestation_aad_url)
        (policy_set_response, _) = attest_client.set_policy(attestation_type, attestation_policy)
        new_policy = attest_client.get_policy(attestation_type)[0]
        assert new_policy == attestation_policy
        expected_policy = AttestationPolicyToken(attestation_policy)
        hasher = hashes.Hash(hashes.SHA256(), backend=default_backend())
        hasher.update(expected_policy.to_jwt_string().encode('utf-8'))
        expected_hash = hasher.finalize()
        assert expected_hash == policy_set_response.policy_token_hash

    @AttestationPreparer()
    @AllAttestationTypes
    @recorded_by_proxy
    def test_aad_reset_policy_unsecured(self, attestation_aad_url, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        attestation_type = kwargs.pop('attestation_type')
        attest_client = self.create_admin_client(attestation_aad_url)
        (policy_set_response, _) = attest_client.reset_policy(attestation_type)
        assert policy_set_response.policy_token_hash is None
        assert policy_set_response.policy_resolution == PolicyModification.REMOVED

    @pytest.mark.live_test_only
    @AttestationPreparer()
    @AllAttestationTypes
    def test_aad_reset_policy_secured(self, **kwargs):
        if False:
            print('Hello World!')
        attestation_aad_url = kwargs.pop('attestation_aad_url')
        attestation_policy_signing_key0 = kwargs.pop('attestation_policy_signing_key0')
        attestation_policy_signing_certificate0 = kwargs.pop('attestation_policy_signing_certificate0')
        signing_certificate = pem_from_base64(attestation_policy_signing_certificate0, 'CERTIFICATE')
        key = pem_from_base64(attestation_policy_signing_key0, 'PRIVATE KEY')
        attest_client = self.create_admin_client(attestation_aad_url)
        (policy_set_response, _) = attest_client.reset_policy(kwargs.pop('attestation_type'), signing_key=key, signing_certificate=signing_certificate)
        assert policy_set_response.policy_token_hash is None
        assert policy_set_response.policy_resolution == PolicyModification.REMOVED

    @pytest.mark.live_test_only
    @AllAttestationTypes
    @AttestationPreparer()
    def test_aad_set_policy_secured(self, **kwargs):
        if False:
            return 10
        attestation_aad_url = kwargs.pop('attestation_aad_url')
        attestation_policy_signing_key0 = kwargs.pop('attestation_policy_signing_key0')
        attestation_policy_signing_certificate0 = kwargs.pop('attestation_policy_signing_certificate0')
        attestation_policy = u'version=1.0; authorizationrules{=> permit();}; issuancerules{};'
        signing_certificate = pem_from_base64(attestation_policy_signing_certificate0, 'CERTIFICATE')
        key = pem_from_base64(attestation_policy_signing_key0, 'PRIVATE KEY')
        attest_client = self.create_admin_client(attestation_aad_url)
        (policy_set_response, _) = attest_client.set_policy(kwargs.pop('attestation_type'), attestation_policy, signing_key=key, signing_certificate=signing_certificate)
        (policy, _) = attest_client.get_policy(AttestationType.SGX_ENCLAVE)
        assert policy == attestation_policy
        expected_policy = AttestationPolicyToken(attestation_policy, signing_key=key, signing_certificate=signing_certificate)
        hasher = hashes.Hash(hashes.SHA256(), backend=default_backend())
        hasher.update(expected_policy.to_jwt_string().encode('utf-8'))
        expected_hash = hasher.finalize()
        assert expected_hash == policy_set_response.policy_token_hash

    @pytest.mark.live_test_only
    @AttestationPreparer()
    @AllAttestationTypes
    def test_isolated_set_policy_secured(self, **kwargs):
        if False:
            return 10
        attestation_isolated_url = kwargs.pop('attestation_isolated_url')
        attestation_isolated_signing_key = kwargs.pop('attestation_isolated_signing_key')
        attestation_isolated_signing_certificate = kwargs.pop('attestation_isolated_signing_certificate')
        attestation_policy = u'version=1.0; authorizationrules{=> permit();}; issuancerules{};'
        signing_certificate = pem_from_base64(attestation_isolated_signing_certificate, 'CERTIFICATE')
        key = pem_from_base64(attestation_isolated_signing_key, 'PRIVATE KEY')
        attest_client = self.create_admin_client(attestation_isolated_url)
        (policy_set_response, _) = attest_client.set_policy(kwargs.pop('attestation_type'), attestation_policy, signing_key=key, signing_certificate=signing_certificate)
        (new_policy, _) = attest_client.get_policy(AttestationType.SGX_ENCLAVE)
        assert new_policy == attestation_policy
        expected_policy = AttestationPolicyToken(attestation_policy, signing_key=key, signing_certificate=signing_certificate)
        hasher = hashes.Hash(hashes.SHA256(), backend=default_backend())
        hasher.update(expected_policy.to_jwt_string().encode('utf-8'))
        expected_hash = hasher.finalize()
        assert expected_hash == policy_set_response.policy_token_hash

    @pytest.mark.live_test_only
    @AttestationPreparer()
    @AllAttestationTypes
    def test_isolated_reset_policy_secured(self, **kwargs):
        if False:
            while True:
                i = 10
        attestation_aad_url = kwargs.pop('attestation_aad_url')
        attestation_isolated_signing_key = kwargs.pop('attestation_isolated_signing_key')
        attestation_isolated_signing_certificate = kwargs.pop('attestation_isolated_signing_certificate')
        signing_certificate = pem_from_base64(attestation_isolated_signing_certificate, 'CERTIFICATE')
        key = pem_from_base64(attestation_isolated_signing_key, 'PRIVATE KEY')
        attest_client = self.create_admin_client(attestation_aad_url)
        (policy_set_response, _) = attest_client.reset_policy(kwargs.pop('attestation_type'), signing_key=key, signing_certificate=signing_certificate)
        assert policy_set_response.policy_token_hash is None
        assert policy_set_response.policy_resolution == PolicyModification.REMOVED

    def _test_get_policy_management_certificates(self, base_uri, expected_certificate):
        if False:
            while True:
                i = 10
        admin_client = self.create_admin_client(base_uri)
        (policy_signers, _) = admin_client.get_policy_management_certificates()
        if expected_certificate is not None:
            found_cert = False
            for signer in policy_signers:
                if signer[0] == expected_certificate:
                    found_cert = True
                    break
            assert found_cert
        else:
            assert len(policy_signers) == 0

    @staticmethod
    def is_isolated_url(instance_url, **kwargs):
        if False:
            i = 10
            return i + 15
        return instance_url == kwargs.get('attestation_isolated_url')

    @pytest.mark.live_test_only
    @AttestationPreparer()
    @AllInstanceTypes
    def test_get_policy_management_certificates(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        instance_url = kwargs.pop('instance_url')
        expected_certificate = None
        if self.is_isolated_url(instance_url, **kwargs):
            expected_certificate = pem_from_base64(kwargs.get('attestation_isolated_signing_certificate'), 'CERTIFICATE')
        self._test_get_policy_management_certificates(instance_url, expected_certificate)

    @pytest.mark.live_test_only
    @AttestationPreparer()
    def test_add_remove_policy_certificate(self, **kwargs):
        if False:
            print('Hello World!')
        attestation_isolated_url = kwargs.pop('attestation_isolated_url')
        attestation_isolated_signing_certificate = kwargs.pop('attestation_isolated_signing_certificate')
        attestation_isolated_signing_key = kwargs.pop('attestation_isolated_signing_key')
        attestation_policy_signing_key0 = kwargs.pop('attestation_policy_signing_key0')
        attestation_policy_signing_certificate0 = kwargs.pop('attestation_policy_signing_certificate0')
        pem_signing_cert = pem_from_base64(attestation_isolated_signing_certificate, 'CERTIFICATE')
        pem_signing_key = pem_from_base64(attestation_isolated_signing_key, 'PRIVATE KEY')
        pem_certificate_to_add = pem_from_base64(attestation_policy_signing_certificate0, 'CERTIFICATE')
        admin_client = self.create_admin_client(attestation_isolated_url, signing_key=pem_signing_key, signing_certificate=pem_signing_cert)
        with pytest.raises(TypeError):
            admin_client.add_policy_management_certificate()
        with pytest.raises(TypeError):
            admin_client.add_policy_management_certificate(pem_certificate_to_add, pem_certificate_to_add)
        (result, _) = admin_client.add_policy_management_certificate(pem_certificate_to_add)
        assert result.certificate_resolution == CertificateModification.IS_PRESENT
        (result, _) = admin_client.add_policy_management_certificate(pem_certificate_to_add, signing_key=pem_signing_key, signing_certificate=pem_signing_cert)
        assert result.certificate_resolution == CertificateModification.IS_PRESENT
        self._test_get_policy_management_certificates(attestation_isolated_url, pem_certificate_to_add)
        (result, _) = admin_client.remove_policy_management_certificate(pem_certificate_to_add)
        assert result.certificate_resolution == CertificateModification.IS_ABSENT
        (result, _) = admin_client.remove_policy_management_certificate(pem_certificate_to_add)
        assert result.certificate_resolution == CertificateModification.IS_ABSENT
        self._test_get_policy_management_certificates(attestation_isolated_url, pem_signing_cert)

    def create_admin_client(self, base_uri, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        docstring\n        '
        credential = self.get_credential(AttestationAdministrationClient)
        attest_client = self.create_client_from_credential(AttestationAdministrationClient, credential=credential, endpoint=base_uri, validate_token=True, validate_signature=True, validate_issuer=self.is_live, issuer=base_uri, validate_expiration=self.is_live, **kwargs)
        return attest_client