from typing import Dict, List, Any, TYPE_CHECKING, Tuple, Union
import base64
from threading import Lock
from cryptography.hazmat.primitives import serialization
from cryptography.x509 import load_pem_x509_certificate
from azure.core.tracing.decorator import distributed_trace
from ._generated import AzureAttestationRestClient
from ._generated.models import AttestationType, PolicyResult as GeneratedPolicyResult, PolicyCertificatesResult as GeneratedPolicyCertificatesResult, JSONWebKey, AttestationCertificateManagementBody, StoredAttestationPolicy as GeneratedStoredAttestationPolicy, PolicyCertificatesModificationResult as GeneratedPolicyCertificatesModificationResult
from ._configuration import AttestationClientConfiguration
from ._models import AttestationSigner, AttestationToken, AttestationPolicyResult, AttestationPolicyCertificateResult
from ._common import pem_from_base64, validate_signing_keys, merge_validation_args
if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class AttestationAdministrationClient(object):
    """Provides administrative APIs for managing an instance of the Attestation Service.

    The :class:`~azure.security.attestation.AttestationAdministrationClient` object implements the policy
    management and policy certificate management functions.

    :param str endpoint: The attestation instance base URI, for example https://mytenant.attest.azure.net.
    :param credential: Credentials for the caller used to interact with the service.
    :type credential: :class:`~azure.core.credentials.TokenCredential`
    :keyword str signing_key: PEM encoded signing key to be used for all
        operations.
    :keyword str signing_certificate: PEM encoded X.509 certificate to be used for all
        operations.
    :keyword bool validate_token: If True, validate the token, otherwise return the token unvalidated.
    :keyword validation_callback: Function callback to allow clients to perform custom validation of the token.
        if the token is invalid, the `validation_callback` function should throw
        an exception.
    :paramtype validation_callback: ~typing.Callable[[~azure.security.attestation.AttestationToken, ~azure.security.attestation.AttestationSigner], None]
    :keyword bool validate_signature: If True, validate the signature of the token being validated.
    :keyword bool validate_expiration: If True, validate the expiration time of the token being validated.
    :keyword str issuer: Expected issuer, used if `validate_issuer` is true.
    :keyword float validation_slack: Slack time for validation - tolerance applied
        to help account for clock drift between the issuer and the current machine.
    :keyword bool validate_issuer: If True, validate that the issuer of the token matches the expected issuer.
    :keyword bool validate_not_before_time: If true, validate the "Not Before" time in the token.

    If the `signing_key` and `signing_certificate` parameters
    are provided, they will be applied to the following APIs:

    * :py:func:`set_policy`
    * :py:func:`reset_policy`
    * :py:func:`add_policy_management_certificate`
    * :py:func:`remove_policy_management_certificate`

    .. note::
        The `signing_key` and `signing_certificate` parameters are a pair. If one
        is present, the other must also be provided. In addition, the public key
        in the `signing_key` and the public key in the `signing_certificate` must
        match to ensure that the `signing_certificate` can be used to validate an
        object signed by `signing_key`.

    .. tip::
        The `validate_token`, `validation_callback`, `validate_signature`,
        `validate_expiration`, `validate_not_before_time`, `validate_issuer`, and
        `issuer` keyword arguments are default values applied to each API call within
        the :py:class:`AttestationAdministrationClient` class. These values can be
        overridden on individual API calls as needed.

    For additional client creation configuration options, please see `Python Request
    Options <https://aka.ms/azsdk/python/options>`_.

    """

    def __init__(self, endpoint, credential, **kwargs):
        if False:
            return 10
        if not credential:
            raise ValueError('Missing credential.')
        self._config = AttestationClientConfiguration(**kwargs)
        self._client = AzureAttestationRestClient(credential, endpoint, **kwargs)
        self._statelock = Lock()
        self._signing_certificates = None
        self._signing_key = None
        self._signing_certificate = None
        signing_key = kwargs.pop('signing_key', None)
        signing_certificate = kwargs.pop('signing_certificate', None)
        if signing_key or signing_certificate:
            (self._signing_key, self._signing_certificate) = validate_signing_keys(signing_key, signing_certificate)

    @distributed_trace
    def get_policy(self, attestation_type, **kwargs):
        if False:
            while True:
                i = 10
        'Retrieves the attestation policy for a specified attestation type.\n\n        :param attestation_type: :class:`azure.security.attestation.AttestationType` for\n            which to retrieve the policy.\n        :type attestation_type: Union[str, ~azure.security.attestation.AttestationType]\n        :keyword bool validate_token: If True, validate the token, otherwise return the token unvalidated.\n        :keyword validation_callback: Function callback to allow clients to perform custom validation of the token.\n            if the token is invalid, the `validation_callback` function should throw\n            an exception.\n        :paramtype validation_callback: ~typing.Callable[[~azure.security.attestation.AttestationToken, ~azure.security.attestation.AttestationSigner], None]\n        :keyword bool validate_signature: If True, validate the signature of the token being validated.\n        :keyword bool validate_expiration: If True, validate the expiration time of the token being validated.\n        :keyword str issuer: Expected issuer, used if `validate_issuer` is true.\n        :keyword float validation_slack: Slack time for validation - tolerance applied\n            to help account for clock drift between the issuer and the current machine.\n        :keyword bool validate_issuer: If True, validate that the issuer of the token matches the expected issuer.\n        :keyword bool validate_not_before_time: If true, validate the "Not Before" time in the token.\n\n        :return: A tuple containing the attestation policy and the token returned\n            by the service..\n\n        :rtype: Tuple[str, ~azure.security.attestation.AttestationToken]\n\n        :raises ~azure.security.attestation.AttestationTokenValidationException: Raised\n            when an attestation token is invalid.\n\n        .. note::\n            The Azure Attestation Policy language is defined `here\n            <https://docs.microsoft.com/azure/attestation/author-sign-policy>`_\n\n        .. admonition:: Example: Retrieving the current policy on an attestation instance.\n\n            .. literalinclude:: ../samples/sample_get_set_policy.py\n                :start-after: [BEGIN get_policy]\n                :end-before: [END get_policy]\n                :language: python\n                :dedent: 8\n                :caption: Getting the current policy document.\n\n        '
        options = merge_validation_args(self._config._kwargs, kwargs)
        policyResult = self._client.policy.get(attestation_type, **kwargs)
        token = AttestationToken(token=policyResult.token, body_type=GeneratedPolicyResult)
        token_body = token._get_body()
        stored_policy = AttestationToken(token=token_body.policy, body_type=GeneratedStoredAttestationPolicy)
        policy_body = stored_policy._get_body()
        actual_policy = policy_body.attestation_policy if policy_body else ''.encode('ascii')
        if options.get('validate_token', True):
            token._validate_token(self._get_signers(**kwargs), **options)
        return (actual_policy.decode('utf-8'), token)

    @distributed_trace
    def set_policy(self, attestation_type, attestation_policy, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Sets the attestation policy for the specified attestation type.\n\n        :param attestation_type: :class:`azure.security.attestation.AttestationType` for\n            which to set the policy.\n        :type attestation_type: Union[str, ~azure.security.attestation.AttestationType]\n        :param str attestation_policy: Attestation policy to be set.\n        :keyword str signing_key: PEM encoded signing key to be used to sign the policy\n            before sending it to the service.\n        :keyword str signing_certificate: PEM encoded X.509 certificate to be sent to the\n            service along with the policy.\n        :keyword bool validate_token: If True, validate the token, otherwise return the token unvalidated.\n        :keyword validation_callback: Function callback to allow clients to perform custom validation of the token.\n            if the token is invalid, the `validation_callback` function should throw\n            an exception.\n        :paramtype validation_callback: ~typing.Callable[[~azure.security.attestation.AttestationToken, ~azure.security.attestation.AttestationSigner], None]\n        :keyword bool validate_signature: If True, validate the signature of the token being validated.\n        :keyword bool validate_expiration: If True, validate the expiration time of the token being validated.\n        :keyword str issuer: Expected issuer, used if `validate_issuer` is true.\n        :keyword float validation_slack: Slack time for validation - tolerance applied\n            to help account for clock drift between the issuer and the current machine.\n        :keyword bool validate_issuer: If True, validate that the issuer of the token matches the expected issuer.\n        :keyword bool validate_not_before_time: If true, validate the "Not Before" time in the token.\n\n        :return: Result of set policy operation.\n\n        :rtype: Tuple[~azure.security.attestation.AttestationPolicyResult, ~azure.security.attestation.AttestationToken]\n\n        :raises ~azure.security.attestation.AttestationTokenValidationException: Raised\n            when an attestation token is invalid.\n\n        .. admonition:: Example: Setting the attestation policy on an AAD mode\n            attestation instance (no signing key required).\n\n            .. literalinclude:: ../samples/sample_get_set_policy.py\n                :start-after: [BEGIN set_policy_unsecured]\n                :end-before: [END set_policy_unsecured]\n                :language: python\n                :dedent: 0\n                :caption: Setting a security policy without a signing key.\n\n        .. admonition:: Example: Setting the attestation policy and verifying\n            that the policy was received by the service.\n\n            .. literalinclude:: ../samples/sample_get_set_policy.py\n                :start-after: [START validate_policy_hash]\n                :end-before: [END validate_policy_hash]\n                :language: python\n                :dedent: 0\n                :caption: Setting the attestation policy with hash verification.\n\n        .. note::\n            If the attestation instance is in *Isolated* mode, then the\n            `signing_key` parameter MUST be a signing key containing one of the\n            certificates returned by :meth:`get_policy_management_certificates`.\n\n            If the attestation instance is in *AAD* mode, then the `signing_key`\n            parameter does not need to be provided.\n\n        '
        signing_key = kwargs.pop('signing_key', self._signing_key)
        signing_certificate = kwargs.pop('signing_certificate', self._signing_certificate)
        policy_token = AttestationToken(body=GeneratedStoredAttestationPolicy(attestation_policy=attestation_policy.encode('ascii')), signing_key=signing_key, signing_certificate=signing_certificate, body_type=GeneratedStoredAttestationPolicy)
        options = merge_validation_args(self._config._kwargs, kwargs)
        policyResult = self._client.policy.set(attestation_type=attestation_type, new_attestation_policy=policy_token.to_jwt_string(), **kwargs)
        token = AttestationToken(token=policyResult.token, body_type=GeneratedPolicyResult)
        if options.get('validate_token', True):
            token._validate_token(self._get_signers(**kwargs), **options)
        return (AttestationPolicyResult._from_generated(token._get_body()), token)

    @distributed_trace
    def reset_policy(self, attestation_type, **kwargs):
        if False:
            i = 10
            return i + 15
        'Resets the attestation policy for the specified attestation type to the default value.\n\n        :param attestation_type: :class:`azure.security.attestation.AttestationType` for\n            which to set the policy.\n        :type attestation_type: Union[str, ~azure.security.attestation.AttestationType]\n        :keyword str signing_key: PEM encoded signing key to be used to sign the policy\n            before sending it to the service.\n        :keyword str signing_certificate: PEM encoded X.509 certificate to be sent to the\n            service along with the policy.\n        :keyword bool validate_token: If True, validate the token, otherwise return the token unvalidated.\n        :keyword validation_callback: Function callback to allow clients to perform custom validation of the token.\n            if the token is invalid, the `validation_callback` function should throw\n            an exception.\n        :paramtype validation_callback: ~typing.Callable[[~azure.security.attestation.AttestationToken, ~azure.security.attestation.AttestationSigner], None]\n        :keyword bool validate_signature: If True, validate the signature of the token being validated.\n        :keyword bool validate_expiration: If True, validate the expiration time of the token being validated.\n        :keyword str issuer: Expected issuer, used if `validate_issuer` is true.\n        :keyword float validation_slack: Slack time for validation - tolerance applied\n            to help account for clock drift between the issuer and the current machine.\n        :keyword bool validate_issuer: If True, validate that the issuer of the token matches the expected issuer.\n        :keyword bool validate_not_before_time: If true, validate the "Not Before" time in the token.\n\n        :return: A policy set result reflecting the outcome of the policy removal and\n            the token which contained the result.\n\n        :rtype: Tuple[~azure.security.attestation.AttestationPolicyResult, ~azure.security.attestation.AttestationToken]\n\n        :raises ~azure.security.attestation.AttestationTokenValidationException: Raised\n            when an attestation token is invalid.\n\n        .. note::\n            If the attestation instance is in *Isolated* mode, then the\n            `signing_key` parameter MUST be a signing key containing one of the\n            certificates returned by :meth:`get_policy_management_certificates`.\n\n            If the attestation instance is in *AAD* mode, then the `signing_key`\n            parameter does not need to be provided.\n\n        .. admonition:: Example: Resetting the attestation policy on an AAD mode\n            attestation instance (no signing key required).\n\n            .. literalinclude:: ../samples/sample_get_set_policy.py\n                :start-after: [BEGIN reset_aad_policy]\n                :end-before: [END reset_aad_policy]\n                :language: python\n                :dedent: 8\n                :caption: Resetting an AAD mode attestation instance.\n\n        .. admonition:: Example: Resetting the attestation policy on an Isolated mode\n            attestation instance (signing key required).\n\n            .. literalinclude:: ../samples/sample_get_set_policy.py\n                :start-after: [BEGIN reset_isolated_policy]\n                :end-before: [END reset_isolated_policy]\n                :language: python\n                :dedent: 8\n                :caption: Resetting an AAD mode attestation instance.\n\n        '
        signing_key = kwargs.pop('signing_key', self._signing_key)
        signing_certificate = kwargs.pop('signing_certificate', self._signing_certificate)
        policy_token = AttestationToken(body=None, signing_key=signing_key, signing_certificate=signing_certificate)
        options = merge_validation_args(self._config._kwargs, kwargs)
        policyResult = self._client.policy.reset(attestation_type=attestation_type, policy_jws=policy_token.to_jwt_string(), **kwargs)
        token = AttestationToken(token=policyResult.token, body_type=GeneratedPolicyResult)
        if options.get('validate_token', True):
            token._validate_token(self._get_signers(**kwargs), **options)
        return (AttestationPolicyResult._from_generated(token._get_body()), token)

    @distributed_trace
    def get_policy_management_certificates(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Retrieves the set of policy management certificates for the instance.\n\n        The list of policy management certificates will only have values if the\n        attestation service instance is in Isolated mode.\n\n        :keyword bool validate_token: If True, validate the token, otherwise\n            return the token unvalidated.\n        :keyword validation_callback: Function callback to allow clients to\n            perform custom validation of the token. If the token is invalid,\n            the `validation_callback` function should throw an exception to cause\n            the API call to fail.\n        :paramtype validation_callback: ~typing.Callable[[~azure.security.attestation.AttestationToken, ~azure.security.attestation.AttestationSigner], None]\n        :keyword bool validate_signature: If True, validate the signature of the\n            token being validated.\n        :keyword bool validate_expiration: If True, validate the expiration time\n            of the token being validated.\n        :keyword float validation_slack: Slack time for validation - tolerance\n            applied to help account for clock drift between the issuer and\n            the current machine.\n        :keyword str issuer: Expected issuer, used if `validate_issuer` is true.\n        :keyword bool validate_issuer: If True, validate that the issuer of the\n            token matches the expected issuer.\n        :keyword bool validate_not_before_time: If true, validate the\n            "Not Before" time in the token.\n\n        :return: A tuple containing the list of PEM encoded X.509 certificate chains and an attestation token.\n        :rtype: Tuple[List[List[str]], ~azure.security.attestation.AttestationToken]\n\n        .. admonition:: Example: Retrieving the set of policy management certificates\n            for an isolated attestation instance.\n\n            .. literalinclude:: ../samples/sample_get_set_policy.py\n                :start-after: [BEGIN get_policy_management_certificate]\n                :end-before: [END get_policy_management_certificate]\n                :language: python\n                :dedent: 8\n                :caption: Retrieving the policy management certificates.\n\n        '
        options = merge_validation_args(self._config._kwargs, kwargs)
        cert_response = self._client.policy_certificates.get(**kwargs)
        token = AttestationToken(token=cert_response.token, body_type=GeneratedPolicyCertificatesResult)
        if options.get('validate_token', True):
            token._validate_token(self._get_signers(**kwargs), **options)
        certificates = []
        cert_list = token._get_body()
        for key in cert_list.policy_certificates.keys:
            key_certs = [pem_from_base64(cert, 'CERTIFICATE') for cert in key.x5_c]
            certificates.append(key_certs)
        return (certificates, token)

    @distributed_trace
    def add_policy_management_certificate(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Adds a new policy management certificate to the set of policy management certificates for the instance.\n\n        :param str certificate_to_add: Required. PEM encoded X.509 certificate to add to\n            the list of attestation policy management certificates.\n        :keyword str signing_key: PEM encoded signing Key representing the key\n            associated with one of the *existing* attestation signing certificates.\n        :keyword str signing_certificate: PEM encoded signing certificate which is one of\n            the *existing* attestation signing certificates.\n        :keyword bool validate_token: If True, validate the token, otherwise return the token unvalidated.\n        :keyword validation_callback: Function callback to allow clients to perform custom validation of the token.\n            if the token is invalid, the `validation_callback` function should throw\n            an exception.\n        :paramtype validation_callback: ~typing.Callable[[~azure.security.attestation.AttestationToken, ~azure.security.attestation.AttestationSigner], None]\n        :keyword bool validate_signature: If True, validate the signature of the token being validated.\n        :keyword bool validate_expiration: If True, validate the expiration time of the token being validated.\n        :keyword str issuer: Expected issuer, used if `validate_issuer` is true.\n        :keyword float validation_slack: Slack time for validation - tolerance applied\n            to help account for clock drift between the issuer and the current machine.\n        :keyword bool validate_issuer: If True, validate that the issuer of the token matches the expected issuer.\n        :keyword bool validate_not_before_time: If true, validate the "Not Before" time in the token.\n\n        :return: AttestationPolicyCertificateResult object describing the status\n            of the add request and the token sent from the service which\n            contained the response.\n\n        :rtype: Tuple[~azure.security.attestation.AttestationPolicyCertificateResult, ~azure.security.attestation.AttestationToken]\n\n        The :class:`AttestationPolicyCertificatesResult` response to the\n        :meth:`add_policy_management_certificate` API contains two attributes\n        of interest.\n\n        The first is `certificate_resolution`, which indicates\n        whether the certificate in question is present in the set of policy\n        management certificates after the operation has completed, or if it is\n        absent.\n\n        The second is the `thumbprint` of the certificate added. The `thumbprint`\n        for the certificate is the SHA1 hash of the DER encoding of the\n        certificate.\n\n        .. admonition:: Example: Generating and adding a new policy management\n            certificates for an isolated attestation instance.\n\n            .. literalinclude:: ../samples/sample_get_set_policy.py\n                :start-after: [BEGIN add_policy_management_certificate]\n                :end-before: [END add_policy_management_certificate]\n                :language: python\n                :dedent: 12\n                :caption: Adding a policy management certificate.\n\n        '
        if len(args) != 1:
            raise TypeError('add_policy_management_certificate takes a single positional parameter. found {}'.format(len(args)))
        certificate_to_add = args[0]
        signing_key = kwargs.pop('signing_key', self._signing_key)
        signing_certificate = kwargs.pop('signing_certificate', self._signing_certificate)
        if not signing_key or not signing_certificate:
            raise ValueError('A signing certificate and key must be provided to add_policy_management_certificate.')
        certificate_to_add = load_pem_x509_certificate(certificate_to_add.encode('ascii'))
        jwk = JSONWebKey(kty='RSA', x5_c=[base64.b64encode(certificate_to_add.public_bytes(serialization.Encoding.DER)).decode('ascii')])
        add_body = AttestationCertificateManagementBody(policy_certificate=jwk)
        cert_add_token = AttestationToken(body=add_body, signing_key=signing_key, signing_certificate=signing_certificate, body_type=AttestationCertificateManagementBody)
        options = merge_validation_args(self._config._kwargs, kwargs)
        cert_response = self._client.policy_certificates.add(cert_add_token.to_jwt_string(), **kwargs)
        token = AttestationToken(token=cert_response.token, body_type=GeneratedPolicyCertificatesModificationResult)
        if options.get('validate_token', True):
            token._validate_token(self._get_signers(**kwargs), **options)
        return (AttestationPolicyCertificateResult._from_generated(token._get_body()), token)

    @distributed_trace
    def remove_policy_management_certificate(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Removes a policy management certificate from the set of policy management certificates for the instance.\n\n        :param str certificate_to_remove: Required. PEM encoded X.509 certificate to remove from\n            the list of attestation policy management certificates.\n        :keyword str signing_key: PEM encoded signing Key representing the key\n            associated with one of the *existing* attestation signing certificates.\n        :keyword str signing_certificate: PEM encoded signing certificate which is one of\n            the *existing* attestation signing certificates.\n        :keyword bool validate_token: If True, validate the token, otherwise return the token unvalidated.\n        :keyword validation_callback: Function callback to allow clients to perform custom validation of the token.\n            if the token is invalid, the `validation_callback` function should throw\n            an exception.\n        :paramtype validation_callback: ~typing.Callable[[~azure.security.attestation.AttestationToken, ~azure.security.attestation.AttestationSigner], None]\n        :keyword bool validate_signature: If True, validate the signature of the token being validated.\n        :keyword bool validate_expiration: If True, validate the expiration time of the token being validated.\n        :keyword str issuer: Expected issuer, used if `validate_issuer` is true.\n        :keyword float validation_slack: Slack time for validation - tolerance applied\n            to help account for clock drift between the issuer and the current machine.\n        :keyword bool validate_issuer: If True, validate that the issuer of the token matches the expected issuer.\n        :keyword bool validate_not_before_time: If true, validate the "Not Before" time in the token.\n        :return: Result describing the outcome of the certificate removal.\n        :rtype: Tuple[~azure.security.attestation.AttestationPolicyCertificateResult, ~azure.security.attestation.AttestationToken]\n\n        The :class:`AttestationPolicyCertificateResult` response to the\n        :meth:`remove_policy_management_certificate` API contains two attributes\n        of interest.\n\n        The first is `certificate_resolution`, which indicates\n        whether the certificate in question is present in the set of policy\n        management certificates after the operation has completed, or if it is\n        absent.\n\n        The second is the `thumbprint` of the certificate added. The `thumbprint`\n        for the certificate is the SHA1 hash of the DER encoding of the\n        certificate.\n\n        .. admonition:: Example: Removing an added policy management\n            certificate for an isolated attestation instance.\n\n            .. literalinclude:: ../samples/sample_get_set_policy.py\n                :start-after: [BEGIN remove_policy_management_certificate]\n                :end-before: [END remove_policy_management_certificate]\n                :language: python\n                :dedent: 8\n                :caption: Removing a policy management certificate.\n\n        '
        if len(args) != 1:
            raise TypeError('remove_policy_management_certificate takes a single positional parameter. found {}'.format(len(args)))
        certificate_to_remove = args[0]
        signing_key = kwargs.pop('signing_key', self._signing_key)
        signing_certificate = kwargs.pop('signing_certificate', self._signing_certificate)
        if not signing_key or not signing_certificate:
            raise ValueError('A signing certificate and key must be provided to remove_policy_management_certificate.')
        certificate_to_remove = load_pem_x509_certificate(certificate_to_remove.encode('ascii'))
        jwk = JSONWebKey(kty='RSA', x5_c=[base64.b64encode(certificate_to_remove.public_bytes(serialization.Encoding.DER)).decode('ascii')])
        add_body = AttestationCertificateManagementBody(policy_certificate=jwk)
        cert_add_token = AttestationToken(body=add_body, signing_key=signing_key, signing_certificate=signing_certificate, body_type=AttestationCertificateManagementBody)
        options = merge_validation_args(self._config._kwargs, kwargs)
        cert_response = self._client.policy_certificates.remove(cert_add_token.to_jwt_string(), **kwargs)
        token = AttestationToken(token=cert_response.token, body_type=GeneratedPolicyCertificatesModificationResult)
        if options.get('validate_token', True):
            token._validate_token(self._get_signers(**kwargs), **options)
        return (AttestationPolicyCertificateResult._from_generated(token._get_body()), token)

    def _get_signers(self, **kwargs):
        if False:
            return 10
        'Returns the set of signing certificates used to sign attestation tokens.'
        with self._statelock:
            if not self._signing_certificates:
                signing_certificates = self._client.signing_certificates.get(**kwargs)
                self._signing_certificates = []
                for key in signing_certificates.keys:
                    self._signing_certificates.append(AttestationSigner._from_generated(key))
            signers = self._signing_certificates
        return signers

    def close(self):
        if False:
            i = 10
            return i + 15
        self._client.close()

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self._client.__enter__()
        return self

    def __exit__(self, *exc_details):
        if False:
            i = 10
            return i + 15
        self._client.__exit__(*exc_details)