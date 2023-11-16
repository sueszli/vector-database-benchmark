import base64
import json
import webauthn as pywebauthn
from webauthn.helpers import base64url_to_bytes, generate_challenge
from webauthn.helpers.exceptions import InvalidAuthenticationResponse, InvalidAuthenticatorDataStructure, InvalidRegistrationResponse, UnsupportedPublicKeyType
from webauthn.helpers.options_to_json import options_to_json
from webauthn.helpers.structs import AttestationConveyancePreference, AuthenticationCredential, AuthenticatorSelectionCriteria, PublicKeyCredentialDescriptor, RegistrationCredential, UserVerificationRequirement

class AuthenticationRejectedError(Exception):
    pass

class RegistrationRejectedError(Exception):
    pass

def _get_webauthn_user_public_key_credential_descriptors(user, *, rp_id):
    if False:
        while True:
            i = 10
    '\n    Returns a webauthn.WebAuthnUser instance corresponding\n    to the given user model, with properties suitable for\n    usage within the webauthn API.\n    '
    return [PublicKeyCredentialDescriptor(id=base64url_to_bytes(credential.credential_id)) for credential in user.webauthn]

def _get_webauthn_user_public_keys(user, *, rp_id):
    if False:
        print('Hello World!')
    return [(base64url_to_bytes(credential.public_key), credential.sign_count) for credential in user.webauthn]

def _webauthn_b64encode(source):
    if False:
        i = 10
        return i + 15
    return base64.urlsafe_b64encode(source).rstrip(b'=')

def generate_webauthn_challenge():
    if False:
        while True:
            i = 10
    "\n    Returns a random challenge suitable for use within\n    Webauthn's credential and configuration option objects.\n\n    See: https://w3c.github.io/webauthn/#cryptographic-challenges\n    "
    return generate_challenge()

def get_credential_options(user, *, challenge, rp_name, rp_id):
    if False:
        print('Hello World!')
    '\n    Returns a dictionary of options for credential creation\n    on the client side.\n    '
    _authenticator_selection = AuthenticatorSelectionCriteria()
    _authenticator_selection.user_verification = UserVerificationRequirement.DISCOURAGED
    options = pywebauthn.generate_registration_options(rp_id=rp_id, rp_name=rp_name, user_id=str(user.id), user_name=user.username, user_display_name=user.name or user.username, challenge=challenge, attestation=AttestationConveyancePreference.NONE, authenticator_selection=_authenticator_selection)
    return json.loads(options_to_json(options))

def get_assertion_options(user, *, challenge, rp_id):
    if False:
        i = 10
        return i + 15
    '\n    Returns a dictionary of options for assertion retrieval\n    on the client side.\n    '
    options = pywebauthn.generate_authentication_options(rp_id=rp_id, challenge=challenge, allow_credentials=_get_webauthn_user_public_key_credential_descriptors(user, rp_id=rp_id), user_verification=UserVerificationRequirement.DISCOURAGED)
    return json.loads(options_to_json(options))

def verify_registration_response(response, challenge, *, rp_id, origin):
    if False:
        return 10
    '\n    Validates the challenge and attestation information\n    sent from the client during device registration.\n\n    Returns a WebAuthnCredential on success.\n    Raises RegistrationRejectedError on failire.\n    '
    encoded_challenge = _webauthn_b64encode(challenge)
    try:
        _credential = RegistrationCredential.model_validate_json(response)
        return pywebauthn.verify_registration_response(credential=_credential, expected_challenge=encoded_challenge, expected_rp_id=rp_id, expected_origin=origin, require_user_verification=False)
    except (InvalidAuthenticatorDataStructure, InvalidRegistrationResponse, UnsupportedPublicKeyType) as e:
        raise RegistrationRejectedError(str(e))

def verify_assertion_response(assertion, *, challenge, user, origin, rp_id):
    if False:
        i = 10
        return i + 15
    '\n    Validates the challenge and assertion information\n    sent from the client during authentication.\n\n    Returns an updated signage count on success.\n    Raises AuthenticationRejectedError on failure.\n    '
    encoded_challenge = _webauthn_b64encode(challenge)
    webauthn_user_public_keys = _get_webauthn_user_public_keys(user, rp_id=rp_id)
    for (public_key, current_sign_count) in webauthn_user_public_keys:
        try:
            _credential = AuthenticationCredential.model_validate_json(assertion)
            return pywebauthn.verify_authentication_response(credential=_credential, expected_challenge=encoded_challenge, expected_rp_id=rp_id, expected_origin=origin, credential_public_key=public_key, credential_current_sign_count=current_sign_count, require_user_verification=False)
        except InvalidAuthenticationResponse:
            pass
    raise AuthenticationRejectedError('Invalid WebAuthn credential')