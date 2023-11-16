import json
import time
from calendar import timegm
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import pytest
from jwt.api_jwt import PyJWT
from jwt.exceptions import DecodeError, ExpiredSignatureError, ImmatureSignatureError, InvalidAudienceError, InvalidIssuedAtError, InvalidIssuerError, MissingRequiredClaimError
from jwt.utils import base64url_decode
from jwt.warnings import RemovedInPyjwt3Warning
from .utils import crypto_required, key_path, utc_timestamp

@pytest.fixture
def jwt():
    if False:
        while True:
            i = 10
    return PyJWT()

@pytest.fixture
def payload():
    if False:
        print('Hello World!')
    'Creates a sample JWT claimset for use as a payload during tests'
    return {'iss': 'jeff', 'exp': utc_timestamp() + 15, 'claim': 'insanity'}

class TestJWT:

    def test_decodes_valid_jwt(self, jwt):
        if False:
            while True:
                i = 10
        example_payload = {'hello': 'world'}
        example_secret = 'secret'
        example_jwt = b'eyJhbGciOiAiSFMyNTYiLCAidHlwIjogIkpXVCJ9.eyJoZWxsbyI6ICJ3b3JsZCJ9.tvagLDLoaiJKxOKqpBXSEGy7SYSifZhjntgm9ctpyj8'
        decoded_payload = jwt.decode(example_jwt, example_secret, algorithms=['HS256'])
        assert decoded_payload == example_payload

    def test_decodes_complete_valid_jwt(self, jwt):
        if False:
            i = 10
            return i + 15
        example_payload = {'hello': 'world'}
        example_secret = 'secret'
        example_jwt = b'eyJhbGciOiAiSFMyNTYiLCAidHlwIjogIkpXVCJ9.eyJoZWxsbyI6ICJ3b3JsZCJ9.tvagLDLoaiJKxOKqpBXSEGy7SYSifZhjntgm9ctpyj8'
        decoded = jwt.decode_complete(example_jwt, example_secret, algorithms=['HS256'])
        assert decoded == {'header': {'alg': 'HS256', 'typ': 'JWT'}, 'payload': example_payload, 'signature': b'\xb6\xf6\xa0,2\xe8j"J\xc4\xe2\xaa\xa4\x15\xd2\x10l\xbbI\x84\xa2}\x98c\x9e\xd8&\xf5\xcbi\xca?'}

    def test_load_verify_valid_jwt(self, jwt):
        if False:
            print('Hello World!')
        example_payload = {'hello': 'world'}
        example_secret = 'secret'
        example_jwt = b'eyJhbGciOiAiSFMyNTYiLCAidHlwIjogIkpXVCJ9.eyJoZWxsbyI6ICJ3b3JsZCJ9.tvagLDLoaiJKxOKqpBXSEGy7SYSifZhjntgm9ctpyj8'
        decoded_payload = jwt.decode(example_jwt, key=example_secret, algorithms=['HS256'])
        assert decoded_payload == example_payload

    def test_decode_invalid_payload_string(self, jwt):
        if False:
            for i in range(10):
                print('nop')
        example_jwt = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.aGVsbG8gd29ybGQ.SIr03zM64awWRdPrAM_61QWsZchAtgDV3pphfHPPWkI'
        example_secret = 'secret'
        with pytest.raises(DecodeError) as exc:
            jwt.decode(example_jwt, example_secret, algorithms=['HS256'])
        assert 'Invalid payload string' in str(exc.value)

    def test_decode_with_non_mapping_payload_throws_exception(self, jwt):
        if False:
            i = 10
            return i + 15
        secret = 'secret'
        example_jwt = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.MQ.AbcSR3DWum91KOgfKxUHm78rLs_DrrZ1CrDgpUFFzls'
        with pytest.raises(DecodeError) as context:
            jwt.decode(example_jwt, secret, algorithms=['HS256'])
        exception = context.value
        assert str(exception) == 'Invalid payload string: must be a json object'

    def test_decode_with_invalid_audience_param_throws_exception(self, jwt):
        if False:
            for i in range(10):
                print('nop')
        secret = 'secret'
        example_jwt = 'eyJhbGciOiAiSFMyNTYiLCAidHlwIjogIkpXVCJ9.eyJoZWxsbyI6ICJ3b3JsZCJ9.tvagLDLoaiJKxOKqpBXSEGy7SYSifZhjntgm9ctpyj8'
        with pytest.raises(TypeError) as context:
            jwt.decode(example_jwt, secret, audience=1, algorithms=['HS256'])
        exception = context.value
        assert str(exception) == 'audience must be a string, iterable or None'

    def test_decode_with_nonlist_aud_claim_throws_exception(self, jwt):
        if False:
            for i in range(10):
                print('nop')
        secret = 'secret'
        example_jwt = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJoZWxsbyI6IndvcmxkIiwiYXVkIjoxfQ.Rof08LBSwbm8Z_bhA2N3DFY-utZR1Gi9rbIS5Zthnnc'
        with pytest.raises(InvalidAudienceError) as context:
            jwt.decode(example_jwt, secret, audience='my_audience', algorithms=['HS256'])
        exception = context.value
        assert str(exception) == 'Invalid claim format in token'

    def test_decode_with_invalid_aud_list_member_throws_exception(self, jwt):
        if False:
            for i in range(10):
                print('nop')
        secret = 'secret'
        example_jwt = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJoZWxsbyI6IndvcmxkIiwiYXVkIjpbMV19.iQgKpJ8shetwNMIosNXWBPFB057c2BHs-8t1d2CCM2A'
        with pytest.raises(InvalidAudienceError) as context:
            jwt.decode(example_jwt, secret, audience='my_audience', algorithms=['HS256'])
        exception = context.value
        assert str(exception) == 'Invalid claim format in token'

    def test_encode_bad_type(self, jwt):
        if False:
            return 10
        types = ['string', tuple(), list(), 42, set()]
        for t in types:
            pytest.raises(TypeError, lambda : jwt.encode(t, 'secret', algorithms=['HS256']))

    def test_encode_with_typ(self, jwt):
        if False:
            i = 10
            return i + 15
        payload = {'iss': 'https://scim.example.com', 'iat': 1458496404, 'jti': '4d3559ec67504aaba65d40b0363faad8', 'aud': ['https://scim.example.com/Feeds/98d52461fa5bbc879593b7754', 'https://scim.example.com/Feeds/5d7604516b1d08641d7676ee7'], 'events': {'urn:ietf:params:scim:event:create': {'ref': 'https://scim.example.com/Users/44f6142df96bd6ab61e7521d9', 'attributes': ['id', 'name', 'userName', 'password', 'emails']}}}
        token = jwt.encode(payload, 'secret', algorithm='HS256', headers={'typ': 'secevent+jwt'})
        header = token[0:token.index('.')].encode()
        header = base64url_decode(header)
        header_obj = json.loads(header)
        assert 'typ' in header_obj
        assert header_obj['typ'] == 'secevent+jwt'

    def test_decode_raises_exception_if_exp_is_not_int(self, jwt):
        if False:
            while True:
                i = 10
        example_jwt = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOiJub3QtYW4taW50In0.P65iYgoHtBqB07PMtBSuKNUEIPPPfmjfJG217cEE66s'
        with pytest.raises(DecodeError) as exc:
            jwt.decode(example_jwt, 'secret', algorithms=['HS256'])
        assert 'exp' in str(exc.value)

    def test_decode_raises_exception_if_iat_is_not_int(self, jwt):
        if False:
            for i in range(10):
                print('nop')
        example_jwt = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOiJub3QtYW4taW50In0.H1GmcQgSySa5LOKYbzGm--b1OmRbHFkyk8pq811FzZM'
        with pytest.raises(InvalidIssuedAtError):
            jwt.decode(example_jwt, 'secret', algorithms=['HS256'])

    def test_decode_raises_exception_if_iat_is_greater_than_now(self, jwt, payload):
        if False:
            return 10
        payload['iat'] = utc_timestamp() + 10
        secret = 'secret'
        jwt_message = jwt.encode(payload, secret)
        with pytest.raises(ImmatureSignatureError):
            jwt.decode(jwt_message, secret, algorithms=['HS256'])

    def test_decode_works_if_iat_is_str_of_a_number(self, jwt, payload):
        if False:
            return 10
        payload['iat'] = '1638202770'
        secret = 'secret'
        jwt_message = jwt.encode(payload, secret)
        data = jwt.decode(jwt_message, secret, algorithms=['HS256'])
        assert data['iat'] == '1638202770'

    def test_decode_raises_exception_if_nbf_is_not_int(self, jwt):
        if False:
            i = 10
            return i + 15
        example_jwt = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYmYiOiJub3QtYW4taW50In0.c25hldC8G2ZamC8uKpax9sYMTgdZo3cxrmzFHaAAluw'
        with pytest.raises(DecodeError):
            jwt.decode(example_jwt, 'secret', algorithms=['HS256'])

    def test_decode_raises_exception_if_aud_is_none(self, jwt):
        if False:
            while True:
                i = 10
        example_jwt = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhdWQiOm51bGx9.-Peqc-pTugGvrc5C8Bnl0-X1V_5fv-aVb_7y7nGBVvQ'
        decoded = jwt.decode(example_jwt, 'secret', algorithms=['HS256'])
        assert decoded['aud'] is None

    def test_encode_datetime(self, jwt):
        if False:
            return 10
        secret = 'secret'
        current_datetime = datetime.now(tz=timezone.utc)
        payload = {'exp': current_datetime, 'iat': current_datetime, 'nbf': current_datetime}
        jwt_message = jwt.encode(payload, secret)
        decoded_payload = jwt.decode(jwt_message, secret, leeway=1, algorithms=['HS256'])
        assert decoded_payload['exp'] == timegm(current_datetime.utctimetuple())
        assert decoded_payload['iat'] == timegm(current_datetime.utctimetuple())
        assert decoded_payload['nbf'] == timegm(current_datetime.utctimetuple())
        assert payload == {'exp': current_datetime, 'iat': current_datetime, 'nbf': current_datetime}

    @crypto_required
    def test_decodes_valid_es256_jwt(self, jwt):
        if False:
            while True:
                i = 10
        example_payload = {'hello': 'world'}
        with open(key_path('testkey_ec.pub')) as fp:
            example_pubkey = fp.read()
        example_jwt = b'eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJoZWxsbyI6IndvcmxkIn0.TORyNQab_MoXM7DvNKaTwbrJr4UYd2SsX8hhlnWelQFmPFSf_JzC2EbLnar92t-bXsDovzxp25ExazrVHkfPkQ'
        decoded_payload = jwt.decode(example_jwt, example_pubkey, algorithms=['ES256'])
        assert decoded_payload == example_payload

    @crypto_required
    def test_decodes_valid_rs384_jwt(self, jwt):
        if False:
            for i in range(10):
                print('nop')
        example_payload = {'hello': 'world'}
        with open(key_path('testkey_rsa.pub')) as fp:
            example_pubkey = fp.read()
        example_jwt = b'eyJhbGciOiJSUzM4NCIsInR5cCI6IkpXVCJ9.eyJoZWxsbyI6IndvcmxkIn0.yNQ3nI9vEDs7lEh-Cp81McPuiQ4ZRv6FL4evTYYAh1XlRTTR3Cz8pPA9Stgso8Ra9xGB4X3rlra1c8Jz10nTUjuO06OMm7oXdrnxp1KIiAJDerWHkQ7l3dlizIk1bmMA457W2fNzNfHViuED5ISM081dgf_a71qBwJ_yShMMrSOfxDxmX9c4DjRogRJG8SM5PvpLqI_Cm9iQPGMvmYK7gzcq2cJurHRJDJHTqIdpLWXkY7zVikeen6FhuGyn060Dz9gYq9tuwmrtSWCBUjiN8sqJ00CDgycxKqHfUndZbEAOjcCAhBrqWW3mSVivUfubsYbwUdUG3fSRPjaUPcpe8A'
        decoded_payload = jwt.decode(example_jwt, example_pubkey, algorithms=['RS384'])
        assert decoded_payload == example_payload

    def test_decode_with_expiration(self, jwt, payload):
        if False:
            while True:
                i = 10
        payload['exp'] = utc_timestamp() - 1
        secret = 'secret'
        jwt_message = jwt.encode(payload, secret)
        with pytest.raises(ExpiredSignatureError):
            jwt.decode(jwt_message, secret, algorithms=['HS256'])

    def test_decode_with_notbefore(self, jwt, payload):
        if False:
            while True:
                i = 10
        payload['nbf'] = utc_timestamp() + 10
        secret = 'secret'
        jwt_message = jwt.encode(payload, secret)
        with pytest.raises(ImmatureSignatureError):
            jwt.decode(jwt_message, secret, algorithms=['HS256'])

    def test_decode_skip_expiration_verification(self, jwt, payload):
        if False:
            for i in range(10):
                print('nop')
        payload['exp'] = time.time() - 1
        secret = 'secret'
        jwt_message = jwt.encode(payload, secret)
        jwt.decode(jwt_message, secret, algorithms=['HS256'], options={'verify_exp': False})

    def test_decode_skip_notbefore_verification(self, jwt, payload):
        if False:
            while True:
                i = 10
        payload['nbf'] = time.time() + 10
        secret = 'secret'
        jwt_message = jwt.encode(payload, secret)
        jwt.decode(jwt_message, secret, algorithms=['HS256'], options={'verify_nbf': False})

    def test_decode_with_expiration_with_leeway(self, jwt, payload):
        if False:
            return 10
        payload['exp'] = utc_timestamp() - 2
        secret = 'secret'
        jwt_message = jwt.encode(payload, secret)
        for leeway in (5, timedelta(seconds=5)):
            decoded = jwt.decode(jwt_message, secret, leeway=leeway, algorithms=['HS256'])
            assert decoded == payload
        for leeway in (1, timedelta(seconds=1)):
            with pytest.raises(ExpiredSignatureError):
                jwt.decode(jwt_message, secret, leeway=leeway, algorithms=['HS256'])

    def test_decode_with_notbefore_with_leeway(self, jwt, payload):
        if False:
            return 10
        payload['nbf'] = utc_timestamp() + 10
        secret = 'secret'
        jwt_message = jwt.encode(payload, secret)
        jwt.decode(jwt_message, secret, leeway=13, algorithms=['HS256'])
        with pytest.raises(ImmatureSignatureError):
            jwt.decode(jwt_message, secret, leeway=1, algorithms=['HS256'])

    def test_check_audience_when_valid(self, jwt):
        if False:
            print('Hello World!')
        payload = {'some': 'payload', 'aud': 'urn:me'}
        token = jwt.encode(payload, 'secret')
        jwt.decode(token, 'secret', audience='urn:me', algorithms=['HS256'])

    def test_check_audience_list_when_valid(self, jwt):
        if False:
            i = 10
            return i + 15
        payload = {'some': 'payload', 'aud': 'urn:me'}
        token = jwt.encode(payload, 'secret')
        jwt.decode(token, 'secret', audience=['urn:you', 'urn:me'], algorithms=['HS256'])

    def test_check_audience_none_specified(self, jwt):
        if False:
            for i in range(10):
                print('nop')
        payload = {'some': 'payload', 'aud': 'urn:me'}
        token = jwt.encode(payload, 'secret')
        with pytest.raises(InvalidAudienceError):
            jwt.decode(token, 'secret', algorithms=['HS256'])

    def test_raise_exception_invalid_audience_list(self, jwt):
        if False:
            for i in range(10):
                print('nop')
        payload = {'some': 'payload', 'aud': 'urn:me'}
        token = jwt.encode(payload, 'secret')
        with pytest.raises(InvalidAudienceError):
            jwt.decode(token, 'secret', audience=['urn:you', 'urn:him'], algorithms=['HS256'])

    def test_check_audience_in_array_when_valid(self, jwt):
        if False:
            for i in range(10):
                print('nop')
        payload = {'some': 'payload', 'aud': ['urn:me', 'urn:someone-else']}
        token = jwt.encode(payload, 'secret')
        jwt.decode(token, 'secret', audience='urn:me', algorithms=['HS256'])

    def test_raise_exception_invalid_audience(self, jwt):
        if False:
            print('Hello World!')
        payload = {'some': 'payload', 'aud': 'urn:someone-else'}
        token = jwt.encode(payload, 'secret')
        with pytest.raises(InvalidAudienceError):
            jwt.decode(token, 'secret', audience='urn-me', algorithms=['HS256'])

    def test_raise_exception_audience_as_bytes(self, jwt):
        if False:
            for i in range(10):
                print('nop')
        payload = {'some': 'payload', 'aud': ['urn:me', 'urn:someone-else']}
        token = jwt.encode(payload, 'secret')
        with pytest.raises(InvalidAudienceError):
            jwt.decode(token, 'secret', audience=b'urn:me', algorithms=['HS256'])

    def test_raise_exception_invalid_audience_in_array(self, jwt):
        if False:
            for i in range(10):
                print('nop')
        payload = {'some': 'payload', 'aud': ['urn:someone', 'urn:someone-else']}
        token = jwt.encode(payload, 'secret')
        with pytest.raises(InvalidAudienceError):
            jwt.decode(token, 'secret', audience='urn:me', algorithms=['HS256'])

    def test_raise_exception_token_without_issuer(self, jwt):
        if False:
            i = 10
            return i + 15
        issuer = 'urn:wrong'
        payload = {'some': 'payload'}
        token = jwt.encode(payload, 'secret')
        with pytest.raises(MissingRequiredClaimError) as exc:
            jwt.decode(token, 'secret', issuer=issuer, algorithms=['HS256'])
        assert exc.value.claim == 'iss'

    def test_raise_exception_token_without_audience(self, jwt):
        if False:
            for i in range(10):
                print('nop')
        payload = {'some': 'payload'}
        token = jwt.encode(payload, 'secret')
        with pytest.raises(MissingRequiredClaimError) as exc:
            jwt.decode(token, 'secret', audience='urn:me', algorithms=['HS256'])
        assert exc.value.claim == 'aud'

    def test_raise_exception_token_with_aud_none_and_without_audience(self, jwt):
        if False:
            return 10
        payload = {'some': 'payload', 'aud': None}
        token = jwt.encode(payload, 'secret')
        with pytest.raises(MissingRequiredClaimError) as exc:
            jwt.decode(token, 'secret', audience='urn:me', algorithms=['HS256'])
        assert exc.value.claim == 'aud'

    def test_check_issuer_when_valid(self, jwt):
        if False:
            print('Hello World!')
        issuer = 'urn:foo'
        payload = {'some': 'payload', 'iss': 'urn:foo'}
        token = jwt.encode(payload, 'secret')
        jwt.decode(token, 'secret', issuer=issuer, algorithms=['HS256'])

    def test_check_issuer_list_when_valid(self, jwt):
        if False:
            for i in range(10):
                print('nop')
        issuer = ['urn:foo', 'urn:bar']
        payload = {'some': 'payload', 'iss': 'urn:foo'}
        token = jwt.encode(payload, 'secret')
        jwt.decode(token, 'secret', issuer=issuer, algorithms=['HS256'])

    def test_raise_exception_invalid_issuer(self, jwt):
        if False:
            return 10
        issuer = 'urn:wrong'
        payload = {'some': 'payload', 'iss': 'urn:foo'}
        token = jwt.encode(payload, 'secret')
        with pytest.raises(InvalidIssuerError):
            jwt.decode(token, 'secret', issuer=issuer, algorithms=['HS256'])

    def test_raise_exception_invalid_issuer_list(self, jwt):
        if False:
            return 10
        issuer = ['urn:wrong', 'urn:bar', 'urn:baz']
        payload = {'some': 'payload', 'iss': 'urn:foo'}
        token = jwt.encode(payload, 'secret')
        with pytest.raises(InvalidIssuerError):
            jwt.decode(token, 'secret', issuer=issuer, algorithms=['HS256'])

    def test_skip_check_audience(self, jwt):
        if False:
            i = 10
            return i + 15
        payload = {'some': 'payload', 'aud': 'urn:me'}
        token = jwt.encode(payload, 'secret')
        jwt.decode(token, 'secret', options={'verify_aud': False}, algorithms=['HS256'])

    def test_skip_check_exp(self, jwt):
        if False:
            while True:
                i = 10
        payload = {'some': 'payload', 'exp': datetime.now(tz=timezone.utc) - timedelta(days=1)}
        token = jwt.encode(payload, 'secret')
        jwt.decode(token, 'secret', options={'verify_exp': False}, algorithms=['HS256'])

    def test_decode_should_raise_error_if_exp_required_but_not_present(self, jwt):
        if False:
            while True:
                i = 10
        payload = {'some': 'payload'}
        token = jwt.encode(payload, 'secret')
        with pytest.raises(MissingRequiredClaimError) as exc:
            jwt.decode(token, 'secret', options={'require': ['exp']}, algorithms=['HS256'])
        assert exc.value.claim == 'exp'

    def test_decode_should_raise_error_if_iat_required_but_not_present(self, jwt):
        if False:
            for i in range(10):
                print('nop')
        payload = {'some': 'payload'}
        token = jwt.encode(payload, 'secret')
        with pytest.raises(MissingRequiredClaimError) as exc:
            jwt.decode(token, 'secret', options={'require': ['iat']}, algorithms=['HS256'])
        assert exc.value.claim == 'iat'

    def test_decode_should_raise_error_if_nbf_required_but_not_present(self, jwt):
        if False:
            i = 10
            return i + 15
        payload = {'some': 'payload'}
        token = jwt.encode(payload, 'secret')
        with pytest.raises(MissingRequiredClaimError) as exc:
            jwt.decode(token, 'secret', options={'require': ['nbf']}, algorithms=['HS256'])
        assert exc.value.claim == 'nbf'

    def test_skip_check_signature(self, jwt):
        if False:
            while True:
                i = 10
        token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb21lIjoicGF5bG9hZCJ9.4twFt5NiznN84AWoo1d7KO1T_yoc0Z6XOpOVswacPZA'
        jwt.decode(token, 'secret', options={'verify_signature': False}, algorithms=['HS256'])

    def test_skip_check_iat(self, jwt):
        if False:
            for i in range(10):
                print('nop')
        payload = {'some': 'payload', 'iat': datetime.now(tz=timezone.utc) + timedelta(days=1)}
        token = jwt.encode(payload, 'secret')
        jwt.decode(token, 'secret', options={'verify_iat': False}, algorithms=['HS256'])

    def test_skip_check_nbf(self, jwt):
        if False:
            i = 10
            return i + 15
        payload = {'some': 'payload', 'nbf': datetime.now(tz=timezone.utc) + timedelta(days=1)}
        token = jwt.encode(payload, 'secret')
        jwt.decode(token, 'secret', options={'verify_nbf': False}, algorithms=['HS256'])

    def test_custom_json_encoder(self, jwt):
        if False:
            for i in range(10):
                print('nop')

        class CustomJSONEncoder(json.JSONEncoder):

            def default(self, o):
                if False:
                    while True:
                        i = 10
                if isinstance(o, Decimal):
                    return 'it worked'
                return super().default(o)
        data = {'some_decimal': Decimal('2.2')}
        with pytest.raises(TypeError):
            jwt.encode(data, 'secret', algorithms=['HS256'])
        token = jwt.encode(data, 'secret', json_encoder=CustomJSONEncoder)
        payload = jwt.decode(token, 'secret', algorithms=['HS256'])
        assert payload == {'some_decimal': 'it worked'}

    def test_decode_with_verify_exp_option(self, jwt, payload):
        if False:
            print('Hello World!')
        payload['exp'] = utc_timestamp() - 1
        secret = 'secret'
        jwt_message = jwt.encode(payload, secret)
        jwt.decode(jwt_message, secret, algorithms=['HS256'], options={'verify_exp': False})
        with pytest.raises(ExpiredSignatureError):
            jwt.decode(jwt_message, secret, algorithms=['HS256'], options={'verify_exp': True})

    def test_decode_with_verify_exp_option_and_signature_off(self, jwt, payload):
        if False:
            return 10
        payload['exp'] = utc_timestamp() - 1
        secret = 'secret'
        jwt_message = jwt.encode(payload, secret)
        jwt.decode(jwt_message, options={'verify_signature': False})
        with pytest.raises(ExpiredSignatureError):
            jwt.decode(jwt_message, options={'verify_signature': False, 'verify_exp': True})

    def test_decode_with_optional_algorithms(self, jwt, payload):
        if False:
            for i in range(10):
                print('nop')
        secret = 'secret'
        jwt_message = jwt.encode(payload, secret)
        with pytest.raises(DecodeError) as exc:
            jwt.decode(jwt_message, secret)
        assert 'It is required that you pass in a value for the "algorithms" argument when calling decode().' in str(exc.value)

    def test_decode_no_algorithms_verify_signature_false(self, jwt, payload):
        if False:
            while True:
                i = 10
        secret = 'secret'
        jwt_message = jwt.encode(payload, secret)
        jwt.decode(jwt_message, secret, options={'verify_signature': False})

    def test_decode_legacy_verify_warning(self, jwt, payload):
        if False:
            for i in range(10):
                print('nop')
        secret = 'secret'
        jwt_message = jwt.encode(payload, secret)
        with pytest.deprecated_call():
            jwt.decode(jwt_message, secret, verify=False, algorithms=['HS256'])
        with pytest.deprecated_call():
            jwt.decode(jwt_message, secret, verify=True, options={'verify_signature': False})

    def test_decode_no_options_mutation(self, jwt, payload):
        if False:
            i = 10
            return i + 15
        options = {'verify_signature': True}
        orig_options = options.copy()
        secret = 'secret'
        jwt_message = jwt.encode(payload, secret)
        jwt.decode(jwt_message, secret, options=options, algorithms=['HS256'])
        assert options == orig_options

    def test_decode_warns_on_unsupported_kwarg(self, jwt, payload):
        if False:
            for i in range(10):
                print('nop')
        secret = 'secret'
        jwt_message = jwt.encode(payload, secret)
        with pytest.warns(RemovedInPyjwt3Warning) as record:
            jwt.decode(jwt_message, secret, algorithms=['HS256'], foo='bar')
        assert len(record) == 1
        assert 'foo' in str(record[0].message)

    def test_decode_complete_warns_on_unsupported_kwarg(self, jwt, payload):
        if False:
            for i in range(10):
                print('nop')
        secret = 'secret'
        jwt_message = jwt.encode(payload, secret)
        with pytest.warns(RemovedInPyjwt3Warning) as record:
            jwt.decode_complete(jwt_message, secret, algorithms=['HS256'], foo='bar')
        assert len(record) == 1
        assert 'foo' in str(record[0].message)

    def test_decode_strict_aud_forbids_list_audience(self, jwt, payload):
        if False:
            while True:
                i = 10
        secret = 'secret'
        payload['aud'] = 'urn:foo'
        jwt_message = jwt.encode(payload, secret)
        jwt.decode(jwt_message, secret, audience=['urn:foo', 'urn:bar'], options={'strict_aud': False}, algorithms=['HS256'])
        with pytest.raises(InvalidAudienceError, match='Invalid audience \\(strict\\)'):
            jwt.decode(jwt_message, secret, audience=['urn:foo', 'urn:bar'], options={'strict_aud': True}, algorithms=['HS256'])

    def test_decode_strict_aud_forbids_list_claim(self, jwt, payload):
        if False:
            while True:
                i = 10
        secret = 'secret'
        payload['aud'] = ['urn:foo', 'urn:bar']
        jwt_message = jwt.encode(payload, secret)
        jwt.decode(jwt_message, secret, audience='urn:foo', options={'strict_aud': False}, algorithms=['HS256'])
        with pytest.raises(InvalidAudienceError, match='Invalid claim format in token \\(strict\\)'):
            jwt.decode(jwt_message, secret, audience='urn:foo', options={'strict_aud': True}, algorithms=['HS256'])

    def test_decode_strict_aud_does_not_match(self, jwt, payload):
        if False:
            while True:
                i = 10
        secret = 'secret'
        payload['aud'] = 'urn:foo'
        jwt_message = jwt.encode(payload, secret)
        with pytest.raises(InvalidAudienceError, match="Audience doesn't match \\(strict\\)"):
            jwt.decode(jwt_message, secret, audience='urn:bar', options={'strict_aud': True}, algorithms=['HS256'])

    def test_decode_strict_ok(self, jwt, payload):
        if False:
            return 10
        secret = 'secret'
        payload['aud'] = 'urn:foo'
        jwt_message = jwt.encode(payload, secret)
        jwt.decode(jwt_message, secret, audience='urn:foo', options={'strict_aud': True}, algorithms=['HS256'])