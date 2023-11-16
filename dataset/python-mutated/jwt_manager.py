import json
import logging
from os.path import exists, join
from typing import Optional, Union, cast
import jwt
from authlib.jose import JsonWebKey
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.management.color import color_style
from django.urls import reverse
from django.utils.module_loading import import_string
from jwt import api_jws
from jwt.algorithms import RSAAlgorithm
from .utils import build_absolute_uri, get_domain
logger = logging.getLogger(__name__)
PUBLIC_KEY: Optional[rsa.RSAPublicKey] = None

class JWTManagerBase:

    @classmethod
    def get_domain(cls) -> str:
        if False:
            for i in range(10):
                print('nop')
        return NotImplemented

    @classmethod
    def get_private_key(cls) -> rsa.RSAPrivateKey:
        if False:
            print('Hello World!')
        return NotImplemented

    @classmethod
    def get_public_key(cls) -> rsa.RSAPublicKey:
        if False:
            return 10
        return NotImplemented

    @classmethod
    def encode(cls, payload: dict) -> str:
        if False:
            for i in range(10):
                print('nop')
        return NotImplemented

    @classmethod
    def jws_encode(cls, payload: bytes, is_payload_detached: bool=True) -> str:
        if False:
            return 10
        return NotImplemented

    @classmethod
    def decode(cls, token: str, verify_expiration: bool=True, verify_aud: bool=False) -> dict:
        if False:
            print('Hello World!')
        return NotImplemented

    @classmethod
    def validate_configuration(cls):
        if False:
            i = 10
            return i + 15
        return NotImplemented

    @classmethod
    def get_jwks(cls) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return NotImplemented

    @classmethod
    def get_key_id(cls) -> str:
        if False:
            i = 10
            return i + 15
        return NotImplemented

    @classmethod
    def get_issuer(cls) -> str:
        if False:
            for i in range(10):
                print('nop')
        return NotImplemented

class JWTManager(JWTManagerBase):
    KEY_FILE_FOR_DEBUG = '.jwt_key.pem'

    @classmethod
    def get_domain(cls) -> str:
        if False:
            return 10
        return get_domain()

    @classmethod
    def get_private_key(cls) -> rsa.RSAPrivateKey:
        if False:
            print('Hello World!')
        pem = settings.RSA_PRIVATE_KEY
        if not pem:
            if settings.DEBUG:
                return cls._load_debug_private_key()
            raise ImproperlyConfigured('RSA_PRIVATE_KEY is required when DEBUG mode is disabled.')
        return cls._get_private_key(pem)

    @classmethod
    def _get_private_key(cls, pem: Union[str, bytes]) -> rsa.RSAPrivateKey:
        if False:
            i = 10
            return i + 15
        if isinstance(pem, str):
            pem = pem.encode('utf-8')
        password: Union[str, bytes, None] = settings.RSA_PRIVATE_PASSWORD
        if isinstance(password, str):
            password = password.encode('utf-8')
        return cast(rsa.RSAPrivateKey, serialization.load_pem_private_key(pem, password=password))

    @classmethod
    def _load_debug_private_key(cls) -> rsa.RSAPrivateKey:
        if False:
            return 10
        key_path = join(settings.PROJECT_ROOT, cls.KEY_FILE_FOR_DEBUG)
        if exists(key_path):
            return cls._load_local_private_key(key_path)
        return cls._create_local_private_key(key_path)

    @classmethod
    def _load_local_private_key(cls, path) -> rsa.RSAPrivateKey:
        if False:
            for i in range(10):
                print('nop')
        with open(path, 'rb') as key_file:
            return cast(rsa.RSAPrivateKey, serialization.load_pem_private_key(key_file.read(), password=None))

    @classmethod
    def _create_local_private_key(cls, path) -> rsa.RSAPrivateKey:
        if False:
            while True:
                i = 10
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        with open(path, 'wb') as p_key_file:
            p_key_file.write(private_key.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.TraditionalOpenSSL, encryption_algorithm=serialization.NoEncryption()))
        return private_key

    @classmethod
    def get_public_key(cls) -> rsa.RSAPublicKey:
        if False:
            for i in range(10):
                print('nop')
        global PUBLIC_KEY
        if PUBLIC_KEY is None:
            private_key = cls.get_private_key()
            PUBLIC_KEY = private_key.public_key()
        return PUBLIC_KEY

    @classmethod
    def get_jwks(cls) -> dict:
        if False:
            while True:
                i = 10
        jwk_dict = json.loads(RSAAlgorithm.to_jwk(cls.get_public_key()))
        jwk_dict.update({'use': 'sig', 'kid': cls.get_key_id()})
        return {'keys': [jwk_dict]}

    @classmethod
    def get_key_id(cls) -> str:
        if False:
            while True:
                i = 10
        'Generate JWT key ID for the public key.\n\n        This generates a "thumbprint" as \'kid\' field using RFC 7638 implementation\n        based on the RSA public key.\n        '
        public_key_pem = cls.get_public_key().public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)
        jwk = JsonWebKey.import_key(public_key_pem)
        return jwk.thumbprint()

    @classmethod
    def encode(cls, payload):
        if False:
            for i in range(10):
                print('nop')
        return jwt.encode(payload, cls.get_private_key(), algorithm='RS256', headers={'kid': cls.get_key_id()})

    @classmethod
    def jws_encode(cls, payload: bytes, is_payload_detached: bool=True) -> str:
        if False:
            print('Hello World!')
        return api_jws.encode(payload, key=cls.get_private_key(), algorithm='RS256', headers={'kid': cls.get_key_id()}, is_payload_detached=is_payload_detached)

    @classmethod
    def decode(cls, token, verify_expiration: bool=True, verify_aud: bool=False):
        if False:
            i = 10
            return i + 15
        headers = jwt.get_unverified_header(token)
        if headers.get('alg') == 'RS256':
            return jwt.decode(token, cls.get_public_key(), algorithms=['RS256'], options={'verify_exp': verify_expiration, 'verify_aud': verify_aud})
        return jwt.decode(token, cast(str, settings.SECRET_KEY), algorithms=['HS256'], options={'verify_exp': verify_expiration, 'verify_aud': verify_aud})

    @classmethod
    def validate_configuration(cls):
        if False:
            while True:
                i = 10
        if not settings.RSA_PRIVATE_KEY:
            if not settings.DEBUG:
                raise ImproperlyConfigured('Variable RSA_PRIVATE_KEY is not provided. It is required for running in not DEBUG mode.')
            else:
                msg = 'RSA_PRIVATE_KEY is missing. Using temporary key for local development with DEBUG mode.'
                logger.warning(color_style().WARNING(msg))
        try:
            cls.get_private_key()
        except Exception as e:
            raise ImproperlyConfigured(f'Unable to load provided PEM private key. {e}')

    @classmethod
    def get_issuer(cls) -> str:
        if False:
            return 10
        return build_absolute_uri(reverse('api'), domain=cls.get_domain())

def get_jwt_manager() -> JWTManagerBase:
    if False:
        return 10
    return import_string(settings.JWT_MANAGER_PATH)