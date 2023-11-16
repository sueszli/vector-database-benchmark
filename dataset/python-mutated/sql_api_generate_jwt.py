from __future__ import annotations
import base64
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
import jwt
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
logger = logging.getLogger(__name__)
ISSUER = 'iss'
EXPIRE_TIME = 'exp'
ISSUE_TIME = 'iat'
SUBJECT = 'sub'

class JWTGenerator:
    """
    Creates and signs a JWT with the specified private key file, username, and account identifier.

    The JWTGenerator keeps the generated token and only regenerates the token if a specified period
    of time has passed.

    Creates an object that generates JWTs for the specified user, account identifier, and private key

    :param account: Your Snowflake account identifier.
        See https://docs.snowflake.com/en/user-guide/admin-account-identifier.html. Note that if you are using
        the account locator, exclude any region information from the account locator.
    :param user: The Snowflake username.
    :param private_key: Private key from the file path for signing the JWTs.
    :param lifetime: The number of minutes (as a timedelta) during which the key will be valid.
    :param renewal_delay: The number of minutes (as a timedelta) from now after which the JWT
        generator should renew the JWT.
    """
    LIFETIME = timedelta(minutes=59)
    RENEWAL_DELTA = timedelta(minutes=54)
    ALGORITHM = 'RS256'

    def __init__(self, account: str, user: str, private_key: Any, lifetime: timedelta=LIFETIME, renewal_delay: timedelta=RENEWAL_DELTA):
        if False:
            print('Hello World!')
        logger.info('Creating JWTGenerator with arguments\n            account : %s, user : %s, lifetime : %s, renewal_delay : %s', account, user, lifetime, renewal_delay)
        self.account = self.prepare_account_name_for_jwt(account)
        self.user = user.upper()
        self.qualified_username = self.account + '.' + self.user
        self.lifetime = lifetime
        self.renewal_delay = renewal_delay
        self.private_key = private_key
        self.renew_time = datetime.now(timezone.utc)
        self.token: str | None = None

    def prepare_account_name_for_jwt(self, raw_account: str) -> str:
        if False:
            return 10
        '\n        Prepare the account identifier for use in the JWT.\n\n        For the JWT, the account identifier must not include the subdomain or any region or cloud provider\n        information.\n\n        :param raw_account: The specified account identifier.\n        '
        account = raw_account
        if '.global' not in account:
            account = account.partition('.')[0]
        else:
            account = account.partition('-')[0]
        return account.upper()

    def get_token(self) -> str | None:
        if False:
            i = 10
            return i + 15
        '\n        Generates a new JWT.\n\n        If a JWT has been already been generated earlier, return the previously\n        generated token unless the specified renewal time has passed.\n        '
        now = datetime.now(timezone.utc)
        if self.token is None or self.renew_time <= now:
            logger.info('Generating a new token because the present time (%s) is later than the renewal time (%s)', now, self.renew_time)
            self.renew_time = now + self.renewal_delay
            public_key_fp = self.calculate_public_key_fingerprint(self.private_key)
            payload = {ISSUER: self.qualified_username + '.' + public_key_fp, SUBJECT: self.qualified_username, ISSUE_TIME: now, EXPIRE_TIME: now + self.lifetime}
            token = jwt.encode(payload, key=self.private_key, algorithm=JWTGenerator.ALGORITHM)
            if isinstance(token, bytes):
                token = token.decode('utf-8')
            self.token = token
        return self.token

    def calculate_public_key_fingerprint(self, private_key: Any) -> str:
        if False:
            print('Hello World!')
        '\n        Given a private key in PEM format, return the public key fingerprint.\n\n        :param private_key: private key\n        '
        public_key_raw = private_key.public_key().public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
        sha256hash = hashlib.sha256()
        sha256hash.update(public_key_raw)
        public_key_fp = 'SHA256:' + base64.b64encode(sha256hash.digest()).decode('utf-8')
        return public_key_fp