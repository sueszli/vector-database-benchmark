"""
Credential managers for L{twisted.mail}.
"""
import hashlib
import hmac
from zope.interface import implementer
from twisted.cred import credentials
from twisted.mail._except import IllegalClientResponse
from twisted.mail.interfaces import IChallengeResponse, IClientAuthentication
from twisted.python.compat import nativeString

@implementer(IClientAuthentication)
class CramMD5ClientAuthenticator:

    def __init__(self, user):
        if False:
            return 10
        self.user = user

    def getName(self):
        if False:
            print('Hello World!')
        return b'CRAM-MD5'

    def challengeResponse(self, secret, chal):
        if False:
            for i in range(10):
                print('nop')
        response = hmac.HMAC(secret, chal, digestmod=hashlib.md5).hexdigest()
        return self.user + b' ' + response.encode('ascii')

@implementer(IClientAuthentication)
class LOGINAuthenticator:

    def __init__(self, user):
        if False:
            print('Hello World!')
        self.user = user
        self.challengeResponse = self.challengeUsername

    def getName(self):
        if False:
            while True:
                i = 10
        return b'LOGIN'

    def challengeUsername(self, secret, chal):
        if False:
            for i in range(10):
                print('nop')
        self.challengeResponse = self.challengeSecret
        return self.user

    def challengeSecret(self, secret, chal):
        if False:
            print('Hello World!')
        return secret

@implementer(IClientAuthentication)
class PLAINAuthenticator:

    def __init__(self, user):
        if False:
            i = 10
            return i + 15
        self.user = user

    def getName(self):
        if False:
            while True:
                i = 10
        return b'PLAIN'

    def challengeResponse(self, secret, chal):
        if False:
            while True:
                i = 10
        return b'\x00' + self.user + b'\x00' + secret

@implementer(IChallengeResponse)
class LOGINCredentials(credentials.UsernamePassword):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.challenges = [b'Password\x00', b'User Name\x00']
        self.responses = [b'password', b'username']
        credentials.UsernamePassword.__init__(self, None, None)

    def getChallenge(self):
        if False:
            return 10
        return self.challenges.pop()

    def setResponse(self, response):
        if False:
            print('Hello World!')
        setattr(self, nativeString(self.responses.pop()), response)

    def moreChallenges(self):
        if False:
            print('Hello World!')
        return bool(self.challenges)

@implementer(IChallengeResponse)
class PLAINCredentials(credentials.UsernamePassword):

    def __init__(self):
        if False:
            return 10
        credentials.UsernamePassword.__init__(self, None, None)

    def getChallenge(self):
        if False:
            return 10
        return b''

    def setResponse(self, response):
        if False:
            while True:
                i = 10
        parts = response.split(b'\x00')
        if len(parts) != 3:
            raise IllegalClientResponse('Malformed Response - wrong number of parts')
        (useless, self.username, self.password) = parts

    def moreChallenges(self):
        if False:
            for i in range(10):
                print('nop')
        return False
__all__ = ['CramMD5ClientAuthenticator', 'LOGINCredentials', 'LOGINAuthenticator', 'PLAINCredentials', 'PLAINAuthenticator']