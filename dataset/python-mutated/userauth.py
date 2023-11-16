"""
Implementation of the ssh-userauth service.
Currently implemented authentication types are public-key and password.

Maintainer: Paul Swartz
"""
import struct
from twisted.conch import error, interfaces
from twisted.conch.ssh import keys, service, transport
from twisted.conch.ssh.common import NS, getNS
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, reactor
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString

class SSHUserAuthServer(service.SSHService):
    """
    A service implementing the server side of the 'ssh-userauth' service.  It
    is used to authenticate the user on the other side as being able to access
    this server.

    @ivar name: the name of this service: 'ssh-userauth'
    @type name: L{bytes}
    @ivar authenticatedWith: a list of authentication methods that have
        already been used.
    @type authenticatedWith: L{list}
    @ivar loginTimeout: the number of seconds we wait before disconnecting
        the user for taking too long to authenticate
    @type loginTimeout: L{int}
    @ivar attemptsBeforeDisconnect: the number of failed login attempts we
        allow before disconnecting.
    @type attemptsBeforeDisconnect: L{int}
    @ivar loginAttempts: the number of login attempts that have been made
    @type loginAttempts: L{int}
    @ivar passwordDelay: the number of seconds to delay when the user gives
        an incorrect password
    @type passwordDelay: L{int}
    @ivar interfaceToMethod: a L{dict} mapping credential interfaces to
        authentication methods.  The server checks to see which of the
        cred interfaces have checkers and tells the client that those methods
        are valid for authentication.
    @type interfaceToMethod: L{dict}
    @ivar supportedAuthentications: A list of the supported authentication
        methods.
    @type supportedAuthentications: L{list} of L{bytes}
    @ivar user: the last username the client tried to authenticate with
    @type user: L{bytes}
    @ivar method: the current authentication method
    @type method: L{bytes}
    @ivar nextService: the service the user wants started after authentication
        has been completed.
    @type nextService: L{bytes}
    @ivar portal: the L{twisted.cred.portal.Portal} we are using for
        authentication
    @type portal: L{twisted.cred.portal.Portal}
    @ivar clock: an object with a callLater method.  Stubbed out for testing.
    """
    name = b'ssh-userauth'
    loginTimeout = 10 * 60 * 60
    attemptsBeforeDisconnect = 20
    passwordDelay = 1
    clock = reactor
    interfaceToMethod = {credentials.ISSHPrivateKey: b'publickey', credentials.IUsernamePassword: b'password'}
    _log = Logger()

    def serviceStarted(self):
        if False:
            i = 10
            return i + 15
        '\n        Called when the userauth service is started.  Set up instance\n        variables, check if we should allow password authentication (only\n        allow if the outgoing connection is encrypted) and set up a login\n        timeout.\n        '
        self.authenticatedWith = []
        self.loginAttempts = 0
        self.user = None
        self.nextService = None
        self.portal = self.transport.factory.portal
        self.supportedAuthentications = []
        for i in self.portal.listCredentialsInterfaces():
            if i in self.interfaceToMethod:
                self.supportedAuthentications.append(self.interfaceToMethod[i])
        if not self.transport.isEncrypted('in'):
            if b'password' in self.supportedAuthentications:
                self.supportedAuthentications.remove(b'password')
        self._cancelLoginTimeout = self.clock.callLater(self.loginTimeout, self.timeoutAuthentication)

    def serviceStopped(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Called when the userauth service is stopped.  Cancel the login timeout\n        if it's still going.\n        "
        if self._cancelLoginTimeout:
            self._cancelLoginTimeout.cancel()
            self._cancelLoginTimeout = None

    def timeoutAuthentication(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called when the user has timed out on authentication.  Disconnect\n        with a DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE message.\n        '
        self._cancelLoginTimeout = None
        self.transport.sendDisconnect(transport.DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE, b'you took too long')

    def tryAuth(self, kind, user, data):
        if False:
            while True:
                i = 10
        '\n        Try to authenticate the user with the given method.  Dispatches to a\n        auth_* method.\n\n        @param kind: the authentication method to try.\n        @type kind: L{bytes}\n        @param user: the username the client is authenticating with.\n        @type user: L{bytes}\n        @param data: authentication specific data sent by the client.\n        @type data: L{bytes}\n        @return: A Deferred called back if the method succeeded, or erred back\n            if it failed.\n        @rtype: C{defer.Deferred}\n        '
        self._log.debug('{user!r} trying auth {kind!r}', user=user, kind=kind)
        if kind not in self.supportedAuthentications:
            return defer.fail(error.ConchError('unsupported authentication, failing'))
        kind = nativeString(kind.replace(b'-', b'_'))
        f = getattr(self, f'auth_{kind}', None)
        if f:
            ret = f(data)
            if not ret:
                return defer.fail(error.ConchError(f'{kind} return None instead of a Deferred'))
            else:
                return ret
        return defer.fail(error.ConchError(f'bad auth type: {kind}'))

    def ssh_USERAUTH_REQUEST(self, packet):
        if False:
            return 10
        '\n        The client has requested authentication.  Payload::\n            string user\n            string next service\n            string method\n            <authentication specific data>\n\n        @type packet: L{bytes}\n        '
        (user, nextService, method, rest) = getNS(packet, 3)
        if user != self.user or nextService != self.nextService:
            self.authenticatedWith = []
        self.user = user
        self.nextService = nextService
        self.method = method
        d = self.tryAuth(method, user, rest)
        if not d:
            self._ebBadAuth(failure.Failure(error.ConchError('auth returned none')))
            return
        d.addCallback(self._cbFinishedAuth)
        d.addErrback(self._ebMaybeBadAuth)
        d.addErrback(self._ebBadAuth)
        return d

    def _cbFinishedAuth(self, result):
        if False:
            i = 10
            return i + 15
        '\n        The callback when user has successfully been authenticated.  For a\n        description of the arguments, see L{twisted.cred.portal.Portal.login}.\n        We start the service requested by the user.\n        '
        (interface, avatar, logout) = result
        self.transport.avatar = avatar
        self.transport.logoutFunction = logout
        service = self.transport.factory.getService(self.transport, self.nextService)
        if not service:
            raise error.ConchError(f'could not get next service: {self.nextService}')
        self._log.debug('{user!r} authenticated with {method!r}', user=self.user, method=self.method)
        self.transport.sendPacket(MSG_USERAUTH_SUCCESS, b'')
        self.transport.setService(service())

    def _ebMaybeBadAuth(self, reason):
        if False:
            return 10
        '\n        An intermediate errback.  If the reason is\n        error.NotEnoughAuthentication, we send a MSG_USERAUTH_FAILURE, but\n        with the partial success indicator set.\n\n        @type reason: L{twisted.python.failure.Failure}\n        '
        reason.trap(error.NotEnoughAuthentication)
        self.transport.sendPacket(MSG_USERAUTH_FAILURE, NS(b','.join(self.supportedAuthentications)) + b'\xff')

    def _ebBadAuth(self, reason):
        if False:
            while True:
                i = 10
        "\n        The final errback in the authentication chain.  If the reason is\n        error.IgnoreAuthentication, we simply return; the authentication\n        method has sent its own response.  Otherwise, send a failure message\n        and (if the method is not 'none') increment the number of login\n        attempts.\n\n        @type reason: L{twisted.python.failure.Failure}\n        "
        if reason.check(error.IgnoreAuthentication):
            return
        if self.method != b'none':
            self._log.debug('{user!r} failed auth {method!r}', user=self.user, method=self.method)
            if reason.check(UnauthorizedLogin):
                self._log.debug('unauthorized login: {message}', message=reason.getErrorMessage())
            elif reason.check(error.ConchError):
                self._log.debug('reason: {reason}', reason=reason.getErrorMessage())
            else:
                self._log.failure('Error checking auth for user {user}', failure=reason, user=self.user)
            self.loginAttempts += 1
            if self.loginAttempts > self.attemptsBeforeDisconnect:
                self.transport.sendDisconnect(transport.DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE, b'too many bad auths')
                return
        self.transport.sendPacket(MSG_USERAUTH_FAILURE, NS(b','.join(self.supportedAuthentications)) + b'\x00')

    def auth_publickey(self, packet):
        if False:
            i = 10
            return i + 15
        '\n        Public key authentication.  Payload::\n            byte has signature\n            string algorithm name\n            string key blob\n            [string signature] (if has signature is True)\n\n        Create a SSHPublicKey credential and verify it using our portal.\n        '
        hasSig = ord(packet[0:1])
        (algName, blob, rest) = getNS(packet[1:], 2)
        try:
            keys.Key.fromString(blob)
        except keys.BadKeyError:
            error = 'Unsupported key type {} or bad key'.format(algName.decode('ascii'))
            self._log.error(error)
            return defer.fail(UnauthorizedLogin(error))
        signature = hasSig and getNS(rest)[0] or None
        if hasSig:
            b = NS(self.transport.sessionID) + bytes((MSG_USERAUTH_REQUEST,)) + NS(self.user) + NS(self.nextService) + NS(b'publickey') + bytes((hasSig,)) + NS(algName) + NS(blob)
            c = credentials.SSHPrivateKey(self.user, algName, blob, b, signature)
            return self.portal.login(c, None, interfaces.IConchUser)
        else:
            c = credentials.SSHPrivateKey(self.user, algName, blob, None, None)
            return self.portal.login(c, None, interfaces.IConchUser).addErrback(self._ebCheckKey, packet[1:])

    def _ebCheckKey(self, reason, packet):
        if False:
            return 10
        '\n        Called back if the user did not sent a signature.  If reason is\n        error.ValidPublicKey then this key is valid for the user to\n        authenticate with.  Send MSG_USERAUTH_PK_OK.\n        '
        reason.trap(error.ValidPublicKey)
        self.transport.sendPacket(MSG_USERAUTH_PK_OK, packet)
        return failure.Failure(error.IgnoreAuthentication())

    def auth_password(self, packet):
        if False:
            for i in range(10):
                print('nop')
        '\n        Password authentication.  Payload::\n            string password\n\n        Make a UsernamePassword credential and verify it with our portal.\n        '
        password = getNS(packet[1:])[0]
        c = credentials.UsernamePassword(self.user, password)
        return self.portal.login(c, None, interfaces.IConchUser).addErrback(self._ebPassword)

    def _ebPassword(self, f):
        if False:
            print('Hello World!')
        '\n        If the password is invalid, wait before sending the failure in order\n        to delay brute-force password guessing.\n        '
        d = defer.Deferred()
        self.clock.callLater(self.passwordDelay, d.callback, f)
        return d

class SSHUserAuthClient(service.SSHService):
    """
    A service implementing the client side of 'ssh-userauth'.

    This service will try all authentication methods provided by the server,
    making callbacks for more information when necessary.

    @ivar name: the name of this service: 'ssh-userauth'
    @type name: L{str}
    @ivar preferredOrder: a list of authentication methods that should be used
        first, in order of preference, if supported by the server
    @type preferredOrder: L{list}
    @ivar user: the name of the user to authenticate as
    @type user: L{bytes}
    @ivar instance: the service to start after authentication has finished
    @type instance: L{service.SSHService}
    @ivar authenticatedWith: a list of strings of authentication methods we've tried
    @type authenticatedWith: L{list} of L{bytes}
    @ivar triedPublicKeys: a list of public key objects that we've tried to
        authenticate with
    @type triedPublicKeys: L{list} of L{Key}
    @ivar lastPublicKey: the last public key object we've tried to authenticate
        with
    @type lastPublicKey: L{Key}
    """
    name = b'ssh-userauth'
    preferredOrder = [b'publickey', b'password', b'keyboard-interactive']

    def __init__(self, user, instance):
        if False:
            for i in range(10):
                print('nop')
        self.user = user
        self.instance = instance

    def serviceStarted(self):
        if False:
            return 10
        self.authenticatedWith = []
        self.triedPublicKeys = []
        self.lastPublicKey = None
        self.askForAuth(b'none', b'')

    def askForAuth(self, kind, extraData):
        if False:
            for i in range(10):
                print('nop')
        '\n        Send a MSG_USERAUTH_REQUEST.\n\n        @param kind: the authentication method to try.\n        @type kind: L{bytes}\n        @param extraData: method-specific data to go in the packet\n        @type extraData: L{bytes}\n        '
        self.lastAuth = kind
        self.transport.sendPacket(MSG_USERAUTH_REQUEST, NS(self.user) + NS(self.instance.name) + NS(kind) + extraData)

    def tryAuth(self, kind):
        if False:
            for i in range(10):
                print('nop')
        '\n        Dispatch to an authentication method.\n\n        @param kind: the authentication method\n        @type kind: L{bytes}\n        '
        kind = nativeString(kind.replace(b'-', b'_'))
        self._log.debug('trying to auth with {kind}', kind=kind)
        f = getattr(self, 'auth_' + kind, None)
        if f:
            return f()

    def _ebAuth(self, ignored, *args):
        if False:
            return 10
        "\n        Generic callback for a failed authentication attempt.  Respond by\n        asking for the list of accepted methods (the 'none' method)\n        "
        self.askForAuth(b'none', b'')

    def ssh_USERAUTH_SUCCESS(self, packet):
        if False:
            for i in range(10):
                print('nop')
        '\n        We received a MSG_USERAUTH_SUCCESS.  The server has accepted our\n        authentication, so start the next service.\n        '
        self.transport.setService(self.instance)

    def ssh_USERAUTH_FAILURE(self, packet):
        if False:
            for i in range(10):
                print('nop')
        '\n        We received a MSG_USERAUTH_FAILURE.  Payload::\n            string methods\n            byte partial success\n\n        If partial success is C{True}, then the previous method succeeded but is\n        not sufficient for authentication. C{methods} is a comma-separated list\n        of accepted authentication methods.\n\n        We sort the list of methods by their position in C{self.preferredOrder},\n        removing methods that have already succeeded. We then call\n        C{self.tryAuth} with the most preferred method.\n\n        @param packet: the C{MSG_USERAUTH_FAILURE} payload.\n        @type packet: L{bytes}\n\n        @return: a L{defer.Deferred} that will be callbacked with L{None} as\n            soon as all authentication methods have been tried, or L{None} if no\n            more authentication methods are available.\n        @rtype: C{defer.Deferred} or L{None}\n        '
        (canContinue, partial) = getNS(packet)
        partial = ord(partial)
        if partial:
            self.authenticatedWith.append(self.lastAuth)

        def orderByPreference(meth):
            if False:
                i = 10
                return i + 15
            '\n            Invoked once per authentication method in order to extract a\n            comparison key which is then used for sorting.\n\n            @param meth: the authentication method.\n            @type meth: L{bytes}\n\n            @return: the comparison key for C{meth}.\n            @rtype: L{int}\n            '
            if meth in self.preferredOrder:
                return self.preferredOrder.index(meth)
            else:
                return len(self.preferredOrder)
        canContinue = sorted((meth for meth in canContinue.split(b',') if meth not in self.authenticatedWith), key=orderByPreference)
        self._log.debug('can continue with: {methods}', methods=canContinue)
        return self._cbUserauthFailure(None, iter(canContinue))

    def _cbUserauthFailure(self, result, iterator):
        if False:
            for i in range(10):
                print('nop')
        if result:
            return
        try:
            method = next(iterator)
        except StopIteration:
            self.transport.sendDisconnect(transport.DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE, b'no more authentication methods available')
        else:
            d = defer.maybeDeferred(self.tryAuth, method)
            d.addCallback(self._cbUserauthFailure, iterator)
            return d

    def ssh_USERAUTH_PK_OK(self, packet):
        if False:
            while True:
                i = 10
        '\n        This message (number 60) can mean several different messages depending\n        on the current authentication type.  We dispatch to individual methods\n        in order to handle this request.\n        '
        func = getattr(self, 'ssh_USERAUTH_PK_OK_%s' % nativeString(self.lastAuth.replace(b'-', b'_')), None)
        if func is not None:
            return func(packet)
        else:
            self.askForAuth(b'none', b'')

    def ssh_USERAUTH_PK_OK_publickey(self, packet):
        if False:
            while True:
                i = 10
        '\n        This is MSG_USERAUTH_PK.  Our public key is valid, so we create a\n        signature and try to authenticate with it.\n        '
        publicKey = self.lastPublicKey
        b = NS(self.transport.sessionID) + bytes((MSG_USERAUTH_REQUEST,)) + NS(self.user) + NS(self.instance.name) + NS(b'publickey') + b'\x01' + NS(publicKey.sshType()) + NS(publicKey.blob())
        d = self.signData(publicKey, b)
        if not d:
            self.askForAuth(b'none', b'')
            return
        d.addCallback(self._cbSignedData)
        d.addErrback(self._ebAuth)

    def ssh_USERAUTH_PK_OK_password(self, packet):
        if False:
            return 10
        '\n        This is MSG_USERAUTH_PASSWD_CHANGEREQ.  The password given has expired.\n        We ask for an old password and a new password, then send both back to\n        the server.\n        '
        (prompt, language, rest) = getNS(packet, 2)
        self._oldPass = self._newPass = None
        d = self.getPassword(b'Old Password: ')
        d = d.addCallbacks(self._setOldPass, self._ebAuth)
        d.addCallback(lambda ignored: self.getPassword(prompt))
        d.addCallbacks(self._setNewPass, self._ebAuth)

    def ssh_USERAUTH_PK_OK_keyboard_interactive(self, packet):
        if False:
            i = 10
            return i + 15
        '\n        This is MSG_USERAUTH_INFO_RESPONSE.  The server has sent us the\n        questions it wants us to answer, so we ask the user and sent the\n        responses.\n        '
        (name, instruction, lang, data) = getNS(packet, 3)
        numPrompts = struct.unpack('!L', data[:4])[0]
        data = data[4:]
        prompts = []
        for i in range(numPrompts):
            (prompt, data) = getNS(data)
            echo = bool(ord(data[0:1]))
            data = data[1:]
            prompts.append((prompt, echo))
        d = self.getGenericAnswers(name, instruction, prompts)
        d.addCallback(self._cbGenericAnswers)
        d.addErrback(self._ebAuth)

    def _cbSignedData(self, signedData):
        if False:
            print('Hello World!')
        "\n        Called back out of self.signData with the signed data.  Send the\n        authentication request with the signature.\n\n        @param signedData: the data signed by the user's private key.\n        @type signedData: L{bytes}\n        "
        publicKey = self.lastPublicKey
        self.askForAuth(b'publickey', b'\x01' + NS(publicKey.sshType()) + NS(publicKey.blob()) + NS(signedData))

    def _setOldPass(self, op):
        if False:
            while True:
                i = 10
        '\n        Called back when we are choosing a new password.  Simply store the old\n        password for now.\n\n        @param op: the old password as entered by the user\n        @type op: L{bytes}\n        '
        self._oldPass = op

    def _setNewPass(self, np):
        if False:
            i = 10
            return i + 15
        '\n        Called back when we are choosing a new password.  Get the old password\n        and send the authentication message with both.\n\n        @param np: the new password as entered by the user\n        @type np: L{bytes}\n        '
        op = self._oldPass
        self._oldPass = None
        self.askForAuth(b'password', b'\xff' + NS(op) + NS(np))

    def _cbGenericAnswers(self, responses):
        if False:
            return 10
        '\n        Called back when we are finished answering keyboard-interactive\n        questions.  Send the info back to the server in a\n        MSG_USERAUTH_INFO_RESPONSE.\n\n        @param responses: a list of L{bytes} responses\n        @type responses: L{list}\n        '
        data = struct.pack('!L', len(responses))
        for r in responses:
            data += NS(r.encode('UTF8'))
        self.transport.sendPacket(MSG_USERAUTH_INFO_RESPONSE, data)

    def auth_publickey(self):
        if False:
            i = 10
            return i + 15
        '\n        Try to authenticate with a public key.  Ask the user for a public key;\n        if the user has one, send the request to the server and return True.\n        Otherwise, return False.\n\n        @rtype: L{bool}\n        '
        d = defer.maybeDeferred(self.getPublicKey)
        d.addBoth(self._cbGetPublicKey)
        return d

    def _cbGetPublicKey(self, publicKey):
        if False:
            i = 10
            return i + 15
        if not isinstance(publicKey, keys.Key):
            publicKey = None
        if publicKey is not None:
            self.lastPublicKey = publicKey
            self.triedPublicKeys.append(publicKey)
            self._log.debug('using key of type {keyType}', keyType=publicKey.type())
            self.askForAuth(b'publickey', b'\x00' + NS(publicKey.sshType()) + NS(publicKey.blob()))
            return True
        else:
            return False

    def auth_password(self):
        if False:
            i = 10
            return i + 15
        '\n        Try to authenticate with a password.  Ask the user for a password.\n        If the user will return a password, return True.  Otherwise, return\n        False.\n\n        @rtype: L{bool}\n        '
        d = self.getPassword()
        if d:
            d.addCallbacks(self._cbPassword, self._ebAuth)
            return True
        else:
            return False

    def auth_keyboard_interactive(self):
        if False:
            i = 10
            return i + 15
        '\n        Try to authenticate with keyboard-interactive authentication.  Send\n        the request to the server and return True.\n\n        @rtype: L{bool}\n        '
        self._log.debug('authing with keyboard-interactive')
        self.askForAuth(b'keyboard-interactive', NS(b'') + NS(b''))
        return True

    def _cbPassword(self, password):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called back when the user gives a password.  Send the request to the\n        server.\n\n        @param password: the password the user entered\n        @type password: L{bytes}\n        '
        self.askForAuth(b'password', b'\x00' + NS(password))

    def signData(self, publicKey, signData):
        if False:
            return 10
        "\n        Sign the given data with the given public key.\n\n        By default, this will call getPrivateKey to get the private key,\n        then sign the data using Key.sign().\n\n        This method is factored out so that it can be overridden to use\n        alternate methods, such as a key agent.\n\n        @param publicKey: The public key object returned from L{getPublicKey}\n        @type publicKey: L{keys.Key}\n\n        @param signData: the data to be signed by the private key.\n        @type signData: L{bytes}\n        @return: a Deferred that's called back with the signature\n        @rtype: L{defer.Deferred}\n        "
        key = self.getPrivateKey()
        if not key:
            return
        return key.addCallback(self._cbSignData, signData)

    def _cbSignData(self, privateKey, signData):
        if False:
            print('Hello World!')
        '\n        Called back when the private key is returned.  Sign the data and\n        return the signature.\n\n        @param privateKey: the private key object\n        @type privateKey: L{keys.Key}\n        @param signData: the data to be signed by the private key.\n        @type signData: L{bytes}\n        @return: the signature\n        @rtype: L{bytes}\n        '
        return privateKey.sign(signData)

    def getPublicKey(self):
        if False:
            return 10
        '\n        Return a public key for the user.  If no more public keys are\n        available, return L{None}.\n\n        This implementation always returns L{None}.  Override it in a\n        subclass to actually find and return a public key object.\n\n        @rtype: L{Key} or L{None}\n        '
        return None

    def getPrivateKey(self):
        if False:
            return 10
        '\n        Return a L{Deferred} that will be called back with the private key\n        object corresponding to the last public key from getPublicKey().\n        If the private key is not available, errback on the Deferred.\n\n        @rtype: L{Deferred} called back with L{Key}\n        '
        return defer.fail(NotImplementedError())

    def getPassword(self, prompt=None):
        if False:
            i = 10
            return i + 15
        "\n        Return a L{Deferred} that will be called back with a password.\n        prompt is a string to display for the password, or None for a generic\n        'user@hostname's password: '.\n\n        @type prompt: L{bytes}/L{None}\n        @rtype: L{defer.Deferred}\n        "
        return defer.fail(NotImplementedError())

    def getGenericAnswers(self, name, instruction, prompts):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns a L{Deferred} with the responses to the promopts.\n\n        @param name: The name of the authentication currently in progress.\n        @param instruction: Describes what the authentication wants.\n        @param prompts: A list of (prompt, echo) pairs, where prompt is a\n        string to display and echo is a boolean indicating whether the\n        user's response should be echoed as they type it.\n        "
        return defer.fail(NotImplementedError())
MSG_USERAUTH_REQUEST = 50
MSG_USERAUTH_FAILURE = 51
MSG_USERAUTH_SUCCESS = 52
MSG_USERAUTH_BANNER = 53
MSG_USERAUTH_INFO_RESPONSE = 61
MSG_USERAUTH_PK_OK = 60
messages = {}
for (k, v) in list(locals().items()):
    if k[:4] == 'MSG_':
        messages[v] = k
SSHUserAuthServer.protocolMessages = messages
SSHUserAuthClient.protocolMessages = messages
del messages
del v
MSG_USERAUTH_PASSWD_CHANGEREQ = 60
MSG_USERAUTH_INFO_REQUEST = 60