"""
Cred plugin for UNIX user accounts.
"""
from zope.interface import implementer
from twisted import plugin
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import IUsernamePassword
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.strcred import ICheckerFactory
from twisted.internet import defer

def verifyCryptedPassword(crypted, pw):
    if False:
        i = 10
        return i + 15
    '\n    Use L{crypt.crypt} to Verify that an unencrypted\n    password matches the encrypted password.\n\n    @param crypted: The encrypted password, obtained from\n                    the Unix password database or Unix shadow\n                    password database.\n    @param pw: The unencrypted password.\n    @return: L{True} if there is successful match, else L{False}.\n    @rtype: L{bool}\n    '
    try:
        import crypt
    except ImportError:
        crypt = None
    if crypt is None:
        raise NotImplementedError('cred_unix not supported on this platform')
    if isinstance(pw, bytes):
        pw = pw.decode('utf-8')
    if isinstance(crypted, bytes):
        crypted = crypted.decode('utf-8')
    try:
        crypted_check = crypt.crypt(pw, crypted)
        if isinstance(crypted_check, bytes):
            crypted_check = crypted_check.decode('utf-8')
        return crypted_check == crypted
    except OSError:
        return False

@implementer(ICredentialsChecker)
class UNIXChecker:
    """
    A credentials checker for a UNIX server. This will check that
    an authenticating username/password is a valid user on the system.

    Does not work on Windows.

    Right now this supports Python's pwd and spwd modules, if they are
    installed. It does not support PAM.
    """
    credentialInterfaces = (IUsernamePassword,)

    def checkPwd(self, pwd, username, password):
        if False:
            i = 10
            return i + 15
        '\n        Obtain the encrypted password for C{username} from the Unix password\n        database using L{pwd.getpwnam}, and see if it it matches it matches\n        C{password}.\n\n        @param pwd: Module which provides functions which\n                    access to the Unix password database.\n        @type pwd: C{module}\n        @param username: The user to look up in the Unix password database.\n        @type username: L{unicode}/L{str} or L{bytes}\n        @param password: The password to compare.\n        @type username: L{unicode}/L{str} or L{bytes}\n        '
        try:
            if isinstance(username, bytes):
                username = username.decode('utf-8')
            cryptedPass = pwd.getpwnam(username).pw_passwd
        except KeyError:
            return defer.fail(UnauthorizedLogin())
        else:
            if cryptedPass in ('*', 'x'):
                return None
            elif verifyCryptedPassword(cryptedPass, password):
                return defer.succeed(username)

    def checkSpwd(self, spwd, username, password):
        if False:
            while True:
                i = 10
        '\n        Obtain the encrypted password for C{username} from the\n        Unix shadow password database using L{spwd.getspnam},\n        and see if it it matches it matches C{password}.\n\n        @param spwd: Module which provides functions which\n                     access to the Unix shadow password database.\n        @type spwd: C{module}\n        @param username: The user to look up in the Unix password database.\n        @type username: L{unicode}/L{str} or L{bytes}\n        @param password: The password to compare.\n        @type username: L{unicode}/L{str} or L{bytes}\n        '
        try:
            if isinstance(username, bytes):
                username = username.decode('utf-8')
            if getattr(spwd.struct_spwd, 'sp_pwdp', None):
                cryptedPass = spwd.getspnam(username).sp_pwdp
            else:
                cryptedPass = spwd.getspnam(username).sp_pwd
        except KeyError:
            return defer.fail(UnauthorizedLogin())
        else:
            if verifyCryptedPassword(cryptedPass, password):
                return defer.succeed(username)

    def requestAvatarId(self, credentials):
        if False:
            for i in range(10):
                print('nop')
        (username, password) = (credentials.username, credentials.password)
        try:
            import pwd
        except ImportError:
            pwd = None
        if pwd is not None:
            checked = self.checkPwd(pwd, username, password)
            if checked is not None:
                return checked
        try:
            import spwd
        except ImportError:
            spwd = None
        if spwd is not None:
            checked = self.checkSpwd(spwd, username, password)
            if checked is not None:
                return checked
        return defer.fail(UnauthorizedLogin())
unixCheckerFactoryHelp = "\nThis checker will attempt to use every resource available to\nauthenticate against the list of users on the local UNIX system.\n(This does not support Windows servers for very obvious reasons.)\n\nRight now, this includes support for:\n\n  * Python's pwd module (which checks /etc/passwd)\n  * Python's spwd module (which checks /etc/shadow)\n\nFuture versions may include support for PAM authentication.\n"

@implementer(ICheckerFactory, plugin.IPlugin)
class UNIXCheckerFactory:
    """
    A factory for L{UNIXChecker}.
    """
    authType = 'unix'
    authHelp = unixCheckerFactoryHelp
    argStringFormat = 'No argstring required.'
    credentialInterfaces = UNIXChecker.credentialInterfaces

    def generateChecker(self, argstring):
        if False:
            for i in range(10):
                print('nop')
        '\n        This checker factory ignores the argument string. Everything\n        needed to generate a user database is pulled out of the local\n        UNIX environment.\n        '
        return UNIXChecker()
theUnixCheckerFactory = UNIXCheckerFactory()