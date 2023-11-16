"""
Cred plugin for ssh key login.
"""
from zope.interface import implementer
from twisted import plugin
from twisted.cred.strcred import ICheckerFactory
sshKeyCheckerFactoryHelp = '\nThis allows SSH public key authentication, based on public keys listed in\nauthorized_keys and authorized_keys2 files in user .ssh/ directories.\n'
try:
    from twisted.conch.checkers import SSHPublicKeyChecker, UNIXAuthorizedKeysFiles

    @implementer(ICheckerFactory, plugin.IPlugin)
    class SSHKeyCheckerFactory:
        """
        Generates checkers that will authenticate a SSH public key
        """
        authType = 'sshkey'
        authHelp = sshKeyCheckerFactoryHelp
        argStringFormat = 'No argstring required.'
        credentialInterfaces = SSHPublicKeyChecker.credentialInterfaces

        def generateChecker(self, argstring=''):
            if False:
                print('Hello World!')
            '\n            This checker factory ignores the argument string. Everything\n            needed to authenticate users is pulled out of the public keys\n            listed in user .ssh/ directories.\n            '
            return SSHPublicKeyChecker(UNIXAuthorizedKeysFiles())
    theSSHKeyCheckerFactory = SSHKeyCheckerFactory()
except ImportError:
    pass