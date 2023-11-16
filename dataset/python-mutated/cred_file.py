"""
Cred plugin for a file of the format 'username:password'.
"""
import sys
from zope.interface import implementer
from twisted import plugin
from twisted.cred.checkers import FilePasswordDB
from twisted.cred.credentials import IUsernameHashedPassword, IUsernamePassword
from twisted.cred.strcred import ICheckerFactory
fileCheckerFactoryHelp = "\nThis checker expects to receive the location of a file that\nconforms to the FilePasswordDB format. Each line in the file\nshould be of the format 'username:password', in plain text.\n"
invalidFileWarning = 'Warning: not a valid file'

@implementer(ICheckerFactory, plugin.IPlugin)
class FileCheckerFactory:
    """
    A factory for instances of L{FilePasswordDB}.
    """
    authType = 'file'
    authHelp = fileCheckerFactoryHelp
    argStringFormat = 'Location of a FilePasswordDB-formatted file.'
    credentialInterfaces = (IUsernamePassword, IUsernameHashedPassword)
    errorOutput = sys.stderr

    def generateChecker(self, argstring):
        if False:
            print('Hello World!')
        '\n        This checker factory expects to get the location of a file.\n        The file should conform to the format required by\n        L{FilePasswordDB} (using defaults for all\n        initialization parameters).\n        '
        from twisted.python.filepath import FilePath
        if not argstring.strip():
            raise ValueError('%r requires a filename' % self.authType)
        elif not FilePath(argstring).isfile():
            self.errorOutput.write(f'{invalidFileWarning}: {argstring}\n')
        return FilePasswordDB(argstring)
theFileCheckerFactory = FileCheckerFactory()