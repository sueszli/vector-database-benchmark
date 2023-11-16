import os
import sys
from twisted import copyright
from twisted.application import app, internet, service
from twisted.python import log

class ServerOptions(app.ServerOptions):
    synopsis = 'Usage: twistd [options]'
    optFlags = [['nodaemon', 'n', '(for backwards compatibility).']]

    def opt_version(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Print version information and exit.\n        '
        print(f'twistd (the Twisted Windows runner) {copyright.version}', file=self.stdout)
        print(copyright.copyright, file=self.stdout)
        sys.exit()

class WindowsApplicationRunner(app.ApplicationRunner):
    """
    An ApplicationRunner which avoids unix-specific things. No
    forking, no PID files, no privileges.
    """

    def preApplication(self):
        if False:
            return 10
        '\n        Do pre-application-creation setup.\n        '
        self.oldstdout = sys.stdout
        self.oldstderr = sys.stderr
        os.chdir(self.config['rundir'])

    def postApplication(self):
        if False:
            return 10
        '\n        Start the application and run the reactor.\n        '
        service.IService(self.application).privilegedStartService()
        app.startApplication(self.application, not self.config['no_save'])
        app.startApplication(internet.TimerService(0.1, lambda : None), 0)
        self.startReactor(None, self.oldstdout, self.oldstderr)
        log.msg('Server Shut Down.')