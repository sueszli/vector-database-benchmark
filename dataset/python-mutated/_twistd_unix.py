import errno
import os
import pwd
import sys
import traceback
from twisted import copyright, logger
from twisted.application import app, service
from twisted.internet.interfaces import IReactorDaemonize
from twisted.python import log, logfile, usage
from twisted.python.runtime import platformType
from twisted.python.util import gidFromString, switchUID, uidFromString, untilConcludes
if platformType == 'win32':
    raise ImportError("_twistd_unix doesn't work on Windows.")

def _umask(value):
    if False:
        while True:
            i = 10
    return int(value, 8)

class ServerOptions(app.ServerOptions):
    synopsis = 'Usage: twistd [options]'
    optFlags = [['nodaemon', 'n', "don't daemonize, don't use default umask of 0077"], ['originalname', None, "Don't try to change the process name"], ['syslog', None, 'Log to syslog, not to file'], ['euid', '', 'Set only effective user-id rather than real user-id. (This option has no effect unless the server is running as root, in which case it means not to shed all privileges after binding ports, retaining the option to regain privileges in cases such as spawning processes. Use with caution.)']]
    optParameters = [['prefix', None, 'twisted', 'use the given prefix when syslogging'], ['pidfile', '', 'twistd.pid', 'Name of the pidfile'], ['chroot', None, None, 'Chroot to a supplied directory before running'], ['uid', 'u', None, 'The uid to run as.', uidFromString], ['gid', 'g', None, 'The gid to run as.  If not specified, the default gid associated with the specified --uid is used.', gidFromString], ['umask', None, None, 'The (octal) file creation mask to apply.', _umask]]
    compData = usage.Completions(optActions={'pidfile': usage.CompleteFiles('*.pid'), 'chroot': usage.CompleteDirs(descr='chroot directory'), 'gid': usage.CompleteGroups(descr='gid to run as'), 'uid': usage.CompleteUsernames(descr='uid to run as'), 'prefix': usage.Completer(descr='syslog prefix')})

    def opt_version(self):
        if False:
            while True:
                i = 10
        '\n        Print version information and exit.\n        '
        print(f'twistd (the Twisted daemon) {copyright.version}', file=self.stdout)
        print(copyright.copyright, file=self.stdout)
        sys.exit()

    def postOptions(self):
        if False:
            i = 10
            return i + 15
        app.ServerOptions.postOptions(self)
        if self['pidfile']:
            self['pidfile'] = os.path.abspath(self['pidfile'])

def checkPID(pidfile):
    if False:
        while True:
            i = 10
    if not pidfile:
        return
    if os.path.exists(pidfile):
        try:
            with open(pidfile) as f:
                pid = int(f.read())
        except ValueError:
            sys.exit(f'Pidfile {pidfile} contains non-numeric value')
        try:
            os.kill(pid, 0)
        except OSError as why:
            if why.errno == errno.ESRCH:
                log.msg(f'Removing stale pidfile {pidfile}', isError=True)
                os.remove(pidfile)
            else:
                sys.exit("Can't check status of PID {} from pidfile {}: {}".format(pid, pidfile, why))
        else:
            sys.exit('Another twistd server is running, PID {}\n\nThis could either be a previously started instance of your application or a\ndifferent application entirely. To start a new one, either run it in some other\ndirectory, or use the --pidfile and --logfile parameters to avoid clashes.\n'.format(pid))

class UnixAppLogger(app.AppLogger):
    """
    A logger able to log to syslog, to files, and to stdout.

    @ivar _syslog: A flag indicating whether to use syslog instead of file
        logging.
    @type _syslog: C{bool}

    @ivar _syslogPrefix: If C{sysLog} is C{True}, the string prefix to use for
        syslog messages.
    @type _syslogPrefix: C{str}

    @ivar _nodaemon: A flag indicating the process will not be daemonizing.
    @type _nodaemon: C{bool}
    """

    def __init__(self, options):
        if False:
            while True:
                i = 10
        app.AppLogger.__init__(self, options)
        self._syslog = options.get('syslog', False)
        self._syslogPrefix = options.get('prefix', '')
        self._nodaemon = options.get('nodaemon', False)

    def _getLogObserver(self):
        if False:
            return 10
        '\n        Create and return a suitable log observer for the given configuration.\n\n        The observer will go to syslog using the prefix C{_syslogPrefix} if\n        C{_syslog} is true.  Otherwise, it will go to the file named\n        C{_logfilename} or, if C{_nodaemon} is true and C{_logfilename} is\n        C{"-"}, to stdout.\n\n        @return: An object suitable to be passed to C{log.addObserver}.\n        '
        if self._syslog:
            from twisted.python import syslog
            return syslog.SyslogObserver(self._syslogPrefix).emit
        if self._logfilename == '-':
            if not self._nodaemon:
                sys.exit('Daemons cannot log to stdout, exiting!')
            logFile = sys.stdout
        elif self._nodaemon and (not self._logfilename):
            logFile = sys.stdout
        else:
            if not self._logfilename:
                self._logfilename = 'twistd.log'
            logFile = logfile.LogFile.fromFullPath(self._logfilename)
            try:
                import signal
            except ImportError:
                pass
            else:
                if not signal.getsignal(signal.SIGUSR1):

                    def rotateLog(signal, frame):
                        if False:
                            print('Hello World!')
                        from twisted.internet import reactor
                        reactor.callFromThread(logFile.rotate)
                    signal.signal(signal.SIGUSR1, rotateLog)
        return logger.textFileLogObserver(logFile)

def launchWithName(name):
    if False:
        print('Hello World!')
    if name and name != sys.argv[0]:
        exe = os.path.realpath(sys.executable)
        log.msg('Changing process name to ' + name)
        os.execv(exe, [name, sys.argv[0], '--originalname'] + sys.argv[1:])

class UnixApplicationRunner(app.ApplicationRunner):
    """
    An ApplicationRunner which does Unix-specific things, like fork,
    shed privileges, and maintain a PID file.
    """
    loggerFactory = UnixAppLogger

    def preApplication(self):
        if False:
            return 10
        '\n        Do pre-application-creation setup.\n        '
        checkPID(self.config['pidfile'])
        self.config['nodaemon'] = self.config['nodaemon'] or self.config['debug']
        self.oldstdout = sys.stdout
        self.oldstderr = sys.stderr

    def _formatChildException(self, exception):
        if False:
            print('Hello World!')
        "\n        Format the C{exception} in preparation for writing to the\n        status pipe.  This does the right thing on Python 2 if the\n        exception's message is Unicode, and in all cases limits the\n        length of the message afte* encoding to 100 bytes.\n\n        This means the returned message may be truncated in the middle\n        of a unicode escape.\n\n        @type exception: L{Exception}\n        @param exception: The exception to format.\n\n        @return: The formatted message, suitable for writing to the\n            status pipe.\n        @rtype: L{bytes}\n        "
        exceptionLine = traceback.format_exception_only(exception.__class__, exception)[-1]
        formattedMessage = f'1 {exceptionLine.strip()}'
        formattedMessage = formattedMessage.encode('ascii', 'backslashreplace')
        return formattedMessage[:100]

    def postApplication(self):
        if False:
            while True:
                i = 10
        '\n        To be called after the application is created: start the application\n        and run the reactor. After the reactor stops, clean up PID files and\n        such.\n        '
        try:
            self.startApplication(self.application)
        except Exception as ex:
            statusPipe = self.config.get('statusPipe', None)
            if statusPipe is not None:
                message = self._formatChildException(ex)
                untilConcludes(os.write, statusPipe, message)
                untilConcludes(os.close, statusPipe)
            self.removePID(self.config['pidfile'])
            raise
        else:
            statusPipe = self.config.get('statusPipe', None)
            if statusPipe is not None:
                untilConcludes(os.write, statusPipe, b'0')
                untilConcludes(os.close, statusPipe)
        self.startReactor(None, self.oldstdout, self.oldstderr)
        self.removePID(self.config['pidfile'])

    def removePID(self, pidfile):
        if False:
            i = 10
            return i + 15
        '\n        Remove the specified PID file, if possible.  Errors are logged, not\n        raised.\n\n        @type pidfile: C{str}\n        @param pidfile: The path to the PID tracking file.\n        '
        if not pidfile:
            return
        try:
            os.unlink(pidfile)
        except OSError as e:
            if e.errno == errno.EACCES or e.errno == errno.EPERM:
                log.msg('Warning: No permission to delete pid file')
            else:
                log.err(e, 'Failed to unlink PID file:')
        except BaseException:
            log.err(None, 'Failed to unlink PID file:')

    def setupEnvironment(self, chroot, rundir, nodaemon, umask, pidfile):
        if False:
            i = 10
            return i + 15
        '\n        Set the filesystem root, the working directory, and daemonize.\n\n        @type chroot: C{str} or L{None}\n        @param chroot: If not None, a path to use as the filesystem root (using\n            L{os.chroot}).\n\n        @type rundir: C{str}\n        @param rundir: The path to set as the working directory.\n\n        @type nodaemon: C{bool}\n        @param nodaemon: A flag which, if set, indicates that daemonization\n            should not be done.\n\n        @type umask: C{int} or L{None}\n        @param umask: The value to which to change the process umask.\n\n        @type pidfile: C{str} or L{None}\n        @param pidfile: If not L{None}, the path to a file into which to put\n            the PID of this process.\n        '
        daemon = not nodaemon
        if chroot is not None:
            os.chroot(chroot)
            if rundir == '.':
                rundir = '/'
        os.chdir(rundir)
        if daemon and umask is None:
            umask = 63
        if umask is not None:
            os.umask(umask)
        if daemon:
            from twisted.internet import reactor
            self.config['statusPipe'] = self.daemonize(reactor)
        if pidfile:
            with open(pidfile, 'wb') as f:
                f.write(b'%d' % (os.getpid(),))

    def daemonize(self, reactor):
        if False:
            i = 10
            return i + 15
        '\n        Daemonizes the application on Unix. This is done by the usual double\n        forking approach.\n\n        @see: U{http://code.activestate.com/recipes/278731/}\n        @see: W. Richard Stevens,\n            "Advanced Programming in the Unix Environment",\n            1992, Addison-Wesley, ISBN 0-201-56317-7\n\n        @param reactor: The reactor in use.  If it provides\n            L{IReactorDaemonize}, its daemonization-related callbacks will be\n            invoked.\n\n        @return: A writable pipe to be used to report errors.\n        @rtype: C{int}\n        '
        if IReactorDaemonize.providedBy(reactor):
            reactor.beforeDaemonize()
        (r, w) = os.pipe()
        if os.fork():
            code = self._waitForStart(r)
            os.close(r)
            os._exit(code)
        os.setsid()
        if os.fork():
            os._exit(0)
        null = os.open('/dev/null', os.O_RDWR)
        for i in range(3):
            try:
                os.dup2(null, i)
            except OSError as e:
                if e.errno != errno.EBADF:
                    raise
        os.close(null)
        if IReactorDaemonize.providedBy(reactor):
            reactor.afterDaemonize()
        return w

    def _waitForStart(self, readPipe: int) -> int:
        if False:
            while True:
                i = 10
        '\n        Wait for the daemonization success.\n\n        @param readPipe: file descriptor to read start information from.\n        @type readPipe: C{int}\n\n        @return: code to be passed to C{os._exit}: 0 for success, 1 for error.\n        @rtype: C{int}\n        '
        data = untilConcludes(os.read, readPipe, 100)
        dataRepr = repr(data[2:])
        if data != b'0':
            msg = 'An error has occurred: {}\nPlease look at log file for more information.\n'.format(dataRepr)
            untilConcludes(sys.__stderr__.write, msg)
            return 1
        return 0

    def shedPrivileges(self, euid, uid, gid):
        if False:
            print('Hello World!')
        '\n        Change the UID and GID or the EUID and EGID of this process.\n\n        @type euid: C{bool}\n        @param euid: A flag which, if set, indicates that only the I{effective}\n            UID and GID should be set.\n\n        @type uid: C{int} or L{None}\n        @param uid: If not L{None}, the UID to which to switch.\n\n        @type gid: C{int} or L{None}\n        @param gid: If not L{None}, the GID to which to switch.\n        '
        if uid is not None or gid is not None:
            extra = euid and 'e' or ''
            desc = f'{extra}uid/{extra}gid {uid}/{gid}'
            try:
                switchUID(uid, gid, euid)
            except OSError as e:
                log.msg('failed to set {}: {} (are you root?) -- exiting.'.format(desc, e))
                sys.exit(1)
            else:
                log.msg(f'set {desc}')

    def startApplication(self, application):
        if False:
            for i in range(10):
                print('nop')
        '\n        Configure global process state based on the given application and run\n        the application.\n\n        @param application: An object which can be adapted to\n            L{service.IProcess} and L{service.IService}.\n        '
        process = service.IProcess(application)
        if not self.config['originalname']:
            launchWithName(process.processName)
        self.setupEnvironment(self.config['chroot'], self.config['rundir'], self.config['nodaemon'], self.config['umask'], self.config['pidfile'])
        service.IService(application).privilegedStartService()
        (uid, gid) = (self.config['uid'], self.config['gid'])
        if uid is None:
            uid = process.uid
        if gid is None:
            gid = process.gid
        if uid is not None and gid is None:
            gid = pwd.getpwuid(uid).pw_gid
        self.shedPrivileges(self.config['euid'], uid, gid)
        app.startApplication(application, not self.config['no_save'])