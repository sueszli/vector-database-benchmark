__author__ = 'Fail2Ban Developers'
__copyright__ = 'Copyright (c) 2004-2008 Cyril Jaquier, 2012-2014 Yaroslav Halchenko, 2014-2016 Serg G. Brester'
__license__ = 'GPL'
import os
import sys
from .fail2bancmdline import Fail2banCmdLine, ServerExecutionException, logSys, PRODUCTION, exit
SERVER = 'fail2ban-server'

class Fail2banServer(Fail2banCmdLine):

    @staticmethod
    def startServerDirect(conf, daemon=True, setServer=None):
        if False:
            while True:
                i = 10
        logSys.debug('  direct starting of server in %s, deamon: %s', os.getpid(), daemon)
        from ..server.server import Server
        server = None
        try:
            server = Server(daemon)
            if setServer:
                setServer(server)
            server.start(conf['socket'], conf['pidfile'], conf['force'], conf=conf)
        except Exception as e:
            try:
                if server:
                    server.quit()
            except Exception as e2:
                if conf['verbose'] > 1:
                    logSys.exception(e2)
            raise
        finally:
            if conf.get('onstart'):
                conf['onstart']()
        return server

    @staticmethod
    def startServerAsync(conf):
        if False:
            while True:
                i = 10
        pid = 0
        frk = not conf['async'] and PRODUCTION
        if frk:
            pid = os.fork()
        logSys.debug('  async starting of server in %s, fork: %s - %s', os.getpid(), frk, pid)
        if pid == 0:
            args = list()
            args.append(SERVER)
            args.append('--async')
            args.append('-b')
            args.append('-s')
            args.append(conf['socket'])
            args.append('-p')
            args.append(conf['pidfile'])
            if conf['force']:
                args.append('-x')
            if conf['verbose'] > 1:
                args.append('-' + 'v' * (conf['verbose'] - 1))
            for o in ('loglevel', 'logtarget', 'syslogsocket'):
                args.append('--' + o)
                args.append(conf[o])
            try:
                exe = Fail2banServer.getServerPath()
                if not frk:
                    args[0] = exe
                    exe = sys.executable
                    args[0:0] = [exe]
                logSys.debug('Starting %r with args %r', exe, args)
                if frk:
                    os.execv(exe, args)
                else:
                    ret = os.spawnv(os.P_WAIT, exe, args)
                    if ret != 0:
                        raise OSError(ret, 'Unknown error by executing server %r with %r' % (args[1], exe))
            except OSError as e:
                if not frk:
                    raise
                logSys.warning('Initial start attempt failed (%s). Starting %r with the same args', e, SERVER)
                if frk:
                    os.execvp(SERVER, args)

    @staticmethod
    def getServerPath():
        if False:
            return 10
        startdir = sys.path[0]
        exe = os.path.abspath(os.path.join(startdir, SERVER))
        if not os.path.isfile(exe):
            startdir = os.path.dirname(sys.argv[0])
            exe = os.path.abspath(os.path.join(startdir, SERVER))
            if not os.path.isfile(exe):
                startdir = os.path.dirname(os.path.abspath(__file__))
                startdir = os.path.join(os.path.dirname(os.path.dirname(startdir)), 'bin')
                exe = os.path.abspath(os.path.join(startdir, SERVER))
        return exe

    def _Fail2banClient(self):
        if False:
            return 10
        from .fail2banclient import Fail2banClient
        cli = Fail2banClient()
        cli.applyMembers(self)
        return cli

    def start(self, argv):
        if False:
            return 10
        server = None
        try:
            ret = self.initCmdLine(argv)
            if ret is not None:
                return ret
            args = self._args
            cli = None
            if len(args) == 1 and args[0] == 'start' and (not self._conf.get('interactive', False)):
                pass
            elif len(args) or self._conf.get('interactive', False):
                cli = self._Fail2banClient()
                return cli.start(argv)
            background = self._conf['background']
            nonsync = self._conf.get('async', False)
            if not nonsync:
                from ..server.utils import Utils
                cli = self._Fail2banClient()
                cli._conf = self._conf
                phase = dict()
                logSys.debug('Configure via async client thread')
                cli.configureServer(phase=phase)
            pid = os.getpid()
            server = Fail2banServer.startServerDirect(self._conf, background, cli._set_server if cli else None)
            if pid != os.getpid():
                os._exit(0)
            if cli:
                cli._server = server
            if not nonsync and cli:
                Utils.wait_for(lambda : phase.get('done', None) is not None, self._conf['timeout'], 0.001)
                if not phase.get('done', False):
                    if server:
                        server.quit()
                    exit(255)
                if background:
                    logSys.debug('Starting server done')
        except Exception as e:
            if self._conf['verbose'] > 1:
                logSys.exception(e)
            else:
                logSys.error(e)
            if server:
                server.quit()
            exit(255)
        return True

    @staticmethod
    def exit(code=0):
        if False:
            for i in range(10):
                print('nop')
        if code != 0:
            logSys.error('Could not start %s', SERVER)
        exit(code)

def exec_command_line(argv):
    if False:
        return 10
    server = Fail2banServer()
    if server.start(argv):
        exit(0)
    else:
        exit(255)