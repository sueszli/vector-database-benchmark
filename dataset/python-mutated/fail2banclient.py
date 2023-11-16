__author__ = 'Fail2Ban Developers'
__copyright__ = 'Copyright (c) 2004-2008 Cyril Jaquier, 2012-2014 Yaroslav Halchenko, 2014-2016 Serg G. Brester'
__license__ = 'GPL'
import os
import shlex
import signal
import socket
import sys
import time
import threading
from threading import Thread
from ..version import version
from .csocket import CSocket
from .beautifier import Beautifier
from .fail2bancmdline import Fail2banCmdLine, ServerExecutionException, ExitException, logSys, exit, output
from ..server.utils import Utils
PROMPT = 'fail2ban> '

def _thread_name():
    if False:
        while True:
            i = 10
    return threading.current_thread().__class__.__name__

def input_command():
    if False:
        return 10
    return input(PROMPT)

class Fail2banClient(Fail2banCmdLine, Thread):

    def __init__(self):
        if False:
            print('Hello World!')
        Fail2banCmdLine.__init__(self)
        Thread.__init__(self)
        self._alive = True
        self._server = None
        self._beautifier = None

    def dispInteractive(self):
        if False:
            while True:
                i = 10
        output('Fail2Ban v' + version + ' reads log file that contains password failure report')
        output('and bans the corresponding IP addresses using firewall rules.')
        output('')

    def __sigTERMhandler(self, signum, frame):
        if False:
            return 10
        output('')
        logSys.warning('Caught signal %d. Exiting' % signum)
        exit(255)

    def __ping(self, timeout=0.1):
        if False:
            print('Hello World!')
        return self.__processCmd([['ping'] + ([timeout] if timeout != -1 else [])], False, timeout=timeout)

    @property
    def beautifier(self):
        if False:
            i = 10
            return i + 15
        if self._beautifier:
            return self._beautifier
        self._beautifier = Beautifier()
        return self._beautifier

    def __processCmd(self, cmd, showRet=True, timeout=-1):
        if False:
            while True:
                i = 10
        client = None
        try:
            beautifier = self.beautifier
            streamRet = True
            for c in cmd:
                beautifier.setInputCmd(c)
                try:
                    if not client:
                        client = CSocket(self._conf['socket'], timeout=timeout)
                    elif timeout != -1:
                        client.settimeout(timeout)
                    if self._conf['verbose'] > 2:
                        logSys.log(5, 'CMD: %r', c)
                    ret = client.send(c)
                    if ret[0] == 0:
                        logSys.log(5, 'OK : %r', ret[1])
                        if showRet or c[0] in ('echo', 'server-status'):
                            output(beautifier.beautify(ret[1]))
                    else:
                        logSys.error('NOK: %r', ret[1].args)
                        if showRet:
                            output(beautifier.beautifyError(ret[1]))
                        streamRet = False
                except socket.error as e:
                    if showRet or self._conf['verbose'] > 1:
                        if showRet or c[0] != 'ping':
                            self.__logSocketError(e, c[0] == 'ping')
                        else:
                            logSys.log(5, ' -- %s failed -- %r', c, e)
                    return False
                except Exception as e:
                    if showRet or self._conf['verbose'] > 1:
                        if self._conf['verbose'] > 1:
                            logSys.exception(e)
                        else:
                            logSys.error(e)
                    return False
        finally:
            if client:
                try:
                    client.close()
                except Exception as e:
                    if showRet or self._conf['verbose'] > 1:
                        logSys.debug(e)
            if showRet or c[0] in ('echo', 'server-status'):
                sys.stdout.flush()
        return streamRet

    def __logSocketError(self, prevError='', errorOnly=False):
        if False:
            return 10
        try:
            if os.access(self._conf['socket'], os.F_OK):
                if os.access(self._conf['socket'], os.W_OK):
                    if errorOnly:
                        logSys.error(prevError)
                    else:
                        logSys.error('%sUnable to contact server. Is it running?', '[%s] ' % prevError if prevError else '')
                else:
                    logSys.error('Permission denied to socket: %s, (you must be root)', self._conf['socket'])
            else:
                logSys.error('Failed to access socket path: %s. Is fail2ban running?', self._conf['socket'])
        except Exception as e:
            logSys.error('Exception while checking socket access: %s', self._conf['socket'])
            logSys.error(e)

    def __prepareStartServer(self):
        if False:
            while True:
                i = 10
        if self.__ping():
            logSys.error('Server already running')
            return None
        (ret, stream) = self.readConfig()
        if not ret:
            return None
        if not self._conf['force'] and os.path.exists(self._conf['socket']):
            logSys.error('Fail2ban seems to be in unexpected state (not running but the socket exists)')
            return None
        return [['server-stream', stream], ['server-status']]

    def _set_server(self, s):
        if False:
            return 10
        self._server = s

    def __startServer(self, background=True):
        if False:
            return 10
        from .fail2banserver import Fail2banServer
        stream = self.__prepareStartServer()
        self._alive = True
        if not stream:
            return False
        try:
            if background:
                Fail2banServer.startServerAsync(self._conf)
                if not self.__processStartStreamAfterWait(stream, False):
                    return False
            else:
                phase = dict()
                self.configureServer(phase=phase, stream=stream)
                self.daemon = True
                self._server = Fail2banServer.startServerDirect(self._conf, False, self._set_server)
                if not phase.get('done', False):
                    if self._server:
                        self._server.quit()
                        self._server = None
                    exit(255)
        except ExitException:
            raise
        except Exception as e:
            output('')
            logSys.error('Exception while starting server ' + ('background' if background else 'foreground'))
            if self._conf['verbose'] > 1:
                logSys.exception(e)
            else:
                logSys.error(e)
            return False
        return True

    def configureServer(self, nonsync=True, phase=None, stream=None):
        if False:
            print('Hello World!')
        if nonsync:
            if phase is not None:

                def _server_ready():
                    if False:
                        for i in range(10):
                            print('nop')
                    phase['start-ready'] = True
                    logSys.log(5, '  server phase %s', phase)
                self._conf['onstart'] = _server_ready
            th = Thread(target=Fail2banClient.configureServer, args=(self, False, phase, stream))
            th.daemon = True
            th.start()
            if stream is None and phase is not None:
                Utils.wait_for(lambda : phase.get('ready', None) is not None, self._conf['timeout'], 0.001)
                logSys.log(5, '  server phase %s', phase)
                if not phase.get('start', False):
                    raise ServerExecutionException('Async configuration of server failed')
            return True
        if phase is not None:
            phase['start'] = True
            logSys.log(5, '  client phase %s', phase)
        if stream is None:
            stream = self.__prepareStartServer()
        if phase is not None:
            phase['ready'] = phase['start'] = True if stream else False
            logSys.log(5, '  client phase %s', phase)
        if not stream:
            return False
        if phase is not None:
            Utils.wait_for(lambda : phase.get('start-ready', None) is not None, 0.5, 0.001)
            phase['configure'] = True if stream else False
            logSys.log(5, '  client phase %s', phase)
        ret = self.__processStartStreamAfterWait(stream, False)
        if phase is not None:
            phase['done'] = ret
        return ret

    def __processCommand(self, cmd):
        if False:
            return 10
        if not isinstance(cmd, list):
            cmd = list(cmd)
        if len(cmd) == 1 and cmd[0] == 'start':
            ret = self.__startServer(self._conf['background'])
            if not ret:
                return False
            return ret
        elif len(cmd) >= 1 and cmd[0] == 'restart':
            if len(cmd) > 1:
                cmd[0:1] = ['reload', '--restart']
                return self.__processCommand(cmd)
            if self._conf.get('interactive', False):
                output('  ## stop ... ')
            self.__processCommand(['stop'])
            if not self.__waitOnServer(False):
                logSys.error('Could not stop server')
                return False
            if self._conf.get('interactive', False):
                output('  ## load configuration ... ')
                self.resetConf()
                ret = self.initCmdLine(self._argv)
                if ret is not None:
                    return ret
            if self._conf.get('interactive', False):
                output('  ## start ... ')
            return self.__processCommand(['start'])
        elif len(cmd) >= 1 and cmd[0] == 'reload':
            opts = []
            while len(cmd) >= 2:
                if cmd[1] in ('--restart', '--unban', '--if-exists'):
                    opts.append(cmd[1])
                    del cmd[1]
                else:
                    if len(cmd) > 2:
                        logSys.error('Unexpected argument(s) for reload: %r', cmd[1:])
                        return False
                    break
            if self.__ping(timeout=-1):
                if len(cmd) == 1 or cmd[1] == '--all':
                    jail = '--all'
                    (ret, stream) = self.readConfig()
                else:
                    jail = cmd[1]
                    (ret, stream) = self.readConfig(jail)
                if not ret:
                    return False
                if self._conf.get('interactive', False):
                    output('  ## reload ... ')
                return self.__processCmd([['reload', jail, opts, stream]], True)
            else:
                logSys.error('Could not find server')
                return False
        elif len(cmd) > 1 and cmd[0] == 'ping':
            return self.__processCmd([cmd], timeout=float(cmd[1]))
        else:
            return self.__processCmd([cmd])

    def __processStartStreamAfterWait(self, *args):
        if False:
            return 10
        ret = False
        try:
            if not self.__waitOnServer():
                logSys.error('Could not find server, waiting failed')
                return False
            ret = self.__processCmd(*args)
        except ServerExecutionException as e:
            if self._conf['verbose'] > 1:
                logSys.exception(e)
            logSys.error('Could not start server. Maybe an old socket file is still present. Try to remove ' + self._conf['socket'] + '. If you used fail2ban-client to start the server, adding the -x option will do it')
        if not ret and self._server:
            self._server.quit()
            self._server = None
        return ret

    def __waitOnServer(self, alive=True, maxtime=None):
        if False:
            return 10
        if maxtime is None:
            maxtime = self._conf['timeout']
        starttime = time.time()
        logSys.log(5, '__waitOnServer: %r', (alive, maxtime))
        sltime = 0.0125 / 2
        test = lambda : os.path.exists(self._conf['socket']) and self.__ping(timeout=sltime)
        with VisualWait(self._conf['verbose']) as vis:
            while self._alive:
                runf = test()
                if runf == alive:
                    return True
                waittime = time.time() - starttime
                logSys.log(5, '  wait-time: %s', waittime)
                if waittime > 1:
                    vis.heartbeat()
                if waittime >= maxtime:
                    raise ServerExecutionException('Failed to start server')
                sltime = min(sltime * 2, 0.5 if waittime > 0.2 else 0.1)
                time.sleep(sltime)
        return False

    def start(self, argv):
        if False:
            while True:
                i = 10
        _prev_signals = {}
        if _thread_name() == '_MainThread':
            for s in (signal.SIGTERM, signal.SIGINT):
                _prev_signals[s] = signal.getsignal(s)
                signal.signal(s, self.__sigTERMhandler)
        try:
            if self._argv is None:
                ret = self.initCmdLine(argv)
                if ret is not None:
                    if ret:
                        return True
                    raise ServerExecutionException('Init of command line failed')
            args = self._args
            if self._conf.get('interactive', False):
                try:
                    import readline
                except ImportError:
                    raise ServerExecutionException('Readline not available')
                try:
                    ret = True
                    if len(args) > 0:
                        ret = self.__processCommand(args)
                    if ret:
                        readline.parse_and_bind('tab: complete')
                        self.dispInteractive()
                        while True:
                            cmd = input_command()
                            if cmd == 'exit' or cmd == 'quit':
                                return True
                            if cmd == 'help':
                                self.dispUsage()
                            elif not cmd == '':
                                try:
                                    self.__processCommand(shlex.split(cmd))
                                except Exception as e:
                                    if self._conf['verbose'] > 1:
                                        logSys.exception(e)
                                    else:
                                        logSys.error(e)
                except (EOFError, KeyboardInterrupt):
                    output('')
                    raise
            else:
                if len(args) < 1:
                    self.dispUsage()
                    return False
                return self.__processCommand(args)
        except Exception as e:
            if self._conf['verbose'] > 1:
                logSys.exception(e)
            else:
                logSys.error(e)
            return False
        finally:
            self._alive = False
            for (s, sh) in _prev_signals.items():
                signal.signal(s, sh)

class _VisualWait:
    """Small progress indication (as "wonderful visual") during waiting process
	"""
    pos = 0
    delta = 1

    def __init__(self, maxpos=10):
        if False:
            print('Hello World!')
        self.maxpos = maxpos

    def __enter__(self):
        if False:
            return 10
        return self

    def __exit__(self, *args):
        if False:
            i = 10
            return i + 15
        if self.pos:
            sys.stdout.write('\r' + ' ' * (35 + self.maxpos) + '\r')
            sys.stdout.flush()

    def heartbeat(self):
        if False:
            i = 10
            return i + 15
        'Show or step for progress indicator\n\t\t'
        if not self.pos:
            sys.stdout.write('\nINFO   [#' + ' ' * self.maxpos + '] Waiting on the server...\r\x1b[8C')
        self.pos += self.delta
        if self.delta > 0:
            s = ' #\x1b[1D' if self.pos > 1 else '# \x1b[2D'
        else:
            s = '\x1b[1D# \x1b[2D'
        sys.stdout.write(s)
        sys.stdout.flush()
        if self.pos > self.maxpos:
            self.delta = -1
        elif self.pos < 2:
            self.delta = 1

class _NotVisualWait:
    """Mockup for invisible progress indication (not verbose)
	"""

    def __enter__(self):
        if False:
            print('Hello World!')
        return self

    def __exit__(self, *args):
        if False:
            print('Hello World!')
        pass

    def heartbeat(self):
        if False:
            return 10
        pass

def VisualWait(verbose, *args, **kwargs):
    if False:
        while True:
            i = 10
    'Wonderful visual progress indication (if verbose)\n\t'
    return _VisualWait(*args, **kwargs) if verbose > 1 else _NotVisualWait()

def exec_command_line(argv):
    if False:
        print('Hello World!')
    client = Fail2banClient()
    if client.start(argv):
        exit(0)
    else:
        exit(255)