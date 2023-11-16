import errno
import os
import random
import select
import signal
import sys
import time
import traceback
from gunicorn.errors import HaltServer, AppImportError
from gunicorn.pidfile import Pidfile
from gunicorn import sock, systemd, util
from gunicorn import __version__, SERVER_SOFTWARE

class Arbiter(object):
    """
    Arbiter maintain the workers processes alive. It launches or
    kills them if needed. It also manages application reloading
    via SIGHUP/USR2.
    """
    WORKER_BOOT_ERROR = 3
    APP_LOAD_ERROR = 4
    START_CTX = {}
    LISTENERS = []
    WORKERS = {}
    PIPE = []
    SIG_QUEUE = []
    SIGNALS = [getattr(signal, 'SIG%s' % x) for x in 'HUP QUIT INT TERM TTIN TTOU USR1 USR2 WINCH'.split()]
    SIG_NAMES = dict(((getattr(signal, name), name[3:].lower()) for name in dir(signal) if name[:3] == 'SIG' and name[3] != '_'))

    def __init__(self, app):
        if False:
            print('Hello World!')
        os.environ['SERVER_SOFTWARE'] = SERVER_SOFTWARE
        self._num_workers = None
        self._last_logged_active_worker_count = None
        self.log = None
        self.setup(app)
        self.pidfile = None
        self.systemd = False
        self.worker_age = 0
        self.reexec_pid = 0
        self.master_pid = 0
        self.master_name = 'Master'
        cwd = util.getcwd()
        args = sys.argv[:]
        args.insert(0, sys.executable)
        self.START_CTX = {'args': args, 'cwd': cwd, 0: sys.executable}

    def _get_num_workers(self):
        if False:
            while True:
                i = 10
        return self._num_workers

    def _set_num_workers(self, value):
        if False:
            i = 10
            return i + 15
        old_value = self._num_workers
        self._num_workers = value
        self.cfg.nworkers_changed(self, value, old_value)
    num_workers = property(_get_num_workers, _set_num_workers)

    def setup(self, app):
        if False:
            for i in range(10):
                print('nop')
        self.app = app
        self.cfg = app.cfg
        if self.log is None:
            self.log = self.cfg.logger_class(app.cfg)
        if 'GUNICORN_FD' in os.environ:
            self.log.reopen_files()
        self.worker_class = self.cfg.worker_class
        self.address = self.cfg.address
        self.num_workers = self.cfg.workers
        self.timeout = self.cfg.timeout
        self.proc_name = self.cfg.proc_name
        self.log.debug('Current configuration:\n{0}'.format('\n'.join(('  {0}: {1}'.format(config, value.value) for (config, value) in sorted(self.cfg.settings.items(), key=lambda setting: setting[1])))))
        if self.cfg.env:
            for (k, v) in self.cfg.env.items():
                os.environ[k] = v
        if self.cfg.preload_app:
            self.app.wsgi()

    def start(self):
        if False:
            i = 10
            return i + 15
        '        Initialize the arbiter. Start listening and set pidfile if needed.\n        '
        self.log.info('Starting gunicorn %s', __version__)
        if 'GUNICORN_PID' in os.environ:
            self.master_pid = int(os.environ.get('GUNICORN_PID'))
            self.proc_name = self.proc_name + '.2'
            self.master_name = 'Master.2'
        self.pid = os.getpid()
        if self.cfg.pidfile is not None:
            pidname = self.cfg.pidfile
            if self.master_pid != 0:
                pidname += '.2'
            self.pidfile = Pidfile(pidname)
            self.pidfile.create(self.pid)
        self.cfg.on_starting(self)
        self.init_signals()
        if not self.LISTENERS:
            fds = None
            listen_fds = systemd.listen_fds()
            if listen_fds:
                self.systemd = True
                fds = range(systemd.SD_LISTEN_FDS_START, systemd.SD_LISTEN_FDS_START + listen_fds)
            elif self.master_pid:
                fds = []
                for fd in os.environ.pop('GUNICORN_FD').split(','):
                    fds.append(int(fd))
            self.LISTENERS = sock.create_sockets(self.cfg, self.log, fds)
        listeners_str = ','.join([str(lnr) for lnr in self.LISTENERS])
        self.log.debug('Arbiter booted')
        self.log.info('Listening at: %s (%s)', listeners_str, self.pid)
        self.log.info('Using worker: %s', self.cfg.worker_class_str)
        systemd.sd_notify('READY=1\nSTATUS=Gunicorn arbiter booted', self.log)
        if hasattr(self.worker_class, 'check_config'):
            self.worker_class.check_config(self.cfg, self.log)
        self.cfg.when_ready(self)

    def init_signals(self):
        if False:
            while True:
                i = 10
        '        Initialize master signal handling. Most of the signals\n        are queued. Child signals only wake up the master.\n        '
        for p in self.PIPE:
            os.close(p)
        self.PIPE = pair = os.pipe()
        for p in pair:
            util.set_non_blocking(p)
            util.close_on_exec(p)
        self.log.close_on_exec()
        for s in self.SIGNALS:
            signal.signal(s, self.signal)
        signal.signal(signal.SIGCHLD, self.handle_chld)

    def signal(self, sig, frame):
        if False:
            return 10
        if len(self.SIG_QUEUE) < 5:
            self.SIG_QUEUE.append(sig)
            self.wakeup()

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        'Main master loop.'
        self.start()
        util._setproctitle('master [%s]' % self.proc_name)
        try:
            self.manage_workers()
            while True:
                self.maybe_promote_master()
                sig = self.SIG_QUEUE.pop(0) if self.SIG_QUEUE else None
                if sig is None:
                    self.sleep()
                    self.murder_workers()
                    self.manage_workers()
                    continue
                if sig not in self.SIG_NAMES:
                    self.log.info('Ignoring unknown signal: %s', sig)
                    continue
                signame = self.SIG_NAMES.get(sig)
                handler = getattr(self, 'handle_%s' % signame, None)
                if not handler:
                    self.log.error('Unhandled signal: %s', signame)
                    continue
                self.log.info('Handling signal: %s', signame)
                handler()
                self.wakeup()
        except (StopIteration, KeyboardInterrupt):
            self.halt()
        except HaltServer as inst:
            self.halt(reason=inst.reason, exit_status=inst.exit_status)
        except SystemExit:
            raise
        except Exception:
            self.log.error('Unhandled exception in main loop', exc_info=True)
            self.stop(False)
            if self.pidfile is not None:
                self.pidfile.unlink()
            sys.exit(-1)

    def handle_chld(self, sig, frame):
        if False:
            while True:
                i = 10
        'SIGCHLD handling'
        self.reap_workers()
        self.wakeup()

    def handle_hup(self):
        if False:
            i = 10
            return i + 15
        '        HUP handling.\n        - Reload configuration\n        - Start the new worker processes with a new configuration\n        - Gracefully shutdown the old worker processes\n        '
        self.log.info('Hang up: %s', self.master_name)
        self.reload()

    def handle_term(self):
        if False:
            for i in range(10):
                print('nop')
        'SIGTERM handling'
        raise StopIteration

    def handle_int(self):
        if False:
            i = 10
            return i + 15
        'SIGINT handling'
        self.stop(False)
        raise StopIteration

    def handle_quit(self):
        if False:
            while True:
                i = 10
        'SIGQUIT handling'
        self.stop(False)
        raise StopIteration

    def handle_ttin(self):
        if False:
            while True:
                i = 10
        '        SIGTTIN handling.\n        Increases the number of workers by one.\n        '
        self.num_workers += 1
        self.manage_workers()

    def handle_ttou(self):
        if False:
            i = 10
            return i + 15
        '        SIGTTOU handling.\n        Decreases the number of workers by one.\n        '
        if self.num_workers <= 1:
            return
        self.num_workers -= 1
        self.manage_workers()

    def handle_usr1(self):
        if False:
            print('Hello World!')
        '        SIGUSR1 handling.\n        Kill all workers by sending them a SIGUSR1\n        '
        self.log.reopen_files()
        self.kill_workers(signal.SIGUSR1)

    def handle_usr2(self):
        if False:
            print('Hello World!')
        '        SIGUSR2 handling.\n        Creates a new arbiter/worker set as a fork of the current\n        arbiter without affecting old workers. Use this to do live\n        deployment with the ability to backout a change.\n        '
        self.reexec()

    def handle_winch(self):
        if False:
            print('Hello World!')
        'SIGWINCH handling'
        if self.cfg.daemon:
            self.log.info('graceful stop of workers')
            self.num_workers = 0
            self.kill_workers(signal.SIGTERM)
        else:
            self.log.debug('SIGWINCH ignored. Not daemonized')

    def maybe_promote_master(self):
        if False:
            print('Hello World!')
        if self.master_pid == 0:
            return
        if self.master_pid != os.getppid():
            self.log.info('Master has been promoted.')
            self.master_name = 'Master'
            self.master_pid = 0
            self.proc_name = self.cfg.proc_name
            del os.environ['GUNICORN_PID']
            if self.pidfile is not None:
                self.pidfile.rename(self.cfg.pidfile)
            util._setproctitle('master [%s]' % self.proc_name)

    def wakeup(self):
        if False:
            while True:
                i = 10
        '        Wake up the arbiter by writing to the PIPE\n        '
        try:
            os.write(self.PIPE[1], b'.')
        except IOError as e:
            if e.errno not in [errno.EAGAIN, errno.EINTR]:
                raise

    def halt(self, reason=None, exit_status=0):
        if False:
            while True:
                i = 10
        ' halt arbiter '
        self.stop()
        log_func = self.log.info if exit_status == 0 else self.log.error
        log_func('Shutting down: %s', self.master_name)
        if reason is not None:
            log_func('Reason: %s', reason)
        if self.pidfile is not None:
            self.pidfile.unlink()
        self.cfg.on_exit(self)
        sys.exit(exit_status)

    def sleep(self):
        if False:
            for i in range(10):
                print('nop')
        '        Sleep until PIPE is readable or we timeout.\n        A readable PIPE means a signal occurred.\n        '
        try:
            ready = select.select([self.PIPE[0]], [], [], 1.0)
            if not ready[0]:
                return
            while os.read(self.PIPE[0], 1):
                pass
        except (select.error, OSError) as e:
            error_number = getattr(e, 'errno', e.args[0])
            if error_number not in [errno.EAGAIN, errno.EINTR]:
                raise
        except KeyboardInterrupt:
            sys.exit()

    def stop(self, graceful=True):
        if False:
            i = 10
            return i + 15
        '        Stop workers\n\n        :attr graceful: boolean, If True (the default) workers will be\n        killed gracefully  (ie. trying to wait for the current connection)\n        '
        unlink = self.reexec_pid == self.master_pid == 0 and (not self.systemd) and (not self.cfg.reuse_port)
        sock.close_sockets(self.LISTENERS, unlink)
        self.LISTENERS = []
        sig = signal.SIGTERM
        if not graceful:
            sig = signal.SIGQUIT
        limit = time.time() + self.cfg.graceful_timeout
        self.kill_workers(sig)
        while self.WORKERS and time.time() < limit:
            time.sleep(0.1)
        self.kill_workers(signal.SIGKILL)

    def reexec(self):
        if False:
            for i in range(10):
                print('nop')
        '        Relaunch the master and workers.\n        '
        if self.reexec_pid != 0:
            self.log.warning('USR2 signal ignored. Child exists.')
            return
        if self.master_pid != 0:
            self.log.warning('USR2 signal ignored. Parent exists.')
            return
        master_pid = os.getpid()
        self.reexec_pid = os.fork()
        if self.reexec_pid != 0:
            return
        self.cfg.pre_exec(self)
        environ = self.cfg.env_orig.copy()
        environ['GUNICORN_PID'] = str(master_pid)
        if self.systemd:
            environ['LISTEN_PID'] = str(os.getpid())
            environ['LISTEN_FDS'] = str(len(self.LISTENERS))
        else:
            environ['GUNICORN_FD'] = ','.join((str(lnr.fileno()) for lnr in self.LISTENERS))
        os.chdir(self.START_CTX['cwd'])
        os.execvpe(self.START_CTX[0], self.START_CTX['args'], environ)

    def reload(self):
        if False:
            print('Hello World!')
        old_address = self.cfg.address
        for k in self.cfg.env:
            if k in self.cfg.env_orig:
                os.environ[k] = self.cfg.env_orig[k]
            else:
                try:
                    del os.environ[k]
                except KeyError:
                    pass
        self.app.reload()
        self.setup(self.app)
        self.log.reopen_files()
        if old_address != self.cfg.address:
            for lnr in self.LISTENERS:
                lnr.close()
            self.LISTENERS = sock.create_sockets(self.cfg, self.log)
            listeners_str = ','.join([str(lnr) for lnr in self.LISTENERS])
            self.log.info('Listening at: %s', listeners_str)
        self.cfg.on_reload(self)
        if self.pidfile is not None:
            self.pidfile.unlink()
        if self.cfg.pidfile is not None:
            self.pidfile = Pidfile(self.cfg.pidfile)
            self.pidfile.create(self.pid)
        util._setproctitle('master [%s]' % self.proc_name)
        for _ in range(self.cfg.workers):
            self.spawn_worker()
        self.manage_workers()

    def murder_workers(self):
        if False:
            for i in range(10):
                print('nop')
        '        Kill unused/idle workers\n        '
        if not self.timeout:
            return
        workers = list(self.WORKERS.items())
        for (pid, worker) in workers:
            try:
                if time.time() - worker.tmp.last_update() <= self.timeout:
                    continue
            except (OSError, ValueError):
                continue
            if not worker.aborted:
                self.log.critical('WORKER TIMEOUT (pid:%s)', pid)
                worker.aborted = True
                self.kill_worker(pid, signal.SIGABRT)
            else:
                self.kill_worker(pid, signal.SIGKILL)

    def reap_workers(self):
        if False:
            for i in range(10):
                print('nop')
        '        Reap workers to avoid zombie processes\n        '
        try:
            while True:
                (wpid, status) = os.waitpid(-1, os.WNOHANG)
                if not wpid:
                    break
                if self.reexec_pid == wpid:
                    self.reexec_pid = 0
                else:
                    exitcode = status >> 8
                    if exitcode != 0:
                        self.log.error('Worker (pid:%s) exited with code %s', wpid, exitcode)
                    if exitcode == self.WORKER_BOOT_ERROR:
                        reason = 'Worker failed to boot.'
                        raise HaltServer(reason, self.WORKER_BOOT_ERROR)
                    if exitcode == self.APP_LOAD_ERROR:
                        reason = 'App failed to load.'
                        raise HaltServer(reason, self.APP_LOAD_ERROR)
                    if exitcode > 0:
                        self.log.error('Worker (pid:%s) exited with code %s.', wpid, exitcode)
                    elif status > 0:
                        try:
                            sig_name = signal.Signals(status).name
                        except ValueError:
                            sig_name = 'code {}'.format(status)
                        msg = 'Worker (pid:{}) was sent {}!'.format(wpid, sig_name)
                        if status == signal.SIGKILL:
                            msg += ' Perhaps out of memory?'
                        self.log.error(msg)
                    worker = self.WORKERS.pop(wpid, None)
                    if not worker:
                        continue
                    worker.tmp.close()
                    self.cfg.child_exit(self, worker)
        except OSError as e:
            if e.errno != errno.ECHILD:
                raise

    def manage_workers(self):
        if False:
            return 10
        '        Maintain the number of workers by spawning or killing\n        as required.\n        '
        if len(self.WORKERS) < self.num_workers:
            self.spawn_workers()
        workers = self.WORKERS.items()
        workers = sorted(workers, key=lambda w: w[1].age)
        while len(workers) > self.num_workers:
            (pid, _) = workers.pop(0)
            self.kill_worker(pid, signal.SIGTERM)
        active_worker_count = len(workers)
        if self._last_logged_active_worker_count != active_worker_count:
            self._last_logged_active_worker_count = active_worker_count
            self.log.debug('{0} workers'.format(active_worker_count), extra={'metric': 'gunicorn.workers', 'value': active_worker_count, 'mtype': 'gauge'})

    def spawn_worker(self):
        if False:
            while True:
                i = 10
        self.worker_age += 1
        worker = self.worker_class(self.worker_age, self.pid, self.LISTENERS, self.app, self.timeout / 2.0, self.cfg, self.log)
        self.cfg.pre_fork(self, worker)
        pid = os.fork()
        if pid != 0:
            worker.pid = pid
            self.WORKERS[pid] = worker
            return pid
        for sibling in self.WORKERS.values():
            sibling.tmp.close()
        worker.pid = os.getpid()
        try:
            util._setproctitle('worker [%s]' % self.proc_name)
            self.log.info('Booting worker with pid: %s', worker.pid)
            self.cfg.post_fork(self, worker)
            worker.init_process()
            sys.exit(0)
        except SystemExit:
            raise
        except AppImportError as e:
            self.log.debug('Exception while loading the application', exc_info=True)
            print('%s' % e, file=sys.stderr)
            sys.stderr.flush()
            sys.exit(self.APP_LOAD_ERROR)
        except Exception:
            self.log.exception('Exception in worker process')
            if not worker.booted:
                sys.exit(self.WORKER_BOOT_ERROR)
            sys.exit(-1)
        finally:
            self.log.info('Worker exiting (pid: %s)', worker.pid)
            try:
                worker.tmp.close()
                self.cfg.worker_exit(self, worker)
            except Exception:
                self.log.warning('Exception during worker exit:\n%s', traceback.format_exc())

    def spawn_workers(self):
        if False:
            print('Hello World!')
        '        Spawn new workers as needed.\n\n        This is where a worker process leaves the main loop\n        of the master process.\n        '
        for _ in range(self.num_workers - len(self.WORKERS)):
            self.spawn_worker()
            time.sleep(0.1 * random.random())

    def kill_workers(self, sig):
        if False:
            while True:
                i = 10
        '        Kill all workers with the signal `sig`\n        :attr sig: `signal.SIG*` value\n        '
        worker_pids = list(self.WORKERS.keys())
        for pid in worker_pids:
            self.kill_worker(pid, sig)

    def kill_worker(self, pid, sig):
        if False:
            while True:
                i = 10
        '        Kill a worker\n\n        :attr pid: int, worker pid\n        :attr sig: `signal.SIG*` value\n         '
        try:
            os.kill(pid, sig)
        except OSError as e:
            if e.errno == errno.ESRCH:
                try:
                    worker = self.WORKERS.pop(pid)
                    worker.tmp.close()
                    self.cfg.worker_exit(self, worker)
                    return
                except (KeyError, OSError):
                    return
            raise