"""Start/stop/manage workers."""
import errno
import os
import shlex
import signal
import sys
from collections import OrderedDict, UserList, defaultdict
from functools import partial
from subprocess import Popen
from time import sleep
from kombu.utils.encoding import from_utf8
from kombu.utils.objects import cached_property
from celery.platforms import IS_WINDOWS, Pidfile, signal_name
from celery.utils.nodenames import gethostname, host_format, node_format, nodesplit
from celery.utils.saferepr import saferepr
__all__ = ('Cluster', 'Node')
CELERY_EXE = 'celery'

def celery_exe(*args):
    if False:
        i = 10
        return i + 15
    return ' '.join((CELERY_EXE,) + args)

def build_nodename(name, prefix, suffix):
    if False:
        for i in range(10):
            print('nop')
    hostname = suffix
    if '@' in name:
        nodename = host_format(name)
        (shortname, hostname) = nodesplit(nodename)
        name = shortname
    else:
        shortname = f'{prefix}{name}'
        nodename = host_format(f'{shortname}@{hostname}')
    return (name, nodename, hostname)

def build_expander(nodename, shortname, hostname):
    if False:
        print('Hello World!')
    return partial(node_format, name=nodename, N=shortname, d=hostname, h=nodename, i='%i', I='%I')

def format_opt(opt, value):
    if False:
        for i in range(10):
            print('nop')
    if not value:
        return opt
    if opt.startswith('--'):
        return f'{opt}={value}'
    return f'{opt} {value}'

def _kwargs_to_command_line(kwargs):
    if False:
        while True:
            i = 10
    return {'--{}'.format(k.replace('_', '-')) if len(k) > 1 else f'-{k}': f'{v}' for (k, v) in kwargs.items()}

class NamespacedOptionParser:

    def __init__(self, args):
        if False:
            for i in range(10):
                print('nop')
        self.args = args
        self.options = OrderedDict()
        self.values = []
        self.passthrough = ''
        self.namespaces = defaultdict(lambda : OrderedDict())

    def parse(self):
        if False:
            print('Hello World!')
        rargs = [arg for arg in self.args if arg]
        pos = 0
        while pos < len(rargs):
            arg = rargs[pos]
            if arg == '--':
                self.passthrough = ' '.join(rargs[pos:])
                break
            elif arg[0] == '-':
                if arg[1] == '-':
                    self.process_long_opt(arg[2:])
                else:
                    value = None
                    if len(rargs) > pos + 1 and rargs[pos + 1][0] != '-':
                        value = rargs[pos + 1]
                        pos += 1
                    self.process_short_opt(arg[1:], value)
            else:
                self.values.append(arg)
            pos += 1

    def process_long_opt(self, arg, value=None):
        if False:
            i = 10
            return i + 15
        if '=' in arg:
            (arg, value) = arg.split('=', 1)
        self.add_option(arg, value, short=False)

    def process_short_opt(self, arg, value=None):
        if False:
            return 10
        self.add_option(arg, value, short=True)

    def optmerge(self, ns, defaults=None):
        if False:
            i = 10
            return i + 15
        if defaults is None:
            defaults = self.options
        return OrderedDict(defaults, **self.namespaces[ns])

    def add_option(self, name, value, short=False, ns=None):
        if False:
            return 10
        prefix = short and '-' or '--'
        dest = self.options
        if ':' in name:
            (name, ns) = name.split(':')
            dest = self.namespaces[ns]
        dest[prefix + name] = value

class Node:
    """Represents a node in a cluster."""

    def __init__(self, name, cmd=None, append=None, options=None, extra_args=None):
        if False:
            print('Hello World!')
        self.name = name
        self.cmd = cmd or f"-m {celery_exe('worker', '--detach')}"
        self.append = append
        self.extra_args = extra_args or ''
        self.options = self._annotate_with_default_opts(options or OrderedDict())
        self.expander = self._prepare_expander()
        self.argv = self._prepare_argv()
        self._pid = None

    def _annotate_with_default_opts(self, options):
        if False:
            return 10
        options['-n'] = self.name
        self._setdefaultopt(options, ['--pidfile', '-p'], '/var/run/celery/%n.pid')
        self._setdefaultopt(options, ['--logfile', '-f'], '/var/log/celery/%n%I.log')
        self._setdefaultopt(options, ['--executable'], sys.executable)
        return options

    def _setdefaultopt(self, d, alt, value):
        if False:
            return 10
        for opt in alt[1:]:
            try:
                return d[opt]
            except KeyError:
                pass
        value = d.setdefault(alt[0], os.path.normpath(value))
        dir_path = os.path.dirname(value)
        if dir_path and (not os.path.exists(dir_path)):
            os.makedirs(dir_path)
        return value

    def _prepare_expander(self):
        if False:
            print('Hello World!')
        (shortname, hostname) = self.name.split('@', 1)
        return build_expander(self.name, shortname, hostname)

    def _prepare_argv(self):
        if False:
            return 10
        cmd = self.expander(self.cmd).split(' ')
        i = cmd.index('celery') + 1
        options = self.options.copy()
        for (opt, value) in self.options.items():
            if opt in ('-A', '--app', '-b', '--broker', '--result-backend', '--loader', '--config', '--workdir', '-C', '--no-color', '-q', '--quiet'):
                cmd.insert(i, format_opt(opt, self.expander(value)))
                options.pop(opt)
        cmd = [' '.join(cmd)]
        argv = tuple(cmd + [format_opt(opt, self.expander(value)) for (opt, value) in options.items()] + [self.extra_args])
        if self.append:
            argv += (self.expander(self.append),)
        return argv

    def alive(self):
        if False:
            return 10
        return self.send(0)

    def send(self, sig, on_error=None):
        if False:
            for i in range(10):
                print('nop')
        pid = self.pid
        if pid:
            try:
                os.kill(pid, sig)
            except OSError as exc:
                if exc.errno != errno.ESRCH:
                    raise
                maybe_call(on_error, self)
                return False
            return True
        maybe_call(on_error, self)

    def start(self, env=None, **kwargs):
        if False:
            while True:
                i = 10
        return self._waitexec(self.argv, path=self.executable, env=env, **kwargs)

    def _waitexec(self, argv, path=sys.executable, env=None, on_spawn=None, on_signalled=None, on_failure=None):
        if False:
            for i in range(10):
                print('nop')
        argstr = self.prepare_argv(argv, path)
        maybe_call(on_spawn, self, argstr=' '.join(argstr), env=env)
        pipe = Popen(argstr, env=env)
        return self.handle_process_exit(pipe.wait(), on_signalled=on_signalled, on_failure=on_failure)

    def handle_process_exit(self, retcode, on_signalled=None, on_failure=None):
        if False:
            return 10
        if retcode < 0:
            maybe_call(on_signalled, self, -retcode)
            return -retcode
        elif retcode > 0:
            maybe_call(on_failure, self, retcode)
        return retcode

    def prepare_argv(self, argv, path):
        if False:
            print('Hello World!')
        args = ' '.join([path] + list(argv))
        return shlex.split(from_utf8(args), posix=not IS_WINDOWS)

    def getopt(self, *alt):
        if False:
            while True:
                i = 10
        for opt in alt:
            try:
                return self.options[opt]
            except KeyError:
                pass
        raise KeyError(alt[0])

    def __repr__(self):
        if False:
            return 10
        return f'<{type(self).__name__}: {self.name}>'

    @cached_property
    def pidfile(self):
        if False:
            for i in range(10):
                print('nop')
        return self.expander(self.getopt('--pidfile', '-p'))

    @cached_property
    def logfile(self):
        if False:
            for i in range(10):
                print('nop')
        return self.expander(self.getopt('--logfile', '-f'))

    @property
    def pid(self):
        if False:
            while True:
                i = 10
        if self._pid is not None:
            return self._pid
        try:
            return Pidfile(self.pidfile).read_pid()
        except ValueError:
            pass

    @pid.setter
    def pid(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._pid = value

    @cached_property
    def executable(self):
        if False:
            print('Hello World!')
        return self.options['--executable']

    @cached_property
    def argv_with_executable(self):
        if False:
            return 10
        return (self.executable,) + self.argv

    @classmethod
    def from_kwargs(cls, name, **kwargs):
        if False:
            i = 10
            return i + 15
        return cls(name, options=_kwargs_to_command_line(kwargs))

def maybe_call(fun, *args, **kwargs):
    if False:
        while True:
            i = 10
    if fun is not None:
        fun(*args, **kwargs)

class MultiParser:
    Node = Node

    def __init__(self, cmd='celery worker', append='', prefix='', suffix='', range_prefix='celery'):
        if False:
            return 10
        self.cmd = cmd
        self.append = append
        self.prefix = prefix
        self.suffix = suffix
        self.range_prefix = range_prefix

    def parse(self, p):
        if False:
            while True:
                i = 10
        names = p.values
        options = dict(p.options)
        ranges = len(names) == 1
        prefix = self.prefix
        cmd = options.pop('--cmd', self.cmd)
        append = options.pop('--append', self.append)
        hostname = options.pop('--hostname', options.pop('-n', gethostname()))
        prefix = options.pop('--prefix', prefix) or ''
        suffix = options.pop('--suffix', self.suffix) or hostname
        suffix = '' if suffix in ('""', "''") else suffix
        range_prefix = options.pop('--range-prefix', '') or self.range_prefix
        if ranges:
            try:
                (names, prefix) = (self._get_ranges(names), range_prefix)
            except ValueError:
                pass
        self._update_ns_opts(p, names)
        self._update_ns_ranges(p, ranges)
        return (self._node_from_options(p, name, prefix, suffix, cmd, append, options) for name in names)

    def _node_from_options(self, p, name, prefix, suffix, cmd, append, options):
        if False:
            while True:
                i = 10
        (namespace, nodename, _) = build_nodename(name, prefix, suffix)
        namespace = nodename if nodename in p.namespaces else namespace
        return Node(nodename, cmd, append, p.optmerge(namespace, options), p.passthrough)

    def _get_ranges(self, names):
        if False:
            for i in range(10):
                print('nop')
        noderange = int(names[0])
        return [str(n) for n in range(1, noderange + 1)]

    def _update_ns_opts(self, p, names):
        if False:
            print('Hello World!')
        for (ns_name, ns_opts) in list(p.namespaces.items()):
            if ns_name.isdigit():
                ns_index = int(ns_name) - 1
                if ns_index < 0:
                    raise KeyError(f'Indexes start at 1 got: {ns_name!r}')
                try:
                    p.namespaces[names[ns_index]].update(ns_opts)
                except IndexError:
                    raise KeyError(f'No node at index {ns_name!r}')

    def _update_ns_ranges(self, p, ranges):
        if False:
            return 10
        for (ns_name, ns_opts) in list(p.namespaces.items()):
            if ',' in ns_name or (ranges and '-' in ns_name):
                for subns in self._parse_ns_range(ns_name, ranges):
                    p.namespaces[subns].update(ns_opts)
                p.namespaces.pop(ns_name)

    def _parse_ns_range(self, ns, ranges=False):
        if False:
            return 10
        ret = []
        for space in ',' in ns and ns.split(',') or [ns]:
            if ranges and '-' in space:
                (start, stop) = space.split('-')
                ret.extend((str(n) for n in range(int(start), int(stop) + 1)))
            else:
                ret.append(space)
        return ret

class Cluster(UserList):
    """Represent a cluster of workers."""

    def __init__(self, nodes, cmd=None, env=None, on_stopping_preamble=None, on_send_signal=None, on_still_waiting_for=None, on_still_waiting_progress=None, on_still_waiting_end=None, on_node_start=None, on_node_restart=None, on_node_shutdown_ok=None, on_node_status=None, on_node_signal=None, on_node_signal_dead=None, on_node_down=None, on_child_spawn=None, on_child_signalled=None, on_child_failure=None):
        if False:
            print('Hello World!')
        self.nodes = nodes
        self.cmd = cmd or celery_exe('worker')
        self.env = env
        self.on_stopping_preamble = on_stopping_preamble
        self.on_send_signal = on_send_signal
        self.on_still_waiting_for = on_still_waiting_for
        self.on_still_waiting_progress = on_still_waiting_progress
        self.on_still_waiting_end = on_still_waiting_end
        self.on_node_start = on_node_start
        self.on_node_restart = on_node_restart
        self.on_node_shutdown_ok = on_node_shutdown_ok
        self.on_node_status = on_node_status
        self.on_node_signal = on_node_signal
        self.on_node_signal_dead = on_node_signal_dead
        self.on_node_down = on_node_down
        self.on_child_spawn = on_child_spawn
        self.on_child_signalled = on_child_signalled
        self.on_child_failure = on_child_failure

    def start(self):
        if False:
            while True:
                i = 10
        return [self.start_node(node) for node in self]

    def start_node(self, node):
        if False:
            while True:
                i = 10
        maybe_call(self.on_node_start, node)
        retcode = self._start_node(node)
        maybe_call(self.on_node_status, node, retcode)
        return retcode

    def _start_node(self, node):
        if False:
            return 10
        return node.start(self.env, on_spawn=self.on_child_spawn, on_signalled=self.on_child_signalled, on_failure=self.on_child_failure)

    def send_all(self, sig):
        if False:
            return 10
        for node in self.getpids(on_down=self.on_node_down):
            maybe_call(self.on_node_signal, node, signal_name(sig))
            node.send(sig, self.on_node_signal_dead)

    def kill(self):
        if False:
            while True:
                i = 10
        return self.send_all(signal.SIGKILL)

    def restart(self, sig=signal.SIGTERM):
        if False:
            return 10
        retvals = []

        def restart_on_down(node):
            if False:
                return 10
            maybe_call(self.on_node_restart, node)
            retval = self._start_node(node)
            maybe_call(self.on_node_status, node, retval)
            retvals.append(retval)
        self._stop_nodes(retry=2, on_down=restart_on_down, sig=sig)
        return retvals

    def stop(self, retry=None, callback=None, sig=signal.SIGTERM):
        if False:
            print('Hello World!')
        return self._stop_nodes(retry=retry, on_down=callback, sig=sig)

    def stopwait(self, retry=2, callback=None, sig=signal.SIGTERM):
        if False:
            return 10
        return self._stop_nodes(retry=retry, on_down=callback, sig=sig)

    def _stop_nodes(self, retry=None, on_down=None, sig=signal.SIGTERM):
        if False:
            while True:
                i = 10
        on_down = on_down if on_down is not None else self.on_node_down
        nodes = list(self.getpids(on_down=on_down))
        if nodes:
            for node in self.shutdown_nodes(nodes, sig=sig, retry=retry):
                maybe_call(on_down, node)

    def shutdown_nodes(self, nodes, sig=signal.SIGTERM, retry=None):
        if False:
            print('Hello World!')
        P = set(nodes)
        maybe_call(self.on_stopping_preamble, nodes)
        to_remove = set()
        for node in P:
            maybe_call(self.on_send_signal, node, signal_name(sig))
            if not node.send(sig, self.on_node_signal_dead):
                to_remove.add(node)
                yield node
        P -= to_remove
        if retry:
            maybe_call(self.on_still_waiting_for, P)
            its = 0
            while P:
                to_remove = set()
                for node in P:
                    its += 1
                    maybe_call(self.on_still_waiting_progress, P)
                    if not node.alive():
                        maybe_call(self.on_node_shutdown_ok, node)
                        to_remove.add(node)
                        yield node
                        maybe_call(self.on_still_waiting_for, P)
                        break
                P -= to_remove
                if P and (not its % len(P)):
                    sleep(float(retry))
            maybe_call(self.on_still_waiting_end)

    def find(self, name):
        if False:
            for i in range(10):
                print('nop')
        for node in self:
            if node.name == name:
                return node
        raise KeyError(name)

    def getpids(self, on_down=None):
        if False:
            while True:
                i = 10
        for node in self:
            if node.pid:
                yield node
            else:
                maybe_call(on_down, node)

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<{name}({0}): {1}>'.format(len(self), saferepr([n.name for n in self]), name=type(self).__name__)

    @property
    def data(self):
        if False:
            i = 10
            return i + 15
        return self.nodes