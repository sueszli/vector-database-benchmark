"""
Node objects for Mininet.

Nodes provide a simple abstraction for interacting with hosts, switches
and controllers. Local nodes are simply one or more processes on the local
machine.

Node: superclass for all (primarily local) network nodes.

Host: a virtual host. By default, a host is simply a shell; commands
    may be sent using Cmd (which waits for output), or using sendCmd(),
    which returns immediately, allowing subsequent monitoring using
    monitor(). Examples of how to run experiments using this
    functionality are provided in the examples/ directory. By default,
    hosts share the root file system, but they may also specify private
    directories.

CPULimitedHost: a virtual host whose CPU bandwidth is limited by
    RT or CFS bandwidth limiting.

Switch: superclass for switch nodes.

UserSwitch: a switch using the user-space switch from the OpenFlow
    reference implementation.

OVSSwitch: a switch using the Open vSwitch OpenFlow-compatible switch
    implementation (openvswitch.org).

OVSBridge: an Ethernet bridge implemented using Open vSwitch.
    Supports STP.

IVSSwitch: OpenFlow switch using the Indigo Virtual Switch.

Controller: superclass for OpenFlow controllers. The default controller
    is controller(8) from the reference implementation.

OVSController: The test controller from Open vSwitch.

NOXController: a controller node using NOX (noxrepo.org).

Ryu: The Ryu controller (https://osrg.github.io/ryu/)

RemoteController: a remote controller node, which may use any
    arbitrary OpenFlow-compatible controller, and which is not
    created or managed by Mininet.

Future enhancements:

- Possibly make Node, Switch and Controller more abstract so that
  they can be used for both local and remote nodes

- Create proxy objects for remote nodes (Mininet: Cluster Edition)
"""
import os
import pty
import re
import signal
import select
from re import findall
from subprocess import Popen, PIPE
from sys import exit
from time import sleep
from mininet.log import info, error, warn, debug
from mininet.util import quietRun, errRun, errFail, moveIntf, isShellBuiltin, numCores, retry, mountCgroups, BaseString, decode, encode, getincrementaldecoder, Python3, which, StrictVersion
from mininet.moduledeps import moduleDeps, pathCheck, TUN
from mininet.link import Link, Intf, TCIntf, OVSIntf

class Node(object):
    """A virtual network node is simply a shell in a network namespace.
       We communicate with it using pipes."""
    portBase = 0

    def __init__(self, name, inNamespace=True, **params):
        if False:
            return 10
        'name: name of node\n           inNamespace: in network namespace?\n           privateDirs: list of private directory strings or tuples\n           params: Node parameters (see config() for details)'
        self.checkSetup()
        self.name = params.get('name', name)
        self.privateDirs = params.get('privateDirs', [])
        self.inNamespace = params.get('inNamespace', inNamespace)
        self.waitExited = params.get('waitExited', Python3)
        self.params = params
        self.intfs = {}
        self.ports = {}
        self.nameToIntf = {}
        (self.shell, self.execed, self.pid, self.stdin, self.stdout, self.lastPid, self.lastCmd, self.pollOut) = (None, None, None, None, None, None, None, None)
        self.waiting = False
        self.readbuf = ''
        self.decoder = getincrementaldecoder()
        (self.master, self.slave) = (None, None)
        self.startShell()
        self.mountPrivateDirs()
    inToNode = {}
    outToNode = {}

    @classmethod
    def fdToNode(cls, fd):
        if False:
            while True:
                i = 10
        'Return node corresponding to given file descriptor.\n           fd: file descriptor\n           returns: node'
        node = cls.outToNode.get(fd)
        return node or cls.inToNode.get(fd)

    def startShell(self, mnopts=None):
        if False:
            while True:
                i = 10
        'Start a shell process for running commands'
        if self.shell:
            error('%s: shell is already running\n' % self.name)
            return
        opts = '-cd' if mnopts is None else mnopts
        if self.inNamespace:
            opts += 'n'
        cmd = ['mnexec', opts, 'env', 'PS1=' + chr(127), 'bash', '--norc', '--noediting', '-is', 'mininet:' + self.name]
        (self.master, self.slave) = pty.openpty()
        self.shell = self._popen(cmd, stdin=self.slave, stdout=self.slave, stderr=self.slave, close_fds=False)
        self.stdin = os.fdopen(self.master, 'r')
        self.stdout = self.stdin
        self.pid = self.shell.pid
        self.pollOut = select.poll()
        self.pollOut.register(self.stdout)
        self.outToNode[self.stdout.fileno()] = self
        self.inToNode[self.stdin.fileno()] = self
        self.execed = False
        self.lastCmd = None
        self.lastPid = None
        self.readbuf = ''
        while True:
            data = self.read(1024)
            if data[-1] == chr(127):
                break
            self.pollOut.poll()
        self.waiting = False
        self.cmd('unset HISTFILE; stty -echo; set +m')

    def mountPrivateDirs(self):
        if False:
            print('Hello World!')
        'mount private directories'
        assert not isinstance(self.privateDirs, BaseString)
        for directory in self.privateDirs:
            if isinstance(directory, tuple):
                privateDir = directory[1] % self.__dict__
                mountPoint = directory[0]
                self.cmd('mkdir -p %s' % privateDir)
                self.cmd('mkdir -p %s' % mountPoint)
                self.cmd('mount --bind %s %s' % (privateDir, mountPoint))
            else:
                self.cmd('mkdir -p %s' % directory)
                self.cmd('mount -n -t tmpfs tmpfs %s' % directory)

    def unmountPrivateDirs(self):
        if False:
            for i in range(10):
                print('nop')
        'mount private directories'
        for directory in self.privateDirs:
            if isinstance(directory, tuple):
                self.cmd('umount ', directory[0])
            else:
                self.cmd('umount ', directory)

    def _popen(self, cmd, **params):
        if False:
            return 10
        'Internal method: spawn and return a process\n            cmd: command to run (list)\n            params: parameters to Popen()'
        assert self
        popen = Popen(cmd, **params)
        debug('_popen', cmd, popen.pid)
        return popen

    def cleanup(self):
        if False:
            print('Hello World!')
        'Help python collect its garbage.'
        if self.shell:
            self.stdin.close()
            os.close(self.slave)
            if self.waitExited:
                debug('waiting for', self.pid, 'to terminate\n')
                self.shell.wait()
        self.shell = None

    def read(self, size=1024):
        if False:
            return 10
        'Buffered read from node, potentially blocking.\n           size: maximum number of characters to return'
        count = len(self.readbuf)
        if count < size:
            data = os.read(self.stdout.fileno(), size - count)
            self.readbuf += self.decoder.decode(data)
        if size >= len(self.readbuf):
            result = self.readbuf
            self.readbuf = ''
        else:
            result = self.readbuf[:size]
            self.readbuf = self.readbuf[size:]
        return result

    def readline(self):
        if False:
            i = 10
            return i + 15
        'Buffered readline from node, potentially blocking.\n           returns: line (minus newline) or None'
        self.readbuf += self.read(1024)
        if '\n' not in self.readbuf:
            return None
        pos = self.readbuf.find('\n')
        line = self.readbuf[0:pos]
        self.readbuf = self.readbuf[pos + 1:]
        return line

    def write(self, data):
        if False:
            for i in range(10):
                print('nop')
        'Write data to node.\n           data: string'
        os.write(self.stdin.fileno(), encode(data))

    def terminate(self):
        if False:
            i = 10
            return i + 15
        'Send kill signal to Node and clean up after it.'
        self.unmountPrivateDirs()
        if self.shell:
            if self.shell.poll() is None:
                os.killpg(self.shell.pid, signal.SIGHUP)
        self.cleanup()

    def stop(self, deleteIntfs=False):
        if False:
            while True:
                i = 10
        'Stop node.\n           deleteIntfs: delete interfaces? (False)'
        if deleteIntfs:
            self.deleteIntfs()
        self.terminate()

    def waitReadable(self, timeoutms=None):
        if False:
            return 10
        "Wait until node's output is readable.\n           timeoutms: timeout in ms or None to wait indefinitely.\n           returns: result of poll()"
        if len(self.readbuf) == 0:
            return self.pollOut.poll(timeoutms)
        return None

    def sendCmd(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "Send a command, followed by a command to echo a sentinel,\n           and return without waiting for the command to complete.\n           args: command and arguments, or string\n           printPid: print command's PID? (False)"
        assert self.shell and (not self.waiting)
        printPid = kwargs.get('printPid', False)
        if len(args) == 1 and isinstance(args[0], list):
            cmd = args[0]
        elif len(args) > 0:
            cmd = args
        if not isinstance(cmd, str):
            cmd = ' '.join([str(c) for c in cmd])
        if not re.search('\\w', cmd):
            cmd = 'echo -n'
        self.lastCmd = cmd
        if len(cmd) > 0 and cmd[-1] == '&':
            cmd += ' printf "\\001%d\\012" $! '
        elif printPid and (not isShellBuiltin(cmd)):
            cmd = 'mnexec -p ' + cmd
        self.write(cmd + '\n')
        self.lastPid = None
        self.waiting = True

    def sendInt(self, intr=chr(3)):
        if False:
            print('Hello World!')
        'Interrupt running command.'
        debug('sendInt: writing chr(%d)\n' % ord(intr))
        self.write(intr)

    def monitor(self, timeoutms=None, findPid=True):
        if False:
            while True:
                i = 10
        'Monitor and return the output of a command.\n           Set self.waiting to False if command has completed.\n           timeoutms: timeout in ms or None to wait indefinitely\n           findPid: look for PID from mnexec -p'
        ready = self.waitReadable(timeoutms)
        if not ready:
            return ''
        data = self.read(1024)
        pidre = '\\[\\d+\\] \\d+\\r\\n'
        marker = chr(1) + '\\d+\\r\\n'
        if findPid and chr(1) in data:
            if re.findall(pidre, data):
                data = re.sub(pidre, '', data)
            while not re.findall(marker, data):
                data += self.read(1024)
            markers = re.findall(marker, data)
            if markers:
                self.lastPid = int(markers[0][1:])
                data = re.sub(marker, '', data)
        if len(data) > 0 and data[-1] == chr(127):
            self.waiting = False
            data = data[:-1]
        elif chr(127) in data:
            self.waiting = False
            data = data.replace(chr(127), '')
        return data

    def waitOutput(self, verbose=False, findPid=True):
        if False:
            return 10
        'Wait for a command to complete.\n           Completion is signaled by a sentinel character, ASCII(127)\n           appearing in the output stream.  Wait for the sentinel and return\n           the output, including trailing newline.\n           verbose: print output interactively'
        log = info if verbose else debug
        output = ''
        while self.waiting:
            data = self.monitor(findPid=findPid)
            output += data
            log(data)
        return output

    def cmd(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Send a command, wait for output, and return it.\n           cmd: string'
        verbose = kwargs.get('verbose', False)
        log = info if verbose else debug
        log('*** %s : %s\n' % (self.name, args))
        if self.shell:
            self.sendCmd(*args, **kwargs)
            return self.waitOutput(verbose)
        else:
            warn('(%s exited - ignoring cmd%s)\n' % (self, args))
        return None

    def cmdPrint(self, *args):
        if False:
            for i in range(10):
                print('nop')
        'Call cmd and printing its output\n           cmd: string'
        return self.cmd(*args, **{'verbose': True})

    def popen(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Return a Popen() object in our namespace\n           args: Popen() args, single list, or string\n           kwargs: Popen() keyword args'
        defaults = {'stdout': PIPE, 'stderr': PIPE, 'mncmd': ['mnexec', '-da', str(self.pid)]}
        defaults.update(kwargs)
        shell = defaults.pop('shell', False)
        if len(args) == 1:
            if isinstance(args[0], list):
                cmd = args[0]
            elif isinstance(args[0], BaseString):
                cmd = [args[0]] if shell else args[0].split()
            else:
                raise Exception('popen() requires a string or list')
        elif len(args) > 0:
            cmd = list(args)
        if shell:
            cmd = [os.environ['SHELL'], '-c'] + [' '.join(cmd)]
        cmd = defaults.pop('mncmd') + cmd
        popen = self._popen(cmd, **defaults)
        return popen

    def pexec(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Execute a command using popen\n           returns: out, err, exitcode'
        popen = self.popen(*args, stdin=PIPE, stdout=PIPE, stderr=PIPE, **kwargs)
        (out, err) = popen.communicate()
        exitcode = popen.wait()
        return (decode(out), decode(err), exitcode)

    def newPort(self):
        if False:
            while True:
                i = 10
        'Return the next port number to allocate.'
        if len(self.ports) > 0:
            return max(self.ports.values()) + 1
        return self.portBase

    def addIntf(self, intf, port=None, moveIntfFn=moveIntf):
        if False:
            i = 10
            return i + 15
        'Add an interface.\n           intf: interface\n           port: port number (optional, typically OpenFlow port number)\n           moveIntfFn: function to move interface (optional)'
        if port is None:
            port = self.newPort()
        self.intfs[port] = intf
        self.ports[intf] = port
        self.nameToIntf[intf.name] = intf
        debug('\n')
        debug('added intf %s (%d) to node %s\n' % (intf, port, self.name))
        if self.inNamespace:
            debug('moving', intf, 'into namespace for', self.name, '\n')
            moveIntfFn(intf.name, self)

    def delIntf(self, intf):
        if False:
            return 10
        "Remove interface from Node's known interfaces\n           Note: to fully delete interface, call intf.delete() instead"
        port = self.ports.get(intf)
        if port is not None:
            del self.intfs[port]
            del self.ports[intf]
            del self.nameToIntf[intf.name]

    def defaultIntf(self):
        if False:
            while True:
                i = 10
        'Return interface for lowest port'
        ports = self.intfs.keys()
        if ports:
            return self.intfs[min(ports)]
        else:
            warn('*** defaultIntf: warning:', self.name, 'has no interfaces\n')
        return None

    def intf(self, intf=None):
        if False:
            print('Hello World!')
        'Return our interface object with given string name,\n           default intf if name is falsy (None, empty string, etc).\n           or the input intf arg.\n\n        Having this fcn return its arg for Intf objects makes it\n        easier to construct functions with flexible input args for\n        interfaces (those that accept both string names and Intf objects).\n        '
        if not intf:
            return self.defaultIntf()
        elif isinstance(intf, BaseString):
            return self.nameToIntf[intf]
        else:
            return intf

    def connectionsTo(self, node):
        if False:
            for i in range(10):
                print('nop')
        'Return [ intf1, intf2... ] for all intfs that connect self to node.'
        connections = []
        for intf in self.intfList():
            link = intf.link
            if link:
                (node1, node2) = (link.intf1.node, link.intf2.node)
                if node1 == self and node2 == node:
                    connections += [(intf, link.intf2)]
                elif node1 == node and node2 == self:
                    connections += [(intf, link.intf1)]
        return connections

    def deleteIntfs(self, checkName=True):
        if False:
            for i in range(10):
                print('nop')
        'Delete all of our interfaces.\n           checkName: only delete interfaces that contain our name'
        for intf in list(self.intfs.values()):
            if self.name in intf.name or not checkName:
                intf.delete()
                info('.')

    def setARP(self, ip, mac):
        if False:
            for i in range(10):
                print('nop')
        'Add an ARP entry.\n           ip: IP address as string\n           mac: MAC address as string'
        result = self.cmd('arp', '-s', ip, mac)
        return result

    def setHostRoute(self, ip, intf):
        if False:
            for i in range(10):
                print('nop')
        'Add route to host.\n           ip: IP address as dotted decimal\n           intf: string, interface name'
        return self.cmd('route add -host', ip, 'dev', intf)

    def setDefaultRoute(self, intf=None):
        if False:
            print('Hello World!')
        'Set the default route to go through intf.\n           intf: Intf or {dev <intfname> via <gw-ip> ...}'
        if isinstance(intf, BaseString) and ' ' in intf:
            params = intf
        else:
            params = 'dev %s' % intf
        self.cmd('ip route del default; ip route add default', params)

    def setMAC(self, mac, intf=None):
        if False:
            for i in range(10):
                print('nop')
        'Set the MAC address for an interface.\n           intf: intf or intf name\n           mac: MAC address as string'
        return self.intf(intf).setMAC(mac)

    def setIP(self, ip, prefixLen=8, intf=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Set the IP address for an interface.\n           intf: intf or intf name\n           ip: IP address as a string\n           prefixLen: prefix length, e.g. 8 for /8 or 16M addrs\n           kwargs: any additional arguments for intf.setIP'
        return self.intf(intf).setIP(ip, prefixLen, **kwargs)

    def IP(self, intf=None):
        if False:
            for i in range(10):
                print('nop')
        'Return IP address of a node or specific interface.'
        return self.intf(intf).IP()

    def MAC(self, intf=None):
        if False:
            while True:
                i = 10
        'Return MAC address of a node or specific interface.'
        return self.intf(intf).MAC()

    def intfIsUp(self, intf=None):
        if False:
            print('Hello World!')
        'Check if an interface is up.'
        return self.intf(intf).isUp()

    def setParam(self, results, method, **param):
        if False:
            while True:
                i = 10
        'Internal method: configure a *single* parameter\n           results: dict of results to update\n           method: config method name\n           param: arg=value (ignore if value=None)\n           value may also be list or dict'
        (name, value) = list(param.items())[0]
        if value is None:
            return None
        f = getattr(self, method, None)
        if not f:
            return None
        if isinstance(value, list):
            result = f(*value)
        elif isinstance(value, dict):
            result = f(**value)
        else:
            result = f(value)
        results[name] = result
        return result

    def config(self, mac=None, ip=None, defaultRoute=None, lo='up', **_params):
        if False:
            print('Hello World!')
        "Configure Node according to (optional) parameters:\n           mac: MAC address for default interface\n           ip: IP address for default interface\n           ifconfig: arbitrary interface configuration\n           Subclasses should override this method and call\n           the parent class's config(**params)"
        r = {}
        self.setParam(r, 'setMAC', mac=mac)
        self.setParam(r, 'setIP', ip=ip)
        self.setParam(r, 'setDefaultRoute', defaultRoute=defaultRoute)
        self.cmd('ifconfig lo ' + lo)
        return r

    def configDefault(self, **moreParams):
        if False:
            print('Hello World!')
        'Configure with default parameters'
        self.params.update(moreParams)
        self.config(**self.params)

    def linkTo(self, node, link=Link):
        if False:
            while True:
                i = 10
        '(Deprecated) Link to another node\n           replace with Link( node1, node2)'
        return link(self, node)

    def intfList(self):
        if False:
            return 10
        'List of our interfaces sorted by port number'
        return [self.intfs[p] for p in sorted(self.intfs.keys())]

    def intfNames(self):
        if False:
            while True:
                i = 10
        'The names of our interfaces sorted by port number'
        return [str(i) for i in self.intfList()]

    def __repr__(self):
        if False:
            return 10
        'More informative string representation'
        intfs = ','.join(['%s:%s' % (i.name, i.IP()) for i in self.intfList()])
        return '<%s %s: %s pid=%s> ' % (self.__class__.__name__, self.name, intfs, self.pid)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'Abbreviated string representation'
        return self.name
    isSetup = False

    @classmethod
    def checkSetup(cls):
        if False:
            i = 10
            return i + 15
        'Make sure our class and superclasses are set up'
        clas = cls
        while clas and (not getattr(clas, 'isSetup', True)):
            clas.setup()
            clas.isSetup = True
            clas = getattr(type(clas), '__base__', None)

    @classmethod
    def setup(cls):
        if False:
            i = 10
            return i + 15
        'Make sure our class dependencies are available'
        pathCheck('mnexec', 'ifconfig', moduleName='Mininet')

class Host(Node):
    """A host is simply a Node"""
    pass

class CPULimitedHost(Host):
    """CPU limited host"""

    def __init__(self, name, sched='cfs', **params):
        if False:
            return 10
        Host.__init__(self, name, **params)
        self.period_us = params.get('period_us', 100000)
        self.sched = sched
        self.cgroupsInited = False
        (self.cgroup, self.rtprio) = (None, None)

    def initCgroups(self):
        if False:
            i = 10
            return i + 15
        'Deferred cgroup initialization'
        if self.cgroupsInited:
            return
        if not CPULimitedHost.inited:
            CPULimitedHost.init()
        self.cgroup = 'cpu,cpuacct,cpuset:/' + self.name
        errFail('cgcreate -g ' + self.cgroup)
        errFail('cgclassify -g cpu,cpuacct:/%s %s' % (self.name, self.pid))
        if self.sched == 'rt':
            self.checkRtGroupSched()
            self.rtprio = 20

    def cgroupSet(self, param, value, resource='cpu'):
        if False:
            for i in range(10):
                print('nop')
        'Set a cgroup parameter and return its value'
        cmd = ['cgset', '-r', '%s.%s=%s' % (resource, param, value), '/' + self.name]
        errFail(cmd)
        nvalue = self.cgroupGet(param, resource)
        if nvalue != str(value):
            error('*** error: cgroupSet: %s set to %s instead of %s\n' % (param, nvalue, value))
        return nvalue

    def cgroupGet(self, param, resource='cpu'):
        if False:
            print('Hello World!')
        'Return value of cgroup parameter'
        pname = '%s.%s' % (resource, param)
        cmd = 'cgget -n -r %s /%s' % (pname, self.name)
        return quietRun(cmd)[len(pname) + 1:].strip()

    def cgroupDel(self):
        if False:
            return 10
        'Clean up our cgroup'
        (_out, _err, exitcode) = errRun('cgdelete -r ' + self.cgroup)
        return exitcode == 0 or 'no such file' in _err.lower()

    def popen(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "Return a Popen() object in node's namespace\n           args: Popen() args, single list, or string\n           kwargs: Popen() keyword args"
        mncmd = kwargs.pop('mncmd', ['mnexec', '-g', self.name, '-da', str(self.pid)])
        if self.sched == 'rt':
            if int(self.cgroupGet('rt_runtime_us', 'cpu')) <= 0:
                mncmd += ['-r', str(self.rtprio)]
            else:
                debug('*** error: not enough cpu time available for %s.' % self.name, 'Using cfs scheduler for subprocess\n')
        return Host.popen(self, *args, mncmd=mncmd, **kwargs)

    def cleanup(self):
        if False:
            i = 10
            return i + 15
        'Clean up Node, then clean up our cgroup'
        super(CPULimitedHost, self).cleanup()
        retry(retries=3, delaySecs=0.1, fn=self.cgroupDel)
    _rtGroupSched = False

    @classmethod
    def checkRtGroupSched(cls):
        if False:
            while True:
                i = 10
        'Check (Ubuntu,Debian) kernel config for CONFIG_RT_GROUP_SCHED for RT'
        if not cls._rtGroupSched:
            release = quietRun('uname -r').strip('\r\n')
            output = quietRun('grep CONFIG_RT_GROUP_SCHED /boot/config-%s' % release)
            if output == '# CONFIG_RT_GROUP_SCHED is not set\n':
                error('\n*** error: please enable RT_GROUP_SCHED in your kernel\n')
                exit(1)
            cls._rtGroupSched = True

    def chrt(self):
        if False:
            print('Hello World!')
        'Set RT scheduling priority'
        quietRun('chrt -p %s %s' % (self.rtprio, self.pid))
        result = quietRun('chrt -p %s' % self.pid)
        firstline = result.split('\n')[0]
        lastword = firstline.split(' ')[-1]
        if lastword != 'SCHED_RR':
            error('*** error: could not assign SCHED_RR to %s\n' % self.name)
        return lastword

    def rtInfo(self, f):
        if False:
            i = 10
            return i + 15
        'Internal method: return parameters for RT bandwidth'
        (pstr, qstr) = ('rt_period_us', 'rt_runtime_us')
        quota = int(self.period_us * f)
        return (pstr, qstr, self.period_us, quota)

    def cfsInfo(self, f):
        if False:
            return 10
        'Internal method: return parameters for CFS bandwidth'
        (pstr, qstr) = ('cfs_period_us', 'cfs_quota_us')
        if self.cgversion == 'cgroup2':
            (pstr, qstr) = ('max', '')
        quota = int(self.period_us * f * numCores())
        period = self.period_us
        if f > 0 and quota < 1000:
            debug('(cfsInfo: increasing default period) ')
            quota = 1000
            period = int(quota / f / numCores())
        if quota < 0:
            quota = 'max' if self.cgversion == 'cgroup2' else -1
        return (pstr, qstr, period, quota)

    def setCPUFrac(self, f, sched=None):
        if False:
            for i in range(10):
                print('nop')
        "Set overall CPU fraction for this host\n           f: CPU bandwidth limit (positive fraction, or -1 for cfs unlimited)\n           sched: 'rt' or 'cfs'\n           Note 'cfs' requires CONFIG_CFS_BANDWIDTH,\n           and 'rt' requires CONFIG_RT_GROUP_SCHED"
        if not sched:
            sched = self.sched
        if sched == 'rt':
            if not f or f < 0:
                raise Exception('Please set a positive CPU fraction for sched=rt\n')
            (pstr, qstr, period, quota) = self.rtInfo(f)
        elif sched == 'cfs':
            (pstr, qstr, period, quota) = self.cfsInfo(f)
        else:
            return
        if self.cgversion == 'cgroup':
            setPeriod = self.cgroupSet(pstr, period)
            setQuota = self.cgroupSet(qstr, quota)
        else:
            (setQuota, setPeriod) = self.cgroupSet(pstr, '%s %s' % (quota, period)).split()
        if sched == 'rt':
            sched = self.chrt()
        info('(%s %s/%dus) ' % (sched, setQuota, int(setPeriod)))

    def setCPUs(self, cores, mems=0):
        if False:
            return 10
        'Specify (real) cores that our cgroup can run on'
        if not cores:
            return
        if isinstance(cores, list):
            cores = ','.join([str(c) for c in cores])
        self.cgroupSet(resource='cpuset', param='cpus', value=cores)
        self.cgroupSet(resource='cpuset', param='mems', value=mems)
        errFail('cgclassify -g cpuset:/%s %s' % (self.name, self.pid))

    def config(self, cpu=-1, cores=None, **params):
        if False:
            for i in range(10):
                print('nop')
        'cpu: desired overall system CPU fraction\n           cores: (real) core(s) this host can run on\n           params: parameters for Node.config()'
        r = Node.config(self, **params)
        self.initCgroups()
        self.setParam(r, 'setCPUFrac', cpu=cpu)
        self.setParam(r, 'setCPUs', cores=cores)
        return r
    inited = False
    cgversion = 'cgroup2'

    @classmethod
    def init(cls):
        if False:
            return 10
        'Initialization for CPULimitedHost class'
        cls.cgversion = mountCgroups()
        cls.inited = True

    def unlimit(self):
        if False:
            print('Hello World!')
        'Unlimit cpu for cfs'
        if self.sched == 'cfs' and self.params.get('cpu', -1) != -1:
            self.setCPUFrac(-1, sched=self.sched)

class Switch(Node):
    """A Switch is a Node that is running (or has execed?)
       an OpenFlow switch."""
    portBase = 1
    dpidLen = 16

    def __init__(self, name, dpid=None, opts='', listenPort=None, **params):
        if False:
            i = 10
            return i + 15
        'dpid: dpid hex string (or None to derive from name, e.g. s1 -> 1)\n           opts: additional switch options\n           listenPort: port to listen on for dpctl connections'
        Node.__init__(self, name, **params)
        self.dpid = self.defaultDpid(dpid)
        self.opts = opts
        self.listenPort = listenPort
        if not self.inNamespace:
            self.controlIntf = Intf('lo', self, port=0)

    def defaultDpid(self, dpid=None):
        if False:
            i = 10
            return i + 15
        'Return correctly formatted dpid from dpid or switch name (s1 -> 1)'
        if dpid:
            dpid = dpid.replace(':', '')
            assert len(dpid) <= self.dpidLen and int(dpid, 16) >= 0
        else:
            nums = re.findall('\\d+', self.name)
            if nums:
                dpid = hex(int(nums[0]))[2:]
            else:
                self.terminate()
                raise Exception('Unable to derive default datapath ID - please either specify a dpid or use a canonical switch name such as s23.')
        return '0' * (self.dpidLen - len(dpid)) + dpid

    def defaultIntf(self):
        if False:
            for i in range(10):
                print('nop')
        'Return control interface'
        if self.controlIntf:
            return self.controlIntf
        else:
            return Node.defaultIntf(self)

    def sendCmd(self, *cmd, **kwargs):
        if False:
            while True:
                i = 10
        'Send command to Node.\n           cmd: string'
        kwargs.setdefault('printPid', False)
        if not self.execed:
            return Node.sendCmd(self, *cmd, **kwargs)
        else:
            error('*** Error: %s has execed and cannot accept commands' % self.name)
        return None

    def connected(self):
        if False:
            print('Hello World!')
        'Is the switch connected to a controller? (override this method)'
        debug('Assuming', repr(self), 'is connected to a controller\n')
        return True

    def stop(self, deleteIntfs=True):
        if False:
            while True:
                i = 10
        'Stop switch\n           deleteIntfs: delete interfaces? (True)'
        if deleteIntfs:
            self.deleteIntfs()

    def __repr__(self):
        if False:
            return 10
        'More informative string representation'
        intfs = ','.join(['%s:%s' % (i.name, i.IP()) for i in self.intfList()])
        return '<%s %s: %s pid=%s> ' % (self.__class__.__name__, self.name, intfs, self.pid)

class UserSwitch(Switch):
    """User-space switch."""
    dpidLen = 12

    def __init__(self, name, dpopts='--no-slicing', **kwargs):
        if False:
            while True:
                i = 10
        'Init.\n           name: name for the switch\n           dpopts: additional arguments to ofdatapath (--no-slicing)'
        Switch.__init__(self, name, **kwargs)
        pathCheck('ofdatapath', 'ofprotocol', moduleName='the OpenFlow reference user switch' + '(openflow.org)')
        if self.listenPort:
            self.opts += ' --listen=ptcp:%i ' % self.listenPort
        else:
            self.opts += ' --listen=punix:/tmp/%s.listen' % self.name
        self.dpopts = dpopts

    @classmethod
    def setup(cls):
        if False:
            for i in range(10):
                print('nop')
        'Ensure any dependencies are loaded; if not, try to load them.'
        if not os.path.exists('/dev/net/tun'):
            moduleDeps(add=TUN)

    def dpctl(self, *args):
        if False:
            while True:
                i = 10
        'Run dpctl command'
        listenAddr = None
        if not self.listenPort:
            listenAddr = 'unix:/tmp/%s.listen' % self.name
        else:
            listenAddr = 'tcp:127.0.0.1:%i' % self.listenPort
        return self.cmd('dpctl ' + ' '.join(args) + ' ' + listenAddr)

    def connected(self):
        if False:
            return 10
        'Is the switch connected to a controller?'
        status = self.dpctl('status')
        return 'remote.is-connected=true' in status and 'local.is-connected=true' in status

    @staticmethod
    def TCReapply(intf):
        if False:
            while True:
                i = 10
        "Unfortunately user switch and Mininet are fighting\n           over tc queuing disciplines. To resolve the conflict,\n           we re-create the user switch's configuration, but as a\n           leaf of the TCIntf-created configuration."
        if isinstance(intf, TCIntf):
            ifspeed = 10000000000
            minspeed = ifspeed * 0.001
            res = intf.config(**intf.params)
            if res is None:
                return
            parent = res['parent']
            intf.tc('%s qdisc add dev %s ' + parent + ' handle 1: htb default 0xfffe')
            intf.tc('%s class add dev %s classid 1:0xffff parent 1: htb rate ' + str(ifspeed))
            intf.tc('%s class add dev %s classid 1:0xfffe parent 1:0xffff ' + 'htb rate ' + str(minspeed) + ' ceil ' + str(ifspeed))

    def start(self, controllers):
        if False:
            i = 10
            return i + 15
        'Start OpenFlow reference user datapath.\n           Log to /tmp/sN-{ofd,ofp}.log.\n           controllers: list of controller objects'
        clist = ','.join(['tcp:%s:%d' % (c.IP(), c.port) for c in controllers])
        ofdlog = '/tmp/' + self.name + '-ofd.log'
        ofplog = '/tmp/' + self.name + '-ofp.log'
        intfs = [str(i) for i in self.intfList() if not i.IP()]
        self.cmd('ofdatapath -i ' + ','.join(intfs) + ' punix:/tmp/' + self.name + ' -d %s ' % self.dpid + self.dpopts + ' 1> ' + ofdlog + ' 2> ' + ofdlog + ' &')
        self.cmd('ofprotocol unix:/tmp/' + self.name + ' ' + clist + ' --fail=closed ' + self.opts + ' 1> ' + ofplog + ' 2>' + ofplog + ' &')
        if 'no-slicing' not in self.dpopts:
            sleep(1)
            for intf in self.intfList():
                if not intf.IP():
                    self.TCReapply(intf)

    def stop(self, deleteIntfs=True):
        if False:
            i = 10
            return i + 15
        'Stop OpenFlow reference user datapath.\n           deleteIntfs: delete interfaces? (True)'
        self.cmd('kill %ofdatapath')
        self.cmd('kill %ofprotocol')
        super(UserSwitch, self).stop(deleteIntfs)

class OVSSwitch(Switch):
    """Open vSwitch switch. Depends on ovs-vsctl."""

    def __init__(self, name, failMode='secure', datapath='kernel', inband=False, protocols=None, reconnectms=1000, stp=False, batch=False, **params):
        if False:
            for i in range(10):
                print('nop')
        'name: name for switch\n           failMode: controller loss behavior (secure|standalone)\n           datapath: userspace or kernel mode (kernel|user)\n           inband: use in-band control (False)\n           protocols: use specific OpenFlow version(s) (e.g. OpenFlow13)\n                      Unspecified (or old OVS version) uses OVS default\n           reconnectms: max reconnect timeout in ms (0/None for default)\n           stp: enable STP (False, requires failMode=standalone)\n           batch: enable batch startup (False)'
        Switch.__init__(self, name, **params)
        self.failMode = failMode
        self.datapath = datapath
        self.inband = inband
        self.protocols = protocols
        self.reconnectms = reconnectms
        self.stp = stp
        self._uuids = []
        self.batch = batch
        self.commands = []

    @classmethod
    def setup(cls):
        if False:
            return 10
        'Make sure Open vSwitch is installed and working'
        pathCheck('ovs-vsctl', moduleName='Open vSwitch (openvswitch.org)')
        (out, err, exitcode) = errRun('ovs-vsctl -t 1 show')
        if exitcode:
            error(out + err + 'ovs-vsctl exited with code %d\n' % exitcode + '*** Error connecting to ovs-db with ovs-vsctl\nMake sure that Open vSwitch is installed, that ovsdb-server is running, and that\n"ovs-vsctl show" works correctly.\nYou may wish to try "service openvswitch-switch start".\n')
            exit(1)
        version = quietRun('ovs-vsctl --version')
        cls.OVSVersion = findall('\\d+\\.\\d+', version)[0]

    @classmethod
    def isOldOVS(cls):
        if False:
            i = 10
            return i + 15
        'Is OVS ersion < 1.10?'
        return StrictVersion(cls.OVSVersion) < StrictVersion('1.10')

    def dpctl(self, *args):
        if False:
            i = 10
            return i + 15
        'Run ovs-ofctl command'
        return self.cmd('ovs-ofctl', args[0], self, *args[1:])

    def vsctl(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Run ovs-vsctl command (or queue for later execution)'
        if self.batch:
            cmd = ' '.join((str(arg).strip() for arg in args))
            self.commands.append(cmd)
            return None
        else:
            return self.cmd('ovs-vsctl', *args, **kwargs)

    @staticmethod
    def TCReapply(intf):
        if False:
            print('Hello World!')
        "Unfortunately OVS and Mininet are fighting\n           over tc queuing disciplines. As a quick hack/\n           workaround, we clear OVS's and reapply our own."
        if isinstance(intf, TCIntf):
            intf.config(**intf.params)

    def attach(self, intf):
        if False:
            for i in range(10):
                print('nop')
        'Connect a data port'
        self.vsctl('add-port', self, intf)
        self.cmd('ifconfig', intf, 'up')
        self.TCReapply(intf)

    def detach(self, intf):
        if False:
            for i in range(10):
                print('nop')
        'Disconnect a data port'
        self.vsctl('del-port', self, intf)

    def controllerUUIDs(self, update=False):
        if False:
            while True:
                i = 10
        'Return ovsdb UUIDs for our controllers\n           update: update cached value'
        if not self._uuids or update:
            controllers = self.cmd('ovs-vsctl -- get Bridge', self, 'Controller').strip()
            if controllers.startswith('[') and controllers.endswith(']'):
                controllers = controllers[1:-1]
                if controllers:
                    self._uuids = [c.strip() for c in controllers.split(',')]
        return self._uuids

    def connected(self):
        if False:
            i = 10
            return i + 15
        'Are we connected to at least one of our controllers?'
        for uuid in self.controllerUUIDs():
            if 'true' in self.vsctl('-- get Controller', uuid, 'is_connected'):
                return True
        return self.failMode == 'standalone'

    def intfOpts(self, intf):
        if False:
            return 10
        'Return OVS interface options for intf'
        opts = ''
        if not self.isOldOVS():
            opts += ' ofport_request=%s' % self.ports[intf]
            if isinstance(intf, OVSIntf):
                (intf1, intf2) = (intf.link.intf1, intf.link.intf2)
                peer = intf1 if intf1 != intf else intf2
                opts += ' type=patch options:peer=%s' % peer
        return '' if not opts else ' -- set Interface %s' % intf + opts

    def bridgeOpts(self):
        if False:
            return 10
        'Return OVS bridge options'
        opts = ' other_config:datapath-id=%s' % self.dpid + ' fail_mode=%s' % self.failMode
        if not self.inband:
            opts += ' other-config:disable-in-band=true'
        if self.datapath == 'user':
            opts += ' datapath_type=netdev'
        if self.protocols and (not self.isOldOVS()):
            opts += ' protocols=%s' % self.protocols
        if self.stp and self.failMode == 'standalone':
            opts += ' stp_enable=true'
        opts += ' other-config:dp-desc=%s' % self.name
        return opts

    def start(self, controllers):
        if False:
            i = 10
            return i + 15
        'Start up a new OVS OpenFlow switch using ovs-vsctl'
        if self.inNamespace:
            raise Exception('OVS kernel switch does not work in a namespace')
        int(self.dpid, 16)
        intfs = ''.join((' -- add-port %s %s' % (self, intf) + self.intfOpts(intf) for intf in self.intfList() if self.ports[intf] and (not intf.IP())))
        clist = [(self.name + c.name, '%s:%s:%d' % (c.protocol, c.IP(), c.port)) for c in controllers]
        if self.listenPort:
            clist.append((self.name + '-listen', 'ptcp:%s' % self.listenPort))
        ccmd = '-- --id=@%s create Controller target=\\"%s\\"'
        if self.reconnectms:
            ccmd += ' max_backoff=%d' % self.reconnectms
        cargs = ' '.join((ccmd % (name, target) for (name, target) in clist))
        cids = ','.join(('@%s' % name for (name, _target) in clist))
        if not self.isOldOVS():
            cargs += ' -- --if-exists del-br %s' % self
        self.vsctl(cargs + ' -- add-br %s' % self + ' -- set bridge %s controller=[%s]' % (self, cids) + self.bridgeOpts() + intfs)
        if not self.batch:
            for intf in self.intfList():
                self.TCReapply(intf)
    argmax = 128000

    @classmethod
    def batchStartup(cls, switches, run=errRun):
        if False:
            for i in range(10):
                print('nop')
        'Batch startup for OVS\n           switches: switches to start up\n           run: function to run commands (errRun)'
        info('...')
        cmds = 'ovs-vsctl'
        for switch in switches:
            if switch.isOldOVS():
                run('ovs-vsctl del-br %s' % switch)
            for cmd in switch.commands:
                cmd = cmd.strip()
                if len(cmds) + len(cmd) >= cls.argmax:
                    run(cmds, shell=True)
                    cmds = 'ovs-vsctl'
                cmds += ' ' + cmd
                switch.cmds = []
                switch.batch = False
        if cmds:
            run(cmds, shell=True)
        for switch in switches:
            for intf in switch.intfs.values():
                if isinstance(intf, TCIntf):
                    intf.config(**intf.params)
        return switches

    def stop(self, deleteIntfs=True):
        if False:
            while True:
                i = 10
        'Terminate OVS switch.\n           deleteIntfs: delete interfaces? (True)'
        self.cmd('ovs-vsctl del-br', self)
        if self.datapath == 'user':
            self.cmd('ip link del', self)
        super(OVSSwitch, self).stop(deleteIntfs)

    @classmethod
    def batchShutdown(cls, switches, run=errRun):
        if False:
            i = 10
            return i + 15
        'Shut down a list of OVS switches'
        delcmd = 'del-br %s'
        if switches and (not switches[0].isOldOVS()):
            delcmd = '--if-exists ' + delcmd
        run('ovs-vsctl ' + ' -- '.join((delcmd % s for s in switches)))
        pids = ' '.join((str(switch.pid) for switch in switches))
        run('kill -HUP ' + pids)
        for switch in switches:
            switch.terminate()
        return switches
OVSKernelSwitch = OVSSwitch

class OVSBridge(OVSSwitch):
    """OVSBridge is an OVSSwitch in standalone/bridge mode"""

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'stp: enable Spanning Tree Protocol (False)\n           see OVSSwitch for other options'
        kwargs.update(failMode='standalone')
        OVSSwitch.__init__(self, *args, **kwargs)

    def start(self, controllers):
        if False:
            while True:
                i = 10
        'Start bridge, ignoring controllers argument'
        OVSSwitch.start(self, controllers=[])

    def connected(self):
        if False:
            print('Hello World!')
        'Are we forwarding yet?'
        if self.stp:
            status = self.dpctl('show')
            return 'STP_FORWARD' in status and 'STP_LEARN' not in status
        else:
            return True

class IVSSwitch(Switch):
    """Indigo Virtual Switch"""

    def __init__(self, name, verbose=False, **kwargs):
        if False:
            i = 10
            return i + 15
        Switch.__init__(self, name, **kwargs)
        self.verbose = verbose

    @classmethod
    def setup(cls):
        if False:
            for i in range(10):
                print('nop')
        'Make sure IVS is installed'
        pathCheck('ivs-ctl', 'ivs', moduleName='Indigo Virtual Switch (projectfloodlight.org)')
        (out, err, exitcode) = errRun('ivs-ctl show')
        if exitcode:
            error(out + err + 'ivs-ctl exited with code %d\n' % exitcode + '*** The openvswitch kernel module might not be loaded. Try modprobe openvswitch.\n')
            exit(1)

    @classmethod
    def batchShutdown(cls, switches):
        if False:
            print('Hello World!')
        'Kill each IVS switch, to be waited on later in stop()'
        for switch in switches:
            switch.cmd('kill %ivs')
        return switches

    def start(self, controllers):
        if False:
            i = 10
            return i + 15
        'Start up a new IVS switch'
        args = ['ivs']
        args.extend(['--name', self.name])
        args.extend(['--dpid', self.dpid])
        if self.verbose:
            args.extend(['--verbose'])
        for intf in self.intfs.values():
            if not intf.IP():
                args.extend(['-i', intf.name])
        for c in controllers:
            args.extend(['-c', '%s:%d' % (c.IP(), c.port)])
        if self.listenPort:
            args.extend(['--listen', '127.0.0.1:%i' % self.listenPort])
        args.append(self.opts)
        logfile = '/tmp/ivs.%s.log' % self.name
        self.cmd(' '.join(args) + ' >' + logfile + ' 2>&1 </dev/null &')

    def stop(self, deleteIntfs=True):
        if False:
            return 10
        'Terminate IVS switch.\n           deleteIntfs: delete interfaces? (True)'
        self.cmd('kill %ivs')
        self.cmd('wait')
        super(IVSSwitch, self).stop(deleteIntfs)

    def attach(self, intf):
        if False:
            while True:
                i = 10
        'Connect a data port'
        self.cmd('ivs-ctl', 'add-port', '--datapath', self.name, intf)

    def detach(self, intf):
        if False:
            i = 10
            return i + 15
        'Disconnect a data port'
        self.cmd('ivs-ctl', 'del-port', '--datapath', self.name, intf)

    def dpctl(self, *args):
        if False:
            print('Hello World!')
        'Run dpctl command'
        if not self.listenPort:
            return "can't run dpctl without passive listening port"
        return self.cmd('ovs-ofctl ' + ' '.join(args) + ' tcp:127.0.0.1:%i' % self.listenPort)

class Controller(Node):
    """A Controller is a Node that is running (or has execed?) an
       OpenFlow controller."""

    def __init__(self, name, inNamespace=False, command='controller', cargs='ptcp:%d', cdir=None, ip='127.0.0.1', port=6653, protocol='tcp', verbose=False, **params):
        if False:
            i = 10
            return i + 15
        self.command = command
        self.cargs = cargs
        if verbose:
            cargs = '-v ' + cargs
        self.cdir = cdir
        if ':' in ip:
            (ip, port) = ip.split(':')
            port = int(port)
        self.ip = ip
        self.port = port
        self.protocol = protocol
        Node.__init__(self, name, inNamespace=inNamespace, ip=ip, **params)
        self.checkListening()

    def checkListening(self):
        if False:
            return 10
        'Make sure no controllers are running on our port'
        (out, _err, returnCode) = errRun('which telnet')
        if 'telnet' not in out or returnCode != 0:
            raise Exception('Error running telnet to check for listening controllers; please check that it is installed.')
        listening = self.cmd('echo A | telnet -e A %s %d' % (self.ip, self.port))
        if 'Connected' in listening:
            servers = self.cmd('netstat -natp').split('\n')
            pstr = ':%d ' % self.port
            clist = servers[0:1] + [s for s in servers if pstr in s]
            raise Exception('Please shut down the controller which is running on port %d:\n' % self.port + '\n'.join(clist))

    def start(self):
        if False:
            i = 10
            return i + 15
        'Start <controller> <args> on controller.\n           Log to /tmp/cN.log'
        pathCheck(self.command)
        cout = '/tmp/' + self.name + '.log'
        if self.cdir is not None:
            self.cmd('cd ' + self.cdir)
        self.cmd(self.command + ' ' + self.cargs % self.port + ' 1>' + cout + ' 2>' + cout + ' &')
        self.execed = False

    def stop(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Stop controller.'
        self.cmd('kill %' + self.command)
        self.cmd('wait %' + self.command)
        super(Controller, self).stop(*args, **kwargs)

    def IP(self, intf=None):
        if False:
            return 10
        'Return IP address of the Controller'
        if self.intfs:
            ip = Node.IP(self, intf)
        else:
            ip = self.ip
        return ip

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'More informative string representation'
        return '<%s %s: %s:%s pid=%s> ' % (self.__class__.__name__, self.name, self.IP(), self.port, self.pid)

    @classmethod
    def isAvailable(cls):
        if False:
            i = 10
            return i + 15
        'Is controller available?'
        return which('controller')

class OVSController(Controller):
    """Open vSwitch controller"""

    def __init__(self, name, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        kwargs.setdefault('command', self.isAvailable() or 'ovs-controller')
        Controller.__init__(self, name, **kwargs)

    @classmethod
    def isAvailable(cls):
        if False:
            print('Hello World!')
        return which('ovs-controller') or which('test-controller') or which('ovs-testcontroller')

class NOX(Controller):
    """Controller to run a NOX application."""

    def __init__(self, name, *noxArgs, **kwargs):
        if False:
            i = 10
            return i + 15
        'Init.\n           name: name to give controller\n           noxArgs: arguments (strings) to pass to NOX'
        if not noxArgs:
            warn('warning: no NOX modules specified; running packetdump only\n')
            noxArgs = ['packetdump']
        elif not isinstance(noxArgs, (list, tuple)):
            noxArgs = [noxArgs]
        if 'NOX_CORE_DIR' not in os.environ:
            exit('exiting; please set missing NOX_CORE_DIR env var')
        noxCoreDir = os.environ['NOX_CORE_DIR']
        Controller.__init__(self, name, command=noxCoreDir + '/nox_core', cargs='--libdir=/usr/local/lib -v -i ptcp:%s ' + ' '.join(noxArgs), cdir=noxCoreDir, **kwargs)

class Ryu(Controller):
    """Ryu OpenFlow Controller"""

    def __init__(self, name, ryuArgs='ryu.app.simple_switch', command='ryu run', **kwargs):
        if False:
            i = 10
            return i + 15
        "Init.\n           name: name to give controller.\n           ryuArgs: modules to pass to Ryu (ryu.app.simple_switch)\n           command: command to run Ryu ('ryu run')"
        if isinstance(ryuArgs, (list, tuple)):
            ryuArgs = ' '.join(ryuArgs)
        cargs = kwargs.pop('cargs', ryuArgs + ' --ofp-tcp-listen-port %s')
        Controller.__init__(self, name, command=command, cargs=cargs, **kwargs)

class RemoteController(Controller):
    """Controller running outside of Mininet's control."""

    def __init__(self, name, ip='127.0.0.1', port=None, **kwargs):
        if False:
            print('Hello World!')
        'Init.\n           name: name to give controller\n           ip: the IP address where the remote controller is\n           listening\n           port: the port where the remote controller is listening'
        Controller.__init__(self, name, ip=ip, port=port, **kwargs)

    def start(self):
        if False:
            while True:
                i = 10
        'Overridden to do nothing.'
        return

    def stop(self):
        if False:
            i = 10
            return i + 15
        'Overridden to do nothing.'
        return

    def checkListening(self):
        if False:
            while True:
                i = 10
        'Warn if remote controller is not accessible'
        if self.port is not None:
            self.isListening(self.ip, self.port)
        else:
            for port in (6653, 6633):
                if self.isListening(self.ip, port):
                    self.port = port
                    info('Connecting to remote controller at %s:%d\n' % (self.ip, self.port))
                    break
        if self.port is None:
            self.port = 6653
            warn('Setting remote controller to %s:%d\n' % (self.ip, self.port))

    def isListening(self, ip, port):
        if False:
            i = 10
            return i + 15
        'Check if a remote controller is listening at a specific ip and port'
        listening = self.cmd('echo A | telnet -e A %s %d' % (ip, port))
        if 'Connected' not in listening:
            warn('Unable to contact the remote controller at %s:%d\n' % (ip, port))
            return False
        else:
            return True
DefaultControllers = (Controller, OVSController)

def findController(controllers=DefaultControllers):
    if False:
        return 10
    'Return first available controller from list, if any'
    for controller in controllers:
        if controller.isAvailable():
            return controller
    return None

def DefaultController(name, controllers=DefaultControllers, **kwargs):
    if False:
        i = 10
        return i + 15
    'Find a controller that is available and instantiate it'
    controller = findController(controllers)
    if not controller:
        raise Exception('Could not find a default OpenFlow controller')
    return controller(name, **kwargs)

def NullController(*_args, **_kwargs):
    if False:
        print('Hello World!')
    'Nonexistent controller - simply returns None'
    return None