"""
cluster.py: prototyping/experimentation for distributed Mininet,
            aka Mininet: Cluster Edition

Author: Bob Lantz

Core classes:

    RemoteNode: a Node() running on a remote server
    RemoteOVSSwitch(): an OVSSwitch() running on a remote server
    RemoteLink: a Link() on a remote server
    Tunnel: a Link() between a local Node() and a RemoteNode()

These are largely interoperable with local objects.

- One Mininet to rule them all

It is important that the same topologies, APIs, and CLI can be used
with minimal or no modification in both local and distributed environments.

- Multiple placement models

Placement should be as easy as possible. We should provide basic placement
support and also allow for explicit placement.

Questions:

What is the basic communication mechanism?

To start with? Probably a single multiplexed ssh connection between each
pair of mininet servers that needs to communicate.

How are tunnels created?

We have several options including ssh, GRE, OF capsulator, socat, VDE, l2tp,
etc..  It's not clear what the best one is.  For now, we use ssh tunnels since
they are encrypted and semi-automatically shared.  We will probably want to
support GRE as well because it's very easy to set up with OVS.

How are tunnels destroyed?

They are destroyed when the links are deleted in Mininet.stop()

How does RemoteNode.popen() work?

It opens a shared ssh connection to the remote server and attaches to
the namespace using mnexec -a -g.

Is there any value to using Paramiko vs. raw ssh?

Maybe, but it doesn't seem to support L2 tunneling.

Should we preflight the entire network, including all server-to-server
connections?

Yes! We don't yet do this with remote server-to-server connections yet.

Should we multiplex the link ssh connections?

Yes, this is done automatically with ControlMaster=auto.

Note on ssh and DNS:
Please add UseDNS: no to your /etc/ssh/sshd_config!!!

Things to do:

- asynchronous/pipelined/parallel startup
- ssh debugging/profiling
- make connections into real objects
- support for other tunneling schemes
- tests and benchmarks
- hifi support (e.g. delay compensation)
"""
from signal import signal, SIGINT, SIG_IGN
from subprocess import Popen, PIPE, STDOUT
import os
from random import randrange
import sys
import re
from itertools import groupby
from operator import attrgetter
from mininet.node import Node, Host, OVSSwitch, Controller
from mininet.link import Link, Intf
from mininet.net import Mininet
from mininet.topo import LinearTopo
from mininet.topolib import TreeTopo
from mininet.util import quietRun, errRun, decode, StrictVersion
from mininet.examples.clustercli import CLI
from mininet.log import setLogLevel, debug, info, error
from mininet.clean import addCleanupCallback

def findUser():
    if False:
        i = 10
        return i + 15
    'Try to return logged-in (usually non-root) user'
    return os.environ.get('SUDO_USER', False) or (quietRun('who am i').split() or [False])[0] or quietRun('whoami').strip()

class ClusterCleanup(object):
    """Cleanup callback"""
    inited = False
    serveruser = {}

    @classmethod
    def add(cls, server, user=''):
        if False:
            for i in range(10):
                print('nop')
        'Add an entry to server: user dict'
        if not cls.inited:
            addCleanupCallback(cls.cleanup)
        if not user:
            user = findUser()
        cls.serveruser[server] = user

    @classmethod
    def cleanup(cls):
        if False:
            while True:
                i = 10
        'Clean up'
        info('*** Cleaning up cluster\n')
        for (server, user) in cls.serveruser.items():
            if server == 'localhost':
                continue
            else:
                cmd = ['su', user, '-c', 'ssh %s@%s sudo mn -c' % (user, server)]
                info(cmd, '\n')
                info(quietRun(cmd))

class RemoteMixin(object):
    """A mix-in class to turn local nodes into remote nodes"""
    sshbase = ['ssh', '-q', '-o', 'BatchMode=yes', '-o', 'ForwardAgent=yes', '-tt']

    def __init__(self, name, server='localhost', user=None, serverIP=None, controlPath=False, splitInit=False, **kwargs):
        if False:
            i = 10
            return i + 15
        'Instantiate a remote node\n           name: name of remote node\n           server: remote server (optional)\n           user: user on remote server (optional)\n           controlPath: specify shared ssh control path (optional)\n           splitInit: split initialization?\n           **kwargs: see Node()'
        self.server = server if server else 'localhost'
        self.serverIP = serverIP if serverIP else self.findServerIP(self.server)
        self.user = user if user else findUser()
        ClusterCleanup.add(server=server, user=user)
        if controlPath is True:
            controlPath = '/tmp/mn-%r@%h:%p'
        self.controlPath = controlPath
        self.splitInit = splitInit
        if self.user and self.server != 'localhost':
            self.dest = '%s@%s' % (self.user, self.serverIP)
            self.sshcmd = ['sudo', '-E', '-u', self.user] + self.sshbase
            if self.controlPath:
                self.sshcmd += ['-o', 'ControlPath=' + self.controlPath, '-o', 'ControlMaster=auto', '-o', 'ControlPersist=' + '1']
            self.sshcmd += [self.dest]
            self.isRemote = True
        else:
            self.dest = None
            self.sshcmd = []
            self.isRemote = False
        (self.shell, self.pid) = (None, None)
        super(RemoteMixin, self).__init__(name, **kwargs)
    _ipMatchRegex = re.compile('\\d+\\.\\d+\\.\\d+\\.\\d+')

    @classmethod
    def findServerIP(cls, server):
        if False:
            i = 10
            return i + 15
        "Return our server's IP address"
        ipmatch = cls._ipMatchRegex.findall(server)
        if ipmatch:
            return ipmatch[0]
        output = quietRun('getent ahostsv4 %s' % server)
        ips = cls._ipMatchRegex.findall(output)
        ip = ips[0] if ips else None
        return ip

    def startShell(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Start a shell process for running commands'
        if self.isRemote:
            kwargs.update(mnopts='-c')
        super(RemoteMixin, self).startShell(*args, **kwargs)
        self.sendCmd('echo $$')
        if not self.splitInit:
            self.finishInit()

    def finishInit(self):
        if False:
            print('Hello World!')
        'Wait for split initialization to complete'
        self.pid = int(self.waitOutput())

    def rpopen(self, *cmd, **opts):
        if False:
            print('Hello World!')
        'Return a Popen object on underlying server in root namespace'
        params = {'stdin': PIPE, 'stdout': PIPE, 'stderr': STDOUT, 'sudo': True}
        params.update(opts)
        return self._popen(*cmd, **params)

    def rcmd(self, *cmd, **opts):
        if False:
            print('Hello World!')
        'rcmd: run a command on underlying server\n           in root namespace\n           args: string or list of strings\n           returns: stdout and stderr'
        popen = self.rpopen(*cmd, **opts)
        result = ''
        while True:
            poll = popen.poll()
            result += decode(popen.stdout.read())
            if poll is not None:
                break
        return result

    @staticmethod
    def _ignoreSignal():
        if False:
            print('Hello World!')
        'Detach from process group to ignore all signals'
        os.setpgrp()

    def _popen(self, cmd, sudo=True, tt=True, **params):
        if False:
            return 10
        'Spawn a process on a remote node\n            cmd: remote command to run (list)\n            **params: parameters to Popen()\n            returns: Popen() object'
        if isinstance(cmd, str):
            cmd = cmd.split()
        if self.isRemote:
            if sudo:
                cmd = ['sudo', '-E'] + cmd
            if tt:
                cmd = self.sshcmd + cmd
            else:
                sshcmd = list(self.sshcmd)
                sshcmd.remove('-tt')
                cmd = sshcmd + cmd
        elif self.user and (not sudo):
            cmd = ['sudo', '-E', '-u', self.user] + cmd
        params.update(preexec_fn=self._ignoreSignal)
        debug('_popen', cmd, '\n')
        popen = super(RemoteMixin, self)._popen(cmd, **params)
        return popen

    def popen(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Override: disable -tt'
        return super(RemoteMixin, self).popen(*args, tt=False, **kwargs)

    def addIntf(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Override: use RemoteLink.moveIntf'
        return super(RemoteMixin, self).addIntf(*args, **kwargs)

class RemoteNode(RemoteMixin, Node):
    """A node on a remote server"""
    pass

class RemoteHost(RemoteNode):
    """A RemoteHost is simply a RemoteNode"""
    pass

class RemoteOVSSwitch(RemoteMixin, OVSSwitch):
    """Remote instance of Open vSwitch"""
    OVSVersions = {}

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs.update(batch=True)
        super(RemoteOVSSwitch, self).__init__(*args, **kwargs)

    def isOldOVS(self):
        if False:
            while True:
                i = 10
        'Is remote switch using an old OVS version?'
        cls = type(self)
        if self.server not in cls.OVSVersions:
            vers = self.cmd('ovs-vsctl --version')
            cls.OVSVersions[self.server] = re.findall('\\d+\\.\\d+', vers)[0]
        return StrictVersion(cls.OVSVersions[self.server]) < StrictVersion('1.10')

    @classmethod
    def batchStartup(cls, switches, **_kwargs):
        if False:
            while True:
                i = 10
        'Start up switches in per-server batches'
        key = attrgetter('server')
        for (server, switchGroup) in groupby(sorted(switches, key=key), key):
            info('(%s)' % server)
            group = tuple(switchGroup)
            switch = group[0]
            OVSSwitch.batchStartup(group, run=switch.cmd)
        return switches

    @classmethod
    def batchShutdown(cls, switches, **_kwargs):
        if False:
            print('Hello World!')
        'Stop switches in per-server batches'
        key = attrgetter('server')
        for (server, switchGroup) in groupby(sorted(switches, key=key), key):
            info('(%s)' % server)
            group = tuple(switchGroup)
            switch = group[0]
            OVSSwitch.batchShutdown(group, run=switch.rcmd)
        return switches

class RemoteLink(Link):
    """A RemoteLink is a link between nodes which may be on different servers"""

    def __init__(self, node1, node2, **kwargs):
        if False:
            print('Hello World!')
        'Initialize a RemoteLink\n           see Link() for parameters'
        self.node1 = node1
        self.node2 = node2
        self.tunnel = None
        kwargs.setdefault('params1', {})
        kwargs.setdefault('params2', {})
        self.cmd = None
        Link.__init__(self, node1, node2, **kwargs)

    def stop(self):
        if False:
            return 10
        'Stop this link'
        if self.tunnel:
            self.tunnel.terminate()
            self.intf1.delete()
            self.intf2.delete()
        else:
            Link.stop(self)
        self.tunnel = None

    def makeIntfPair(self, intfname1, intfname2, addr1=None, addr2=None, node1=None, node2=None, deleteIntfs=True):
        if False:
            return 10
        'Create pair of interfaces\n            intfname1: name of interface 1\n            intfname2: name of interface 2\n            (override this method [and possibly delete()]\n            to change link type)'
        node1 = self.node1 if node1 is None else node1
        node2 = self.node2 if node2 is None else node2
        server1 = getattr(node1, 'server', 'localhost')
        server2 = getattr(node2, 'server', 'localhost')
        if server1 == server2:
            return Link.makeIntfPair(intfname1, intfname2, addr1, addr2, node1, node2, deleteIntfs=deleteIntfs)
        self.tunnel = self.makeTunnel(node1, node2, intfname1, intfname2, addr1, addr2)
        return self.tunnel

    @staticmethod
    def moveIntf(intf, node):
        if False:
            return 10
        'Move remote interface from root ns to node\n            intf: string, interface\n            dstNode: destination Node\n            srcNode: source Node or None (default) for root ns'
        intf = str(intf)
        cmd = 'ip link set %s netns %s' % (intf, node.pid)
        result = node.rcmd(cmd)
        if result:
            raise Exception('error executing command %s' % cmd)
        return True

    def makeTunnel(self, node1, node2, intfname1, intfname2, addr1=None, addr2=None):
        if False:
            while True:
                i = 10
        'Make a tunnel across switches on different servers'
        assert node1.server != node2.server
        if node2.server == 'localhost':
            return self.makeTunnel(node1=node2, node2=node1, intfname1=intfname2, intfname2=intfname1, addr1=addr2, addr2=addr1)
        debug('\n*** Make SSH tunnel ' + node1.server + ':' + intfname1 + ' == ' + node2.server + ':' + intfname2)
        for node in (node1, node2):
            cmd = 'ip tuntap add dev tap9 mode tap user ' + node.user
            result = node.rcmd(cmd)
            if result:
                raise Exception('error creating tap9 on %s: %s' % (node, result))
        dest = '%s@%s' % (node2.user, node2.serverIP)
        cmd = ['ssh', '-n', '-o', 'Tunnel=Ethernet', '-w', '9:9', dest, 'echo @']
        self.cmd = cmd
        tunnel = node1.rpopen(cmd, sudo=False)
        debug('Waiting for tunnel to come up...\n')
        ch = decode(tunnel.stdout.read(1))
        if ch != '@':
            ch += decode(tunnel.stdout.read())
            cmd = ' '.join(cmd)
            raise Exception('makeTunnel:\nTunnel setup failed for %s:%s' % (node1, node1.dest) + ' to %s:%s\n' % (node2, node2.dest) + 'command was: %s' % cmd + '\n' + 'result was: ' + ch)
        for node in (node1, node2):
            if not self.moveIntf('tap9', node):
                raise Exception('interface move failed on node %s' % node)
        for (node, intf, addr) in ((node1, intfname1, addr1), (node2, intfname2, addr2)):
            if not addr:
                result = node.cmd('ip link set tap9 name', intf)
            else:
                result = node.cmd('ip link set tap9 name', intf, 'address', addr)
            if result:
                raise Exception('error renaming %s: %s' % (intf, result))
        return tunnel

    def status(self):
        if False:
            while True:
                i = 10
        'Detailed representation of link'
        if self.tunnel:
            if self.tunnel.poll() is not None:
                status = 'Tunnel EXITED %s' % self.tunnel.returncode
            else:
                status = 'Tunnel Running (%s: %s)' % (self.tunnel.pid, self.cmd)
        else:
            status = 'OK'
        result = '%s %s' % (Link.status(self), status)
        return result

class RemoteSSHLink(RemoteLink):
    """Remote link using SSH tunnels"""

    def __init__(self, node1, node2, **kwargs):
        if False:
            while True:
                i = 10
        RemoteLink.__init__(self, node1, node2, **kwargs)

class RemoteGRELink(RemoteLink):
    """Remote link using GRE tunnels"""
    GRE_KEY = 0

    def __init__(self, node1, node2, **kwargs):
        if False:
            print('Hello World!')
        RemoteLink.__init__(self, node1, node2, **kwargs)

    def stop(self):
        if False:
            return 10
        'Stop this link'
        if self.tunnel:
            self.intf1.delete()
            self.intf2.delete()
        else:
            Link.stop(self)
        self.tunnel = None

    def makeIntfPair(self, intfname1, intfname2, addr1=None, addr2=None, node1=None, node2=None, deleteIntfs=True):
        if False:
            i = 10
            return i + 15
        'Create pair of interfaces\n            intfname1: name of interface 1\n            intfname2: name of interface 2\n            (override this method [and possibly delete()]\n            to change link type)'
        node1 = self.node1 if node1 is None else node1
        node2 = self.node2 if node2 is None else node2
        server1 = getattr(node1, 'server', 'localhost')
        server2 = getattr(node2, 'server', 'localhost')
        if server1 == server2:
            Link.makeIntfPair(intfname1, intfname2, addr1, addr2, node1, node2, deleteIntfs=deleteIntfs)
            node1.cmd('ip link set dev %s mtu 1450' % intfname1)
            node2.cmd('ip link set dev %s mtu 1450' % intfname2)
        else:
            self.makeTunnel(node1, node2, intfname1, intfname2, addr1, addr2)
            self.tunnel = 1

    def makeTunnel(self, node1, node2, intfname1, intfname2, addr1=None, addr2=None):
        if False:
            return 10
        'Make a tunnel across switches on different servers'
        assert node1.server != node2.server
        if node2.server == 'localhost':
            return self.makeTunnel(node1=node2, node2=node1, intfname1=intfname2, intfname2=intfname1, addr1=addr2, addr2=addr1)
        (IP1, IP2) = (node1.serverIP, node2.serverIP)
        if node1.server == 'localhost':
            output = quietRun('ip route get %s' % node2.serverIP)
            IP1 = output.split(' src ')[1].split()[0]
        debug('\n*** Make GRE tunnel ' + node1.server + ':' + intfname1 + ' == ' + node2.server + ':' + intfname2)
        tun1 = 'local ' + IP1 + ' remote ' + IP2
        tun2 = 'local ' + IP2 + ' remote ' + IP1
        self.__class__.GRE_KEY += 1
        for (node, intfname, addr, tun) in [(node1, intfname1, addr1, tun1), (node2, intfname2, addr2, tun2)]:
            node.rcmd('ip link delete ' + intfname)
            result = node.rcmd('ip link add name ' + intfname + ' type gretap ' + tun + ' ttl 64 key ' + str(self.__class__.GRE_KEY))
            if result:
                raise Exception('error creating gretap on %s: %s' % (node, result))
            if addr:
                node.rcmd('ip link set %s address %s' % (intfname, addr))
            node.rcmd('ip link set dev %s up' % intfname)
            node.rcmd('ip link set dev %s mtu 1450' % intfname)
            if not self.moveIntf(intfname, node):
                raise Exception('interface move failed on node %s' % node)
        return None

class Placer(object):
    """Node placement algorithm for MininetCluster"""

    def __init__(self, servers=None, nodes=None, hosts=None, switches=None, controllers=None, links=None):
        if False:
            i = 10
            return i + 15
        'Initialize placement object\n           servers: list of servers\n           nodes: list of all nodes\n           hosts: list of hosts\n           switches: list of switches\n           controllers: list of controllers\n           links: list of links\n           (all arguments are optional)\n           returns: server'
        self.servers = servers or []
        self.nodes = nodes or []
        self.hosts = hosts or []
        self.switches = switches or []
        self.controllers = controllers or []
        self.links = links or []

    def place(self, node):
        if False:
            print('Hello World!')
        'Return server for a given node'
        assert self, node
        return 'localhost'

class RandomPlacer(Placer):
    """Random placement"""

    def place(self, node):
        if False:
            return 10
        'Random placement function\n            node: node'
        assert node
        return self.servers[randrange(0, len(self.servers))]

class RoundRobinPlacer(Placer):
    """Round-robin placement
       Note this will usually result in cross-server links between
       hosts and switches"""

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        Placer.__init__(self, *args, **kwargs)
        self.next = 0

    def place(self, node):
        if False:
            for i in range(10):
                print('nop')
        'Round-robin placement function\n            node: node'
        assert node
        server = self.servers[self.next]
        self.next = (self.next + 1) % len(self.servers)
        return server

class SwitchBinPlacer(Placer):
    """Place switches (and controllers) into evenly-sized bins,
       and attempt to co-locate hosts and switches"""

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        Placer.__init__(self, *args, **kwargs)
        self.servdict = dict(enumerate(self.servers))
        self.hset = frozenset(self.hosts)
        self.sset = frozenset(self.switches)
        self.cset = frozenset(self.controllers)
        self.placement = self.calculatePlacement()

    @staticmethod
    def bin(nodes, servers):
        if False:
            return 10
        'Distribute nodes evenly over servers'
        nlen = len(nodes)
        slen = len(servers)
        quotient = int(nlen / slen)
        binsizes = {server: quotient for server in servers}
        remainder = nlen % slen
        for server in servers[0:remainder]:
            binsizes[server] += 1
        tickets = sum([binsizes[server] * [server] for server in servers], [])
        return dict(zip(nodes, tickets))

    def calculatePlacement(self):
        if False:
            while True:
                i = 10
        'Pre-calculate node placement'
        placement = {}
        switchFor = {}
        for (src, dst) in self.links:
            if src in self.hset and dst in self.sset:
                switchFor[src] = dst
            if dst in self.hset and src in self.sset:
                switchFor[dst] = src
        placement = self.bin(self.switches, self.servers)
        placement.update(self.bin(self.controllers, self.servers))
        for h in self.hosts:
            if h in placement:
                continue
            if h in switchFor:
                placement[h] = placement[switchFor[h]]
            else:
                raise Exception('SwitchBinPlacer: cannot place isolated host ' + h)
        return placement

    def place(self, node):
        if False:
            i = 10
            return i + 15
        'Simple placement algorithm:\n           place switches into evenly sized bins,\n           and place hosts near their switches'
        return self.placement[node]

class HostSwitchBinPlacer(Placer):
    """Place switches *and hosts* into evenly-sized bins
       Note that this will usually result in cross-server
       links between hosts and switches"""

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        Placer.__init__(self, *args, **kwargs)
        scount = len(self.servers)
        self.hbin = max(int(len(self.hosts) / scount), 1)
        self.sbin = max(int(len(self.switches) / scount), 1)
        self.cbin = max(int(len(self.controllers) / scount), 1)
        info('scount:', scount)
        info('bins:', self.hbin, self.sbin, self.cbin, '\n')
        self.servdict = dict(enumerate(self.servers))
        self.hset = frozenset(self.hosts)
        self.sset = frozenset(self.switches)
        self.cset = frozenset(self.controllers)
        (self.hind, self.sind, self.cind) = (0, 0, 0)

    def place(self, node):
        if False:
            for i in range(10):
                print('nop')
        'Simple placement algorithm:\n            place nodes into evenly sized bins'
        if node in self.hset:
            server = self.servdict[self.hind / self.hbin]
            self.hind += 1
        elif node in self.sset:
            server = self.servdict[self.sind / self.sbin]
            self.sind += 1
        elif node in self.cset:
            server = self.servdict[self.cind / self.cbin]
            self.cind += 1
        else:
            info('warning: unknown node', node)
            server = self.servdict[0]
        return server

class MininetCluster(Mininet):
    """Cluster-enhanced version of Mininet class"""
    sshcmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ForwardAgent=yes']

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'servers: a list of servers to use (note: include\n           localhost or None to use local system as well)\n           user: user name for server ssh\n           placement: Placer() subclass'
        params = {'host': RemoteHost, 'switch': RemoteOVSSwitch, 'link': RemoteLink, 'precheck': True}
        params.update(kwargs)
        servers = params.pop('servers', ['localhost'])
        servers = [s if s else 'localhost' for s in servers]
        self.servers = servers
        self.serverIP = params.pop('serverIP', {})
        if not self.serverIP:
            self.serverIP = {server: RemoteMixin.findServerIP(server) for server in self.servers}
        self.user = params.pop('user', findUser())
        if params.pop('precheck'):
            self.precheck()
        self.connections = {}
        self.placement = params.pop('placement', SwitchBinPlacer)
        self.cdir = os.environ['HOME'] + '/.ssh/mn'
        errRun(['mkdir', '-p', self.cdir])
        Mininet.__init__(self, *args, **params)

    def popen(self, cmd):
        if False:
            print('Hello World!')
        'Popen() for server connections'
        assert self
        old = signal(SIGINT, SIG_IGN)
        conn = Popen(cmd, stdin=PIPE, stdout=PIPE, close_fds=True)
        signal(SIGINT, old)
        return conn

    def baddLink(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'break addlink for testing'
        pass

    def precheck(self):
        if False:
            i = 10
            return i + 15
        'Pre-check to make sure connection works and that\n           we can call sudo without a password'
        result = 0
        info('*** Checking servers\n')
        for server in self.servers:
            ip = self.serverIP[server]
            if not server or server == 'localhost':
                continue
            info(server, '')
            dest = '%s@%s' % (self.user, ip)
            cmd = ['sudo', '-E', '-u', self.user]
            cmd += self.sshcmd + ['-n', dest, 'sudo true']
            debug(' '.join(cmd), '\n')
            (_out, _err, code) = errRun(cmd)
            if code != 0:
                error('\nstartConnection: server connection check failed to %s using command:\n%s\n' % (server, ' '.join(cmd)))
            result |= code
        if result:
            error('*** Server precheck failed.\n*** Make sure that the above ssh command works correctly.\n*** You may also need to run mn -c on all nodes, and/or\n*** use sudo -E.\n')
            sys.exit(1)
        info('\n')

    def modifiedaddHost(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Slightly modify addHost'
        assert self
        kwargs['splitInit'] = True
        return Mininet.addHost(*args, **kwargs)

    def placeNodes(self):
        if False:
            while True:
                i = 10
        "Place nodes on servers (if they don't have a server), and\n           start shell processes"
        if not self.servers or not self.topo:
            return
        nodes = self.topo.nodes()
        placer = self.placement(servers=self.servers, nodes=self.topo.nodes(), hosts=self.topo.hosts(), switches=self.topo.switches(), links=self.topo.links())
        for node in nodes:
            config = self.topo.nodeInfo(node)
            if 'server' in config.keys() and config['server'] is None:
                config['server'] = 'localhost'
            server = config.setdefault('server', placer.place(node))
            if server:
                config.setdefault('serverIP', self.serverIP[server])
            info('%s:%s ' % (node, server))
            key = (None, server)
            (_dest, cfile, _conn) = self.connections.get(key, (None, None, None))
            if cfile:
                config.setdefault('controlPath', cfile)

    @staticmethod
    def isLoopback(ipaddr):
        if False:
            return 10
        'Is ipaddr an IPv4 loopback address?'
        return ipaddr.startswith('127.')

    def addController(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Patch to update IP address to global IP address'
        controller = Mininet.addController(self, *args, **kwargs)
        controllerIP = controller.IP()
        if not isinstance(controller, Controller) or not self.isLoopback(controller.IP()):
            return controller
        serverIPs = [ip for ip in self.serverIP.values() if ip != controllerIP]
        if not serverIPs:
            return None
        for remoteIP in serverIPs:
            route = controller.cmd('ip route get', remoteIP, '| egrep -o "dev\\s[^[:space:]]+"')
            if not route:
                raise Exception('addController: no route from', controller, 'to', remoteIP)
            intf = route.split()[1].strip()
            if intf != 'lo':
                break
        if intf == 'lo':
            raise Exception('addController: could not find external interface/IP for %s' % controller)
        debug('adding', intf, 'to', controller)
        Intf(intf, node=controller).updateIP()
        debug(controller, 'IP address updated to', controller.IP())
        return controller

    def buildFromTopo(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Start network'
        info('*** Placing nodes\n')
        self.placeNodes()
        info('\n')
        Mininet.buildFromTopo(self, *args, **kwargs)
remoteServer = 'ubuntu2'

def testNsTunnels(remote=remoteServer, link=RemoteGRELink):
    if False:
        return 10
    'Test tunnels between nodes in namespaces'
    net = Mininet(host=RemoteHost, link=link, waitConnected=True)
    h1 = net.addHost('h1')
    h2 = net.addHost('h2', server=remote)
    net.addLink(h1, h2)
    net.start()
    net.pingAll()
    net.stop()

def testRemoteNet(remote=remoteServer, link=RemoteGRELink):
    if False:
        print('Hello World!')
    'Test remote Node classes'
    info('*** Remote Node Test\n')
    net = Mininet(host=RemoteHost, switch=RemoteOVSSwitch, link=link, controller=ClusterController, waitConnected=True)
    c0 = net.addController('c0')
    info('*** Creating local h1\n')
    h1 = net.addHost('h1')
    info('*** Creating remote h2\n')
    h2 = net.addHost('h2', server=remote)
    info('*** Creating local s1\n')
    s1 = net.addSwitch('s1')
    info('*** Creating remote s2\n')
    s2 = net.addSwitch('s2', server=remote)
    info('*** Adding links\n')
    net.addLink(h1, s1)
    net.addLink(s1, s2)
    net.addLink(h2, s2)
    net.start()
    info('Mininet is running on', quietRun('hostname').strip(), '\n')
    for node in (c0, h1, h2, s1, s2):
        info('Node', node, 'is running on', node.cmd('hostname').strip(), '\n')
    net.pingAll()
    CLI(net)
    net.stop()
remoteHosts = ['h2']
remoteSwitches = ['s2']

def HostPlacer(name, *args, **params):
    if False:
        for i in range(10):
            print('nop')
    'Custom Host() constructor which places hosts on servers'
    if name in remoteHosts:
        return RemoteHost(name, *args, server=remoteServer, **params)
    else:
        return Host(name, *args, **params)

def SwitchPlacer(name, *args, **params):
    if False:
        while True:
            i = 10
    'Custom Switch() constructor which places switches on servers'
    if name in remoteSwitches:
        return RemoteOVSSwitch(name, *args, server=remoteServer, **params)
    else:
        return RemoteOVSSwitch(name, *args, **params)

def ClusterController(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Custom Controller() constructor which updates its intf IP address'
    intf = kwargs.pop('intf', '')
    controller = Controller(*args, **kwargs)
    if not intf:
        output = controller.cmd("ip a | egrep -o '\\w+:\\s\\w+'").split('\n')
        for line in output:
            intf = line.split()[-1]
            if intf != 'lo':
                break
        if intf == 'lo':
            raise Exception('Could not find non-loopback interfacefor %s' % controller)
    Intf(intf, node=controller).updateIP()
    return controller

def testRemoteTopo(link=RemoteGRELink):
    if False:
        while True:
            i = 10
    'Test remote Node classes using Mininet()/Topo() API'
    topo = LinearTopo(2)
    net = Mininet(topo=topo, host=HostPlacer, switch=SwitchPlacer, link=link, controller=ClusterController)
    net.start()
    net.pingAll()
    net.stop()

def testRemoteSwitches(remote=remoteServer, link=RemoteGRELink):
    if False:
        while True:
            i = 10
    'Test with local hosts and remote switches'
    servers = ['localhost', remote]
    topo = TreeTopo(depth=4, fanout=2)
    net = MininetCluster(topo=topo, servers=servers, link=link, placement=RoundRobinPlacer)
    net.start()
    net.pingAll()
    net.stop()

def testMininetCluster(remote=remoteServer, link=RemoteGRELink):
    if False:
        for i in range(10):
            print('nop')
    'Test MininetCluster()'
    servers = ['localhost', remote]
    topo = TreeTopo(depth=3, fanout=3)
    net = MininetCluster(topo=topo, servers=servers, link=link, placement=SwitchBinPlacer)
    net.start()
    net.pingAll()
    net.stop()

def signalTest(remote=remoteServer):
    if False:
        i = 10
        return i + 15
    'Make sure hosts are robust to signals'
    h = RemoteHost('h0', server=remote)
    h.shell.send_signal(SIGINT)
    h.shell.poll()
    if h.shell.returncode is None:
        info('signalTest: SUCCESS: ', h, 'has not exited after SIGINT', '\n')
    else:
        info('signalTest: FAILURE:', h, 'exited with code', h.shell.returncode, '\n')
    h.stop()
if __name__ == '__main__':
    setLogLevel('info')
    remoteLink = RemoteSSHLink
    testRemoteTopo(link=remoteLink)
    testNsTunnels(remote=remoteServer, link=remoteLink)
    testRemoteNet(remote=remoteServer, link=remoteLink)
    testMininetCluster(remote=remoteServer, link=remoteLink)
    testRemoteSwitches(remote=remoteServer, link=remoteLink)
    signalTest(remote=remoteServer)