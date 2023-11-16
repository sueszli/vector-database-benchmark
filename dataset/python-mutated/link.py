"""
link.py: interface and link abstractions for mininet

It seems useful to bundle functionality for interfaces into a single
class.

Also it seems useful to enable the possibility of multiple flavors of
links, including:

- simple veth pairs
- tunneled links
- patchable links (which can be disconnected and reconnected via a patchbay)
- link simulators (e.g. wireless)

Basic division of labor:

  Nodes: know how to execute commands
  Intfs: know how to configure themselves
  Links: know how to connect nodes together

Intf: basic interface object that can configure itself
TCIntf: interface with bandwidth limiting and delay via tc

Link: basic link class for creating veth pairs
"""
import re
from mininet.log import info, error, debug
from mininet.util import makeIntfPair

class Intf(object):
    """Basic interface object that can configure itself."""

    def __init__(self, name, node=None, port=None, link=None, mac=None, **params):
        if False:
            i = 10
            return i + 15
        "name: interface name (e.g. h1-eth0)\n           node: owning node (where this intf most likely lives)\n           link: parent link if we're part of a link\n           other arguments are passed to config()"
        self.node = node
        self.name = name
        self.link = link
        self.mac = mac
        (self.ip, self.prefixLen) = (None, None)
        if self.name == 'lo':
            self.ip = '127.0.0.1'
            self.prefixLen = 8
        if node:
            moveIntfFn = params.pop('moveIntfFn', None)
            if moveIntfFn:
                node.addIntf(self, port=port, moveIntfFn=moveIntfFn)
            else:
                node.addIntf(self, port=port)
        self.params = params
        self.config(**params)

    def cmd(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Run a command in our owning node'
        return self.node.cmd(*args, **kwargs)

    def ifconfig(self, *args):
        if False:
            while True:
                i = 10
        'Configure ourselves using ifconfig'
        return self.cmd('ifconfig', self.name, *args)

    def setIP(self, ipstr, prefixLen=None):
        if False:
            i = 10
            return i + 15
        'Set our IP address'
        if '/' in ipstr:
            (self.ip, self.prefixLen) = ipstr.split('/')
            return self.ifconfig(ipstr, 'up')
        else:
            if prefixLen is None:
                raise Exception('No prefix length set for IP address %s' % (ipstr,))
            (self.ip, self.prefixLen) = (ipstr, prefixLen)
            return self.ifconfig('%s/%s' % (ipstr, prefixLen))

    def setMAC(self, macstr):
        if False:
            i = 10
            return i + 15
        'Set the MAC address for an interface.\n           macstr: MAC address as string'
        self.mac = macstr
        return self.ifconfig('down') + self.ifconfig('hw', 'ether', macstr) + self.ifconfig('up')
    _ipMatchRegex = re.compile('\\d+\\.\\d+\\.\\d+\\.\\d+')
    _macMatchRegex = re.compile('..:..:..:..:..:..')

    def updateIP(self):
        if False:
            i = 10
            return i + 15
        'Return updated IP address based on ifconfig'
        (ifconfig, _err, _exitCode) = self.node.pexec('ifconfig %s' % self.name)
        ips = self._ipMatchRegex.findall(ifconfig)
        self.ip = ips[0] if ips else None
        return self.ip

    def updateMAC(self):
        if False:
            print('Hello World!')
        'Return updated MAC address based on ifconfig'
        ifconfig = self.ifconfig()
        macs = self._macMatchRegex.findall(ifconfig)
        self.mac = macs[0] if macs else None
        return self.mac

    def updateAddr(self):
        if False:
            return 10
        'Return IP address and MAC address based on ifconfig.'
        ifconfig = self.ifconfig()
        ips = self._ipMatchRegex.findall(ifconfig)
        macs = self._macMatchRegex.findall(ifconfig)
        self.ip = ips[0] if ips else None
        self.mac = macs[0] if macs else None
        return (self.ip, self.mac)

    def IP(self):
        if False:
            return 10
        'Return IP address'
        return self.ip

    def MAC(self):
        if False:
            print('Hello World!')
        'Return MAC address'
        return self.mac

    def isUp(self, setUp=False):
        if False:
            while True:
                i = 10
        'Return whether interface is up'
        if setUp:
            cmdOutput = self.ifconfig('up')
            if cmdOutput:
                error('Error setting %s up: %s ' % (self.name, cmdOutput))
                return False
            else:
                return True
        else:
            return 'UP' in self.ifconfig()

    def rename(self, newname):
        if False:
            return 10
        'Rename interface'
        if self.node and self.name in self.node.nameToIntf:
            self.node.nameToIntf[newname] = self.node.nameToIntf.pop(self.name)
        self.ifconfig('down')
        result = self.cmd('ip link set', self.name, 'name', newname)
        self.name = newname
        self.ifconfig('up')
        return result

    def setParam(self, results, method, **param):
        if False:
            return 10
        'Internal method: configure a *single* parameter\n           results: dict of results to update\n           method: config method name\n           param: arg=value (ignore if value=None)\n           value may also be list or dict'
        (name, value) = list(param.items())[0]
        f = getattr(self, method, None)
        if not f or value is None:
            return None
        if isinstance(value, list):
            result = f(*value)
        elif isinstance(value, dict):
            result = f(**value)
        else:
            result = f(value)
        results[name] = result
        return result

    def config(self, mac=None, ip=None, ifconfig=None, up=True, **_params):
        if False:
            print('Hello World!')
        "Configure Node according to (optional) parameters:\n           mac: MAC address\n           ip: IP address\n           ifconfig: arbitrary interface configuration\n           Subclasses should override this method and call\n           the parent class's config(**params)"
        r = {}
        self.setParam(r, 'setMAC', mac=mac)
        self.setParam(r, 'setIP', ip=ip)
        self.setParam(r, 'isUp', up=up)
        self.setParam(r, 'ifconfig', ifconfig=ifconfig)
        return r

    def delete(self):
        if False:
            print('Hello World!')
        'Delete interface'
        self.cmd('ip link del ' + self.name)
        self.node.delIntf(self)
        self.link = None

    def status(self):
        if False:
            while True:
                i = 10
        'Return intf status as a string'
        (links, _err, _result) = self.node.pexec('ip link show')
        if self.name in links:
            return 'OK'
        else:
            return 'MISSING'

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<%s %s>' % (self.__class__.__name__, self.name)

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.name

class TCIntf(Intf):
    """Interface customized by tc (traffic control) utility
       Allows specification of bandwidth limits (various methods)
       as well as delay, loss and max queue length"""
    bwParamMax = 1000

    def bwCmds(self, bw=None, speedup=0, use_hfsc=False, use_tbf=False, latency_ms=None, enable_ecn=False, enable_red=False):
        if False:
            for i in range(10):
                print('nop')
        'Return tc commands to set bandwidth'
        (cmds, parent) = ([], ' root ')
        if bw and (bw < 0 or bw > self.bwParamMax):
            error('Bandwidth limit', bw, 'is outside supported range 0..%d' % self.bwParamMax, '- ignoring\n')
        elif bw is not None:
            if speedup > 0 and self.node.name[0:1] == 's':
                bw = speedup
            if use_hfsc:
                cmds += ['%s qdisc add dev %s root handle 5:0 hfsc default 1', '%s class add dev %s parent 5:0 classid 5:1 hfsc sc ' + 'rate %fMbit ul rate %fMbit' % (bw, bw)]
            elif use_tbf:
                if latency_ms is None:
                    latency_ms = 15.0 * 8 / bw
                cmds += ['%s qdisc add dev %s root handle 5: tbf ' + 'rate %fMbit burst 15000 latency %fms' % (bw, latency_ms)]
            else:
                cmds += ['%s qdisc add dev %s root handle 5:0 htb default 1', '%s class add dev %s parent 5:0 classid 5:1 htb ' + 'rate %fMbit burst 15k' % bw]
            parent = ' parent 5:1 '
            if enable_ecn:
                cmds += ['%s qdisc add dev %s' + parent + 'handle 6: red limit 1000000 ' + 'min 30000 max 35000 avpkt 1500 ' + 'burst 20 ' + 'bandwidth %fmbit probability 1 ecn' % bw]
                parent = ' parent 6: '
            elif enable_red:
                cmds += ['%s qdisc add dev %s' + parent + 'handle 6: red limit 1000000 ' + 'min 30000 max 35000 avpkt 1500 ' + 'burst 20 ' + 'bandwidth %fmbit probability 1' % bw]
                parent = ' parent 6: '
        return (cmds, parent)

    @staticmethod
    def delayCmds(parent, delay=None, jitter=None, loss=None, max_queue_size=None):
        if False:
            i = 10
            return i + 15
        'Internal method: return tc commands for delay and loss'
        cmds = []
        if loss and (loss < 0 or loss > 100):
            error('Bad loss percentage', loss, '%%\n')
        else:
            netemargs = '%s%s%s%s' % ('delay %s ' % delay if delay is not None else '', '%s ' % jitter if jitter is not None else '', 'loss %.5f ' % loss if loss is not None and loss > 0 else '', 'limit %d' % max_queue_size if max_queue_size is not None else '')
            if netemargs:
                cmds = ['%s qdisc add dev %s ' + parent + ' handle 10: netem ' + netemargs]
                parent = ' parent 10:1 '
        return (cmds, parent)

    def tc(self, cmd, tc='tc'):
        if False:
            return 10
        'Execute tc command for our interface'
        c = cmd % (tc, self)
        debug(' *** executing command: %s\n' % c)
        return self.cmd(c)

    def config(self, bw=None, delay=None, jitter=None, loss=None, gro=False, txo=True, rxo=True, speedup=0, use_hfsc=False, use_tbf=False, latency_ms=None, enable_ecn=False, enable_red=False, max_queue_size=None, **params):
        if False:
            print('Hello World!')
        "Configure the port and set its properties.\n           bw: bandwidth in b/s (e.g. '10m')\n           delay: transmit delay (e.g. '1ms' )\n           jitter: jitter (e.g. '1ms')\n           loss: loss (e.g. '1%' )\n           gro: enable GRO (False)\n           txo: enable transmit checksum offload (True)\n           rxo: enable receive checksum offload (True)\n           speedup: experimental switch-side bw option\n           use_hfsc: use HFSC scheduling\n           use_tbf: use TBF scheduling\n           latency_ms: TBF latency parameter\n           enable_ecn: enable ECN (False)\n           enable_red: enable RED (False)\n           max_queue_size: queue limit parameter for netem"
        gro = not params.pop('disable_gro', not gro)
        result = Intf.config(self, **params)

        def on(isOn):
            if False:
                i = 10
                return i + 15
            "Helper method: bool -> 'on'/'off'"
            return 'on' if isOn else 'off'
        self.cmd('ethtool -K', self, 'gro', on(gro), 'tx', on(txo), 'rx', on(rxo))
        if bw is None and (not delay) and (not loss) and (max_queue_size is None):
            return None
        tcoutput = self.tc('%s qdisc show dev %s')
        if 'priomap' not in tcoutput and 'noqueue' not in tcoutput:
            cmds = ['%s qdisc del dev %s root']
        else:
            cmds = []
        (bwcmds, parent) = self.bwCmds(bw=bw, speedup=speedup, use_hfsc=use_hfsc, use_tbf=use_tbf, latency_ms=latency_ms, enable_ecn=enable_ecn, enable_red=enable_red)
        cmds += bwcmds
        (delaycmds, parent) = self.delayCmds(delay=delay, jitter=jitter, loss=loss, max_queue_size=max_queue_size, parent=parent)
        cmds += delaycmds
        stuff = (['%.2fMbit' % bw] if bw is not None else []) + (['%s delay' % delay] if delay is not None else []) + (['%s jitter' % jitter] if jitter is not None else []) + (['%.5f%% loss' % loss] if loss is not None else []) + (['ECN'] if enable_ecn else ['RED'] if enable_red else [])
        info('(' + ' '.join(stuff) + ') ')
        debug('at map stage w/cmds: %s\n' % cmds)
        tcoutputs = [self.tc(cmd) for cmd in cmds]
        for output in tcoutputs:
            if output != '':
                error('*** Error: %s' % output)
        debug('cmds:', cmds, '\n')
        debug('outputs:', tcoutputs, '\n')
        result['tcoutputs'] = tcoutputs
        result['parent'] = parent
        return result

class Link(object):
    """A basic link is just a veth pair.
       Other types of links could be tunnels, link emulators, etc.."""

    def __init__(self, node1, node2, port1=None, port2=None, intfName1=None, intfName2=None, addr1=None, addr2=None, intf=Intf, cls1=None, cls2=None, params1=None, params2=None, fast=True, **params):
        if False:
            i = 10
            return i + 15
        'Create veth link to another node, making two new interfaces.\n           node1: first node\n           node2: second node\n           port1: node1 port number (optional)\n           port2: node2 port number (optional)\n           intf: default interface class/constructor\n           cls1, cls2: optional interface-specific constructors\n           intfName1: node1 interface name (optional)\n           intfName2: node2  interface name (optional)\n           params1: parameters for interface 1 (optional)\n           params2: parameters for interface 2 (optional)\n           **params: additional parameters for both interfaces'
        params1 = dict(params1) if params1 else {}
        params2 = dict(params2) if params2 else {}
        if port1 is not None:
            params1['port'] = port1
        if port2 is not None:
            params2['port'] = port2
        if 'port' not in params1:
            params1['port'] = node1.newPort()
        if 'port' not in params2:
            params2['port'] = node2.newPort()
        if not intfName1:
            intfName1 = self.intfName(node1, params1['port'])
        if not intfName2:
            intfName2 = self.intfName(node2, params2['port'])
        params1.update(params)
        params2.update(params)
        self.fast = fast
        if fast:
            params1.setdefault('moveIntfFn', self._ignore)
            params2.setdefault('moveIntfFn', self._ignore)
            self.makeIntfPair(intfName1, intfName2, addr1, addr2, node1, node2, deleteIntfs=False)
        else:
            self.makeIntfPair(intfName1, intfName2, addr1, addr2)
        if not cls1:
            cls1 = intf
        if not cls2:
            cls2 = intf
        intf1 = cls1(name=intfName1, node=node1, link=self, mac=addr1, **params1)
        intf2 = cls2(name=intfName2, node=node2, link=self, mac=addr2, **params2)
        (self.intf1, self.intf2) = (intf1, intf2)

    @staticmethod
    def _ignore(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Ignore any arguments'
        pass

    def intfName(self, node, n):
        if False:
            i = 10
            return i + 15
        'Construct a canonical interface name node-ethN for interface n.'
        assert self
        return node.name + '-eth' + repr(n)

    @classmethod
    def makeIntfPair(cls, intfname1, intfname2, addr1=None, addr2=None, node1=None, node2=None, deleteIntfs=True):
        if False:
            for i in range(10):
                print('nop')
        'Create pair of interfaces\n           intfname1: name for interface 1\n           intfname2: name for interface 2\n           addr1: MAC address for interface 1 (optional)\n           addr2: MAC address for interface 2 (optional)\n           node1: home node for interface 1 (optional)\n           node2: home node for interface 2 (optional)\n           (override this method [and possibly delete()]\n           to change link type)'
        assert cls
        return makeIntfPair(intfname1, intfname2, addr1, addr2, node1, node2, deleteIntfs=deleteIntfs)

    def delete(self):
        if False:
            while True:
                i = 10
        'Delete this link'
        self.intf1.delete()
        self.intf1 = None
        self.intf2.delete()
        self.intf2 = None

    def stop(self):
        if False:
            return 10
        'Override to stop and clean up link as needed'
        self.delete()

    def status(self):
        if False:
            return 10
        'Return link status as a string'
        return '(%s %s)' % (self.intf1.status(), self.intf2.status())

    def __str__(self):
        if False:
            while True:
                i = 10
        return '%s<->%s' % (self.intf1, self.intf2)

class OVSIntf(Intf):
    """Patch interface on an OVSSwitch"""

    def ifconfig(self, *args):
        if False:
            print('Hello World!')
        cmd = ' '.join(args)
        if cmd == 'up':
            return
        else:
            raise Exception('OVSIntf cannot do ifconfig ' + cmd)

class OVSLink(Link):
    """Link that makes patch links between OVSSwitches
       Warning: in testing we have found that no more
       than ~64 OVS patch links should be used in row."""

    def __init__(self, node1, node2, **kwargs):
        if False:
            while True:
                i = 10
        'See Link.__init__() for options'
        if 'OVSSwitch' not in globals():
            from mininet.node import OVSSwitch
        self.isPatchLink = False
        if isinstance(node1, OVSSwitch) and isinstance(node2, OVSSwitch):
            self.isPatchLink = True
            kwargs.update(cls1=OVSIntf, cls2=OVSIntf)
        Link.__init__(self, node1, node2, **kwargs)

    def makeIntfPair(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Usually delegated to OVSSwitch'
        if self.isPatchLink:
            return (None, None)
        else:
            return Link.makeIntfPair(*args, **kwargs)

class TCLink(Link):
    """Link with TC interfaces"""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        kwargs.setdefault('cls1', TCIntf)
        kwargs.setdefault('cls2', TCIntf)
        Link.__init__(self, *args, **kwargs)

class TCULink(TCLink):
    """TCLink with default settings optimized for UserSwitch
       (txo=rxo=0/False).  Unfortunately with recent Linux kernels,
       enabling TX and RX checksum offload on veth pairs doesn't work
       well with UserSwitch: either it gets terrible performance or
       TCP packets with bad checksums are generated, forwarded, and
       *dropped* due to having bad checksums! OVS and LinuxBridge seem
       to cope with this somehow, but it is likely to be an issue with
       many software Ethernet bridges."""

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs.update(txo=False, rxo=False)
        TCLink.__init__(self, *args, **kwargs)