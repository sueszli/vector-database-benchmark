"""
Node Library for Mininet

This contains additional Node types which you may find to be useful.
"""
from mininet.node import Node, Switch
from mininet.log import info, warn
from mininet.moduledeps import pathCheck
from mininet.util import quietRun

class LinuxBridge(Switch):
    """Linux Bridge (with optional spanning tree)"""
    nextPrio = 100

    def __init__(self, name, stp=False, prio=None, **kwargs):
        if False:
            while True:
                i = 10
        'stp: use spanning tree protocol? (default False)\n           prio: optional explicit bridge priority for STP'
        self.stp = stp
        if prio:
            self.prio = prio
        else:
            self.prio = LinuxBridge.nextPrio
            LinuxBridge.nextPrio += 1
        Switch.__init__(self, name, **kwargs)

    def connected(self):
        if False:
            while True:
                i = 10
        'Are we forwarding yet?'
        if self.stp:
            return 'forwarding' in self.cmd('brctl showstp', self)
        else:
            return True

    def start(self, _controllers):
        if False:
            i = 10
            return i + 15
        'Start Linux bridge'
        self.cmd('ifconfig', self, 'down')
        self.cmd('brctl delbr', self)
        self.cmd('brctl addbr', self)
        if self.stp:
            self.cmd('brctl setbridgeprio', self.prio)
            self.cmd('brctl stp', self, 'on')
        for i in self.intfList():
            if self.name in i.name:
                self.cmd('brctl addif', self, i)
        self.cmd('ifconfig', self, 'up')

    def stop(self, deleteIntfs=True):
        if False:
            i = 10
            return i + 15
        'Stop Linux bridge\n           deleteIntfs: delete interfaces? (True)'
        self.cmd('ifconfig', self, 'down')
        self.cmd('brctl delbr', self)
        super(LinuxBridge, self).stop(deleteIntfs)

    def dpctl(self, *args):
        if False:
            for i in range(10):
                print('nop')
        'Run brctl command'
        return self.cmd('brctl', *args)

    @classmethod
    def setup(cls):
        if False:
            print('Hello World!')
        'Check dependencies and warn about firewalling'
        pathCheck('brctl', moduleName='bridge-utils')
        for table in ('arp', 'ip', 'ip6'):
            cmd = 'sysctl net.bridge.bridge-nf-call-%stables' % table
            out = quietRun(cmd).strip()
            if out.endswith('1'):
                warn('Warning: Linux bridge may not work with', out, '\n')

class NAT(Node):
    """NAT: Provides connectivity to external network"""

    def __init__(self, name, subnet='10.0/8', localIntf=None, flush=False, **params):
        if False:
            while True:
                i = 10
        'Start NAT/forwarding between Mininet and external network\n           subnet: Mininet subnet (default 10.0/8)\n           flush: flush iptables before installing NAT rules'
        super(NAT, self).__init__(name, **params)
        self.subnet = subnet
        self.localIntf = localIntf
        self.flush = flush
        self.forwardState = self.cmd('sysctl -n net.ipv4.ip_forward').strip()

    def setManualConfig(self, intf):
        if False:
            while True:
                i = 10
        'Prevent network-manager/networkd from messing with our interface\n           by specifying manual configuration in /etc/network/interfaces'
        cfile = '/etc/network/interfaces'
        line = '\niface %s inet manual\n' % intf
        try:
            with open(cfile) as f:
                config = f.read()
        except IOError:
            config = ''
        if line not in config:
            info('*** Adding "' + line.strip() + '" to ' + cfile + '\n')
            with open(cfile, 'a') as f:
                f.write(line)
            self.cmd('service network-manager restart || netplan apply')

    def config(self, **params):
        if False:
            print('Hello World!')
        'Configure the NAT and iptables'
        if not self.localIntf:
            self.localIntf = self.defaultIntf()
        self.setManualConfig(self.localIntf)
        super(NAT, self).config(**params)
        if self.flush:
            self.cmd('sysctl net.ipv4.ip_forward=0')
            self.cmd('iptables -F')
            self.cmd('iptables -t nat -F')
            self.cmd('iptables -P INPUT ACCEPT')
            self.cmd('iptables -P OUTPUT ACCEPT')
            self.cmd('iptables -P FORWARD DROP')
        self.cmd('iptables -I FORWARD', '-i', self.localIntf, '-d', self.subnet, '-j DROP')
        self.cmd('iptables -A FORWARD', '-i', self.localIntf, '-s', self.subnet, '-j ACCEPT')
        self.cmd('iptables -A FORWARD', '-o', self.localIntf, '-d', self.subnet, '-j ACCEPT')
        self.cmd('iptables -t nat -A POSTROUTING', '-s', self.subnet, "'!'", '-d', self.subnet, '-j MASQUERADE')
        self.cmd('sysctl net.ipv4.ip_forward=1')

    def terminate(self):
        if False:
            return 10
        'Stop NAT/forwarding between Mininet and external network'
        self.cmd('iptables -D FORWARD', '-i', self.localIntf, '-d', self.subnet, '-j DROP')
        self.cmd('iptables -D FORWARD', '-i', self.localIntf, '-s', self.subnet, '-j ACCEPT')
        self.cmd('iptables -D FORWARD', '-o', self.localIntf, '-d', self.subnet, '-j ACCEPT')
        self.cmd('iptables -t nat -D POSTROUTING', '-s', self.subnet, "'!'", '-d', self.subnet, '-j MASQUERADE')
        self.cmd('sysctl net.ipv4.ip_forward=%s' % self.forwardState)
        super(NAT, self).terminate()