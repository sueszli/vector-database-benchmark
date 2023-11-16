"""
vlanhost.py: Host subclass that uses a VLAN tag for the default interface.

Dependencies:
    This class depends on the "vlan" package
    $ sudo apt-get install vlan

Usage (example uses VLAN ID=1000):
    From the command line:
        sudo mn --custom vlanhost.py --host vlan,vlan=1000

    From a script (see exampleUsage function below):
        from functools import partial
        from vlanhost import VLANHost

        ....

        host = partial( VLANHost, vlan=1000 )
        net = Mininet( host=host, ... )

    Directly running this script:
        sudo python vlanhost.py 1000

"""
from sys import exit
from mininet.node import Host
from mininet.topo import Topo
from mininet.util import quietRun
from mininet.log import error

class VLANHost(Host):
    """Host connected to VLAN interface"""

    def config(self, vlan=100, **params):
        if False:
            i = 10
            return i + 15
        'Configure VLANHost according to (optional) parameters:\n           vlan: VLAN ID for default interface'
        r = super(VLANHost, self).config(**params)
        intf = self.defaultIntf()
        self.cmd('ifconfig %s inet 0' % intf)
        self.cmd('vconfig add %s %d' % (intf, vlan))
        self.cmd('ifconfig %s.%d inet %s' % (intf, vlan, params['ip']))
        newName = '%s.%d' % (intf, vlan)
        intf.name = newName
        self.nameToIntf[newName] = intf
        return r
hosts = {'vlan': VLANHost}

def exampleAllHosts(vlan):
    if False:
        for i in range(10):
            print('nop')
    'Simple example of how VLANHost can be used in a script'
    host = partial(VLANHost, vlan=vlan)
    topo = SingleSwitchTopo(k=2)
    net = Mininet(host=host, topo=topo, waitConnected=True)
    net.start()
    CLI(net)
    net.stop()

class VLANStarTopo(Topo):
    """Example topology that uses host in multiple VLANs

       The topology has a single switch. There are k VLANs with
       n hosts in each, all connected to the single switch. There
       are also n hosts that are not in any VLAN, also connected to
       the switch."""

    def build(self, k=2, n=2, vlanBase=100):
        if False:
            return 10
        s1 = self.addSwitch('s1')
        for i in range(k):
            vlan = vlanBase + i
            for j in range(n):
                name = 'h%d-%d' % (j + 1, vlan)
                h = self.addHost(name, cls=VLANHost, vlan=vlan)
                self.addLink(h, s1)
        for j in range(n):
            h = self.addHost('h%d' % (j + 1))
            self.addLink(h, s1)

def exampleCustomTags():
    if False:
        while True:
            i = 10
    'Simple example that exercises VLANStarTopo'
    net = Mininet(topo=VLANStarTopo(), waitConnected=True)
    net.start()
    CLI(net)
    net.stop()
if __name__ == '__main__':
    import sys
    from functools import partial
    from mininet.net import Mininet
    from mininet.cli import CLI
    from mininet.topo import SingleSwitchTopo
    from mininet.log import setLogLevel
    setLogLevel('info')
    if not quietRun('which vconfig'):
        error("Cannot find command 'vconfig'\nThe package", "'vlan' is required in Ubuntu or Debian,", "or 'vconfig' in Fedora\n")
        exit()
    if len(sys.argv) >= 2:
        exampleAllHosts(vlan=int(sys.argv[1]))
    else:
        exampleCustomTags()