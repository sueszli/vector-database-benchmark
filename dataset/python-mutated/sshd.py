"""
Create a network and start sshd(8) on each host.

While something like rshd(8) would be lighter and faster,
(and perfectly adequate on an in-machine network)
the advantage of running sshd is that scripts can work
unchanged on mininet and hardware.

In addition to providing ssh access to hosts, this example
demonstrates:

- creating a convenience function to construct networks
- connecting the host network to the root namespace
- running server processes (sshd in this case) on hosts
"""
import sys
from mininet.net import Mininet
from mininet.cli import CLI
from mininet.log import lg, info
from mininet.node import Node
from mininet.topolib import TreeTopo
from mininet.util import waitListening

def TreeNet(depth=1, fanout=2, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Convenience function for creating tree networks.'
    topo = TreeTopo(depth, fanout)
    return Mininet(topo, waitConnected=True, **kwargs)

def connectToRootNS(network, switch, ip, routes):
    if False:
        i = 10
        return i + 15
    'Connect hosts to root namespace via switch. Starts network.\n      network: Mininet() network object\n      switch: switch to connect to root namespace\n      ip: IP address for root namespace node\n      routes: host networks to route to'
    root = Node('root', inNamespace=False)
    intf = network.addLink(root, switch).intf1
    root.setIP(ip, intf=intf)
    network.start()
    for route in routes:
        root.cmd('route add -net ' + route + ' dev ' + str(intf))

def sshd(network, cmd='/usr/sbin/sshd', opts='-D', ip='10.123.123.1/32', routes=None, switch=None):
    if False:
        for i in range(10):
            print('nop')
    'Start a network, connect it to root ns, and run sshd on all hosts.\n       ip: root-eth0 IP address in root namespace (10.123.123.1/32)\n       routes: Mininet host networks to route to (10.0/24)\n       switch: Mininet switch to connect to root namespace (s1)'
    if not switch:
        switch = network['s1']
    if not routes:
        routes = ['10.0.0.0/24']
    connectToRootNS(network, switch, ip, routes)
    for host in network.hosts:
        host.cmd(cmd + ' ' + opts + '&')
    info('*** Waiting for ssh daemons to start\n')
    for server in network.hosts:
        waitListening(server=server, port=22, timeout=5)
    info('\n*** Hosts are running sshd at the following addresses:\n')
    for host in network.hosts:
        info(host.name, host.IP(), '\n')
    info("\n*** Type 'exit' or control-D to shut down network\n")
    CLI(network)
    for host in network.hosts:
        host.cmd('kill %' + cmd)
    network.stop()
if __name__ == '__main__':
    lg.setLogLevel('info')
    net = TreeNet(depth=1, fanout=4)
    argvopts = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else '-D -o UseDNS=no -u0'
    sshd(net, opts=argvopts)