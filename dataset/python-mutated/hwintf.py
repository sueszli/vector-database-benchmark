"""
This example shows how to add an interface (for example a real
hardware interface) to a network after the network is created.
"""
import re
import sys
from sys import exit
from mininet.cli import CLI
from mininet.log import setLogLevel, info, error
from mininet.net import Mininet
from mininet.link import Intf
from mininet.topolib import TreeTopo
from mininet.util import quietRun

def checkIntf(intf):
    if False:
        return 10
    'Make sure intf exists and is not configured.'
    config = quietRun('ifconfig %s 2>/dev/null' % intf, shell=True)
    if not config:
        error('Error:', intf, 'does not exist!\n')
        exit(1)
    ips = re.findall('\\d+\\.\\d+\\.\\d+\\.\\d+', config)
    if ips:
        error('Error:', intf, 'has an IP address,and is probably in use!\n')
        exit(1)
if __name__ == '__main__':
    setLogLevel('info')
    intfName = sys.argv[1] if len(sys.argv) > 1 else 'eth1'
    info('*** Connecting to hw intf: %s' % intfName)
    info('*** Checking', intfName, '\n')
    checkIntf(intfName)
    info('*** Creating network\n')
    net = Mininet(topo=TreeTopo(depth=1, fanout=2), waitConnected=True)
    switch = net.switches[0]
    info('*** Adding hardware interface', intfName, 'to switch', switch.name, '\n')
    _intf = Intf(intfName, node=switch)
    info('*** Note: you may need to reconfigure the interfaces for the Mininet hosts:\n', net.hosts, '\n')
    net.start()
    CLI(net)
    net.stop()