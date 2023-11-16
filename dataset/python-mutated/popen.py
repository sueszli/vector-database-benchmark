"""
This example monitors a number of hosts using host.popen() and
pmonitor()
"""
from mininet.net import Mininet
from mininet.topo import SingleSwitchTopo
from mininet.log import setLogLevel, info
from mininet.util import pmonitor

def monitorhosts(hosts=5):
    if False:
        return 10
    'Start a bunch of pings and monitor them using popen'
    mytopo = SingleSwitchTopo(hosts)
    net = Mininet(topo=mytopo, waitConnected=True)
    net.start()
    popens = {}
    last = net.hosts[-1]
    for host in net.hosts:
        popens[host] = host.popen('ping -c5 %s' % last.IP())
        last = host
    for (host, line) in pmonitor(popens):
        if host:
            info('<%s>: %s' % (host.name, line))
    net.stop()
if __name__ == '__main__':
    setLogLevel('info')
    monitorhosts(hosts=5)