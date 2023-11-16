"""
A sanity check for cluster edition
"""
from mininet.examples.cluster import MininetCluster
from mininet.log import setLogLevel
from mininet.examples.clustercli import ClusterCLI as CLI
from mininet.topo import SingleSwitchTopo

def clusterSanity():
    if False:
        while True:
            i = 10
    'Sanity check for cluster mode'
    topo = SingleSwitchTopo()
    net = MininetCluster(topo=topo)
    net.start()
    CLI(net)
    net.stop()
if __name__ == '__main__':
    setLogLevel('info')
    clusterSanity()