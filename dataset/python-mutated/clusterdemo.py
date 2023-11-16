"""clusterdemo.py: demo of Mininet Cluster Edition prototype"""
from mininet.examples.cluster import MininetCluster, SwitchBinPlacer, RemoteLink
from mininet.topolib import TreeTopo
from mininet.log import setLogLevel
from mininet.examples.clustercli import ClusterCLI as CLI

def demo():
    if False:
        for i in range(10):
            print('nop')
    'Simple Demo of Cluster Mode'
    servers = ['localhost', 'ubuntu2', 'ubuntu3']
    topo = TreeTopo(depth=3, fanout=3)
    net = MininetCluster(topo=topo, servers=servers, link=RemoteLink, placement=SwitchBinPlacer)
    net.start()
    CLI(net)
    net.stop()
if __name__ == '__main__':
    setLogLevel('info')
    demo()