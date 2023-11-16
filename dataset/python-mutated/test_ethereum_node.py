from unittest.mock import patch, Mock
from unittest import TestCase
from golem.ethereum.node import NodeProcess

class TestPublicNodeList(TestCase):

    def test_node_start(self):
        if False:
            while True:
                i = 10
        node = NodeProcess(['addr1', 'addr2'])
        node.web3 = Mock()
        node.web3.isConnected = Mock()
        node.start()
        node.web3.isConnected.assert_called_once()