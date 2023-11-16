from golem_messages.factories.datastructures import p2p as dt_p2p_factory
from golem import testutils
from golem.network import nodeskeeper

class TestNodesKeeper(testutils.DatabaseFixture):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.node = dt_p2p_factory.Node()

    def test_get(self):
        if False:
            return 10
        nodeskeeper.store(self.node)
        self.assertEqual(self.node, nodeskeeper.get(self.node.key))