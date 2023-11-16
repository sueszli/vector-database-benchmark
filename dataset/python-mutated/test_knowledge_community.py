import datetime
from unittest.mock import MagicMock, Mock
from ipv8.keyvault.private.libnaclkey import LibNaCLSK
from pony.orm import db_session
from tribler.core.components.database.db.layers.knowledge_data_access_layer import Operation, ResourceType
from tribler.core.components.ipv8.adapters_tests import TriblerMockIPv8, TriblerTestBase
from tribler.core.components.knowledge.community.knowledge_community import KnowledgeCommunity
from tribler.core.components.knowledge.community.knowledge_payload import StatementOperation
from tribler.core.components.database.db.tribler_database import TriblerDatabase
REQUEST_INTERVAL_FOR_RANDOM_OPERATIONS = 0.1

class TestKnowledgeCommunity(TriblerTestBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.initialize(KnowledgeCommunity, 2)

    async def tearDown(self):
        await super().tearDown()

    def create_node(self, *args, **kwargs):
        if False:
            return 10
        return TriblerMockIPv8('curve25519', KnowledgeCommunity, db=TriblerDatabase(), key=LibNaCLSK(), request_interval=REQUEST_INTERVAL_FOR_RANDOM_OPERATIONS)

    def create_operation(self, subject='1' * 20, obj=''):
        if False:
            for i in range(10):
                print('nop')
        community = self.overlay(0)
        operation = StatementOperation(subject_type=ResourceType.TORRENT, subject=subject, predicate=ResourceType.TAG, object=obj, operation=Operation.ADD, clock=0, creator_public_key=community.key.pub().key_to_bin())
        operation.clock = community.db.knowledge.get_clock(operation) + 1
        return operation

    @db_session
    def fill_db(self):
        if False:
            i = 10
            return i + 15
        community = self.overlay(0)
        for i in range(10):
            message = self.create_operation(obj=f'{i}' * 3)
            signature = community.sign(message)
            if i >= 5:
                signature = b'1' * 64
            community.db.knowledge.add_operation(message, signature)
        cyrillic_message = self.create_operation(subject='Контент', obj='Тэг')
        community.db.knowledge.add_operation(cyrillic_message, community.sign(cyrillic_message))
        for op in community.db.instance.StatementOp.select():
            op.set(updated_at=datetime.datetime.utcnow() - datetime.timedelta(minutes=2))

    async def test_gossip(self):
        self.fill_db()
        await self.introduce_nodes()
        await self.deliver_messages(timeout=REQUEST_INTERVAL_FOR_RANDOM_OPERATIONS * 2)
        with db_session:
            assert self.overlay(0).db.instance.StatementOp.select().count() == 11
            assert self.overlay(1).db.instance.StatementOp.select().count() == 6

    async def test_on_request_eat_exceptions(self):
        self.fill_db()
        self.overlay(0).db.knowledge.get_operations_for_gossip = Mock(return_value=[MagicMock()])
        await self.introduce_nodes()
        await self.deliver_messages(timeout=REQUEST_INTERVAL_FOR_RANDOM_OPERATIONS * 2)
        self.overlay(0).db.knowledge.get_operations_for_gossip.assert_called()

    async def test_no_peers(self):
        self.overlay(0).get_peers = Mock(return_value=[])
        self.fill_db()
        await self.introduce_nodes()
        await self.deliver_messages(timeout=REQUEST_INTERVAL_FOR_RANDOM_OPERATIONS * 2)
        self.overlay(0).get_peers.assert_called()