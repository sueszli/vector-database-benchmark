import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest
from ipv8.keyvault.private.libnaclkey import LibNaCLSK
from pony.orm import db_session
from tribler.core.components.database.db.layers.knowledge_data_access_layer import ResourceType
from tribler.core.components.database.db.tribler_database import TriblerDatabase
from tribler.core.components.knowledge.rules.knowledge_rules_processor import KnowledgeRulesProcessor
from tribler.core.components.metadata_store.db.serialization import REGULAR_TORRENT
from tribler.core.components.metadata_store.db.store import MetadataStore
from tribler.core.utilities.path_util import Path
from tribler.core.utilities.utilities import MEMORY_DB
TEST_BATCH_SIZE = 100
TEST_INTERVAL = 0.1

@pytest.fixture
async def tag_rules_processor(tmp_path: Path):
    mds = MetadataStore(db_filename=MEMORY_DB, channels_dir=tmp_path, my_key=LibNaCLSK())
    db = TriblerDatabase(filename=':memory:')
    processor = KnowledgeRulesProcessor(notifier=MagicMock(), db=db, mds=mds, batch_size=TEST_BATCH_SIZE, batch_interval=TEST_INTERVAL)
    yield processor
    await processor.shutdown()

def test_constructor(tag_rules_processor: KnowledgeRulesProcessor):
    if False:
        while True:
            i = 10
    assert tag_rules_processor.batch_size == TEST_BATCH_SIZE
    assert tag_rules_processor.batch_interval == TEST_INTERVAL

def test_save_tags(tag_rules_processor: KnowledgeRulesProcessor):
    if False:
        i = 10
        return i + 15
    expected_calls = [{'obj': 'tag2', 'predicate': ResourceType.TAG, 'subject': 'infohash', 'subject_type': ResourceType.TORRENT}, {'obj': 'tag1', 'predicate': ResourceType.TAG, 'subject': 'infohash', 'subject_type': ResourceType.TORRENT}]
    tag_rules_processor.db.add_auto_generated_operation = Mock()
    tag_rules_processor.save_statements(subject_type=ResourceType.TORRENT, subject='infohash', predicate=ResourceType.TAG, objects={'tag1', 'tag2'})
    actual_calls = [c.kwargs for c in tag_rules_processor.db.add_auto_generated_operation.mock_calls]
    assert [c for c in actual_calls if c not in expected_calls] == []

@patch.object(KnowledgeRulesProcessor, 'process_torrent_title', new=AsyncMock(return_value=1))
@patch.object(KnowledgeRulesProcessor, 'cancel_pending_task')
async def test_process_batch(mocked_cancel_pending_task: Mock, tag_rules_processor: KnowledgeRulesProcessor):
    with db_session:
        for _ in range(50):
            tag_rules_processor.mds.TorrentMetadata(infohash=os.urandom(20), metadata_type=REGULAR_TORRENT)
    tag_rules_processor.set_last_processed_torrent_id(10)
    tag_rules_processor.batch_size = 30
    assert await tag_rules_processor.process_batch() == 30
    assert tag_rules_processor.get_last_processed_torrent_id() == 40
    assert not mocked_cancel_pending_task.called
    assert await tag_rules_processor.process_batch() == 10
    assert tag_rules_processor.get_last_processed_torrent_id() == 50
    assert mocked_cancel_pending_task.called

@db_session
@patch.object(KnowledgeRulesProcessor, 'register_task', new=MagicMock())
def test_start_current_version(tag_rules_processor: KnowledgeRulesProcessor):
    if False:
        for i in range(10):
            print('nop')
    tag_rules_processor.set_rules_processor_version(tag_rules_processor.version)
    tag_rules_processor.set_last_processed_torrent_id(100)
    tag_rules_processor.start()
    assert tag_rules_processor.get_rules_processor_version() == tag_rules_processor.version
    assert tag_rules_processor.get_last_processed_torrent_id() == 100

@db_session
@patch.object(KnowledgeRulesProcessor, 'register_task')
def test_start_batch_processing(mocked_register_task: Mock, tag_rules_processor: KnowledgeRulesProcessor):
    if False:
        print('Hello World!')
    tag_rules_processor.mds.TorrentMetadata(infohash=os.urandom(20), metadata_type=REGULAR_TORRENT)
    tag_rules_processor.start()
    assert mocked_register_task.called

def test_add_to_queue(tag_rules_processor: KnowledgeRulesProcessor):
    if False:
        return 10
    'Test that add_to_queue adds the title to the queue'
    tag_rules_processor.put_entity_to_the_queue(b'infohash', 'title')
    assert tag_rules_processor.queue.qsize() == 1
    title = tag_rules_processor.queue.get_nowait()
    assert title.infohash == b'infohash'
    assert title.title == 'title'

def test_add_empty_to_queue(tag_rules_processor: KnowledgeRulesProcessor):
    if False:
        for i in range(10):
            print('nop')
    'Test that add_to_queue does not add the empty title to the queue'
    tag_rules_processor.put_entity_to_the_queue(b'infohash', None)
    assert tag_rules_processor.queue.qsize() == 0

async def test_process_queue(tag_rules_processor: KnowledgeRulesProcessor):
    """Test that process_queue processes the queue"""
    tag_rules_processor.put_entity_to_the_queue(b'infohash', 'title')
    tag_rules_processor.put_entity_to_the_queue(b'infohash2', 'title2')
    tag_rules_processor.put_entity_to_the_queue(b'infohash3', 'title3')
    assert await tag_rules_processor.process_queue() == 3
    assert await tag_rules_processor.process_queue() == 0

async def test_process_queue_out_of_limit(tag_rules_processor: KnowledgeRulesProcessor):
    """Test that process_queue processes the queue by using batch size"""
    tag_rules_processor.queue_batch_size = 2
    tag_rules_processor.put_entity_to_the_queue(b'infohash', 'title')
    tag_rules_processor.put_entity_to_the_queue(b'infohash2', 'title2')
    tag_rules_processor.put_entity_to_the_queue(b'infohash3', 'title3')
    assert await tag_rules_processor.process_queue() == 2
    assert await tag_rules_processor.process_queue() == 1

async def test_put_entity_to_the_queue_out_of_limit(tag_rules_processor: KnowledgeRulesProcessor):
    """ Test that put_entity_to_the_queue does not add the title to the queue if the queue is full"""
    tag_rules_processor.queue.maxsize = 1
    tag_rules_processor.put_entity_to_the_queue(b'infohash', 'title')
    tag_rules_processor.put_entity_to_the_queue(b'infohash2', 'title2')
    assert tag_rules_processor.queue.qsize() == 1