from google.cloud import firestore
import pytest
import distributed_counters
shards_list = []
doc_ref = None

@pytest.fixture
def fs_client():
    if False:
        print('Hello World!')
    yield firestore.Client()
    for shard in shards_list:
        shard.delete()
    if doc_ref:
        doc_ref.delete()

def test_distributed_counters(fs_client):
    if False:
        while True:
            i = 10
    col = fs_client.collection('dc_samples')
    doc_ref = col.document('distributed_counter')
    counter = distributed_counters.Counter(2)
    counter.init_counter(doc_ref)
    shards = doc_ref.collection('shards').list_documents()
    shards_list = [shard for shard in shards]
    assert len(shards_list) == 2
    counter.increment_counter(doc_ref)
    counter.increment_counter(doc_ref)
    assert counter.get_count(doc_ref) == 2

def test_distributed_counters_cleanup(fs_client):
    if False:
        print('Hello World!')
    col = fs_client.collection('dc_samples')
    doc_ref = col.document('distributed_counter')
    shards = doc_ref.collection('shards').list_documents()
    shards_list = [shard for shard in shards]
    for shard in shards_list:
        shard.delete()
    doc_ref.delete()