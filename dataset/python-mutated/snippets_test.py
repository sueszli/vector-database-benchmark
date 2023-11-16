import os
import backoff
from google.api_core.retry import Retry
from google.cloud import datastore
import pytest
import snippets
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
retry_policy = Retry()

class CleanupClient(datastore.Client):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.entities_to_delete = []
        self.keys_to_delete = []

    def cleanup(self):
        if False:
            for i in range(10):
                print('nop')
        batch = self.batch()
        batch.begin()
        self.delete_multi(list({x.key for x in self.entities_to_delete if x}) + list(set(self.keys_to_delete)))
        batch.commit(retry=retry_policy)

@pytest.fixture
def client():
    if False:
        print('Hello World!')
    client = CleanupClient(PROJECT)
    yield client
    client.cleanup()

@pytest.mark.flaky
class TestDatastoreSnippets:

    def test_incomplete_key(self, client):
        if False:
            print('Hello World!')
        assert snippets.incomplete_key(client)

    def test_named_key(self, client):
        if False:
            print('Hello World!')
        key = snippets.named_key(client)
        assert key
        assert key.name == 'sampleTask'

    def test_key_with_parent(self, client):
        if False:
            i = 10
            return i + 15
        key = snippets.key_with_parent(client)
        assert key
        assert key.name == 'sampleTask'
        assert key.parent.name == 'default'

    def test_key_with_multilevel_parent(self, client):
        if False:
            while True:
                i = 10
        key = snippets.key_with_multilevel_parent(client)
        assert key
        assert key.name == 'sampleTask'
        assert key.parent.name == 'default'
        assert key.parent.parent.name == 'alice'

    def test_basic_entity(self, client):
        if False:
            for i in range(10):
                print('nop')
        assert snippets.basic_entity(client)

    def test_entity_with_parent(self, client):
        if False:
            return 10
        task = snippets.entity_with_parent(client)
        assert task
        assert task.key.name == 'sampleTask'
        assert snippets.entity_with_parent(client)

    def test_properties(self, client):
        if False:
            i = 10
            return i + 15
        assert snippets.properties(client)

    def test_array_value(self, client):
        if False:
            i = 10
            return i + 15
        assert snippets.array_value(client)

    def test_upsert(self, client):
        if False:
            for i in range(10):
                print('nop')
        task = snippets.upsert(client)
        client.entities_to_delete.append(task)
        assert task
        assert task.key.name == 'sampleTask'

    def test_insert(self, client):
        if False:
            i = 10
            return i + 15
        task = snippets.insert(client)
        client.entities_to_delete.append(task)
        assert task

    def test_update(self, client):
        if False:
            i = 10
            return i + 15
        task = snippets.insert(client)
        client.entities_to_delete.append(task)
        assert task

    def test_lookup(self, client):
        if False:
            return 10
        task = snippets.lookup(client)
        client.entities_to_delete.append(task)
        assert task
        assert task.key.name == 'sampleTask'

    def test_delete(self, client):
        if False:
            while True:
                i = 10
        snippets.delete(client)

    def test_batch_upsert(self, client):
        if False:
            return 10
        tasks = snippets.batch_upsert(client)
        client.entities_to_delete.extend(tasks)
        assert tasks

    def test_batch_lookup(self, client):
        if False:
            return 10
        tasks = snippets.batch_lookup(client)
        client.entities_to_delete.extend(tasks)
        assert tasks

    def test_batch_delete(self, client):
        if False:
            while True:
                i = 10
        snippets.batch_delete(client)

    @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
    def test_unindexed_property_query(self, client):
        if False:
            i = 10
            return i + 15
        tasks = snippets.unindexed_property_query(client)
        client.entities_to_delete.extend(tasks)
        assert tasks

    @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
    def test_basic_query(self, client):
        if False:
            i = 10
            return i + 15
        tasks = snippets.basic_query(client)
        client.entities_to_delete.extend(tasks)
        assert tasks

    @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
    def test_projection_query(self, client):
        if False:
            return 10
        (priorities, percents) = snippets.projection_query(client)
        client.entities_to_delete.extend(client.query(kind='Task').fetch())
        assert priorities
        assert percents

    def test_ancestor_query(self, client):
        if False:
            for i in range(10):
                print('nop')
        tasks = snippets.ancestor_query(client)
        client.entities_to_delete.extend(tasks)
        assert tasks

    def test_run_query(self, client):
        if False:
            print('Hello World!')
        snippets.run_query(client)

    def test_cursor_paging(self, client):
        if False:
            while True:
                i = 10
        for n in range(6):
            client.entities_to_delete.append(snippets.insert(client))

        @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
        def run_sample():
            if False:
                for i in range(10):
                    print('nop')
            results = snippets.cursor_paging(client)
            (page_one, cursor_one, page_two, cursor_two) = results
            assert len(page_one) == 5
            assert len(page_two)
            assert cursor_one
        run_sample()

    @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
    def test_property_filter(self, client):
        if False:
            return 10
        tasks = snippets.property_filter(client)
        client.entities_to_delete.extend(tasks)
        assert tasks

    @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
    def test_composite_filter(self, client):
        if False:
            print('Hello World!')
        tasks = snippets.composite_filter(client)
        client.entities_to_delete.extend(tasks)
        assert tasks

    @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
    def test_key_filter(self, client):
        if False:
            i = 10
            return i + 15
        tasks = snippets.key_filter(client)
        client.entities_to_delete.extend(tasks)
        assert tasks

    @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
    def test_ascending_sort(self, client):
        if False:
            for i in range(10):
                print('nop')
        tasks = snippets.ascending_sort(client)
        client.entities_to_delete.extend(tasks)
        assert tasks

    @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
    def test_descending_sort(self, client):
        if False:
            print('Hello World!')
        tasks = snippets.descending_sort(client)
        client.entities_to_delete.extend(tasks)
        assert tasks

    @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
    def test_multi_sort(self, client):
        if False:
            while True:
                i = 10
        tasks = snippets.multi_sort(client)
        client.entities_to_delete.extend(tasks)
        assert tasks

    @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
    def test_keys_only_query(self, client):
        if False:
            i = 10
            return i + 15
        keys = snippets.keys_only_query(client)
        client.entities_to_delete.extend(client.query(kind='Task').fetch())
        assert keys

    @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
    def test_distinct_on_query(self, client):
        if False:
            return 10
        tasks = snippets.distinct_on_query(client)
        client.entities_to_delete.extend(tasks)
        assert tasks

    def test_kindless_query(self, client):
        if False:
            for i in range(10):
                print('nop')
        tasks = snippets.kindless_query(client)
        assert tasks

    def test_inequality_range(self, client):
        if False:
            i = 10
            return i + 15
        snippets.inequality_range(client)

    def test_inequality_invalid(self, client):
        if False:
            while True:
                i = 10
        snippets.inequality_invalid(client)

    def test_equal_and_inequality_range(self, client):
        if False:
            while True:
                i = 10
        snippets.equal_and_inequality_range(client)

    def test_inequality_sort(self, client):
        if False:
            for i in range(10):
                print('nop')
        snippets.inequality_sort(client)

    def test_inequality_sort_invalid_not_same(self, client):
        if False:
            print('Hello World!')
        snippets.inequality_sort_invalid_not_same(client)

    def test_inequality_sort_invalid_not_first(self, client):
        if False:
            while True:
                i = 10
        snippets.inequality_sort_invalid_not_first(client)

    def test_array_value_inequality_range(self, client):
        if False:
            print('Hello World!')
        snippets.array_value_inequality_range(client)

    def test_array_value_equality(self, client):
        if False:
            print('Hello World!')
        snippets.array_value_equality(client)

    def test_exploding_properties(self, client):
        if False:
            print('Hello World!')
        task = snippets.exploding_properties(client)
        assert task

    def test_transactional_update(self, client):
        if False:
            i = 10
            return i + 15
        keys = snippets.transactional_update(client)
        client.keys_to_delete.extend(keys)

    def test_transactional_get_or_create(self, client):
        if False:
            print('Hello World!')
        task = snippets.transactional_get_or_create(client)
        client.entities_to_delete.append(task)
        assert task

    def transactional_single_entity_group_read_only(self, client):
        if False:
            i = 10
            return i + 15
        (task_list, tasks_in_list) = snippets.transactional_single_entity_group_read_only(client)
        client.entities_to_delete.append(task_list)
        client.entities_to_delete.extend(tasks_in_list)
        assert task_list
        assert tasks_in_list

    @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
    def test_namespace_run_query(self, client):
        if False:
            for i in range(10):
                print('nop')
        (all_namespaces, filtered_namespaces) = snippets.namespace_run_query(client)
        assert all_namespaces
        assert filtered_namespaces
        assert 'google' in filtered_namespaces

    @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
    def test_kind_run_query(self, client):
        if False:
            while True:
                i = 10
        kinds = snippets.kind_run_query(client)
        client.entities_to_delete.extend(client.query(kind='Task').fetch())
        assert kinds
        assert 'Task' in kinds

    @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
    def test_property_run_query(self, client):
        if False:
            i = 10
            return i + 15
        kinds = snippets.property_run_query(client)
        client.entities_to_delete.extend(client.query(kind='Task').fetch())
        assert kinds
        assert 'Task' in kinds

    @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
    def test_property_by_kind_run_query(self, client):
        if False:
            for i in range(10):
                print('nop')
        reprs = snippets.property_by_kind_run_query(client)
        client.entities_to_delete.extend(client.query(kind='Task').fetch())
        assert reprs

    @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
    def test_index_merge_queries(self, client):
        if False:
            return 10
        snippets.index_merge_queries(client)

    @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
    def test_regional_endpoint(self, client):
        if False:
            for i in range(10):
                print('nop')
        client = snippets.regional_endpoint()
        assert client is not None