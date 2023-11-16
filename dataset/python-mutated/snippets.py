import argparse
from collections import defaultdict
import datetime
from pprint import pprint
from google.api_core.client_options import ClientOptions
import google.cloud.exceptions
from google.cloud import datastore

def _preamble():
    if False:
        print('Hello World!')
    from google.cloud import datastore
    client = datastore.Client()
    assert client is not None

def incomplete_key(client):
    if False:
        print('Hello World!')
    key = client.key('Task')
    return key

def named_key(client):
    if False:
        print('Hello World!')
    key = client.key('Task', 'sampleTask')
    return key

def key_with_parent(client):
    if False:
        return 10
    key = client.key('TaskList', 'default', 'Task', 'sampleTask')
    parent_key = client.key('TaskList', 'default')
    key = client.key('Task', 'sampleTask', parent=parent_key)
    return key

def key_with_multilevel_parent(client):
    if False:
        i = 10
        return i + 15
    key = client.key('User', 'alice', 'TaskList', 'default', 'Task', 'sampleTask')
    return key

def basic_entity(client):
    if False:
        while True:
            i = 10
    task = datastore.Entity(client.key('Task'))
    task.update({'category': 'Personal', 'done': False, 'priority': 4, 'description': 'Learn Cloud Datastore'})
    return task

def entity_with_parent(client):
    if False:
        while True:
            i = 10
    key_with_parent = client.key('TaskList', 'default', 'Task', 'sampleTask')
    task = datastore.Entity(key=key_with_parent)
    task.update({'category': 'Personal', 'done': False, 'priority': 4, 'description': 'Learn Cloud Datastore'})
    return task

def properties(client):
    if False:
        return 10
    key = client.key('Task')
    task = datastore.Entity(key, exclude_from_indexes=('description',))
    task.update({'category': 'Personal', 'description': 'Learn Cloud Datastore', 'created': datetime.datetime.now(tz=datetime.timezone.utc), 'done': False, 'priority': 4, 'percent_complete': 10.5})
    client.put(task)
    return task

def array_value(client):
    if False:
        for i in range(10):
            print('nop')
    key = client.key('Task')
    task = datastore.Entity(key)
    task.update({'tags': ['fun', 'programming'], 'collaborators': ['alice', 'bob']})
    return task

def upsert(client):
    if False:
        while True:
            i = 10
    complete_key = client.key('Task', 'sampleTask')
    task = datastore.Entity(key=complete_key)
    task.update({'category': 'Personal', 'done': False, 'priority': 4, 'description': 'Learn Cloud Datastore'})
    client.put(task)
    return task

def insert(client):
    if False:
        i = 10
        return i + 15
    with client.transaction():
        incomplete_key = client.key('Task')
        task = datastore.Entity(key=incomplete_key)
        task.update({'category': 'Personal', 'done': False, 'priority': 4, 'description': 'Learn Cloud Datastore'})
        client.put(task)
    return task

def update(client):
    if False:
        while True:
            i = 10
    upsert(client)
    with client.transaction():
        key = client.key('Task', 'sampleTask')
        task = client.get(key)
        task['done'] = True
        client.put(task)
    return task

def lookup(client):
    if False:
        i = 10
        return i + 15
    upsert(client)
    key = client.key('Task', 'sampleTask')
    task = client.get(key)
    return task

def delete(client):
    if False:
        i = 10
        return i + 15
    upsert(client)
    key = client.key('Task', 'sampleTask')
    client.delete(key)
    return key

def batch_upsert(client):
    if False:
        while True:
            i = 10
    task1 = datastore.Entity(client.key('Task', 1))
    task1.update({'category': 'Personal', 'done': False, 'priority': 4, 'description': 'Learn Cloud Datastore'})
    task2 = datastore.Entity(client.key('Task', 2))
    task2.update({'category': 'Work', 'done': False, 'priority': 8, 'description': 'Integrate Cloud Datastore'})
    client.put_multi([task1, task2])
    return (task1, task2)

def batch_lookup(client):
    if False:
        i = 10
        return i + 15
    batch_upsert(client)
    keys = [client.key('Task', 1), client.key('Task', 2)]
    tasks = client.get_multi(keys)
    return tasks

def batch_delete(client):
    if False:
        return 10
    batch_upsert(client)
    keys = [client.key('Task', 1), client.key('Task', 2)]
    client.delete_multi(keys)
    return keys

def unindexed_property_query(client):
    if False:
        return 10
    upsert(client)
    query = client.query(kind='Task')
    query.add_filter('description', '=', 'Learn Cloud Datastore')
    return list(query.fetch())

def basic_query(client):
    if False:
        while True:
            i = 10
    upsert(client)
    query = client.query(kind='Task')
    query.add_filter('done', '=', False)
    query.add_filter('priority', '>=', 4)
    query.order = ['-priority']
    return list(query.fetch())

def projection_query(client):
    if False:
        while True:
            i = 10
    task = datastore.Entity(client.key('Task'))
    task.update({'category': 'Personal', 'done': False, 'priority': 4, 'description': 'Learn Cloud Datastore', 'percent_complete': 0.5})
    client.put(task)
    query = client.query(kind='Task')
    query.projection = ['priority', 'percent_complete']
    priorities = []
    percent_completes = []
    for task in query.fetch():
        priorities.append(task['priority'])
        percent_completes.append(task['percent_complete'])
    return (priorities, percent_completes)

def ancestor_query(client):
    if False:
        return 10
    task = datastore.Entity(client.key('TaskList', 'default', 'Task'))
    task.update({'category': 'Personal', 'description': 'Learn Cloud Datastore'})
    client.put(task)
    ancestor = client.key('TaskList', 'default')
    query = client.query(kind='Task', ancestor=ancestor)
    return list(query.fetch())

def run_query(client):
    if False:
        i = 10
        return i + 15
    query = client.query()
    results = list(query.fetch())
    return results

def limit(client):
    if False:
        for i in range(10):
            print('nop')
    query = client.query()
    tasks = list(query.fetch(limit=5))
    return tasks

def cursor_paging(client):
    if False:
        return 10

    def get_one_page_of_tasks(cursor=None):
        if False:
            i = 10
            return i + 15
        query = client.query(kind='Task')
        query_iter = query.fetch(start_cursor=cursor, limit=5)
        page = next(query_iter.pages)
        tasks = list(page)
        next_cursor = query_iter.next_page_token
        return (tasks, next_cursor)
    (page_one, cursor_one) = get_one_page_of_tasks()
    (page_two, cursor_two) = get_one_page_of_tasks(cursor=cursor_one)
    return (page_one, cursor_one, page_two, cursor_two)

def property_filter(client):
    if False:
        print('Hello World!')
    upsert(client)
    query = client.query(kind='Task')
    query.add_filter('done', '=', False)
    return list(query.fetch())

def composite_filter(client):
    if False:
        print('Hello World!')
    upsert(client)
    query = client.query(kind='Task')
    query.add_filter('done', '=', False)
    query.add_filter('priority', '=', 4)
    return list(query.fetch())

def key_filter(client):
    if False:
        print('Hello World!')
    upsert(client)
    query = client.query(kind='Task')
    first_key = client.key('Task', 'first_task')
    query.key_filter(first_key, '>')
    return list(query.fetch())

def ascending_sort(client):
    if False:
        print('Hello World!')
    task = upsert(client)
    task['created'] = datetime.datetime.now(tz=datetime.timezone.utc)
    client.put(task)
    query = client.query(kind='Task')
    query.order = ['created']
    return list(query.fetch())

def descending_sort(client):
    if False:
        i = 10
        return i + 15
    task = upsert(client)
    task['created'] = datetime.datetime.now(tz=datetime.timezone.utc)
    client.put(task)
    query = client.query(kind='Task')
    query.order = ['-created']
    return list(query.fetch())

def multi_sort(client):
    if False:
        while True:
            i = 10
    task = upsert(client)
    task['created'] = datetime.datetime.now(tz=datetime.timezone.utc)
    client.put(task)
    query = client.query(kind='Task')
    query.order = ['-priority', 'created']
    return list(query.fetch())

def keys_only_query(client):
    if False:
        for i in range(10):
            print('nop')
    upsert(client)
    query = client.query()
    query.keys_only()
    keys = list([entity.key for entity in query.fetch(limit=10)])
    return keys

def distinct_on_query(client):
    if False:
        print('Hello World!')
    upsert(client)
    query = client.query(kind='Task')
    query.distinct_on = ['category']
    query.order = ['category', 'priority']
    return list(query.fetch())

def kindless_query(client):
    if False:
        for i in range(10):
            print('nop')
    upsert(client)
    last_seen_key = client.key('Task', 'a')
    query = client.query()
    query.key_filter(last_seen_key, '>')
    return list(query.fetch())

def inequality_range(client):
    if False:
        while True:
            i = 10
    start_date = datetime.datetime(1990, 1, 1)
    end_date = datetime.datetime(2000, 1, 1)
    query = client.query(kind='Task')
    query.add_filter('created', '>', start_date)
    query.add_filter('created', '<', end_date)
    return list(query.fetch())

def inequality_invalid(client):
    if False:
        print('Hello World!')
    try:
        start_date = datetime.datetime(1990, 1, 1)
        query = client.query(kind='Task')
        query.add_filter('created', '>', start_date)
        query.add_filter('priority', '>', 3)
        return list(query.fetch())
    except (google.cloud.exceptions.BadRequest, google.cloud.exceptions.GrpcRendezvous):
        pass

def equal_and_inequality_range(client):
    if False:
        for i in range(10):
            print('nop')
    start_date = datetime.datetime(1990, 1, 1)
    end_date = datetime.datetime(2000, 12, 31, 23, 59, 59)
    query = client.query(kind='Task')
    query.add_filter('priority', '=', 4)
    query.add_filter('done', '=', False)
    query.add_filter('created', '>', start_date)
    query.add_filter('created', '<', end_date)
    return list(query.fetch())

def inequality_sort(client):
    if False:
        return 10
    query = client.query(kind='Task')
    query.add_filter('priority', '>', 3)
    query.order = ['priority', 'created']
    return list(query.fetch())

def inequality_sort_invalid_not_same(client):
    if False:
        return 10
    try:
        query = client.query(kind='Task')
        query.add_filter('priority', '>', 3)
        query.order = ['created']
        return list(query.fetch())
    except (google.cloud.exceptions.BadRequest, google.cloud.exceptions.GrpcRendezvous):
        pass

def inequality_sort_invalid_not_first(client):
    if False:
        i = 10
        return i + 15
    try:
        query = client.query(kind='Task')
        query.add_filter('priority', '>', 3)
        query.order = ['created', 'priority']
        return list(query.fetch())
    except (google.cloud.exceptions.BadRequest, google.cloud.exceptions.GrpcRendezvous):
        pass

def array_value_inequality_range(client):
    if False:
        while True:
            i = 10
    query = client.query(kind='Task')
    query.add_filter('tag', '>', 'learn')
    query.add_filter('tag', '<', 'math')
    return list(query.fetch())

def array_value_equality(client):
    if False:
        i = 10
        return i + 15
    query = client.query(kind='Task')
    query.add_filter('tag', '=', 'fun')
    query.add_filter('tag', '=', 'programming')
    return list(query.fetch())

def exploding_properties(client):
    if False:
        for i in range(10):
            print('nop')
    task = datastore.Entity(client.key('Task'))
    task.update({'tags': ['fun', 'programming', 'learn'], 'collaborators': ['alice', 'bob', 'charlie'], 'created': datetime.datetime.now(tz=datetime.timezone.utc)})
    return task

def transactional_update(client):
    if False:
        for i in range(10):
            print('nop')
    account1 = datastore.Entity(client.key('Account'))
    account1['balance'] = 100
    account2 = datastore.Entity(client.key('Account'))
    account2['balance'] = 100
    client.put_multi([account1, account2])

    def transfer_funds(client, from_key, to_key, amount):
        if False:
            while True:
                i = 10
        with client.transaction():
            from_account = client.get(from_key)
            to_account = client.get(to_key)
            from_account['balance'] -= amount
            to_account['balance'] += amount
            client.put_multi([from_account, to_account])
    for _ in range(5):
        try:
            transfer_funds(client, account1.key, account2.key, 50)
            break
        except google.cloud.exceptions.Conflict:
            continue
    else:
        print('Transaction failed.')
    return (account1.key, account2.key)

def transactional_get_or_create(client):
    if False:
        for i in range(10):
            print('nop')
    with client.transaction():
        key = client.key('Task', datetime.datetime.now(tz=datetime.timezone.utc).isoformat())
        task = client.get(key)
        if not task:
            task = datastore.Entity(key)
            task.update({'description': 'Example task'})
            client.put(task)
        return task

def transactional_single_entity_group_read_only(client):
    if False:
        for i in range(10):
            print('nop')
    client.put_multi([datastore.Entity(key=client.key('TaskList', 'default')), datastore.Entity(key=client.key('TaskList', 'default', 'Task', 1))])
    with client.transaction(read_only=True):
        task_list_key = client.key('TaskList', 'default')
        task_list = client.get(task_list_key)
        query = client.query(kind='Task', ancestor=task_list_key)
        tasks_in_list = list(query.fetch())
        return (task_list, tasks_in_list)

def namespace_run_query(client):
    if False:
        i = 10
        return i + 15
    task = datastore.Entity(client.key('Task', 'sample-task', namespace='google'))
    client.put(task)
    query = client.query(kind='__namespace__')
    query.keys_only()
    all_namespaces = [entity.key.id_or_name for entity in query.fetch()]
    start_namespace = client.key('__namespace__', 'g')
    end_namespace = client.key('__namespace__', 'h')
    query = client.query(kind='__namespace__')
    query.key_filter(start_namespace, '>=')
    query.key_filter(end_namespace, '<')
    filtered_namespaces = [entity.key.id_or_name for entity in query.fetch()]
    return (all_namespaces, filtered_namespaces)

def kind_run_query(client):
    if False:
        return 10
    upsert(client)
    query = client.query(kind='__kind__')
    query.keys_only()
    kinds = [entity.key.id_or_name for entity in query.fetch()]
    return kinds

def property_run_query(client):
    if False:
        print('Hello World!')
    upsert(client)
    query = client.query(kind='__property__')
    query.keys_only()
    properties_by_kind = defaultdict(list)
    for entity in query.fetch():
        kind = entity.key.parent.name
        property_ = entity.key.name
        properties_by_kind[kind].append(property_)
    return properties_by_kind

def property_by_kind_run_query(client):
    if False:
        print('Hello World!')
    upsert(client)
    ancestor = client.key('__kind__', 'Task')
    query = client.query(kind='__property__', ancestor=ancestor)
    representations_by_property = {}
    for entity in query.fetch():
        property_name = entity.key.name
        property_types = entity['property_representation']
        representations_by_property[property_name] = property_types
    return representations_by_property

def regional_endpoint():
    if False:
        i = 10
        return i + 15
    ENDPOINT = 'https://nam5-datastore.googleapis.com'
    client_options = ClientOptions(api_endpoint=ENDPOINT)
    client = datastore.Client(client_options=client_options)
    query = client.query(kind='Task')
    results = list(query.fetch())
    for r in results:
        print(r)
    return client

def eventual_consistent_query(client):
    if False:
        print('Hello World!')
    query = client.query(kind='Task')
    query.fetch(eventual=True)

def index_merge_queries(client):
    if False:
        for i in range(10):
            print('nop')
    photo = datastore.Entity(client.key('Photo', 'sample_photo'))
    photo.update({'owner_id': 'user1234', 'size': 2, 'coloration': 2, 'tag': ['family', 'outside', 'camping']})
    client.put(photo)
    queries = []
    query_owner_id = client.query(kind='Photo', filters=[('owner_id', '=', 'user1234')])
    query_size = client.query(kind='Photo', filters=[('size', '=', 2)])
    query_coloration = client.query(kind='Photo', filters=[('coloration', '=', 2)])
    queries.append(query_owner_id)
    queries.append(query_size)
    queries.append(query_coloration)
    query_all_properties = client.query(kind='Photo', filters=[('owner_id', '=', 'user1234'), ('size', '=', 2), ('coloration', '=', 2), ('tag', '=', 'family')])
    queries.append(query_all_properties)
    query_tag = client.query(kind='Photo', filters=[('tag', '=', 'family'), ('tag', '=', 'outside'), ('tag', '=', 'camping')])
    query_owner_size_color_tags = client.query(kind='Photo', filters=[('owner_id', '=', 'user1234'), ('size', '=', 2), ('coloration', '=', 2), ('tag', '=', 'family'), ('tag', '=', 'outside'), ('tag', '=', 'camping')])
    queries.append(query_tag)
    queries.append(query_owner_size_color_tags)
    query_owner_size_tag = client.query(kind='Photo', filters=[('owner_id', '=', 'username'), ('size', '=', 2), ('tag', '=', 'family')])
    queries.append(query_owner_size_tag)
    query_size_coloration = client.query(kind='Photo', filters=[('size', '=', 2), ('coloration', '=', 1)])
    queries.append(query_size_coloration)
    results = []
    for query in queries:
        results.append(query.fetch())
    return results

def main(project_id):
    if False:
        for i in range(10):
            print('nop')
    client = datastore.Client(project_id)
    for (name, function) in globals().items():
        if name in ('main', '_preamble', 'defaultdict') or not callable(function):
            continue
        print(name)
        pprint(function(client))
        print('\n-----------------\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstrates datastore API operations.')
    parser.add_argument('project_id', help='Your cloud project ID.')
    args = parser.parse_args()
    main(args.project_id)