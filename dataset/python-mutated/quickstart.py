def run_quickstart():
    if False:
        print('Hello World!')
    from google.cloud import datastore
    datastore_client = datastore.Client()
    kind = 'Task'
    name = 'sampletask1'
    task_key = datastore_client.key(kind, name)
    task = datastore.Entity(key=task_key)
    task['description'] = 'Buy milk'
    datastore_client.put(task)
    print(f"Saved {task.key.name}: {task['description']}")
if __name__ == '__main__':
    run_quickstart()