import pytest
from gpt_engineer.data.file_repository import FileRepository, FileRepositories

def test_DB_operations(tmp_path):
    if False:
        i = 10
        return i + 15
    db = FileRepository(tmp_path)
    db['test_key'] = 'test_value'
    assert (tmp_path / 'test_key').is_file()
    val = db['test_key']
    assert val == 'test_value'

def test_DBs_initialization(tmp_path):
    if False:
        while True:
            i = 10
    dir_names = ['memory', 'logs', 'preprompts', 'input', 'workspace', 'archive', 'project_metadata']
    directories = [tmp_path / name for name in dir_names]
    dbs = [FileRepository(dir) for dir in directories]
    dbs_instance = FileRepositories(*dbs)
    assert isinstance(dbs_instance.memory, FileRepository)
    assert isinstance(dbs_instance.logs, FileRepository)
    assert isinstance(dbs_instance.preprompts, FileRepository)
    assert isinstance(dbs_instance.input, FileRepository)
    assert isinstance(dbs_instance.workspace, FileRepository)
    assert isinstance(dbs_instance.archive, FileRepository)
    assert isinstance(dbs_instance.project_metadata, FileRepository)

def test_large_files(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    db = FileRepository(tmp_path)
    large_content = 'a' * 10 ** 6
    db['large_file'] = large_content
    assert db['large_file'] == large_content

def test_concurrent_access(tmp_path):
    if False:
        while True:
            i = 10
    import threading
    db = FileRepository(tmp_path)
    num_threads = 10
    num_writes = 1000

    def write_to_db(thread_id):
        if False:
            return 10
        for i in range(num_writes):
            key = f'thread{thread_id}_write{i}'
            db[key] = str(i)
    threads = []
    for thread_id in range(num_threads):
        t = threading.Thread(target=write_to_db, args=(thread_id,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    for thread_id in range(num_threads):
        for i in range(num_writes):
            key = f'thread{thread_id}_write{i}'
            assert key in db
            assert db[key] == str(i)

def test_error_messages(tmp_path):
    if False:
        print('Hello World!')
    db = FileRepository(tmp_path)
    with pytest.raises(KeyError):
        db['non_existent']
    with pytest.raises(AssertionError) as e:
        db['key'] = ['Invalid', 'value']
    assert str(e.value) == 'val must be str'

def test_DBs_dataclass_attributes(tmp_path):
    if False:
        i = 10
        return i + 15
    dir_names = ['memory', 'logs', 'preprompts', 'input', 'workspace', 'archive', 'project_metadata']
    directories = [tmp_path / name for name in dir_names]
    dbs = [FileRepository(dir) for dir in directories]
    dbs_instance = FileRepositories(*dbs)
    assert dbs_instance.memory == dbs[0]
    assert dbs_instance.logs == dbs[1]
    assert dbs_instance.preprompts == dbs[2]
    assert dbs_instance.input == dbs[3]
    assert dbs_instance.workspace == dbs[4]
    assert dbs_instance.archive == dbs[5]
    assert dbs_instance.project_metadata == dbs[6]