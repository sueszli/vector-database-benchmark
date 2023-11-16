import os
import random
import time
from contextlib import contextmanager
MOCK_LOCUSTFILE_CONTENT = '\n"""This is a mock locust file for unit testing"""\n\nfrom locust import HttpUser, TaskSet, task, between, LoadTestShape\n\n\ndef index(l):\n    l.client.get("/")\n\ndef stats(l):\n    l.client.get("/stats/requests")\n\n\nclass UserTasks(TaskSet):\n    # one can specify tasks like this\n    tasks = [index, stats]\n\n\nclass UserSubclass(HttpUser):\n    host = "http://127.0.0.1:8089"\n    wait_time = between(2, 5)\n    tasks = [UserTasks]\n\n\nclass NotUserSubclass():\n    host = "http://localhost:8000"\n\n'

class MockedLocustfile:
    __slots__ = ['filename', 'directory', 'file_path']

@contextmanager
def mock_locustfile(filename_prefix='mock_locustfile', content=MOCK_LOCUSTFILE_CONTENT, dir=None):
    if False:
        return 10
    mocked = MockedLocustfile()
    mocked.directory = dir or os.path.dirname(os.path.abspath(__file__))
    mocked.filename = '%s_%s_%i.py' % (filename_prefix, str(time.time()).replace('.', '_'), random.randint(0, 100000))
    mocked.file_path = os.path.join(mocked.directory, mocked.filename)
    with open(mocked.file_path, 'w') as file:
        file.write(content)
    try:
        yield mocked
    finally:
        os.remove(mocked.file_path)