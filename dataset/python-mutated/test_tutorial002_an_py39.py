import os
from pathlib import Path
from fastapi.testclient import TestClient
from ...utils import needs_py39

@needs_py39
def test():
    if False:
        while True:
            i = 10
    from docs_src.background_tasks.tutorial002_an_py39 import app
    client = TestClient(app)
    log = Path('log.txt')
    if log.is_file():
        os.remove(log)
    response = client.post('/send-notification/foo@example.com?q=some-query')
    assert response.status_code == 200, response.text
    assert response.json() == {'message': 'Message sent'}
    with open('./log.txt') as f:
        assert 'found query: some-query\nmessage to foo@example.com' in f.read()