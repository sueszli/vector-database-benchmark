import json
import os
_TOP_DIR = os.path.abspath(os.path.sep.join((os.path.dirname(__file__), '../')))
_SAMPLES_DIR = os.path.abspath(os.path.sep.join((os.path.dirname(__file__), '../samples/')))
import sys
sys.path.append(_TOP_DIR)
sys.path.append(_SAMPLES_DIR)
from wiringflask import web

def test_wiring_with_flask():
    if False:
        for i in range(10):
            print('nop')
    client = web.app.test_client()
    with web.app.app_context():
        response = client.get('/')
    assert response.status_code == 200
    assert json.loads(response.data) == {'result': 'OK'}