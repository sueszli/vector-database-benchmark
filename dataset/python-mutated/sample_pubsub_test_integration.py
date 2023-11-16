import base64
import os
import subprocess
import uuid
import requests
from requests.packages.urllib3.util.retry import Retry

def test_print_name():
    if False:
        print('Hello World!')
    name = str(uuid.uuid4())
    port = 8088
    encoded_name = base64.b64encode(name.encode('utf-8')).decode('utf-8')
    pubsub_message = {'data': {'data': encoded_name}}
    process = subprocess.Popen(['functions-framework', '--target', 'hello_pubsub', '--signature-type', 'event', '--port', str(port)], cwd=os.path.dirname(__file__), stdout=subprocess.PIPE)
    url = f'http://localhost:{port}/'
    retry_policy = Retry(total=6, backoff_factor=1)
    retry_adapter = requests.adapters.HTTPAdapter(max_retries=retry_policy)
    session = requests.Session()
    session.mount(url, retry_adapter)
    response = session.post(url, json=pubsub_message)
    assert response.status_code == 200
    process.kill()
    process.wait()
    (out, err) = process.communicate()
    print(out, err, response.content)
    assert f'Hello {name}!' in str(out)