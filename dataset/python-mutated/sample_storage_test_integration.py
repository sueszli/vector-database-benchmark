import datetime
import os
import subprocess
import uuid
import requests
from requests.packages.urllib3.util.retry import Retry

def test_print_name():
    if False:
        while True:
            i = 10
    filename = str(uuid.uuid4())
    port = 8089
    example_timestamp = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    storage_message = {'data': {'name': filename, 'bucket': 'my_bucket', 'metageneration': '1', 'timeCreated': example_timestamp, 'updated': example_timestamp}}
    process = subprocess.Popen(['functions-framework', '--target', 'hello_gcs', '--signature-type', 'event', '--port', str(port)], cwd=os.path.dirname(__file__), stdout=subprocess.PIPE)
    url = f'http://localhost:{port}/'
    retry_policy = Retry(total=6, backoff_factor=1)
    retry_adapter = requests.adapters.HTTPAdapter(max_retries=retry_policy)
    session = requests.Session()
    session.mount(url, retry_adapter)
    response = session.post(url, json=storage_message)
    assert response.status_code == 200
    process.kill()
    process.wait()
    (out, err) = process.communicate()
    print(out, err, response.content)
    assert f'File: {filename}' in str(out)