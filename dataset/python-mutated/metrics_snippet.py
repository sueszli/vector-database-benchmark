from ray import serve
import time
import requests

@serve.deployment
def sleeper():
    if False:
        return 10
    time.sleep(1)
s = sleeper.bind()
serve.run(s)
while True:
    requests.get('http://localhost:8000/')
    break
response = requests.get('http://localhost:8000/')
assert response.status_code == 200