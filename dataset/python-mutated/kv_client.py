import time
import httpx

class KVClient:

    def __init__(self, endpoint='localhost:2379'):
        if False:
            while True:
                i = 10
        self.endpoint = endpoint if endpoint.startswith('http://') else f'http://{endpoint}'

    def put(self, key, value):
        if False:
            return 10
        key = key if key.startswith('/') else f'/{key}'
        u = f'{self.endpoint}{key}'
        try:
            r = httpx.post(u, data=value, timeout=None, follow_redirects=True)
            if r.status_code == 200:
                return True
            else:
                return False
        except:
            return False

    def get(self, key):
        if False:
            for i in range(10):
                print('nop')
        key = key if key.startswith('/') else f'/{key}'
        u = f'{self.endpoint}{key}'
        try:
            r = httpx.get(u, timeout=None, follow_redirects=True)
            if r.status_code == 200:
                ret = r.json()
                return ret.get(key, '')
            else:
                return 'error'
        except:
            return ''

    def get_prefix(self, key):
        if False:
            print('Hello World!')
        key = key if key.startswith('/') else f'/{key}'
        u = f'{self.endpoint}{key}'
        try:
            r = httpx.get(u, timeout=None, follow_redirects=True)
            if r.status_code == 200:
                return r.json()
        except:
            return ''

    def delete(self, key):
        if False:
            return 10
        key = key if key.startswith('/') else f'/{key}'
        u = f'{self.endpoint}{key}'
        try:
            r = httpx.delete(u, timeout=None, follow_redirects=True)
            if r.status_code == 200:
                return True
            else:
                return False
        except:
            return False

    def wait_server_ready(self, timeout=3):
        if False:
            print('Hello World!')
        end = time.time() + timeout
        while time.time() < end:
            if self.get('/healthy') == 'ok':
                return True
if __name__ == '__main__':
    cli = KVClient('http://localhost:8090')
    data = {'/workers/1': 'rank1', '/workers/2': 'rank2'}
    for (k, v) in data.items():
        cli.put(k, v)
    x = cli.get_prefix('/workers')
    print(x)
    for (k, v) in data.items():
        assert x[k] == v
    cli.put('key', 'value')
    print(cli.get('key'))
    assert cli.get('key') == 'value'
    cli.delete('key')
    print(cli.get('/key'))
    print(cli.get('/healthy'))
    assert cli.get('/healthy') == 'ok'