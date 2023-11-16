"""
python_requests.py by xianhu
"""
import requests.adapters
r0 = requests.get('https://github.com/timeline.json')
r1 = requests.post('http://httpbin.org/post')
r2 = requests.put('http://httpbin.org/put')
r3 = requests.delete('http://httpbin.org/delete')
r4 = requests.head('http://httpbin.org/get')
r5 = requests.options('http://httpbin.org/get')
r6 = requests.patch('http://httpbin.org/get')
params = {'key1': 'value1', 'key2': 'value2'}
r = requests.get('http://httpbin.org/get', params=params)
print(r.url)
params = {'key1': 'value1', 'key2': ['value2', 'value3']}
r = requests.get('http://httpbin.org/get', params=params)
print(r.url)
r = requests.get('https://github.com/timeline.json')
print(r.text)
print(r.content)
print(r.json())
print(r.encoding)
r.encoding = 'ISO-8859-1'
print(r.text)
r = requests.get('https://github.com/timeline.json', stream=True)
print(r.raw)
print(r.raw.read(10))
with open('test', 'wb') as fd:
    for chunk in r.iter_content(chunk_size=256):
        fd.write(chunk)
headers = {'user-agent': 'my-app/0.0.1'}
r = requests.get('https://api.github.com/some/endpoint', headers=headers)
print(r.request.headers)
print(r.headers)
print(r.headers['Content-Type'])
print(r.headers.get('content-type'))
post_dict = {'key1': 'value1', 'key2': 'value2'}
r = requests.post('http://httpbin.org/post', data=post_dict)
print(r.text)
files = {'file': open('report.xls', 'rb')}
r = requests.post('http://httpbin.org/post', files=files)
print(r.text)
files = {'file': ('report.xls', open('report.xls', 'rb'), 'application/vnd.ms-excel', {'Expires': '0'})}
r = requests.post('http://httpbin.org/post', files=files)
print(r.text)
files = {'file': ('report.csv', 'some,data,to,send\nanother,row,to,send\n')}
r = requests.post('http://httpbin.org/post', files=files)
print(r.text)
r = requests.get('http://httpbin.org/get')
print(r.status_code)
print(r.status_code == requests.codes.ok)
bad_r = requests.get('http://httpbin.org/status/404')
print(bad_r.status_code)
bad_r.raise_for_status()
r = requests.get('http://example.com/some/cookie/setting/url')
print(r.cookies['example_cookie_name'])
cookies = {'cookies_are': 'working'}
r = requests.get('http://httpbin.org/cookies', cookies=cookies)
print(r.text)
s = requests.Session()
s.get('http://httpbin.org/cookies/set/sessioncookie/123456789')
s.get('http://httpbin.org/cookies')
for cookie in s.cookies:
    print(cookie)
requests.utils.add_dict_to_cookiejar(s.cookies, {'cookie_key': 'cookie_value'})
s.auth = ('user', 'pass')
s.headers.update({'x-test': 'true'})
s.get('http://httpbin.org/headers', headers={'x-test2': 'true'})
s.get('http://httpbin.org/cookies', cookies={'from-my': 'browser'})
s.get('http://httpbin.org/cookies')
with requests.Session() as s:
    s.get('http://httpbin.org/cookies/set/sessioncookie/123456789')
r = requests.get('http://github.com', allow_redirects=True)
print(r.status_code)
print(r.history)
r = requests.get('http://github.com', allow_redirects=False)
print(r.status_code)
print(r.history)
requests.get('http://github.com', timeout=0.001)
requests.get('https://github.com', timeout=5)
requests.get('https://github.com', timeout=(3.05, 27))
requests.get('https://kennethreitz.com', verify=True)
requests.get('https://github.com', verify=True)
requests.get('https://kennethreitz.com', cert=('/path/server.crt', '/path/key'))
requests.get('https://kennethreitz.com', cert='/wrong_path/server.pem')
with open('massive-body') as f:
    requests.post('http://some.url/streamed', data=f)

def print_url(resp):
    if False:
        for i in range(10):
            print('nop')
    print(resp.url)
    return
requests.get('http://httpbin.org', hooks=dict(response=print_url))
proxies = {'http': 'http://10.10.1.10:3128', 'https': 'http://10.10.1.10:1080'}
requests.get('http://example.org', proxies=proxies)
proxies = {'http': 'socks5://user:pass@host:port', 'https': 'socks5://user:pass@host:port'}
requests.get('http://example.org', proxies=proxies)
s = requests.Session()
s.mount('http://baidu.com', requests.adapters.HTTPAdapter())
s.mount('https://', requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100))