from __future__ import annotations
from collections.abc import Iterable
import requests
url = 'https://httpbin.org/post'
multiple_files = [('images', ('foo.png', open('foo.png', 'rb'), 'image/png')), ('images', ('bar.png', open('bar.png', 'rb'), 'image/png'))]
r = requests.post(url, files=multiple_files)

def gen() -> Iterable[bytes]:
    if False:
        for i in range(10):
            print('nop')
    yield b'foo'
    yield b'bar'
requests.post('http://httpbin.org/anything', data=gen()).json()['data']
requests.post('http://httpbin.org/anything', data=b'foobar').json()['data']
requests.post('http://httpbin.org/anything', data='foobar').json()['data']
requests.post('http://httpbin.org/anything', data=open('/tmp/foobar', 'rb', encoding='UTF-8')).json()['data']
requests.post('http://httpbin.org/anything', data=open('/tmp/foobar', 'r', encoding='UTF-8')).json()['data']
requests.post('http://httpbin.org/anything', data={b'foo': b'bar'}).json()['form']
requests.post('http://httpbin.org/anything', data={'foo': 'bar'}).json()['form']
requests.post('http://httpbin.org/anything', data=[(b'foo', b'bar')]).json()['form']
requests.post('http://httpbin.org/anything', data=[('foo', 'bar')]).json()['form']
requests.post('http://httpbin.org/anything', data=((b'foo', b'bar'),)).json()['form']
requests.post('http://httpbin.org/anything', data=(('foo', 'bar'),)).json()['form']