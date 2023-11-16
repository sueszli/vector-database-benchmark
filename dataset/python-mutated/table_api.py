import asyncio
from functools import partial
from .dispatch import async_queue, subscribe, unsubscribe
from .view_api import view as make_view

def table(client, data, name, index=None, limit=None):
    if False:
        while True:
            i = 10
    'Create a Perspective `Table` by posting a message to a Perspective\n    server implementation through `client`, returning a `PerspectiveTableProxy`\n    object whose API is entirely async and must be called with `await` or\n    in a `yield`-based generator.'
    options = {}
    if index:
        options['index'] = index
    elif limit:
        options['limit'] = limit
    msg = {'cmd': 'table', 'name': name, 'args': [data], 'options': options}
    future = asyncio.Future()
    client.post(msg, future)
    return future

class PerspectiveTableProxy(object):

    def __init__(self, client, name):
        if False:
            print('Hello World!')
        'A proxy for a Perspective `Table` object elsewhere, i.e. on a remote\n        server accessible through a Websocket.\n\n        All public API methods on this proxy are async, and must be called\n        with `await` or a `yield`-based coroutine.\n\n        Args:\n            client (:obj:`PerspectiveClient`): A `PerspectiveClient` that is\n                set up to send messages to a Perspective server implementation\n                elsewhere.\n\n            name (:obj:`str`): a `str` name for the Table. Automatically\n                generated if using the `table` function defined above.\n        '
        self._client = client
        self._name = name
        self._async_queue = partial(async_queue, self._client, self._name)
        self._subscribe = partial(subscribe, self._client, self._name)
        self._unsubscribe = partial(unsubscribe, self._client, self._name)

    def make_port(self):
        if False:
            print('Hello World!')
        return self._async_queue('make_port', 'table_method')

    def remove_port(self):
        if False:
            i = 10
            return i + 15
        return self._async_queue('remove_port', 'table_method')

    def get_index(self):
        if False:
            print('Hello World!')
        return self._async_queue('get_index', 'table_method')

    def get_limit(self):
        if False:
            return 10
        return self._async_queue('get_limit', 'table_method')

    def get_num_views(self):
        if False:
            while True:
                i = 10
        return self._async_queue('get_num_views', 'table_method')

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        return self._async_queue('clear', 'table_method')

    def replace(self, data):
        if False:
            for i in range(10):
                print('nop')
        return self._async_queue('replace', 'table_method', data)

    def size(self):
        if False:
            i = 10
            return i + 15
        return self._async_queue('size', 'table_method')

    def schema(self, as_string=False):
        if False:
            for i in range(10):
                print('nop')
        return self._async_queue('schema', 'table_method', as_string=as_string)

    def expression_schema(self, expressions, **kwargs):
        if False:
            while True:
                i = 10
        return self._async_queue('expression_schema', 'table_method', expressions, **kwargs)

    def columns(self):
        if False:
            i = 10
            return i + 15
        return self._async_queue('columns', 'table_method')

    def is_valid_filter(self, filter):
        if False:
            return 10
        return self._async_queue('is_valid_filter', 'table_method', filter)

    def on_delete(self, callback):
        if False:
            return 10
        return self._subscribe('on_delete', 'table_method', callback)

    def remove_delete(self, callback):
        if False:
            print('Hello World!')
        return self._unsubscribe('remove_delete', 'table_method', callback)

    def delete(self):
        if False:
            return 10
        return self._async_queue('delete', 'table_method')

    def view(self, columns=None, group_by=None, split_by=None, aggregates=None, sort=None, filter=None, expressions=None):
        if False:
            for i in range(10):
                print('nop')
        return make_view(self._client, self._name, columns, group_by, split_by, aggregates, sort, filter, expressions)

    def update(self, data, port_id=0):
        if False:
            print('Hello World!')
        msg = {'name': self._name, 'cmd': 'table_method', 'method': 'update', 'args': [data, {'port_id': port_id}], 'subscribe': False}
        return self._client.post(msg)

    def remove(self, pkeys, port_id=0):
        if False:
            return 10
        msg = {'name': self._name, 'cmd': 'table_method', 'method': 'remove', 'args': [pkeys, {'port_id': port_id}], 'subscribe': False}
        return self._client.post(msg)