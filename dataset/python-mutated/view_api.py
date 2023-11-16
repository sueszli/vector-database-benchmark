import asyncio
from random import random
from functools import partial
from .dispatch import async_queue, subscribe, unsubscribe

def view(client, table_name, columns=None, group_by=None, split_by=None, aggregates=None, sort=None, filter=None, expressions=None):
    if False:
        for i in range(10):
            print('nop')
    'Create a new View by posting a message to the Perspective server\n    implementation through `client`, returning a Future that will resolve to a\n    `PerspectiveViewProxy` object whose API must be called with `await` or\n    `yield`, or an Exception if the View creation failed.\n    '
    name = str(random())
    config = {'columns': columns, 'group_by': group_by, 'split_by': split_by, 'aggregates': aggregates, 'sort': sort, 'filter': filter, 'expressions': expressions}
    msg = {'cmd': 'view', 'view_name': name, 'table_name': table_name, 'config': config}
    future = asyncio.Future()
    client.post(msg, future)
    return future

class PerspectiveViewProxy(object):

    def __init__(self, client, name):
        if False:
            print('Hello World!')
        'A proxy for a Perspective `View` object elsewhere, i.e. on a remote\n        server accessible through a Websocket.\n\n        All public API methods on this proxy are async, and must be called\n        with `await` or a `yield`-based coroutine.\n\n        Args:\n            client (:obj:`PerspectiveClient`): A `PerspectiveClient` that is\n                set up to send messages to a Perspective server implementation\n                elsewhere.\n\n            name (:obj:`str`): a `str` name for the View. Automatically\n                generated if using the `view` function defined above.\n        '
        self._client = client
        self._name = name
        self._async_queue = partial(async_queue, self._client, self._name)
        self._subscribe = partial(subscribe, self._client, self._name)
        self._unsubscribe = partial(unsubscribe, self._client, self._name)

    def get_config(self):
        if False:
            i = 10
            return i + 15
        return self._async_queue('get_config', 'view_method')

    def sides(self):
        if False:
            while True:
                i = 10
        return self._async_queue('sides', 'view_method')

    def num_rows(self):
        if False:
            print('Hello World!')
        return self._async_queue('num_rows', 'view_method')

    def num_columns(self):
        if False:
            return 10
        return self._async_queue('num_columns', 'view_method')

    def get_min_max(self):
        if False:
            return 10
        return self._async_queue('get_min_max', 'view_method')

    def get_row_expanded(self, idx):
        if False:
            print('Hello World!')
        return self._async_queue('get_row_expanded', 'view_method', idx)

    def expand(self, idx):
        if False:
            print('Hello World!')
        return self._async_queue('expand', 'view_method', idx)

    def collapse(self, idx):
        if False:
            for i in range(10):
                print('nop')
        return self._async_queue('collapse', 'view_method', idx)

    def set_depth(self, idx):
        if False:
            print('Hello World!')
        return self._async_queue('set_depth', 'view_method', idx)

    def column_paths(self):
        if False:
            print('Hello World!')
        return self._async_queue('column_paths', 'view_method')

    def schema(self, as_string=False):
        if False:
            return 10
        return self._async_queue('schema', 'view_method', as_string=as_string)

    def expression_schema(self, as_string=False):
        if False:
            i = 10
            return i + 15
        return self._async_queue('expression_schema', 'view_method', as_string=as_string)

    def on_update(self, callback, mode=None):
        if False:
            while True:
                i = 10
        return self._subscribe('on_update', 'view_method', callback, mode=mode)

    def remove_update(self, callback):
        if False:
            return 10
        return self._unsubscribe('remove_update', 'view_method', callback)

    def on_delete(self, callback):
        if False:
            while True:
                i = 10
        return self._subscribe('on_delete', 'view_method', callback)

    def remove_delete(self, callback):
        if False:
            for i in range(10):
                print('nop')
        return self._unsubscribe('remove_delete', 'view_method', callback)

    def delete(self):
        if False:
            for i in range(10):
                print('nop')
        return self._async_queue('delete', 'view_method')

    def to_arrow(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._async_queue('to_arrow', 'view_method', **kwargs)

    def to_records(self, **kwargs):
        if False:
            print('Hello World!')
        return self._async_queue('to_records', 'view_method', **kwargs)

    def to_dict(self, **kwargs):
        if False:
            i = 10
            return i + 15
        return self._async_queue('to_dict', 'view_method', **kwargs)

    def to_numpy(self, **kwargs):
        if False:
            while True:
                i = 10
        return self._async_queue('to_numpy', 'view_method', **kwargs)

    def to_df(self, **kwargs):
        if False:
            i = 10
            return i + 15
        return self._async_queue('to_df', 'view_method', **kwargs)

    def to_csv(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._async_queue('to_csv', 'view_method', **kwargs)

    def to_json(self, **kwargs):
        if False:
            return 10
        return self._async_queue('to_json', 'view_method', **kwargs)

    def to_columns(self, **kwargs):
        if False:
            while True:
                i = 10
        return self._async_queue('to_columns', 'view_method', **kwargs)

    def to_columns_string(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._async_queue('to_columns_string', 'view_method', **kwargs)