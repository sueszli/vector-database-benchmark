import itertools
import json
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import MutableMapping
from typing import Any
from typing import cast
from typing import TextIO
from mitmproxy import flowfilter
from mitmproxy.http import HTTPFlow

def f_id(x):
    if False:
        while True:
            i = 10
    return x

class URLDict(MutableMapping):
    """Data structure to store information using filters as keys."""

    def __init__(self):
        if False:
            return 10
        self.store: dict[flowfilter.TFilter, Any] = {}

    def __getitem__(self, key, *, count=0):
        if False:
            print('Hello World!')
        if count:
            ret = itertools.islice(self.get_generator(key), 0, count)
        else:
            ret = list(self.get_generator(key))
        if ret:
            return ret
        else:
            raise KeyError

    def __setitem__(self, key: str, value):
        if False:
            print('Hello World!')
        fltr = flowfilter.parse(key)
        if fltr:
            self.store.__setitem__(fltr, value)
        else:
            raise ValueError('Not a valid filter')

    def __delitem__(self, key):
        if False:
            print('Hello World!')
        self.store.__delitem__(key)

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self.store.__iter__()

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.store.__len__()

    def get_generator(self, flow: HTTPFlow) -> Generator[Any, None, None]:
        if False:
            i = 10
            return i + 15
        for (fltr, value) in self.store.items():
            if flowfilter.match(fltr, flow):
                yield value

    def get(self, flow: HTTPFlow, default=None, *, count=0) -> list[Any]:
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.__getitem__(flow, count=count)
        except KeyError:
            return default

    @classmethod
    def _load(cls, json_obj, value_loader: Callable=f_id):
        if False:
            while True:
                i = 10
        url_dict = cls()
        for (fltr, value) in json_obj.items():
            url_dict[fltr] = value_loader(value)
        return url_dict

    @classmethod
    def load(cls, f: TextIO, value_loader: Callable=f_id):
        if False:
            return 10
        json_obj = json.load(f)
        return cls._load(json_obj, value_loader)

    @classmethod
    def loads(cls, json_str: str, value_loader: Callable=f_id):
        if False:
            for i in range(10):
                print('nop')
        json_obj = json.loads(json_str)
        return cls._load(json_obj, value_loader)

    def _dump(self, value_dumper: Callable=f_id) -> dict:
        if False:
            i = 10
            return i + 15
        dumped: dict[flowfilter.TFilter | str, Any] = {}
        for (fltr, value) in self.store.items():
            if hasattr(fltr, 'pattern'):
                dumped[cast(Any, fltr).pattern] = value_dumper(value)
            else:
                dumped[str(fltr)] = value_dumper(value)
        return dumped

    def dump(self, f: TextIO, value_dumper: Callable=f_id):
        if False:
            while True:
                i = 10
        json.dump(self._dump(value_dumper), f)

    def dumps(self, value_dumper: Callable=f_id):
        if False:
            while True:
                i = 10
        return json.dumps(self._dump(value_dumper))