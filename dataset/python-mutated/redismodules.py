from json import JSONDecoder, JSONEncoder

class RedisModuleCommands:
    """This class contains the wrapper functions to bring supported redis
    modules into the command namespace.
    """

    def json(self, encoder=JSONEncoder(), decoder=JSONDecoder()):
        if False:
            i = 10
            return i + 15
        'Access the json namespace, providing support for redis json.'
        from .json import JSON
        jj = JSON(client=self, encoder=encoder, decoder=decoder)
        return jj

    def ft(self, index_name='idx'):
        if False:
            print('Hello World!')
        'Access the search namespace, providing support for redis search.'
        from .search import Search
        s = Search(client=self, index_name=index_name)
        return s

    def ts(self):
        if False:
            print('Hello World!')
        'Access the timeseries namespace, providing support for\n        redis timeseries data.\n        '
        from .timeseries import TimeSeries
        s = TimeSeries(client=self)
        return s

    def bf(self):
        if False:
            while True:
                i = 10
        'Access the bloom namespace.'
        from .bf import BFBloom
        bf = BFBloom(client=self)
        return bf

    def cf(self):
        if False:
            while True:
                i = 10
        'Access the bloom namespace.'
        from .bf import CFBloom
        cf = CFBloom(client=self)
        return cf

    def cms(self):
        if False:
            return 10
        'Access the bloom namespace.'
        from .bf import CMSBloom
        cms = CMSBloom(client=self)
        return cms

    def topk(self):
        if False:
            return 10
        'Access the bloom namespace.'
        from .bf import TOPKBloom
        topk = TOPKBloom(client=self)
        return topk

    def tdigest(self):
        if False:
            print('Hello World!')
        'Access the bloom namespace.'
        from .bf import TDigestBloom
        tdigest = TDigestBloom(client=self)
        return tdigest

    def graph(self, index_name='idx'):
        if False:
            for i in range(10):
                print('nop')
        'Access the graph namespace, providing support for\n        redis graph data.\n        '
        from .graph import Graph
        g = Graph(client=self, name=index_name)
        return g

class AsyncRedisModuleCommands(RedisModuleCommands):

    def ft(self, index_name='idx'):
        if False:
            return 10
        'Access the search namespace, providing support for redis search.'
        from .search import AsyncSearch
        s = AsyncSearch(client=self, index_name=index_name)
        return s

    def graph(self, index_name='idx'):
        if False:
            for i in range(10):
                print('nop')
        'Access the graph namespace, providing support for\n        redis graph data.\n        '
        from .graph import AsyncGraph
        g = AsyncGraph(client=self, name=index_name)
        return g