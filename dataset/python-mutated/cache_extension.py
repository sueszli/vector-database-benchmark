from jinja2 import nodes
from jinja2.ext import Extension

class FragmentCacheExtension(Extension):
    tags = {'cache'}

    def __init__(self, environment):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(environment)
        environment.extend(fragment_cache_prefix='', fragment_cache=None)

    def parse(self, parser):
        if False:
            i = 10
            return i + 15
        lineno = next(parser.stream).lineno
        args = [parser.parse_expression()]
        if parser.stream.skip_if('comma'):
            args.append(parser.parse_expression())
        else:
            args.append(nodes.Const(None))
        body = parser.parse_statements(['name:endcache'], drop_needle=True)
        return nodes.CallBlock(self.call_method('_cache_support', args), [], [], body).set_lineno(lineno)

    def _cache_support(self, name, timeout, caller):
        if False:
            i = 10
            return i + 15
        'Helper callback.'
        key = self.environment.fragment_cache_prefix + name
        rv = self.environment.fragment_cache.get(key)
        if rv is not None:
            return rv
        rv = caller()
        self.environment.fragment_cache.add(key, rv, timeout)
        return rv