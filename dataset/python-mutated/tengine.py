import itertools
import textwrap
from typing import List, Optional, Tuple
import llnl.util.lang
import spack.config
import spack.extensions
from spack.util.path import canonicalize_path

class ContextMeta(type):
    """Meta class for Context. It helps reducing the boilerplate in
    client code.
    """
    _new_context_properties: List[str] = []

    def __new__(cls, name, bases, attr_dict):
        if False:
            i = 10
            return i + 15
        context_properties = list(cls._new_context_properties)
        for x in bases:
            try:
                context_properties.extend(x.context_properties)
            except AttributeError:
                pass
        context_properties = list(llnl.util.lang.dedupe(context_properties))
        cls._new_context_properties = []
        attr_dict['context_properties'] = context_properties
        return super(ContextMeta, cls).__new__(cls, name, bases, attr_dict)

    @classmethod
    def context_property(cls, func):
        if False:
            while True:
                i = 10
        'Decorator that adds a function name to the list of new context\n        properties, and then returns a property.\n        '
        name = func.__name__
        cls._new_context_properties.append(name)
        return property(func)
context_property = ContextMeta.context_property

class Context(metaclass=ContextMeta):
    """Base class for context classes that are used with the template
    engine.
    """

    def to_dict(self):
        if False:
            print('Hello World!')
        'Returns a dictionary containing all the context properties.'
        d = [(name, getattr(self, name)) for name in self.context_properties]
        return dict(d)

@llnl.util.lang.memoized
def make_environment(dirs: Optional[Tuple[str, ...]]=None):
    if False:
        i = 10
        return i + 15
    'Returns a configured environment for template rendering.'
    import jinja2
    if dirs is None:
        builtins = spack.config.get('config:template_dirs', ['$spack/share/spack/templates'])
        extensions = spack.extensions.get_template_dirs()
        dirs = tuple((canonicalize_path(d) for d in itertools.chain(builtins, extensions)))
    loader = jinja2.FileSystemLoader(dirs)
    env = jinja2.Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)
    _set_filters(env)
    return env

def prepend_to_line(text, token):
    if False:
        print('Hello World!')
    'Prepends a token to each line in text'
    return [token + line for line in text]

def quote(text):
    if False:
        while True:
            i = 10
    'Quotes each line in text'
    return ['"{0}"'.format(line) for line in text]

def curly_quote(text):
    if False:
        i = 10
        return i + 15
    'Encloses each line of text in curly braces'
    return ['{{{0}}}'.format(line) for line in text]

def _set_filters(env):
    if False:
        for i in range(10):
            print('nop')
    'Sets custom filters to the template engine environment'
    env.filters['textwrap'] = textwrap.wrap
    env.filters['prepend_to_line'] = prepend_to_line
    env.filters['join'] = '\n'.join
    env.filters['quote'] = quote
    env.filters['curly_quote'] = curly_quote