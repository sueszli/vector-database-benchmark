import re
from dataclasses import dataclass
from typing import List, Optional

class Http:
    """A resource accessed via HTTP."""

    def __init__(self, hostname='', path='', query=None):
        if False:
            return 10
        if query is None:
            query = {}
        self.hostname = hostname
        self.path = path
        self.query = query

    def __repr__(self):
        if False:
            return 10
        return str(self)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        q = dict(self.query.items())
        host_str = f'hostname="{self.hostname}"' if self.hostname else None
        path_str = f'path="{self.path}"' if self.path != '' else None
        query_str = f'query={q}' if q != {} else None
        field_str = ', '.join((x for x in [host_str, path_str, query_str] if x))
        return f'Http({field_str})'

class PathMapper:
    """Map from a template string with capture groups of the form
    ``{name}`` to a dictionary of the form ``{name: captured_value}``

    :param template: the template string to match against
    """

    def __init__(self, template):
        if False:
            return 10
        capture_group = re.compile('({([^}]+)})')
        for (outer, inner) in capture_group.findall(template):
            if inner == '*':
                template = template.replace(outer, '.*')
            else:
                template = template.replace(outer, f'(?P<{inner}>[^/]+)')
        self.pattern = re.compile(f'^{template}$')

    def map(self, string):
        if False:
            print('Hello World!')
        match = self.pattern.match(string)
        if match:
            return match.groupdict()
actors = {'guest': '1', 'president': '1'}
frobbed: List[str] = []

def get_frobbed():
    if False:
        for i in range(10):
            print('nop')
    global frobbed
    return frobbed

def set_frobbed(f):
    if False:
        return 10
    global frobbed
    frobbed = f

class Widget:
    id: str = ''
    name: str = ''
    actions = ('get', 'create')

    def __init__(self, id='', name=''):
        if False:
            i = 10
            return i + 15
        self.id = id
        self.name = name

    def company(self):
        if False:
            print('Hello World!')
        return Company(id=self.id)

    def frob(self, what):
        if False:
            while True:
                i = 10
        global frobbed
        frobbed.append(what)
        return self

class DooDad(Widget):
    pass

@dataclass
class User:
    name: str = ''
    id: int = 0
    widget: Optional[Widget] = None

    def companies(self):
        if False:
            for i in range(10):
                print('nop')
        yield Company(id='0')
        yield Company(id=actors[self.name])

    def groups(self):
        if False:
            return 10
        return ['social', 'dev', 'product']

    def companies_iter(self):
        if False:
            return 10
        return iter([Company(id='acme'), Company(id='Initech')])

@dataclass(frozen=True)
class Company:
    id: str = ''
    default_role: str = ''

    def role(self, actor: User):
        if False:
            for i in range(10):
                print('nop')
        if actor.name == 'president':
            return 'admin'
        else:
            return 'guest'

    def roles(self):
        if False:
            i = 10
            return i + 15
        yield 'guest'
        yield 'admin'