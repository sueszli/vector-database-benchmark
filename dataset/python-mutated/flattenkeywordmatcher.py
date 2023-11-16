from robot.errors import DataError
from robot.model import TagPatterns
from robot.utils import MultiMatcher

def validate_flatten_keyword(options):
    if False:
        return 10
    for opt in options:
        low = opt.lower()
        if low == 'foritem':
            low = 'iteration'
        if not (low in ('for', 'while', 'iteration') or low.startswith('name:') or low.startswith('tag:')):
            raise DataError(f"Expected 'FOR', 'WHILE', 'ITERATION', 'TAG:<pattern>' or 'NAME:<pattern>', got '{opt}'.")

class FlattenByTypeMatcher:

    def __init__(self, flatten):
        if False:
            return 10
        if isinstance(flatten, str):
            flatten = [flatten]
        flatten = [f.lower() for f in flatten]
        self.types = set()
        if 'for' in flatten:
            self.types.add('for')
        if 'while' in flatten:
            self.types.add('while')
        if 'iteration' in flatten or 'foritem' in flatten:
            self.types.add('iter')

    def match(self, tag):
        if False:
            return 10
        return tag in self.types

    def __bool__(self):
        if False:
            print('Hello World!')
        return bool(self.types)

class FlattenByNameMatcher:

    def __init__(self, flatten):
        if False:
            i = 10
            return i + 15
        if isinstance(flatten, str):
            flatten = [flatten]
        names = [n[5:] for n in flatten if n[:5].lower() == 'name:']
        self._matcher = MultiMatcher(names)

    def match(self, name, owner=None):
        if False:
            print('Hello World!')
        name = f'{owner}.{name}' if owner else name
        return self._matcher.match(name)

    def __bool__(self):
        if False:
            i = 10
            return i + 15
        return bool(self._matcher)

class FlattenByTagMatcher:

    def __init__(self, flatten):
        if False:
            while True:
                i = 10
        if isinstance(flatten, str):
            flatten = [flatten]
        patterns = [p[4:] for p in flatten if p[:4].lower() == 'tag:']
        self._matcher = TagPatterns(patterns)

    def match(self, tags):
        if False:
            for i in range(10):
                print('nop')
        return self._matcher.match(tags)

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self._matcher)