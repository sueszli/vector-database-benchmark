from __future__ import annotations
from collections.abc import Collection

class ETags(Collection):
    """A set that can be used to check if one etag is present in a collection
    of etags.
    """

    def __init__(self, strong_etags=None, weak_etags=None, star_tag=False):
        if False:
            print('Hello World!')
        if not star_tag and strong_etags:
            self._strong = frozenset(strong_etags)
        else:
            self._strong = frozenset()
        self._weak = frozenset(weak_etags or ())
        self.star_tag = star_tag

    def as_set(self, include_weak=False):
        if False:
            for i in range(10):
                print('nop')
        'Convert the `ETags` object into a python set.  Per default all the\n        weak etags are not part of this set.'
        rv = set(self._strong)
        if include_weak:
            rv.update(self._weak)
        return rv

    def is_weak(self, etag):
        if False:
            return 10
        'Check if an etag is weak.'
        return etag in self._weak

    def is_strong(self, etag):
        if False:
            i = 10
            return i + 15
        'Check if an etag is strong.'
        return etag in self._strong

    def contains_weak(self, etag):
        if False:
            for i in range(10):
                print('nop')
        'Check if an etag is part of the set including weak and strong tags.'
        return self.is_weak(etag) or self.contains(etag)

    def contains(self, etag):
        if False:
            for i in range(10):
                print('nop')
        'Check if an etag is part of the set ignoring weak tags.\n        It is also possible to use the ``in`` operator.\n        '
        if self.star_tag:
            return True
        return self.is_strong(etag)

    def contains_raw(self, etag):
        if False:
            for i in range(10):
                print('nop')
        'When passed a quoted tag it will check if this tag is part of the\n        set.  If the tag is weak it is checked against weak and strong tags,\n        otherwise strong only.'
        from ..http import unquote_etag
        (etag, weak) = unquote_etag(etag)
        if weak:
            return self.contains_weak(etag)
        return self.contains(etag)

    def to_header(self):
        if False:
            while True:
                i = 10
        'Convert the etags set into a HTTP header string.'
        if self.star_tag:
            return '*'
        return ', '.join([f'"{x}"' for x in self._strong] + [f'W/"{x}"' for x in self._weak])

    def __call__(self, etag=None, data=None, include_weak=False):
        if False:
            print('Hello World!')
        if [etag, data].count(None) != 1:
            raise TypeError('either tag or data required, but at least one')
        if etag is None:
            from ..http import generate_etag
            etag = generate_etag(data)
        if include_weak:
            if etag in self._weak:
                return True
        return etag in self._strong

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self.star_tag or self._strong or self._weak)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.to_header()

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._strong)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self._strong)

    def __contains__(self, etag):
        if False:
            while True:
                i = 10
        return self.contains(etag)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'<{type(self).__name__} {str(self)!r}>'