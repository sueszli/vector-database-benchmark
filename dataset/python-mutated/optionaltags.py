from __future__ import absolute_import, division, unicode_literals
from . import base

class Filter(base.Filter):
    """Removes optional tags from the token stream"""

    def slider(self):
        if False:
            i = 10
            return i + 15
        previous1 = previous2 = None
        for token in self.source:
            if previous1 is not None:
                yield (previous2, previous1, token)
            previous2 = previous1
            previous1 = token
        if previous1 is not None:
            yield (previous2, previous1, None)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        for (previous, token, next) in self.slider():
            type = token['type']
            if type == 'StartTag':
                if token['data'] or not self.is_optional_start(token['name'], previous, next):
                    yield token
            elif type == 'EndTag':
                if not self.is_optional_end(token['name'], next):
                    yield token
            else:
                yield token

    def is_optional_start(self, tagname, previous, next):
        if False:
            while True:
                i = 10
        type = next and next['type'] or None
        if tagname in 'html':
            return type not in ('Comment', 'SpaceCharacters')
        elif tagname == 'head':
            if type in ('StartTag', 'EmptyTag'):
                return True
            elif type == 'EndTag':
                return next['name'] == 'head'
        elif tagname == 'body':
            if type in ('Comment', 'SpaceCharacters'):
                return False
            elif type == 'StartTag':
                return next['name'] not in ('script', 'style')
            else:
                return True
        elif tagname == 'colgroup':
            if type in ('StartTag', 'EmptyTag'):
                return next['name'] == 'col'
            else:
                return False
        elif tagname == 'tbody':
            if type == 'StartTag':
                if previous and previous['type'] == 'EndTag' and (previous['name'] in ('tbody', 'thead', 'tfoot')):
                    return False
                return next['name'] == 'tr'
            else:
                return False
        return False

    def is_optional_end(self, tagname, next):
        if False:
            while True:
                i = 10
        type = next and next['type'] or None
        if tagname in ('html', 'head', 'body'):
            return type not in ('Comment', 'SpaceCharacters')
        elif tagname in ('li', 'optgroup', 'tr'):
            if type == 'StartTag':
                return next['name'] == tagname
            else:
                return type == 'EndTag' or type is None
        elif tagname in ('dt', 'dd'):
            if type == 'StartTag':
                return next['name'] in ('dt', 'dd')
            elif tagname == 'dd':
                return type == 'EndTag' or type is None
            else:
                return False
        elif tagname == 'p':
            if type in ('StartTag', 'EmptyTag'):
                return next['name'] in ('address', 'article', 'aside', 'blockquote', 'datagrid', 'dialog', 'dir', 'div', 'dl', 'fieldset', 'footer', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header', 'hr', 'menu', 'nav', 'ol', 'p', 'pre', 'section', 'table', 'ul')
            else:
                return type == 'EndTag' or type is None
        elif tagname == 'option':
            if type == 'StartTag':
                return next['name'] in ('option', 'optgroup')
            else:
                return type == 'EndTag' or type is None
        elif tagname in ('rt', 'rp'):
            if type == 'StartTag':
                return next['name'] in ('rt', 'rp')
            else:
                return type == 'EndTag' or type is None
        elif tagname == 'colgroup':
            if type in ('Comment', 'SpaceCharacters'):
                return False
            elif type == 'StartTag':
                return next['name'] != 'colgroup'
            else:
                return True
        elif tagname in ('thead', 'tbody'):
            if type == 'StartTag':
                return next['name'] in ['tbody', 'tfoot']
            elif tagname == 'tbody':
                return type == 'EndTag' or type is None
            else:
                return False
        elif tagname == 'tfoot':
            if type == 'StartTag':
                return next['name'] == 'tbody'
            else:
                return type == 'EndTag' or type is None
        elif tagname in ('td', 'th'):
            if type == 'StartTag':
                return next['name'] in ('td', 'th')
            else:
                return type == 'EndTag' or type is None
        return False