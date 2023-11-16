"""Python implementation of _read_stanza_*."""
from __future__ import absolute_import
import re
from bzrlib.rio import Stanza
_tag_re = re.compile('^[-a-zA-Z0-9_]+$')

def _valid_tag(tag):
    if False:
        for i in range(10):
            print('nop')
    if type(tag) != str:
        raise TypeError(tag)
    return bool(_tag_re.match(tag))

def _read_stanza_utf8(line_iter):
    if False:
        return 10

    def iter_unicode_lines():
        if False:
            print('Hello World!')
        for line in line_iter:
            if type(line) != str:
                raise TypeError(line)
            yield line.decode('utf-8')
    return _read_stanza_unicode(iter_unicode_lines())

def _read_stanza_unicode(unicode_iter):
    if False:
        i = 10
        return i + 15
    stanza = Stanza()
    tag = None
    accum_value = None
    for line in unicode_iter:
        if line is None or line == u'':
            break
        if line == u'\n':
            break
        real_l = line
        if line[0] == u'\t':
            if tag is None:
                raise ValueError('invalid continuation line %r' % real_l)
            accum_value.append(u'\n' + line[1:-1])
        else:
            if tag is not None:
                stanza.add(tag, u''.join(accum_value))
            try:
                colon_index = line.index(u': ')
            except ValueError:
                raise ValueError('tag/value separator not found in line %r' % real_l)
            tag = str(line[:colon_index])
            if not _valid_tag(tag):
                raise ValueError('invalid rio tag %r' % (tag,))
            accum_value = [line[colon_index + 2:-1]]
    if tag is not None:
        stanza.add(tag, u''.join(accum_value))
        return stanza
    else:
        return None