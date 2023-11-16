import re
import unicodedata
MANGLE_DELIM = 'X'
normalizes_to_underscore = '_︳︴﹍﹎﹏＿'

def mangle(s):
    if False:
        i = 10
        return i + 15
    'Stringify the argument (with :class:`str`, not :func:`repr` or\n    :hy:func:`hy.repr`) and convert it to a valid Python identifier according\n    to :ref:`Hy\'s mangling rules <mangling>`. ::\n\n        (hy.mangle \'foo-bar)   ; => "foo_bar"\n        (hy.mangle "🦑")       ; => "hyx_XsquidX"\n\n    If the stringified argument is already both legal as a Python identifier\n    and normalized according to Unicode normalization form KC (NFKC), it will\n    be returned unchanged. Thus, ``hy.mangle`` is idempotent. ::\n\n        (setv x \'♦-->♠)\n        (= (hy.mangle (hy.mangle x)) (hy.mangle x))  ; => True\n\n    Generally, the stringifed input is expected to be parsable as a symbol. As\n    a convenience, it can also have the syntax of a :ref:`dotted identifier\n    <dotted-identifiers>`, and ``hy.mangle`` will mangle the dot-delimited\n    parts separately. ::\n\n        (hy.mangle "a.c!.d")  ; => "a.hyx_cXexclamation_markX.d"\n    '
    assert s
    s = str(s)
    if '.' in s and s.strip('.'):
        return '.'.join((mangle(x) if x else '' for x in s.split('.')))
    s2 = s.lstrip(normalizes_to_underscore)
    leading_underscores = '_' * (len(s) - len(s2))
    s = s2
    s = s[0] + s[1:].replace('-', '_') if s else s
    if not (leading_underscores + s).isidentifier():
        s = 'hyx_' + ''.join((c if c != MANGLE_DELIM and ('S' + c).isidentifier() else '{0}{1}{0}'.format(MANGLE_DELIM, unicodedata.name(c, '').lower().replace('-', 'H').replace(' ', '_') or 'U{:x}'.format(ord(c))) for c in s))
    s = leading_underscores + s
    s = unicodedata.normalize('NFKC', s)
    assert s.isidentifier()
    return s

def unmangle(s):
    if False:
        print('Hello World!')
    'Stringify the argument and try to convert it to a pretty unmangled\n    form. See :ref:`Hy\'s mangling rules <mangling>`. ::\n\n        (hy.unmangle "hyx_XsquidX")  ; => "🦑"\n\n    Unmangling may not round-trip, because different Hy symbol names can mangle\n    to the same Python identifier. In particular, Python itself already\n    considers distinct strings that have the same normalized form (according to\n    NFKC), such as ``hello`` and ``𝔥𝔢𝔩𝔩𝔬``, to be the same identifier.\n\n    It\'s an error to call ``hy.unmangle`` on something that looks like a\n    properly mangled name but isn\'t. For example, ``(hy.unmangle\n    "hyx_XpizzazzX")`` is erroneous, because there is no Unicode character\n    named "PIZZAZZ" (yet).'
    s = str(s)
    prefix = ''
    suffix = ''
    m = re.fullmatch('(_+)(.*?)(_*)', s, re.DOTALL)
    if m:
        (prefix, s, suffix) = m.groups()
    if s.startswith('hyx_'):
        s = re.sub('{0}(U)?([_a-z0-9H]+?){0}'.format(MANGLE_DELIM), lambda mo: chr(int(mo.group(2), base=16)) if mo.group(1) else unicodedata.lookup(mo.group(2).replace('_', ' ').replace('H', '-').upper()), s[len('hyx_'):])
    s = s.replace('_', '-')
    return prefix + s + suffix

def slashes2dots(s):
    if False:
        print('Hello World!')
    'Interpret forward slashes as a substitute for periods.'
    return mangle(re.sub('/(-*)', lambda m: '.' + '_' * len(m.group(1)), unmangle(s)))