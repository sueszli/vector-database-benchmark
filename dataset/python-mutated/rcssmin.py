"""
==============
 CSS Minifier
==============

CSS Minifier.

The minifier is based on the semantics of the `YUI compressor`_\\, which itself
is based on `the rule list by Isaac Schlueter`_\\.

This module is a re-implementation aiming for speed instead of maximum
compression, so it can be used at runtime (rather than during a preprocessing
step). RCSSmin does syntactical compression only (removing spaces, comments
and possibly semicolons). It does not provide semantic compression (like
removing empty blocks, collapsing redundant properties etc). It does, however,
support various CSS hacks (by keeping them working as intended).

Here's a feature list:

- Strings are kept, except that escaped newlines are stripped
- Space/Comments before the very end or before various characters are
  stripped: ``:{});=>+],!`` (The colon (``:``) is a special case, a single
  space is kept if it's outside a ruleset.)
- Space/Comments at the very beginning or after various characters are
  stripped: ``{}(=:>+[,!``
- Optional space after unicode escapes is kept, resp. replaced by a simple
  space
- whitespaces inside ``url()`` definitions are stripped
- Comments starting with an exclamation mark (``!``) can be kept optionally.
- All other comments and/or whitespace characters are replaced by a single
  space.
- Multiple consecutive semicolons are reduced to one
- The last semicolon within a ruleset is stripped
- CSS Hacks supported:

  - IE7 hack (``>/**/``)
  - Mac-IE5 hack (``/*\\*/.../**/``)
  - The boxmodelhack is supported naturally because it relies on valid CSS2
    strings
  - Between ``:first-line`` and the following comma or curly brace a space is
    inserted. (apparently it's needed for IE6)
  - Same for ``:first-letter``

rcssmin.c is a reimplementation of rcssmin.py in C and improves runtime up to
factor 50 or so (depending on the input).

Both python 2 (>= 2.4) and python 3 are supported.

.. _YUI compressor: https://github.com/yui/yuicompressor/

.. _the rule list by Isaac Schlueter: https://github.com/isaacs/cssmin/tree/
"""
__author__ = 'André Malo'
__author__ = getattr(__author__, 'decode', lambda x: __author__)('latin-1')
__docformat__ = 'restructuredtext en'
__license__ = 'Apache License, Version 2.0'
__version__ = '1.0.1'
__all__ = ['cssmin']
import re as _re

def _make_cssmin(python_only=False):
    if False:
        while True:
            i = 10
    '\n    Generate CSS minifier.\n\n    :Parameters:\n      `python_only` : ``bool``\n        Use only the python variant. If true, the c extension is not even\n        tried to be loaded.\n\n    :Return: Minifier\n    :Rtype: ``callable``\n    '
    if not python_only:
        try:
            import _rcssmin
        except ImportError:
            pass
        else:
            return _rcssmin.cssmin
    nl = '(?:[\\n\\f]|\\r\\n?)'
    spacechar = '[\\r\\n\\f\\040\\t]'
    unicoded = '[0-9a-fA-F]{1,6}(?:[\\040\\n\\t\\f]|\\r\\n?)?'
    escaped = '[^\\n\\r\\f0-9a-fA-F]'
    escape = '(?:\\\\(?:%(unicoded)s|%(escaped)s))' % locals()
    nmchar = '[^\\000-\\054\\056\\057\\072-\\100\\133-\\136\\140\\173-\\177]'
    comment = '(?:/\\*[^*]*\\*+(?:[^/*][^*]*\\*+)*/)'
    _bang_comment = '(?:/\\*(!?)[^*]*\\*+(?:[^/*][^*]*\\*+)*/)'
    string1 = '(?:\\047[^\\047\\\\\\r\\n\\f]*(?:\\\\[^\\r\\n\\f][^\\047\\\\\\r\\n\\f]*)*\\047)'
    string2 = '(?:"[^"\\\\\\r\\n\\f]*(?:\\\\[^\\r\\n\\f][^"\\\\\\r\\n\\f]*)*")'
    strings = '(?:%s|%s)' % (string1, string2)
    nl_string1 = '(?:\\047[^\\047\\\\\\r\\n\\f]*(?:\\\\(?:[^\\r]|\\r\\n?)[^\\047\\\\\\r\\n\\f]*)*\\047)'
    nl_string2 = '(?:"[^"\\\\\\r\\n\\f]*(?:\\\\(?:[^\\r]|\\r\\n?)[^"\\\\\\r\\n\\f]*)*")'
    nl_strings = '(?:%s|%s)' % (nl_string1, nl_string2)
    uri_nl_string1 = '(?:\\047[^\\047\\\\]*(?:\\\\(?:[^\\r]|\\r\\n?)[^\\047\\\\]*)*\\047)'
    uri_nl_string2 = '(?:"[^"\\\\]*(?:\\\\(?:[^\\r]|\\r\\n?)[^"\\\\]*)*")'
    uri_nl_strings = '(?:%s|%s)' % (uri_nl_string1, uri_nl_string2)
    nl_escaped = '(?:\\\\%(nl)s)' % locals()
    space = '(?:%(spacechar)s|%(comment)s)' % locals()
    ie7hack = '(?:>/\\*\\*/)'
    uri = '(?:(?:[^\\000-\\040"\\047()\\\\\\177]*(?:%(escape)s[^\\000-\\040"\\047()\\\\\\177]*)*)(?:(?:%(spacechar)s+|%(nl_escaped)s+)(?:(?:[^\\000-\\040"\\047()\\\\\\177]|%(escape)s|%(nl_escaped)s)[^\\000-\\040"\\047()\\\\\\177]*(?:%(escape)s[^\\000-\\040"\\047()\\\\\\177]*)*)+)*)' % locals()
    nl_unesc_sub = _re.compile(nl_escaped).sub
    uri_space_sub = _re.compile('(%(escape)s+)|%(spacechar)s+|%(nl_escaped)s+' % locals()).sub
    uri_space_subber = lambda m: m.groups()[0] or ''
    space_sub_simple = _re.compile('[\\r\\n\\f\\040\\t;]+|(%(comment)s+)' % locals()).sub
    space_sub_banged = _re.compile('[\\r\\n\\f\\040\\t;]+|(%(_bang_comment)s+)' % locals()).sub
    post_esc_sub = _re.compile('[\\r\\n\\f\\t]+').sub
    main_sub = _re.compile('([^\\\\"\\047u>@\\r\\n\\f\\040\\t/;:{}]+)|(?<=[{}(=:>+[,!])(%(space)s+)|^(%(space)s+)|(%(space)s+)(?=(([:{});=>+\\],!])|$)?)|;(%(space)s*(?:;%(space)s*)*)(?=(\\})?)|(\\{)|(\\})|(%(strings)s)|(?<!%(nmchar)s)url\\(%(spacechar)s*(%(uri_nl_strings)s|%(uri)s)%(spacechar)s*\\)|(@[mM][eE][dD][iI][aA])(?!%(nmchar)s)|(%(ie7hack)s)(%(space)s*)|(:[fF][iI][rR][sS][tT]-[lL](?:[iI][nN][eE]|[eE][tT][tT][eE][rR]))(%(space)s*)(?=[{,])|(%(nl_strings)s)|(%(escape)s[^\\\\"\\047u>@\\r\\n\\f\\040\\t/;:{}]*)' % locals()).sub

    def main_subber(keep_bang_comments):
        if False:
            print('Hello World!')
        ' Make main subber '
        (in_macie5, in_rule, at_media) = ([0], [0], [0])
        if keep_bang_comments:
            space_sub = space_sub_banged

            def space_subber(match):
                if False:
                    while True:
                        i = 10
                ' Space|Comment subber '
                if match.lastindex:
                    (group1, group2) = match.group(1, 2)
                    if group2:
                        if group1.endswith('\\*/'):
                            in_macie5[0] = 1
                        else:
                            in_macie5[0] = 0
                        return group1
                    elif group1:
                        if group1.endswith('\\*/'):
                            if in_macie5[0]:
                                return ''
                            in_macie5[0] = 1
                            return '/*\\*/'
                        elif in_macie5[0]:
                            in_macie5[0] = 0
                            return '/**/'
                return ''
        else:
            space_sub = space_sub_simple

            def space_subber(match):
                if False:
                    while True:
                        i = 10
                ' Space|Comment subber '
                if match.lastindex:
                    if match.group(1).endswith('\\*/'):
                        if in_macie5[0]:
                            return ''
                        in_macie5[0] = 1
                        return '/*\\*/'
                    elif in_macie5[0]:
                        in_macie5[0] = 0
                        return '/**/'
                return ''

        def fn_space_post(group):
            if False:
                i = 10
                return i + 15
            ' space with token after '
            if group(5) is None or (group(6) == ':' and (not in_rule[0]) and (not at_media[0])):
                return ' ' + space_sub(space_subber, group(4))
            return space_sub(space_subber, group(4))

        def fn_semicolon(group):
            if False:
                i = 10
                return i + 15
            ' ; handler '
            return ';' + space_sub(space_subber, group(7))

        def fn_semicolon2(group):
            if False:
                return 10
            ' ; handler '
            if in_rule[0]:
                return space_sub(space_subber, group(7))
            return ';' + space_sub(space_subber, group(7))

        def fn_open(group):
            if False:
                i = 10
                return i + 15
            ' { handler '
            if at_media[0]:
                at_media[0] -= 1
            else:
                in_rule[0] = 1
            return '{'

        def fn_close(group):
            if False:
                for i in range(10):
                    print('nop')
            ' } handler '
            in_rule[0] = 0
            return '}'

        def fn_media(group):
            if False:
                for i in range(10):
                    print('nop')
            ' @media handler '
            at_media[0] += 1
            return group(13)

        def fn_ie7hack(group):
            if False:
                for i in range(10):
                    print('nop')
            ' IE7 Hack handler '
            if not in_rule[0] and (not at_media[0]):
                in_macie5[0] = 0
                return group(14) + space_sub(space_subber, group(15))
            return '>' + space_sub(space_subber, group(15))
        table = (None, None, None, None, fn_space_post, fn_space_post, fn_space_post, fn_semicolon, fn_semicolon2, fn_open, fn_close, lambda g: g(11), lambda g: 'url(%s)' % uri_space_sub(uri_space_subber, g(12)), fn_media, None, fn_ie7hack, None, lambda g: g(16) + ' ' + space_sub(space_subber, g(17)), lambda g: nl_unesc_sub('', g(18)), lambda g: post_esc_sub(' ', g(19)))

        def func(match):
            if False:
                print('Hello World!')
            ' Main subber '
            (idx, group) = (match.lastindex, match.group)
            if idx > 3:
                return table[idx](group)
            elif idx == 1:
                return group(1)
            return space_sub(space_subber, group(idx))
        return func

    def cssmin(style, keep_bang_comments=False):
        if False:
            while True:
                i = 10
        '\n        Minify CSS.\n\n        :Parameters:\n          `style` : ``str``\n            CSS to minify\n\n          `keep_bang_comments` : ``bool``\n            Keep comments starting with an exclamation mark? (``/*!...*/``)\n\n        :Return: Minified style\n        :Rtype: ``str``\n        '
        return main_sub(main_subber(keep_bang_comments), style)
    return cssmin
cssmin = _make_cssmin()
if __name__ == '__main__':

    def main():
        if False:
            return 10
        ' Main '
        import sys as _sys
        keep_bang_comments = '-b' in _sys.argv[1:] or '-bp' in _sys.argv[1:] or '-pb' in _sys.argv[1:]
        if '-p' in _sys.argv[1:] or '-bp' in _sys.argv[1:] or '-pb' in _sys.argv[1:]:
            global cssmin
            cssmin = _make_cssmin(python_only=True)
        _sys.stdout.write(cssmin(_sys.stdin.read(), keep_bang_comments=keep_bang_comments))
    main()