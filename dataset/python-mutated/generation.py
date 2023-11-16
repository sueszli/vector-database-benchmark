from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import re
import os
import time
import collections
import shutil
import renpy
from renpy.translation import quote_unicode
from renpy.lexer import elide_filename

def scan_comments(filename):
    if False:
        i = 10
        return i + 15
    rv = []
    if filename not in renpy.config.translate_comments:
        return rv
    comment = []
    start = 0
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [i.rstrip() for i in f.read().replace(u'\ufeff', '').split('\n')]
    for (i, l) in enumerate(lines):
        if not comment:
            start = i + 1
        m = re.match('\\s*## (.*)', l)
        if m:
            c = m.group(1)
            if comment:
                c = c.strip()
            comment.append(c)
        elif comment:
            s = '## ' + ' '.join(comment)
            if s.endswith('#'):
                s = s.rstrip('# ')
            comment = []
            rv.append((start, s))
    return rv
tl_file_cache = {}
todo = True

def open_tl_file(fn):
    if False:
        for i in range(10):
            print('nop')
    if fn in tl_file_cache:
        return tl_file_cache[fn]
    if not os.path.exists(fn):
        dn = os.path.dirname(fn)
        try:
            os.makedirs(dn)
        except Exception:
            pass
        f = open(fn, 'a', encoding='utf-8')
        f.write(u'\ufeff')
    else:
        f = open(fn, 'a', encoding='utf-8')
    if todo:
        f.write(u'# TO' + 'DO: Translation updated at {}\n'.format(time.strftime('%Y-%m-%d %H:%M')))
    f.write(u'\n')
    tl_file_cache[fn] = f
    return f

def close_tl_files():
    if False:
        return 10
    for i in tl_file_cache.values():
        i.close()
    tl_file_cache.clear()

def shorten_filename(filename):
    if False:
        for i in range(10):
            print('nop')
    '\n    Shortens a file name. Returns the shortened filename, and a flag that says\n    if the filename is in the common directory.\n    '
    commondir = os.path.normpath(renpy.config.commondir)
    gamedir = os.path.normpath(renpy.config.gamedir)
    if filename.startswith(commondir):
        fn = os.path.relpath(filename, commondir)
        common = True
    elif filename.startswith(gamedir):
        fn = os.path.relpath(filename, gamedir)
        common = False
    else:
        fn = os.path.basename(filename)
        common = False
    return (fn, common)

def is_empty_extend(t):
    if False:
        while True:
            i = 10
    '\n    Reture true if the translation is an empty extend.\n    '
    for t in t.block:
        if t.get_code() != 'extend ""':
            return False
    return True

def write_translates(filename, language, filter):
    if False:
        for i in range(10):
            print('nop')
    (fn, common) = shorten_filename(filename)
    if common:
        return
    tl_filename = os.path.join(renpy.config.gamedir, renpy.config.tl_directory, language, fn)
    if tl_filename[-1] == 'm':
        tl_filename = tl_filename[:-1]
    if language == 'None':
        language = None
    translator = renpy.game.script.translator
    for (label, t) in translator.file_translates[filename]:
        if (t.identifier, language) in translator.language_translates:
            continue
        if hasattr(t, 'alternate'):
            if (t.alternate, language) in translator.language_translates:
                continue
        if is_empty_extend(t):
            continue
        f = open_tl_file(tl_filename)
        if label is None:
            label = ''
        f.write(u'# {}:{}\n'.format(t.filename, t.linenumber))
        f.write(u'translate {} {}:\n'.format(language, t.identifier.replace('.', '_')))
        f.write(u'\n')
        for n in t.block:
            f.write(u'    # ' + n.get_code() + '\n')
        for n in t.block:
            f.write(u'    ' + n.get_code(filter) + '\n')
        f.write(u'\n')

def translation_filename(s):
    if False:
        return 10
    if renpy.config.translate_launcher:
        return s.launcher_file
    if s.common:
        return 'common.rpy'
    filename = s.elided
    if filename[-1] == 'm':
        filename = filename[:-1]
    return filename

def write_strings(language, filter, min_priority, max_priority, common_only, only_strings=[]):
    if False:
        while True:
            i = 10
    '\n    Writes strings to the file.\n    '
    if language == 'None':
        stl = renpy.game.script.translator.strings[None]
    else:
        stl = renpy.game.script.translator.strings[language]
    strings = renpy.translation.scanstrings.scan(min_priority, max_priority, common_only)
    stringfiles = collections.defaultdict(list)
    for s in strings:
        tlfn = translation_filename(s)
        if tlfn is None:
            continue
        if s.text in stl.translations:
            continue
        if language == 'None' and tlfn == 'common.rpy':
            tlfn = 'common.rpym'
        if only_strings and s.text not in only_strings:
            continue
        stringfiles[tlfn].append(s)
    for (tlfn, sl) in stringfiles.items():
        tlfn = os.path.join(renpy.config.gamedir, renpy.config.tl_directory, language, tlfn)
        f = open_tl_file(tlfn)
        f.write(u'translate {} strings:\n'.format(language))
        f.write(u'\n')
        for s in sl:
            text = filter(s.text)
            f.write(u'    # {}:{}\n'.format(elide_filename(s.filename), s.line))
            f.write(u'    old "{}"\n'.format(quote_unicode(s.text)))
            f.write(u'    new "{}"\n'.format(quote_unicode(text)))
            f.write(u'\n')

def null_filter(s):
    if False:
        while True:
            i = 10
    return s

def empty_filter(s):
    if False:
        i = 10
        return i + 15
    return ''

def generic_filter(s, function):
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: text_utility\n\n    Transforms `s`, while leaving text tags and interpolation the same.\n\n    `function`\n        A function that is called with strings corresponding to runs of\n        text, and should return a second string that replaces that run\n        of text.\n\n    ::\n\n        init python:\n            def upper(s):\n                return s.upper()\n\n        $ upper_string = renpy.transform_text("{b}Not Upper{/b}", upper)\n\n    '

    def remove_special(s, start, end, process):
        if False:
            return 10
        specials = 0
        rv = ''
        buf = ''
        for i in s:
            if i == start:
                if i == buf and specials:
                    rv += buf + i
                    specials = 0
                    buf = ''
                    continue
                if specials == 0:
                    rv += process(buf)
                    buf = ''
                buf += i
                specials += 1
            elif i == end and specials:
                buf += i
                specials -= 1
                if specials == 0:
                    rv += buf
                    buf = ''
            else:
                buf += i
        if buf:
            if specials == 0:
                rv += process(buf)
            else:
                rv += buf
        return rv

    def remove_braces(s):
        if False:
            print('Hello World!')
        return remove_special(s, '{', '}', function)
    return remove_special(s, '[', ']', remove_braces)

def rot13_transform(s):
    if False:
        while True:
            i = 10
    ROT13 = {}
    for (i, j) in zip('ABCDEFGHIJKLM', 'NOPQRSTUVWXYZ'):
        ROT13[i] = j
        ROT13[j] = i
        i = i.lower()
        j = j.lower()
        ROT13[i] = j
        ROT13[j] = i
    return ''.join((ROT13.get(i, i) for i in s))

def rot13_filter(s):
    if False:
        i = 10
        return i + 15
    return generic_filter(s, rot13_transform)

def piglatin_transform(s):
    if False:
        return 10
    lst = ['sh', 'gl', 'ch', 'ph', 'tr', 'br', 'fr', 'bl', 'gr', 'st', 'sl', 'cl', 'pl', 'fl']

    def replace(m):
        if False:
            i = 10
            return i + 15
        i = m.group(0)
        if i[0] in '0123456789':
            rv = i
        elif i[0] in ['a', 'e', 'i', 'o', 'u']:
            rv = i + 'ay'
        elif i[:2] in lst:
            rv = i[2:] + i[:2] + 'ay'
        else:
            rv = i[1:] + i[0] + 'ay'
        if i[0].isupper():
            rv = rv.capitalize()
        return rv
    return re.sub('\\w+', replace, s)

def piglatin_filter(s):
    if False:
        i = 10
        return i + 15
    if s == '{#language name and font}':
        return 'Igpay Atinlay'
    rv = generic_filter(s, piglatin_transform)
    rv = re.sub('\\{\\{(.*)?ay\\}', '{{\\1}', rv)
    return rv

def translate_list_files():
    if False:
        return 10
    '\n    Returns a list of files that exist and should be scanned for translations.\n    '
    filenames = list(renpy.config.translate_files)
    for (dirname, filename) in renpy.loader.listdirfiles():
        if dirname is None:
            continue
        if filename.startswith('tl/'):
            continue
        filename = os.path.join(dirname, filename)
        if not (filename.endswith('.rpy') or filename.endswith('.rpym')):
            continue
        filename = os.path.normpath(filename)
        if not os.path.exists(filename):
            continue
        filenames.append(filename)
    return filenames

def count_missing(language, min_priority, max_priority, common_only):
    if False:
        for i in range(10):
            print('nop')
    '\n    Prints a count of missing translations for `language`.\n    '
    translator = renpy.game.script.translator
    missing_translates = 0
    for filename in translate_list_files():
        for (_, t) in translator.file_translates[filename]:
            if is_empty_extend(t):
                continue
            if (t.identifier, language) not in translator.language_translates:
                missing_translates += 1
    missing_strings = 0
    stl = renpy.game.script.translator.strings[language]
    strings = renpy.translation.scanstrings.scan(min_priority, max_priority, common_only)
    for s in strings:
        tlfn = translation_filename(s)
        if tlfn is None:
            continue
        if s.text in stl.translations:
            continue
        missing_strings += 1
    print('{}: {} missing dialogue translations, {} missing string translations.'.format(language, missing_translates, missing_strings))

def translate_command():
    if False:
        while True:
            i = 10
    '\n    The translate command. When called from the command line, this generates\n    the translations.\n    '
    ap = renpy.arguments.ArgumentParser(description='Generates or updates translations.')
    ap.add_argument('language', help='The language to generate translations for.')
    ap.add_argument('--rot13', help='Apply rot13 while generating translations.', dest='rot13', action='store_true')
    ap.add_argument('--piglatin', help='Apply pig latin while generating translations.', dest='piglatin', action='store_true')
    ap.add_argument('--empty', help='Produce empty strings while generating translations.', dest='empty', action='store_true')
    ap.add_argument('--count', help='Instead of generating files, print a count of missing translations.', dest='count', action='store_true')
    ap.add_argument('--min-priority', help='Translate strings with more than this priority.', dest='min_priority', default=0, type=int)
    ap.add_argument('--max-priority', help='Translate strings with less than this priority.', dest='max_priority', default=0, type=int)
    ap.add_argument('--strings-only', help='Only translate strings (not dialogue).', dest='strings_only', default=False, action='store_true')
    ap.add_argument('--common-only', help='Only translate string from the common code.', dest='common_only', default=False, action='store_true')
    ap.add_argument('--no-todo', help='Do not include the TODO flag.', dest='todo', default=True, action='store_false')
    ap.add_argument('--string', help='Translate a single string.', dest='string', action='append')
    args = ap.parse_args()
    global todo
    todo = args.todo
    if renpy.config.translate_launcher:
        max_priority = args.max_priority or 499
    else:
        max_priority = args.max_priority or 299
    if args.count:
        count_missing(args.language, args.min_priority, max_priority, args.common_only)
        return False
    if args.rot13:
        filter = rot13_filter
    elif args.piglatin:
        filter = piglatin_filter
    elif args.empty:
        filter = empty_filter
    else:
        filter = null_filter
    if not args.strings_only:
        for filename in translate_list_files():
            write_translates(filename, args.language, filter)
    write_strings(args.language, filter, args.min_priority, max_priority, args.common_only, args.string)
    close_tl_files()
    if renpy.config.translate_launcher and (not args.strings_only):
        src = os.path.join(renpy.config.renpy_base, 'gui', 'game', 'script.rpy')
        dst = os.path.join(renpy.config.gamedir, 'tl', args.language, 'script.rpym')
        if os.path.exists(src) and (not os.path.exists(dst)):
            shutil.copy(src, dst)
    return False
renpy.arguments.register_command('translate', translate_command)