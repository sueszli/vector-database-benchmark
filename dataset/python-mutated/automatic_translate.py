import os
import deepl
import argparse
import ast
import re
BLOCKLIST = '\nOpendyslexic\ngame\nbase\nimages\naudio\ngui\n(transform)\n(transition)\ngl tearing\nTearing\nlabels\nPlay Bundle\nUniversal APK\nVisual Studio Code\nAtom\njEdit\nTab\nAndroid\niOS\niPhone\niPad\n## Windows: %APPDATA\\RenPy\\<config.save_directory>\n'
BLOCKLIST = set((i.strip() for i in BLOCKLIST.split('\n') if i.strip()))
if 'DEEPL_TOKEN' not in os.environ:
    raise Exception('Please set the DEEPL_TOKEN environment variable to your DeepL API key.')
tl = deepl.Translator(os.environ['DEEPL_TOKEN'])

def add_tags(s):
    if False:
        while True:
            i = 10
    pos = 0
    rv = ''

    def consume():
        if False:
            while True:
                i = 10
        nonlocal pos
        pos += 1
        return s[pos - 1]

    def peek():
        if False:
            print('Hello World!')
        return s[pos]
    while pos < len(s):
        c = consume()
        if c == '{':
            if peek() == '{':
                consume()
                rv += '{{'
                continue
            rv += '<span translate="no">{'
            while True:
                c = consume()
                rv += c
                if c == '}':
                    break
            rv += '</span>'
            continue
        if c == '[':
            if peek() == '[':
                consume()
                rv += '[['
                continue
            rv += '<span translate="no">['
            count = 1
            while count:
                c = consume()
                rv += c
                if c == '[':
                    count += 1
                if c == ']':
                    count -= 1
            rv += '</span>'
            continue
        rv += c
    rv = rv.replace('</span><span translate="no">', '')
    return rv

def translate(s, lang, source_fn):
    if False:
        for i in range(10):
            print('nop')
    if not s:
        return s
    if 'APPDATA' in s:
        return s
    if '$HOME' in s:
        return s
    if s.startswith('## http'):
        return s
    if s in BLOCKLIST:
        return s
    if '00console' in source_fn:
        (prefix, delim, s) = s.rpartition(': ')
    elif s.startswith('## '):
        (prefix, delim, s) = s.partition(' ')
    else:
        prefix = ''
        delim = ''
    s = s.replace('&', '&amp;')
    s = s.replace('<', '&lt;')
    s = s.replace('>', '&gt;')
    s = add_tags(s)
    s = tl.translate_text(s, source_lang='EN', target_lang=lang, tag_handling='html').text
    s = s.replace('<span translate="no">', '')
    s = s.replace('</span>', '')
    s = s.replace('&lt;', '<')
    s = s.replace('&gt;', '>')
    s = s.replace('&amp;', '&')
    s = prefix + delim + s
    return s

def translate_lines(s, lang, source_fn):
    if False:
        for i in range(10):
            print('nop')
    rv = []
    for l in s.split('\n'):
        rv.append(translate(l, lang, source_fn))
    return '\n'.join(rv)

def quote_unicode(s):
    if False:
        i = 10
        return i + 15
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace('\x07', '\\a')
    s = s.replace('\x08', '\\b')
    s = s.replace('\x0c', '\\f')
    s = s.replace('\n', '\\n')
    s = s.replace('\r', '\\r')
    s = s.replace('\t', '\\t')
    s = s.replace('\x0b', '\\v')
    return f'"{s}"'

def process_file(fn, language, only_strings=[]):
    if False:
        for i in range(10):
            print('nop')

    def should_translate(s):
        if False:
            for i in range(10):
                print('nop')
        if '{#' in s:
            return False
        if only_strings and s not in only_strings:
            return False
        return True
    print('Translate', fn, 'to', language)
    result = []
    source_fn = ''
    old = ''
    with open(fn) as f:
        lines = f.readlines()
    for l in lines:
        if l.startswith('    old'):
            result.append(l)
            old = ast.literal_eval(l.strip().partition(' ')[2])
        elif l.startswith('    new'):
            new = ast.literal_eval(l.strip().partition(' ')[2])
            orig_new = new
            if (not new or new == old) and should_translate(old):
                new = translate_lines(old, language, source_fn)
                if new != old or new != orig_new:
                    if new != old:
                        result.append('    # Automatic translation.\n')
                    new = quote_unicode(new)
                    l = f'    new {new}\n'
            result.append(l)
        else:
            if (m := re.match('    (.*):\\d+$', l)):
                source_fn = m.group(1)
            result.append(l)
    with open(fn, 'w') as f:
        f.write(''.join(result))

def main():
    if False:
        while True:
            i = 10
    ap = argparse.ArgumentParser()
    ap.add_argument('language')
    ap.add_argument('files', nargs='+')
    ap.add_argument('--string', help='Translate a single string.', dest='string', action='append')
    args = ap.parse_args()
    for fn in args.files:
        process_file(fn, args.language, args.string)
if __name__ == '__main__':
    main()