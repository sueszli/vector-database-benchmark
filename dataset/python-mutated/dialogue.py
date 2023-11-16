from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import renpy
import os
from renpy.translation import quote_unicode

def create_dialogue_map(language):
    if False:
        return 10
    "\n    :undocumented:\n\n    Creates a map from a dialogue string to a potential translation of the\n    the dialogue. This is meant for the Ren'Py tutorial, as a way of translating\n    strings found in the examples.\n    "
    rv = {}

    def get_text(t):
        if False:
            while True:
                i = 10
        for i in t.block:
            if isinstance(i, renpy.ast.Say):
                return i.what
        return None
    translator = renpy.game.script.translator
    for v in translator.file_translates.values():
        for (_, t) in v:
            lt = translator.language_translates.get((t.identifier, language), None)
            if lt is None:
                continue
            t_text = get_text(t)
            lt_text = get_text(lt)
            if t_text and lt_text:
                rv[t_text] = lt_text
    return rv

def notags_filter(s):
    if False:
        print('Hello World!')

    def tag_pass(s):
        if False:
            print('Hello World!')
        brace = False
        first = False
        rv = ''
        for i in s:
            if i == '{':
                if first:
                    brace = False
                    first = False
                    rv += '{{'
                else:
                    brace = True
                    first = True
            elif i == '}':
                first = False
                if brace:
                    brace = False
                else:
                    rv += i
            else:
                first = False
                if brace:
                    pass
                else:
                    rv += i
        return rv

    def square_pass(s):
        if False:
            return 10
        squares = 0
        first = False
        rv = ''
        buf = ''
        for i in s:
            if i == '[':
                if first:
                    squares = 0
                else:
                    rv += tag_pass(buf)
                    buf = ''
                    if squares == 0:
                        first = True
                    squares += 1
                rv += '['
            elif i == ']':
                first = False
                squares -= 1
                if squares < 0:
                    squares += 1
                rv += ']'
            elif squares:
                rv += i
            else:
                buf += i
        if buf:
            rv += tag_pass(buf)
        return rv
    return square_pass(s)

def combine_filter(s):
    if False:
        return 10
    doubles = ['{{', '%%']
    if renpy.config.lenticular_bracket_ruby:
        doubles.append('【【')
    for double in doubles:
        while True:
            if s.find(double) >= 0:
                i = s.find(double)
                s = s[:i] + s[i + 1:]
            else:
                break
    return s

def what_filter(s):
    if False:
        i = 10
        return i + 15
    return '[what]'

class DialogueFile(object):

    def __init__(self, filename, output, tdf=True, strings=False, notags=True, escape=True, language=None):
        if False:
            print('Hello World!')
        "\n        `filename`\n            The file we're extracting dialogue from.\n\n        `tdf`\n            If true, dialogue is extracted in tab-delimited format. If false,\n            dialogue is extracted by itself.\n\n        `strings`\n            If true, extract all translatable strings, not just dialogue.\n\n        `notags`\n            If true, strip text tags from the extracted dialogue.\n\n        `escape`\n            If true, escape special characters in the dialogue.\n        "
        self.filename = filename
        commondir = os.path.normpath(renpy.config.commondir)
        if filename.startswith(commondir):
            return
        self.tdf = tdf
        self.notags = notags
        self.escape = escape
        self.strings = strings
        self.language = language
        self.f = open(output, 'a', encoding='utf-8')
        with self.f:
            self.write_dialogue()

    def write_dialogue(self):
        if False:
            while True:
                i = 10
        '\n        Writes the dialogue to the file.\n        '
        lines = []
        translator = renpy.game.script.translator
        for (label, t) in translator.file_translates[self.filename]:
            if label is None:
                label = ''
            identifier = t.identifier.replace('.', '_')
            tl = None
            if self.language is not None:
                tl = translator.language_translates.get((identifier, self.language), None)
            if tl is None:
                block = t.block
            else:
                block = tl.block
            for n in block:
                if isinstance(n, renpy.ast.Say):
                    if not n.who:
                        who = ''
                    else:
                        who = n.who
                    what = n.what
                    if self.notags:
                        what = notags_filter(what)
                    what = combine_filter(what)
                    if self.escape:
                        what = quote_unicode(what)
                    elif self.tdf:
                        what = what.replace('\\', '\\\\')
                        what = what.replace('\t', '\\t')
                        what = what.replace('\n', '\\n')
                    if self.tdf:
                        lines.append([t.identifier, who, what, n.filename, str(n.linenumber), n.get_code(what_filter)])
                    else:
                        lines.append([what])
        if self.strings:
            lines.extend(self.get_strings())
            if self.tdf:
                lines.sort(key=lambda x: int(x[4]))
        for line in lines:
            self.f.write('\t'.join(line) + '\n')

    def get_strings(self):
        if False:
            i = 10
            return i + 15
        '\n        Finds the strings in the file.\n        '
        lines = []
        filename = renpy.lexer.elide_filename(self.filename)
        for ss in renpy.translation.scanstrings.scan_strings(self.filename):
            line = ss.line
            s = ss.text
            stl = renpy.game.script.translator.strings[None]
            if s in stl.translations:
                continue
            stl.translations[s] = s
            s = renpy.translation.translate_string(s, self.language)
            if self.notags:
                s = notags_filter(s)
            s = combine_filter(s)
            if self.escape:
                s = quote_unicode(s)
            elif self.tdf:
                s = s.replace('\\', '\\\\')
                s = s.replace('\t', '\\t')
                s = s.replace('\n', '\\n')
            if self.tdf:
                lines.append(['', '', s, filename, str(line)])
            else:
                lines.append([s])
        return lines

def dialogue_command():
    if False:
        print('Hello World!')
    '\n    The dialogue command. This updates dialogue.txt, a file giving all the dialogue\n    in the game.\n    '
    ap = renpy.arguments.ArgumentParser(description='Generates or updates translations.')
    ap.add_argument('language', help='The language to extract dialogue for.')
    ap.add_argument('--text', help='Output the dialogue as plain text, instead of a tab-delimited file.', dest='text', action='store_true')
    ap.add_argument('--strings', help='Output all translatable strings, not just dialogue.', dest='strings', action='store_true')
    ap.add_argument('--notags', help='Strip text tags from the dialogue.', dest='notags', action='store_true')
    ap.add_argument('--escape', help='Escape quotes and other special characters.', dest='escape', action='store_true')
    args = ap.parse_args()
    tdf = not args.text
    if tdf:
        output = os.path.join(renpy.config.basedir, 'dialogue.tab')
    else:
        output = os.path.join(renpy.config.basedir, 'dialogue.txt')
    with open(output, 'w') as f:
        if tdf:
            line = ['Identifier', 'Character', 'Dialogue', 'Filename', 'Line Number', "Ren'Py Script"]
            f.write('\t'.join(line) + '\n')
    for (dirname, filename) in renpy.loader.listdirfiles():
        if dirname is None:
            continue
        filename = os.path.join(dirname, filename)
        if not (filename.endswith('.rpy') or filename.endswith('.rpym')):
            continue
        filename = os.path.normpath(filename)
        language = args.language
        if language in ('None', ''):
            language = None
        DialogueFile(filename, output, tdf=tdf, strings=args.strings, notags=args.notags, escape=args.escape, language=language)
    return False
renpy.arguments.register_command('dialogue', dialogue_command)