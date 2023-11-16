from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import os
import re
import renpy
STRING_RE = '(?x)\n\\b_[_p]?\\s*(\\((?:[\\s\\\\\\n]*[uU]?(?:\n\\"\\"\\"(?:\\\\.|\\\\\\n|\\"{1,2}|[^\\\\"])*?\\"\\"\\"\n|\'\'\'(?:\\\\.|\\\\\\n|\\\'{1,2}|[^\\\\\'])*?\'\'\'\n|"(?:\\\\.|\\\\\\n|[^\\\\"])*"\n|\'(?:\\\\.|\\\\\\n|[^\\\\\'])*\'\n))+\\s*\\))\n'
REGULAR_PRIORITIES = [('script.rpy', 5, 'script.rpy'), ('options.rpy', 10, 'options.rpy'), ('gui.rpy', 20, 'gui.rpy'), ('screens.rpy', 30, 'screens.rpy'), ('', 100, 'launcher.rpy')]
COMMON_PRIORITIES = [('_compat/', 420, 'obsolete.rpy'), ('_layout/', 410, 'obsolete.rpy'), ('00layout.rpy', 400, 'obsolete.rpy'), ('00console.rpy', 320, 'developer.rpy'), ('_developer/', 310, 'developer.rpy'), ('_errorhandling.rpym', 220, 'error.rpy'), ('00gamepad.rpy', 210, 'error.rpy'), ('00gltest.rpy', 200, 'error.rpy'), ('00gallery.rpy', 180, 'common.rpy'), ('00compat.rpy', 180, 'common.rpy'), ('00updater.rpy', 170, 'common.rpy'), ('00gamepad.rpy', 160, 'common.rpy'), ('00iap.rpy', 150, 'common.rpy'), ('', 50, 'common.rpy')]

class String(object):
    """
    This stores information about a translation string or comment.
    """

    def __init__(self, filename, line, text, comment):
        if False:
            i = 10
            return i + 15
        self.filename = filename
        self.line = line
        self.text = text
        self.comment = comment
        (self.elided, self.common) = renpy.translation.generation.shorten_filename(self.filename)
        if self.common:
            pl = COMMON_PRIORITIES
        else:
            pl = REGULAR_PRIORITIES
        normalized_elided = self.elided.replace('\\', '/')
        for (prefix, priority, launcher_file) in pl:
            if normalized_elided.startswith(prefix):
                break
        else:
            priority = 500
            launcher_file = 'unknown.rpy'
        self.priority = priority
        self.sort_key = (priority, self.filename, self.line)
        self.launcher_file = launcher_file

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<String {self.filename}:{self.line} {self.text!r}>'.format(self=self)

def scan_strings(filename):
    if False:
        print('Hello World!')
    "\n    Scans `filename`, a file containing Ren'Py script, for translatable\n    strings.\n\n    Returns a list of TranslationString objects.\n    "
    rv = []
    for (line, s) in renpy.game.script.translator.additional_strings[filename]:
        rv.append(String(filename, line, s, False))
    for (_filename, lineno, text) in renpy.lexer.list_logical_lines(filename):
        for m in re.finditer(STRING_RE, text):
            s = m.group(1)
            s = s.replace('\\\n', '')
            if s is not None:
                s = s.strip()
                s = eval(s)
                if m.group(0).startswith('_p'):
                    s = renpy.minstore._p(s)
                if s:
                    rv.append(String(filename, lineno, s, False))
    return rv

def scan_comments(filename):
    if False:
        print('Hello World!')
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
            rv.append(String(filename, start, s, True))
    return rv

def scan(min_priority=0, max_priority=299, common_only=False):
    if False:
        return 10
    '\n    Scans all files for translatable strings and comments. Returns a list\n    of String objects.\n    '
    filenames = renpy.translation.generation.translate_list_files()
    strings = []
    for filename in filenames:
        filename = os.path.normpath(filename)
        if not os.path.exists(filename):
            continue
        strings.extend(scan_strings(filename))
        strings.extend(scan_comments(filename))
    strings.sort(key=lambda s: s.sort_key)
    rv = []
    seen = set()
    for s in strings:
        if s.priority < min_priority:
            continue
        if s.priority > max_priority:
            continue
        if common_only and (not s.common):
            continue
        if s.text in seen:
            continue
        seen.add(s.text)
        rv.append(s)
    return rv