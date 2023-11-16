import builtins
import keyword
import re
import time
from idlelib.config import idleConf
from idlelib.delegator import Delegator
DEBUG = False

def any(name, alternates):
    if False:
        print('Hello World!')
    'Return a named group pattern matching list of alternates.'
    return '(?P<%s>' % name + '|'.join(alternates) + ')'

def make_pat():
    if False:
        for i in range(10):
            print('nop')
    kw = '\\b' + any('KEYWORD', keyword.kwlist) + '\\b'
    match_softkw = '^[ \\t]*' + '(?P<MATCH_SOFTKW>match)\\b' + '(?![ \\t]*(?:' + '|'.join(['[:,;=^&|@~)\\]}]', '\\b(?:' + '|'.join(keyword.kwlist) + ')\\b']) + '))'
    case_default = '^[ \\t]*' + '(?P<CASE_SOFTKW>case)' + '[ \\t]+(?P<CASE_DEFAULT_UNDERSCORE>_\\b)'
    case_softkw_and_pattern = '^[ \\t]*' + '(?P<CASE_SOFTKW2>case)\\b' + '(?![ \\t]*(?:' + '|'.join(['_\\b', '[:,;=^&|@~)\\]}]', '\\b(?:' + '|'.join(keyword.kwlist) + ')\\b']) + '))'
    builtinlist = [str(name) for name in dir(builtins) if not name.startswith('_') and name not in keyword.kwlist]
    builtin = '([^.\'\\"\\\\#]\\b|^)' + any('BUILTIN', builtinlist) + '\\b'
    comment = any('COMMENT', ['#[^\\n]*'])
    stringprefix = '(?i:r|u|f|fr|rf|b|br|rb)?'
    sqstring = stringprefix + "'[^'\\\\\\n]*(\\\\.[^'\\\\\\n]*)*'?"
    dqstring = stringprefix + '"[^"\\\\\\n]*(\\\\.[^"\\\\\\n]*)*"?'
    sq3string = stringprefix + "'''[^'\\\\]*((\\\\.|'(?!''))[^'\\\\]*)*(''')?"
    dq3string = stringprefix + '"""[^"\\\\]*((\\\\.|"(?!""))[^"\\\\]*)*(""")?'
    string = any('STRING', [sq3string, dq3string, sqstring, dqstring])
    prog = re.compile('|'.join([builtin, comment, string, kw, match_softkw, case_default, case_softkw_and_pattern, any('SYNC', ['\\n'])]), re.DOTALL | re.MULTILINE)
    return prog
prog = make_pat()
idprog = re.compile('\\s+(\\w+)')
prog_group_name_to_tag = {'MATCH_SOFTKW': 'KEYWORD', 'CASE_SOFTKW': 'KEYWORD', 'CASE_DEFAULT_UNDERSCORE': 'KEYWORD', 'CASE_SOFTKW2': 'KEYWORD'}

def matched_named_groups(re_match):
    if False:
        while True:
            i = 10
    'Get only the non-empty named groups from an re.Match object.'
    return ((k, v) for (k, v) in re_match.groupdict().items() if v)

def color_config(text):
    if False:
        for i in range(10):
            print('nop')
    'Set color options of Text widget.\n\n    If ColorDelegator is used, this should be called first.\n    '
    theme = idleConf.CurrentTheme()
    normal_colors = idleConf.GetHighlight(theme, 'normal')
    cursor_color = idleConf.GetHighlight(theme, 'cursor')['foreground']
    select_colors = idleConf.GetHighlight(theme, 'hilite')
    text.config(foreground=normal_colors['foreground'], background=normal_colors['background'], insertbackground=cursor_color, selectforeground=select_colors['foreground'], selectbackground=select_colors['background'], inactiveselectbackground=select_colors['background'])

class ColorDelegator(Delegator):
    """Delegator for syntax highlighting (text coloring).

    Instance variables:
        delegate: Delegator below this one in the stack, meaning the
                one this one delegates to.

        Used to track state:
        after_id: Identifier for scheduled after event, which is a
                timer for colorizing the text.
        allow_colorizing: Boolean toggle for applying colorizing.
        colorizing: Boolean flag when colorizing is in process.
        stop_colorizing: Boolean flag to end an active colorizing
                process.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        Delegator.__init__(self)
        self.init_state()
        self.prog = prog
        self.idprog = idprog
        self.LoadTagDefs()

    def init_state(self):
        if False:
            i = 10
            return i + 15
        'Initialize variables that track colorizing state.'
        self.after_id = None
        self.allow_colorizing = True
        self.stop_colorizing = False
        self.colorizing = False

    def setdelegate(self, delegate):
        if False:
            i = 10
            return i + 15
        'Set the delegate for this instance.\n\n        A delegate is an instance of a Delegator class and each\n        delegate points to the next delegator in the stack.  This\n        allows multiple delegators to be chained together for a\n        widget.  The bottom delegate for a colorizer is a Text\n        widget.\n\n        If there is a delegate, also start the colorizing process.\n        '
        if self.delegate is not None:
            self.unbind('<<toggle-auto-coloring>>')
        Delegator.setdelegate(self, delegate)
        if delegate is not None:
            self.config_colors()
            self.bind('<<toggle-auto-coloring>>', self.toggle_colorize_event)
            self.notify_range('1.0', 'end')
        else:
            self.stop_colorizing = True
            self.allow_colorizing = False

    def config_colors(self):
        if False:
            for i in range(10):
                print('nop')
        'Configure text widget tags with colors from tagdefs.'
        for (tag, cnf) in self.tagdefs.items():
            self.tag_configure(tag, **cnf)
        self.tag_raise('sel')

    def LoadTagDefs(self):
        if False:
            i = 10
            return i + 15
        'Create dictionary of tag names to text colors.'
        theme = idleConf.CurrentTheme()
        self.tagdefs = {'COMMENT': idleConf.GetHighlight(theme, 'comment'), 'KEYWORD': idleConf.GetHighlight(theme, 'keyword'), 'BUILTIN': idleConf.GetHighlight(theme, 'builtin'), 'STRING': idleConf.GetHighlight(theme, 'string'), 'DEFINITION': idleConf.GetHighlight(theme, 'definition'), 'SYNC': {'background': None, 'foreground': None}, 'TODO': {'background': None, 'foreground': None}, 'ERROR': idleConf.GetHighlight(theme, 'error'), 'hit': idleConf.GetHighlight(theme, 'hit')}
        if DEBUG:
            print('tagdefs', self.tagdefs)

    def insert(self, index, chars, tags=None):
        if False:
            for i in range(10):
                print('nop')
        'Insert chars into widget at index and mark for colorizing.'
        index = self.index(index)
        self.delegate.insert(index, chars, tags)
        self.notify_range(index, index + '+%dc' % len(chars))

    def delete(self, index1, index2=None):
        if False:
            i = 10
            return i + 15
        'Delete chars between indexes and mark for colorizing.'
        index1 = self.index(index1)
        self.delegate.delete(index1, index2)
        self.notify_range(index1)

    def notify_range(self, index1, index2=None):
        if False:
            for i in range(10):
                print('nop')
        'Mark text changes for processing and restart colorizing, if active.'
        self.tag_add('TODO', index1, index2)
        if self.after_id:
            if DEBUG:
                print('colorizing already scheduled')
            return
        if self.colorizing:
            self.stop_colorizing = True
            if DEBUG:
                print('stop colorizing')
        if self.allow_colorizing:
            if DEBUG:
                print('schedule colorizing')
            self.after_id = self.after(1, self.recolorize)
        return

    def close(self):
        if False:
            print('Hello World!')
        if self.after_id:
            after_id = self.after_id
            self.after_id = None
            if DEBUG:
                print('cancel scheduled recolorizer')
            self.after_cancel(after_id)
        self.allow_colorizing = False
        self.stop_colorizing = True

    def toggle_colorize_event(self, event=None):
        if False:
            while True:
                i = 10
        'Toggle colorizing on and off.\n\n        When toggling off, if colorizing is scheduled or is in\n        process, it will be cancelled and/or stopped.\n\n        When toggling on, colorizing will be scheduled.\n        '
        if self.after_id:
            after_id = self.after_id
            self.after_id = None
            if DEBUG:
                print('cancel scheduled recolorizer')
            self.after_cancel(after_id)
        if self.allow_colorizing and self.colorizing:
            if DEBUG:
                print('stop colorizing')
            self.stop_colorizing = True
        self.allow_colorizing = not self.allow_colorizing
        if self.allow_colorizing and (not self.colorizing):
            self.after_id = self.after(1, self.recolorize)
        if DEBUG:
            print('auto colorizing turned', 'on' if self.allow_colorizing else 'off')
        return 'break'

    def recolorize(self):
        if False:
            print('Hello World!')
        'Timer event (every 1ms) to colorize text.\n\n        Colorizing is only attempted when the text widget exists,\n        when colorizing is toggled on, and when the colorizing\n        process is not already running.\n\n        After colorizing is complete, some cleanup is done to\n        make sure that all the text has been colorized.\n        '
        self.after_id = None
        if not self.delegate:
            if DEBUG:
                print('no delegate')
            return
        if not self.allow_colorizing:
            if DEBUG:
                print('auto colorizing is off')
            return
        if self.colorizing:
            if DEBUG:
                print('already colorizing')
            return
        try:
            self.stop_colorizing = False
            self.colorizing = True
            if DEBUG:
                print('colorizing...')
            t0 = time.perf_counter()
            self.recolorize_main()
            t1 = time.perf_counter()
            if DEBUG:
                print('%.3f seconds' % (t1 - t0))
        finally:
            self.colorizing = False
        if self.allow_colorizing and self.tag_nextrange('TODO', '1.0'):
            if DEBUG:
                print('reschedule colorizing')
            self.after_id = self.after(1, self.recolorize)

    def recolorize_main(self):
        if False:
            print('Hello World!')
        'Evaluate text and apply colorizing tags.'
        next = '1.0'
        while (todo_tag_range := self.tag_nextrange('TODO', next)):
            self.tag_remove('SYNC', todo_tag_range[0], todo_tag_range[1])
            sync_tag_range = self.tag_prevrange('SYNC', todo_tag_range[0])
            head = sync_tag_range[1] if sync_tag_range else '1.0'
            chars = ''
            next = head
            lines_to_get = 1
            ok = False
            while not ok:
                mark = next
                next = self.index(mark + '+%d lines linestart' % lines_to_get)
                lines_to_get = min(lines_to_get * 2, 100)
                ok = 'SYNC' in self.tag_names(next + '-1c')
                line = self.get(mark, next)
                if not line:
                    return
                for tag in self.tagdefs:
                    self.tag_remove(tag, mark, next)
                chars += line
                self._add_tags_in_section(chars, head)
                if 'SYNC' in self.tag_names(next + '-1c'):
                    head = next
                    chars = ''
                else:
                    ok = False
                if not ok:
                    self.tag_add('TODO', next)
                self.update()
                if self.stop_colorizing:
                    if DEBUG:
                        print('colorizing stopped')
                    return

    def _add_tag(self, start, end, head, matched_group_name):
        if False:
            i = 10
            return i + 15
        'Add a tag to a given range in the text widget.\n\n        This is a utility function, receiving the range as `start` and\n        `end` positions, each of which is a number of characters\n        relative to the given `head` index in the text widget.\n\n        The tag to add is determined by `matched_group_name`, which is\n        the name of a regular expression "named group" as matched by\n        by the relevant highlighting regexps.\n        '
        tag = prog_group_name_to_tag.get(matched_group_name, matched_group_name)
        self.tag_add(tag, f'{head}+{start:d}c', f'{head}+{end:d}c')

    def _add_tags_in_section(self, chars, head):
        if False:
            return 10
        'Parse and add highlighting tags to a given part of the text.\n\n        `chars` is a string with the text to parse and to which\n        highlighting is to be applied.\n\n            `head` is the index in the text widget where the text is found.\n        '
        for m in self.prog.finditer(chars):
            for (name, matched_text) in matched_named_groups(m):
                (a, b) = m.span(name)
                self._add_tag(a, b, head, name)
                if matched_text in ('def', 'class'):
                    if (m1 := self.idprog.match(chars, b)):
                        (a, b) = m1.span(1)
                        self._add_tag(a, b, head, 'DEFINITION')

    def removecolors(self):
        if False:
            print('Hello World!')
        'Remove all colorizing tags.'
        for tag in self.tagdefs:
            self.tag_remove(tag, '1.0', 'end')

def _color_delegator(parent):
    if False:
        for i in range(10):
            print('nop')
    from tkinter import Toplevel, Text
    from idlelib.idle_test.test_colorizer import source
    from idlelib.percolator import Percolator
    top = Toplevel(parent)
    top.title('Test ColorDelegator')
    (x, y) = map(int, parent.geometry().split('+')[1:])
    top.geometry('700x550+%d+%d' % (x + 20, y + 175))
    text = Text(top, background='white')
    text.pack(expand=1, fill='both')
    text.insert('insert', source)
    text.focus_set()
    color_config(text)
    p = Percolator(text)
    d = ColorDelegator()
    p.insertfilter(d)
if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_colorizer', verbosity=2, exit=False)
    from idlelib.idle_test.htest import run
    run(_color_delegator)