"""
This is a parser for Windows .rc files, which are text files which define
dialogs and other Windows UI resources.
"""
__author__ = 'Adam Walker'
__version__ = '0.11'
import os
import pprint
import shlex
import stat
import sys
import commctrl
import win32con
_controlMap = {'DEFPUSHBUTTON': 128, 'PUSHBUTTON': 128, 'Button': 128, 'GROUPBOX': 128, 'Static': 130, 'CTEXT': 130, 'RTEXT': 130, 'LTEXT': 130, 'LISTBOX': 131, 'SCROLLBAR': 132, 'COMBOBOX': 133, 'EDITTEXT': 129, 'ICON': 130, 'RICHEDIT': 'RichEdit20A'}
_addDefaults = {'EDITTEXT': win32con.WS_BORDER | win32con.WS_TABSTOP, 'GROUPBOX': win32con.BS_GROUPBOX, 'LTEXT': win32con.SS_LEFT, 'DEFPUSHBUTTON': win32con.BS_DEFPUSHBUTTON | win32con.WS_TABSTOP, 'PUSHBUTTON': win32con.WS_TABSTOP, 'CTEXT': win32con.SS_CENTER, 'RTEXT': win32con.SS_RIGHT, 'ICON': win32con.SS_ICON, 'LISTBOX': win32con.LBS_NOTIFY}
defaultControlStyle = win32con.WS_CHILD | win32con.WS_VISIBLE
defaultControlStyleEx = 0

class DialogDef:
    name = ''
    id = 0
    style = 0
    styleEx = None
    caption = ''
    font = 'MS Sans Serif'
    fontSize = 8
    x = 0
    y = 0
    w = 0
    h = 0
    template = None

    def __init__(self, n, i):
        if False:
            while True:
                i = 10
        self.name = n
        self.id = i
        self.styles = []
        self.stylesEx = []
        self.controls = []

    def createDialogTemplate(self):
        if False:
            print('Hello World!')
        t = None
        self.template = [[self.caption, (self.x, self.y, self.w, self.h), self.style, self.styleEx, (self.fontSize, self.font)]]
        for control in self.controls:
            self.template.append(control.createDialogTemplate())
        return self.template

class ControlDef:
    id = ''
    controlType = ''
    subType = ''
    idNum = 0
    style = defaultControlStyle
    styleEx = defaultControlStyleEx
    label = ''
    x = 0
    y = 0
    w = 0
    h = 0

    def __init__(self):
        if False:
            return 10
        self.styles = []
        self.stylesEx = []

    def toString(self):
        if False:
            while True:
                i = 10
        s = '<Control id:' + self.id + ' controlType:' + self.controlType + ' subType:' + self.subType + ' idNum:' + str(self.idNum) + ' style:' + str(self.style) + ' styles:' + str(self.styles) + ' label:' + self.label + ' x:' + str(self.x) + ' y:' + str(self.y) + ' w:' + str(self.w) + ' h:' + str(self.h) + '>'
        return s

    def createDialogTemplate(self):
        if False:
            i = 10
            return i + 15
        ct = self.controlType
        if 'CONTROL' == ct:
            ct = self.subType
        if ct in _controlMap:
            ct = _controlMap[ct]
        t = [ct, self.label, self.idNum, (self.x, self.y, self.w, self.h), self.style, self.styleEx]
        return t

class StringDef:

    def __init__(self, id, idNum, value):
        if False:
            return 10
        self.id = id
        self.idNum = idNum
        self.value = value

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'StringDef({self.id!r}, {self.idNum!r}, {self.value!r})'

class RCParser:
    next_id = 1001
    dialogs = {}
    _dialogs = {}
    debugEnabled = False
    token = ''

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.ungot = False
        self.ids = {'IDC_STATIC': -1}
        self.names = {-1: 'IDC_STATIC'}
        self.bitmaps = {}
        self.stringTable = {}
        self.icons = {}

    def debug(self, *args):
        if False:
            while True:
                i = 10
        if self.debugEnabled:
            print(args)

    def getToken(self):
        if False:
            while True:
                i = 10
        if self.ungot:
            self.ungot = False
            self.debug('getToken returns (ungot):', self.token)
            return self.token
        self.token = self.lex.get_token()
        self.debug('getToken returns:', self.token)
        if self.token == '':
            self.token = None
        return self.token

    def ungetToken(self):
        if False:
            while True:
                i = 10
        self.ungot = True

    def getCheckToken(self, expected):
        if False:
            i = 10
            return i + 15
        tok = self.getToken()
        assert tok == expected, f"Expected token '{expected}', but got token '{tok}'!"
        return tok

    def getCommaToken(self):
        if False:
            for i in range(10):
                print('nop')
        return self.getCheckToken(',')

    def currentNumberToken(self):
        if False:
            return 10
        mult = 1
        if self.token == '-':
            mult = -1
            self.getToken()
        return int(self.token) * mult

    def currentQuotedString(self):
        if False:
            i = 10
            return i + 15
        assert self.token.startswith('"'), self.token
        bits = [self.token]
        while 1:
            tok = self.getToken()
            if not tok.startswith('"'):
                self.ungetToken()
                break
            bits.append(tok)
        sval = ''.join(bits)[1:-1]
        for (i, o) in (('""', '"'), ('\\r', '\r'), ('\\n', '\n'), ('\\t', '\t')):
            sval = sval.replace(i, o)
        return sval

    def load(self, rcstream):
        if False:
            for i in range(10):
                print('nop')
        '\n        RCParser.loadDialogs(rcFileName) -> None\n        Load the dialog information into the parser. Dialog Definations can then be accessed\n        using the "dialogs" dictionary member (name->DialogDef). The "ids" member contains the dictionary of id->name.\n        The "names" member contains the dictionary of name->id\n        '
        self.open(rcstream)
        self.getToken()
        while self.token is not None:
            self.parse()
            self.getToken()

    def open(self, rcstream):
        if False:
            return 10
        self.lex = shlex.shlex(rcstream)
        self.lex.commenters = '//#'

    def parseH(self, file):
        if False:
            print('Hello World!')
        lex = shlex.shlex(file)
        lex.commenters = '//'
        token = ' '
        while token is not None:
            token = lex.get_token()
            if token == '' or token is None:
                token = None
            elif token == 'define':
                n = lex.get_token()
                i = int(lex.get_token())
                self.ids[n] = i
                if i in self.names:
                    pass
                else:
                    self.names[i] = n
                if self.next_id <= i:
                    self.next_id = i + 1

    def parse(self):
        if False:
            print('Hello World!')
        noid_parsers = {'STRINGTABLE': self.parse_stringtable}
        id_parsers = {'DIALOG': self.parse_dialog, 'DIALOGEX': self.parse_dialog, 'BITMAP': self.parse_bitmap, 'ICON': self.parse_icon}
        deep = 0
        base_token = self.token
        rp = noid_parsers.get(base_token)
        if rp is not None:
            rp()
        else:
            resource_id = self.token
            self.getToken()
            if self.token is None:
                return
            if 'BEGIN' == self.token:
                deep = 1
                while deep != 0 and self.token is not None:
                    self.getToken()
                    self.debug('Zooming over', self.token)
                    if 'BEGIN' == self.token:
                        deep += 1
                    elif 'END' == self.token:
                        deep -= 1
            else:
                rp = id_parsers.get(self.token)
                if rp is not None:
                    self.debug(f"Dispatching '{self.token}'")
                    rp(resource_id)
                else:
                    self.debug("Skipping top-level '%s'" % base_token)
                    self.ungetToken()

    def addId(self, id_name):
        if False:
            print('Hello World!')
        if id_name in self.ids:
            id = self.ids[id_name]
        else:
            for n in ['IDOK', 'IDCANCEL', 'IDYES', 'IDNO', 'IDABORT']:
                if id_name == n:
                    v = getattr(win32con, n)
                    self.ids[n] = v
                    self.names[v] = n
                    return v
            id = self.next_id
            self.next_id += 1
            self.ids[id_name] = id
            self.names[id] = id_name
        return id

    def lang(self):
        if False:
            return 10
        while self.token[0:4] == 'LANG' or self.token[0:7] == 'SUBLANG' or self.token == ',':
            self.getToken()

    def parse_textinclude(self, res_id):
        if False:
            print('Hello World!')
        while self.getToken() != 'BEGIN':
            pass
        while 1:
            if self.token == 'END':
                break
            s = self.getToken()

    def parse_stringtable(self):
        if False:
            while True:
                i = 10
        while self.getToken() != 'BEGIN':
            pass
        while 1:
            self.getToken()
            if self.token == 'END':
                break
            sid = self.token
            self.getToken()
            sd = StringDef(sid, self.addId(sid), self.currentQuotedString())
            self.stringTable[sid] = sd

    def parse_bitmap(self, name):
        if False:
            return 10
        return self.parse_bitmap_or_icon(name, self.bitmaps)

    def parse_icon(self, name):
        if False:
            print('Hello World!')
        return self.parse_bitmap_or_icon(name, self.icons)

    def parse_bitmap_or_icon(self, name, dic):
        if False:
            for i in range(10):
                print('nop')
        self.getToken()
        while not self.token.startswith('"'):
            self.getToken()
        bmf = self.token[1:-1]
        dic[name] = bmf

    def parse_dialog(self, name):
        if False:
            while True:
                i = 10
        dlg = DialogDef(name, self.addId(name))
        assert len(dlg.controls) == 0
        self._dialogs[name] = dlg
        extras = []
        self.getToken()
        while not self.token.isdigit():
            self.debug('extra', self.token)
            extras.append(self.token)
            self.getToken()
        dlg.x = int(self.token)
        self.getCommaToken()
        self.getToken()
        dlg.y = int(self.token)
        self.getCommaToken()
        self.getToken()
        dlg.w = int(self.token)
        self.getCommaToken()
        self.getToken()
        dlg.h = int(self.token)
        self.getToken()
        while not (self.token is None or self.token == '' or self.token == 'END'):
            if self.token == 'STYLE':
                self.dialogStyle(dlg)
            elif self.token == 'EXSTYLE':
                self.dialogExStyle(dlg)
            elif self.token == 'CAPTION':
                self.dialogCaption(dlg)
            elif self.token == 'FONT':
                self.dialogFont(dlg)
            elif self.token == 'BEGIN':
                self.controls(dlg)
            else:
                break
        self.dialogs[name] = dlg.createDialogTemplate()

    def dialogStyle(self, dlg):
        if False:
            print('Hello World!')
        (dlg.style, dlg.styles) = self.styles([], win32con.DS_SETFONT)

    def dialogExStyle(self, dlg):
        if False:
            for i in range(10):
                print('nop')
        self.getToken()
        (dlg.styleEx, dlg.stylesEx) = self.styles([], 0)

    def styles(self, defaults, defaultStyle):
        if False:
            while True:
                i = 10
        list = defaults
        style = defaultStyle
        if 'STYLE' == self.token:
            self.getToken()
        i = 0
        Not = False
        while (i % 2 == 1 and ('|' == self.token or 'NOT' == self.token) or i % 2 == 0) and (not self.token is None):
            Not = False
            if 'NOT' == self.token:
                Not = True
                self.getToken()
            i += 1
            if self.token != '|':
                if self.token in win32con.__dict__:
                    value = getattr(win32con, self.token)
                elif self.token in commctrl.__dict__:
                    value = getattr(commctrl, self.token)
                else:
                    value = 0
                if Not:
                    list.append('NOT ' + self.token)
                    self.debug('styles add Not', self.token, value)
                    style &= ~value
                else:
                    list.append(self.token)
                    self.debug('styles add', self.token, value)
                    style |= value
            self.getToken()
        self.debug('style is ', style)
        return (style, list)

    def dialogCaption(self, dlg):
        if False:
            i = 10
            return i + 15
        if 'CAPTION' == self.token:
            self.getToken()
        self.token = self.token[1:-1]
        self.debug('Caption is:', self.token)
        dlg.caption = self.token
        self.getToken()

    def dialogFont(self, dlg):
        if False:
            while True:
                i = 10
        if 'FONT' == self.token:
            self.getToken()
        dlg.fontSize = int(self.token)
        self.getCommaToken()
        self.getToken()
        dlg.font = self.token[1:-1]
        self.getToken()
        while 'BEGIN' != self.token:
            self.getToken()

    def controls(self, dlg):
        if False:
            i = 10
            return i + 15
        if self.token == 'BEGIN':
            self.getToken()
        without_text = ['EDITTEXT', 'COMBOBOX', 'LISTBOX', 'SCROLLBAR']
        while self.token != 'END':
            control = ControlDef()
            control.controlType = self.token
            self.getToken()
            if control.controlType not in without_text:
                if self.token[0:1] == '"':
                    control.label = self.currentQuotedString()
                elif self.token == '-' or self.token.isdigit():
                    control.label = str(self.currentNumberToken())
                else:
                    control.label = str(self.addId(self.token))
                self.getCommaToken()
                self.getToken()
            if self.token == '-' or self.token.isdigit():
                control.id = self.currentNumberToken()
                control.idNum = control.id
            else:
                control.id = self.token
                control.idNum = self.addId(control.id)
            self.getCommaToken()
            if control.controlType == 'CONTROL':
                self.getToken()
                control.subType = self.token[1:-1]
                thisDefaultStyle = defaultControlStyle | _addDefaults.get(control.subType, 0)
                self.getCommaToken()
                self.getToken()
                (control.style, control.styles) = self.styles([], thisDefaultStyle)
            else:
                thisDefaultStyle = defaultControlStyle | _addDefaults.get(control.controlType, 0)
                control.style = thisDefaultStyle
            control.x = int(self.getToken())
            self.getCommaToken()
            control.y = int(self.getToken())
            self.getCommaToken()
            control.w = int(self.getToken())
            self.getCommaToken()
            self.getToken()
            control.h = int(self.token)
            self.getToken()
            if self.token == ',':
                self.getToken()
                (control.style, control.styles) = self.styles([], thisDefaultStyle)
            if self.token == ',':
                self.getToken()
                (control.styleEx, control.stylesEx) = self.styles([], defaultControlStyleEx)
            dlg.controls.append(control)

def ParseStreams(rc_file, h_file):
    if False:
        for i in range(10):
            print('nop')
    rcp = RCParser()
    if h_file:
        rcp.parseH(h_file)
    try:
        rcp.load(rc_file)
    except:
        lex = getattr(rcp, 'lex', None)
        if lex:
            print('ERROR parsing dialogs at line', lex.lineno)
            print('Next 10 tokens are:')
            for i in range(10):
                print(lex.get_token(), end=' ')
            print()
        raise
    return rcp

def Parse(rc_name, h_name=None):
    if False:
        i = 10
        return i + 15
    if h_name:
        h_file = open(h_name, 'r')
    else:
        h_name = rc_name[:-2] + 'h'
        try:
            h_file = open(h_name, 'r')
        except OSError:
            h_name = os.path.join(os.path.dirname(rc_name), 'resource.h')
            try:
                h_file = open(h_name, 'r')
            except OSError:
                h_file = None
    rc_file = open(rc_name, 'r')
    try:
        return ParseStreams(rc_file, h_file)
    finally:
        if h_file is not None:
            h_file.close()
        rc_file.close()
    return rcp

def GenerateFrozenResource(rc_name, output_name, h_name=None):
    if False:
        i = 10
        return i + 15
    "Converts an .rc windows resource source file into a python source file\n    with the same basic public interface as the rest of this module.\n    Particularly useful for py2exe or other 'freeze' type solutions,\n    where a frozen .py file can be used inplace of a real .rc file.\n    "
    rcp = Parse(rc_name, h_name)
    in_stat = os.stat(rc_name)
    out = open(output_name, 'wt')
    out.write('#%s\n' % output_name)
    out.write('#This is a generated file. Please edit %s instead.\n' % rc_name)
    out.write('__version__=%r\n' % __version__)
    out.write('_rc_size_=%d\n_rc_mtime_=%d\n' % (in_stat[stat.ST_SIZE], in_stat[stat.ST_MTIME]))
    out.write('class StringDef:\n')
    out.write('\tdef __init__(self, id, idNum, value):\n')
    out.write('\t\tself.id = id\n')
    out.write('\t\tself.idNum = idNum\n')
    out.write('\t\tself.value = value\n')
    out.write('\tdef __repr__(self):\n')
    out.write('\t\treturn "StringDef(%r, %r, %r)" % (self.id, self.idNum, self.value)\n')
    out.write('class FakeParser:\n')
    for name in ('dialogs', 'ids', 'names', 'bitmaps', 'icons', 'stringTable'):
        out.write(f'\t{name} = \\\n')
        pprint.pprint(getattr(rcp, name), out)
        out.write('\n')
    out.write('def Parse(s):\n')
    out.write('\treturn FakeParser()\n')
    out.close()
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print(__doc__)
        print()
        print('See test_win32rcparser.py, and the win32rcparser directory (both')
        print("in the test suite) for an example of this module's usage.")
    else:
        filename = sys.argv[1]
        if '-v' in sys.argv:
            RCParser.debugEnabled = 1
        print("Dumping all resources in '%s'" % filename)
        resources = Parse(filename)
        for (id, ddef) in resources.dialogs.items():
            print('Dialog %s (%d controls)' % (id, len(ddef)))
            pprint.pprint(ddef)
            print()
        for (id, sdef) in resources.stringTable.items():
            print(f'String {id}={sdef.value!r}')
            print()
        for (id, sdef) in resources.bitmaps.items():
            print(f'Bitmap {id}={sdef!r}')
            print()
        for (id, sdef) in resources.icons.items():
            print(f'Icon {id}={sdef!r}')
            print()