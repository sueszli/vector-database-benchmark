"""
Syntax Highlighting
===================

This example shows how to write a syntax-highlighted editor with Qt and Lark

Requirements:

  PyQt5==5.15.8
  QScintilla==2.13.4
"""
import sys
import textwrap
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QColor, QFont, QFontMetrics
from PyQt5.Qsci import QsciScintilla
from PyQt5.Qsci import QsciLexerCustom
from lark import Lark

class LexerJson(QsciLexerCustom):

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.create_parser()
        self.create_styles()

    def create_styles(self):
        if False:
            return 10
        deeppink = QColor(249, 38, 114)
        khaki = QColor(230, 219, 116)
        mediumpurple = QColor(174, 129, 255)
        mediumturquoise = QColor(81, 217, 205)
        yellowgreen = QColor(166, 226, 46)
        lightcyan = QColor(213, 248, 232)
        darkslategrey = QColor(39, 40, 34)
        styles = {0: mediumturquoise, 1: mediumpurple, 2: yellowgreen, 3: deeppink, 4: khaki, 5: lightcyan}
        for (style, color) in styles.items():
            self.setColor(color, style)
            self.setPaper(darkslategrey, style)
            self.setFont(self.parent().font(), style)
        self.token_styles = {'COLON': 5, 'COMMA': 5, 'LBRACE': 5, 'LSQB': 5, 'RBRACE': 5, 'RSQB': 5, 'FALSE': 0, 'NULL': 0, 'TRUE': 0, 'STRING': 4, 'NUMBER': 1}

    def create_parser(self):
        if False:
            return 10
        grammar = '\n            anons: ":" "{" "}" "," "[" "]"\n            TRUE: "true"\n            FALSE: "false"\n            NULL: "NULL"\n            %import common.ESCAPED_STRING -> STRING\n            %import common.SIGNED_NUMBER  -> NUMBER\n            %import common.WS\n            %ignore WS\n        '
        self.lark = Lark(grammar, parser=None, lexer='basic')

    def defaultPaper(self, style):
        if False:
            for i in range(10):
                print('nop')
        return QColor(39, 40, 34)

    def language(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Json'

    def description(self, style):
        if False:
            i = 10
            return i + 15
        return {v: k for (k, v) in self.token_styles.items()}.get(style, '')

    def styleText(self, start, end):
        if False:
            i = 10
            return i + 15
        self.startStyling(start)
        text = self.parent().text()[start:end]
        last_pos = 0
        try:
            for token in self.lark.lex(text):
                ws_len = token.start_pos - last_pos
                if ws_len:
                    self.setStyling(ws_len, 0)
                token_len = len(bytearray(token, 'utf-8'))
                self.setStyling(token_len, self.token_styles.get(token.type, 0))
                last_pos = token.start_pos + token_len
        except Exception as e:
            print(e)

class EditorAll(QsciScintilla):

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        font = QFont()
        font.setFamily('Consolas')
        font.setFixedPitch(True)
        font.setPointSize(8)
        font.setBold(True)
        self.setFont(font)
        fontmetrics = QFontMetrics(font)
        self.setMarginsFont(font)
        self.setMarginWidth(0, fontmetrics.width('000') + 6)
        self.setMarginLineNumbers(0, True)
        self.setMarginsForegroundColor(QColor(128, 128, 128))
        self.setMarginsBackgroundColor(QColor(39, 40, 34))
        self.setMarginType(1, self.SymbolMargin)
        self.setMarginWidth(1, 12)
        self.setIndentationsUseTabs(False)
        self.setIndentationWidth(4)
        self.setBackspaceUnindents(True)
        self.setIndentationGuides(True)
        self.setCaretForegroundColor(QColor(247, 247, 241))
        self.setCaretWidth(2)
        self.setSelectionBackgroundColor(QColor(61, 61, 52))
        self.resetSelectionForegroundColor()
        self.SendScintilla(QsciScintilla.SCI_SETMULTIPLESELECTION, True)
        self.SendScintilla(QsciScintilla.SCI_SETMULTIPASTE, 1)
        self.SendScintilla(QsciScintilla.SCI_SETADDITIONALSELECTIONTYPING, True)
        lexer = LexerJson(self)
        self.setLexer(lexer)
EXAMPLE_TEXT = textwrap.dedent('        {\n            "_id": "5b05ffcbcf8e597939b3f5ca",\n            "about": "Excepteur consequat commodo esse voluptate aute aliquip ad sint deserunt commodo eiusmod irure. Sint aliquip sit magna duis eu est culpa aliqua excepteur ut tempor nulla. Aliqua ex pariatur id labore sit. Quis sit ex aliqua veniam exercitation laboris anim adipisicing. Lorem nisi reprehenderit ullamco labore qui sit ut aliqua tempor consequat pariatur proident.",\n            "address": "665 Malbone Street, Thornport, Louisiana, 243",\n            "age": 23,\n            "balance": "$3,216.91",\n            "company": "BULLJUICE",\n            "email": "elisekelley@bulljuice.com",\n            "eyeColor": "brown",\n            "gender": "female",\n            "guid": "d3a6d865-0f64-4042-8a78-4f53de9b0707",\n            "index": 0,\n            "isActive": false,\n            "isActive2": true,\n            "latitude": -18.660714,\n            "longitude": -85.378048,\n            "name": "Elise Kelley",\n            "phone": "+1 (808) 543-3966",\n            "picture": "http://placehold.it/32x32",\n            "registered": "2017-09-30T03:47:40 -02:00",\n            "tags": [\n                "et",\n                "nostrud",\n                "in",\n                "fugiat",\n                "incididunt",\n                "labore",\n                "nostrud"\n            ]\n        }    ')

def main():
    if False:
        for i in range(10):
            print('nop')
    app = QApplication(sys.argv)
    ex = EditorAll()
    ex.setWindowTitle(__file__)
    ex.setText(EXAMPLE_TEXT)
    ex.resize(800, 600)
    ex.show()
    sys.exit(app.exec_())
if __name__ == '__main__':
    main()