from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtPrintSupport import *
import os
import sys
import uuid
FONT_SIZES = [7, 8, 9, 10, 11, 12, 13, 14, 18, 24, 36, 48, 64, 72, 96, 144, 288]
IMAGE_EXTENSIONS = ['.jpg', '.png', '.bmp']
HTML_EXTENSIONS = ['.htm', '.html']

def hexuuid():
    if False:
        return 10
    return uuid.uuid4().hex

def splitext(p):
    if False:
        i = 10
        return i + 15
    return os.path.splitext(p)[1].lower()

class TextEdit(QTextEdit):

    def canInsertFromMimeData(self, source):
        if False:
            print('Hello World!')
        if source.hasImage():
            return True
        else:
            return super(TextEdit, self).canInsertFromMimeData(source)

    def insertFromMimeData(self, source):
        if False:
            print('Hello World!')
        cursor = self.textCursor()
        document = self.document()
        if source.hasUrls():
            for u in source.urls():
                file_ext = splitext(str(u.toLocalFile()))
                if u.isLocalFile() and file_ext in IMAGE_EXTENSIONS:
                    image = QImage(u.toLocalFile())
                    document.addResource(QTextDocument.ImageResource, u, image)
                    cursor.insertImage(u.toLocalFile())
                else:
                    break
            else:
                return
        elif source.hasImage():
            image = source.imageData()
            uuid = hexuuid()
            document.addResource(QTextDocument.ImageResource, uuid, image)
            cursor.insertImage(uuid)
            return
        super(TextEdit, self).insertFromMimeData(source)

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(MainWindow, self).__init__(*args, **kwargs)
        layout = QVBoxLayout()
        self.editor = TextEdit()
        self.editor.setAutoFormatting(QTextEdit.AutoAll)
        self.editor.selectionChanged.connect(self.update_format)
        font = QFont('Times', 12)
        self.editor.setFont(font)
        self.editor.setFontPointSize(12)
        self.path = None
        layout.addWidget(self.editor)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        file_toolbar = QToolBar('File')
        file_toolbar.setIconSize(QSize(14, 14))
        self.addToolBar(file_toolbar)
        file_menu = self.menuBar().addMenu('&File')
        open_file_action = QAction(QIcon(os.path.join('images', 'blue-folder-open-document.png')), 'Open file...', self)
        open_file_action.setStatusTip('Open file')
        open_file_action.triggered.connect(self.file_open)
        file_menu.addAction(open_file_action)
        file_toolbar.addAction(open_file_action)
        save_file_action = QAction(QIcon(os.path.join('images', 'disk.png')), 'Save', self)
        save_file_action.setStatusTip('Save current page')
        save_file_action.triggered.connect(self.file_save)
        file_menu.addAction(save_file_action)
        file_toolbar.addAction(save_file_action)
        saveas_file_action = QAction(QIcon(os.path.join('images', 'disk--pencil.png')), 'Save As...', self)
        saveas_file_action.setStatusTip('Save current page to specified file')
        saveas_file_action.triggered.connect(self.file_saveas)
        file_menu.addAction(saveas_file_action)
        file_toolbar.addAction(saveas_file_action)
        print_action = QAction(QIcon(os.path.join('images', 'printer.png')), 'Print...', self)
        print_action.setStatusTip('Print current page')
        print_action.triggered.connect(self.file_print)
        file_menu.addAction(print_action)
        file_toolbar.addAction(print_action)
        edit_toolbar = QToolBar('Edit')
        edit_toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(edit_toolbar)
        edit_menu = self.menuBar().addMenu('&Edit')
        undo_action = QAction(QIcon(os.path.join('images', 'arrow-curve-180-left.png')), 'Undo', self)
        undo_action.setStatusTip('Undo last change')
        undo_action.triggered.connect(self.editor.undo)
        edit_menu.addAction(undo_action)
        redo_action = QAction(QIcon(os.path.join('images', 'arrow-curve.png')), 'Redo', self)
        redo_action.setStatusTip('Redo last change')
        redo_action.triggered.connect(self.editor.redo)
        edit_toolbar.addAction(redo_action)
        edit_menu.addAction(redo_action)
        edit_menu.addSeparator()
        cut_action = QAction(QIcon(os.path.join('images', 'scissors.png')), 'Cut', self)
        cut_action.setStatusTip('Cut selected text')
        cut_action.setShortcut(QKeySequence.Cut)
        cut_action.triggered.connect(self.editor.cut)
        edit_toolbar.addAction(cut_action)
        edit_menu.addAction(cut_action)
        copy_action = QAction(QIcon(os.path.join('images', 'document-copy.png')), 'Copy', self)
        copy_action.setStatusTip('Copy selected text')
        cut_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(self.editor.copy)
        edit_toolbar.addAction(copy_action)
        edit_menu.addAction(copy_action)
        paste_action = QAction(QIcon(os.path.join('images', 'clipboard-paste-document-text.png')), 'Paste', self)
        paste_action.setStatusTip('Paste from clipboard')
        cut_action.setShortcut(QKeySequence.Paste)
        paste_action.triggered.connect(self.editor.paste)
        edit_toolbar.addAction(paste_action)
        edit_menu.addAction(paste_action)
        select_action = QAction(QIcon(os.path.join('images', 'selection-input.png')), 'Select all', self)
        select_action.setStatusTip('Select all text')
        cut_action.setShortcut(QKeySequence.SelectAll)
        select_action.triggered.connect(self.editor.selectAll)
        edit_menu.addAction(select_action)
        edit_menu.addSeparator()
        wrap_action = QAction(QIcon(os.path.join('images', 'arrow-continue.png')), 'Wrap text to window', self)
        wrap_action.setStatusTip('Toggle wrap text to window')
        wrap_action.setCheckable(True)
        wrap_action.setChecked(True)
        wrap_action.triggered.connect(self.edit_toggle_wrap)
        edit_menu.addAction(wrap_action)
        format_toolbar = QToolBar('Format')
        format_toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(format_toolbar)
        format_menu = self.menuBar().addMenu('&Format')
        self.fonts = QFontComboBox()
        self.fonts.currentFontChanged.connect(self.editor.setCurrentFont)
        format_toolbar.addWidget(self.fonts)
        self.fontsize = QComboBox()
        self.fontsize.addItems([str(s) for s in FONT_SIZES])
        self.fontsize.currentIndexChanged[str].connect(lambda s: self.editor.setFontPointSize(float(s)))
        format_toolbar.addWidget(self.fontsize)
        self.bold_action = QAction(QIcon(os.path.join('images', 'edit-bold.png')), 'Bold', self)
        self.bold_action.setStatusTip('Bold')
        self.bold_action.setShortcut(QKeySequence.Bold)
        self.bold_action.setCheckable(True)
        self.bold_action.toggled.connect(lambda x: self.editor.setFontWeight(QFont.Bold if x else QFont.Normal))
        format_toolbar.addAction(self.bold_action)
        format_menu.addAction(self.bold_action)
        self.italic_action = QAction(QIcon(os.path.join('images', 'edit-italic.png')), 'Italic', self)
        self.italic_action.setStatusTip('Italic')
        self.italic_action.setShortcut(QKeySequence.Italic)
        self.italic_action.setCheckable(True)
        self.italic_action.toggled.connect(self.editor.setFontItalic)
        format_toolbar.addAction(self.italic_action)
        format_menu.addAction(self.italic_action)
        self.underline_action = QAction(QIcon(os.path.join('images', 'edit-underline.png')), 'Underline', self)
        self.underline_action.setStatusTip('Underline')
        self.underline_action.setShortcut(QKeySequence.Underline)
        self.underline_action.setCheckable(True)
        self.underline_action.toggled.connect(self.editor.setFontUnderline)
        format_toolbar.addAction(self.underline_action)
        format_menu.addAction(self.underline_action)
        format_menu.addSeparator()
        self.alignl_action = QAction(QIcon(os.path.join('images', 'edit-alignment.png')), 'Align left', self)
        self.alignl_action.setStatusTip('Align text left')
        self.alignl_action.setCheckable(True)
        self.alignl_action.triggered.connect(lambda : self.editor.setAlignment(Qt.AlignLeft))
        format_toolbar.addAction(self.alignl_action)
        format_menu.addAction(self.alignl_action)
        self.alignc_action = QAction(QIcon(os.path.join('images', 'edit-alignment-center.png')), 'Align center', self)
        self.alignc_action.setStatusTip('Align text center')
        self.alignc_action.setCheckable(True)
        self.alignc_action.triggered.connect(lambda : self.editor.setAlignment(Qt.AlignCenter))
        format_toolbar.addAction(self.alignc_action)
        format_menu.addAction(self.alignc_action)
        self.alignr_action = QAction(QIcon(os.path.join('images', 'edit-alignment-right.png')), 'Align right', self)
        self.alignr_action.setStatusTip('Align text right')
        self.alignr_action.setCheckable(True)
        self.alignr_action.triggered.connect(lambda : self.editor.setAlignment(Qt.AlignRight))
        format_toolbar.addAction(self.alignr_action)
        format_menu.addAction(self.alignr_action)
        self.alignj_action = QAction(QIcon(os.path.join('images', 'edit-alignment-justify.png')), 'Justify', self)
        self.alignj_action.setStatusTip('Justify text')
        self.alignj_action.setCheckable(True)
        self.alignj_action.triggered.connect(lambda : self.editor.setAlignment(Qt.AlignJustify))
        format_toolbar.addAction(self.alignj_action)
        format_menu.addAction(self.alignj_action)
        format_group = QActionGroup(self)
        format_group.setExclusive(True)
        format_group.addAction(self.alignl_action)
        format_group.addAction(self.alignc_action)
        format_group.addAction(self.alignr_action)
        format_group.addAction(self.alignj_action)
        format_menu.addSeparator()
        self._format_actions = [self.fonts, self.fontsize, self.bold_action, self.italic_action, self.underline_action]
        self.update_format()
        self.update_title()
        self.show()

    def block_signals(self, objects, b):
        if False:
            while True:
                i = 10
        for o in objects:
            o.blockSignals(b)

    def update_format(self):
        if False:
            i = 10
            return i + 15
        '\n        Update the font format toolbar/actions when a new text selection is made. This is necessary to keep\n        toolbars/etc. in sync with the current edit state.\n        :return:\n        '
        self.block_signals(self._format_actions, True)
        self.fonts.setCurrentFont(self.editor.currentFont())
        self.fontsize.setCurrentText(str(int(self.editor.fontPointSize())))
        self.italic_action.setChecked(self.editor.fontItalic())
        self.underline_action.setChecked(self.editor.fontUnderline())
        self.bold_action.setChecked(self.editor.fontWeight() == QFont.Bold)
        self.alignl_action.setChecked(self.editor.alignment() == Qt.AlignLeft)
        self.alignc_action.setChecked(self.editor.alignment() == Qt.AlignCenter)
        self.alignr_action.setChecked(self.editor.alignment() == Qt.AlignRight)
        self.alignj_action.setChecked(self.editor.alignment() == Qt.AlignJustify)
        self.block_signals(self._format_actions, False)

    def dialog_critical(self, s):
        if False:
            for i in range(10):
                print('nop')
        dlg = QMessageBox(self)
        dlg.setText(s)
        dlg.setIcon(QMessageBox.Critical)
        dlg.show()

    def file_open(self):
        if False:
            return 10
        (path, _) = QFileDialog.getOpenFileName(self, 'Open file', '', 'HTML documents (*.html);Text documents (*.txt);All files (*.*)')
        try:
            with open(path, 'rU') as f:
                text = f.read()
        except Exception as e:
            self.dialog_critical(str(e))
        else:
            self.path = path
            self.editor.setText(text)
            self.update_title()

    def file_save(self):
        if False:
            for i in range(10):
                print('nop')
        if self.path is None:
            return self.file_saveas()
        text = self.editor.toHtml() if splitext(self.path) in HTML_EXTENSIONS else self.editor.toPlainText()
        try:
            with open(self.path, 'w') as f:
                f.write(text)
        except Exception as e:
            self.dialog_critical(str(e))

    def file_saveas(self):
        if False:
            return 10
        (path, _) = QFileDialog.getSaveFileName(self, 'Save file', '', 'HTML documents (*.html);Text documents (*.txt);All files (*.*)')
        if not path:
            return
        text = self.editor.toHtml() if splitext(path) in HTML_EXTENSIONS else self.editor.toPlainText()
        try:
            with open(path, 'w') as f:
                f.write(text)
        except Exception as e:
            self.dialog_critical(str(e))
        else:
            self.path = path
            self.update_title()

    def file_print(self):
        if False:
            i = 10
            return i + 15
        dlg = QPrintDialog()
        if dlg.exec_():
            self.editor.print_(dlg.printer())

    def update_title(self):
        if False:
            for i in range(10):
                print('nop')
        self.setWindowTitle('%s - Megasolid Idiom' % (os.path.basename(self.path) if self.path else 'Untitled'))

    def edit_toggle_wrap(self):
        if False:
            while True:
                i = 10
        self.editor.setLineWrapMode(1 if self.editor.lineWrapMode() == 0 else 0)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName('Megasolid Idiom')
    window = MainWindow()
    app.exec_()