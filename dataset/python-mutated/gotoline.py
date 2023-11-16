from qtpy.QtCore import Qt
from qtpy.QtGui import QIntValidator
from qtpy.QtWidgets import QDialog, QLabel, QLineEdit, QGridLayout, QDialogButtonBox, QVBoxLayout, QHBoxLayout
from spyder.api.translations import _

class GoToLineDialog(QDialog):

    def __init__(self, editor):
        if False:
            i = 10
            return i + 15
        QDialog.__init__(self, editor, Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.lineno = None
        self.editor = editor
        self.setWindowTitle(_('Editor'))
        self.setModal(True)
        label = QLabel(_('Go to line:'))
        self.lineedit = QLineEdit()
        validator = QIntValidator(self.lineedit)
        validator.setRange(1, editor.get_line_count())
        self.lineedit.setValidator(validator)
        self.lineedit.textChanged.connect(self.text_has_changed)
        cl_label = QLabel(_('Current line:'))
        cl_label_v = QLabel('<b>%d</b>' % editor.get_cursor_line_number())
        last_label = QLabel(_('Line count:'))
        last_label_v = QLabel('%d' % editor.get_line_count())
        glayout = QGridLayout()
        glayout.addWidget(label, 0, 0, Qt.AlignVCenter | Qt.AlignRight)
        glayout.addWidget(self.lineedit, 0, 1, Qt.AlignVCenter)
        glayout.addWidget(cl_label, 1, 0, Qt.AlignVCenter | Qt.AlignRight)
        glayout.addWidget(cl_label_v, 1, 1, Qt.AlignVCenter)
        glayout.addWidget(last_label, 2, 0, Qt.AlignVCenter | Qt.AlignRight)
        glayout.addWidget(last_label_v, 2, 1, Qt.AlignVCenter)
        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Vertical, self)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        btnlayout = QVBoxLayout()
        btnlayout.addWidget(bbox)
        btnlayout.addStretch(1)
        ok_button = bbox.button(QDialogButtonBox.Ok)
        ok_button.setEnabled(False)
        self.lineedit.textChanged.connect(lambda text: ok_button.setEnabled(len(text) > 0 and text != '+'))
        layout = QHBoxLayout()
        layout.addLayout(glayout)
        layout.addLayout(btnlayout)
        self.setLayout(layout)
        self.lineedit.setFocus()

    def text_has_changed(self, text):
        if False:
            while True:
                i = 10
        "Line edit's text has changed."
        text = str(text)
        if text and text != '+':
            self.lineno = int(text)
        else:
            self.lineno = None
            self.lineedit.clear()

    def get_line_number(self):
        if False:
            i = 10
            return i + 15
        'Return line number.'
        return self.lineno