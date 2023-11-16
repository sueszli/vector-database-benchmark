from qtpy.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QCheckBox, QHBoxLayout, QGridLayout

class EditSrcCodeInfoDialog(QDialog):
    dont_show_again = False

    def __init__(self, parent):
        if False:
            print('Hello World!')
        super(EditSrcCodeInfoDialog, self).__init__(parent)
        self.setLayout(QGridLayout())
        info_text_edit = QTextEdit()
        info_text_edit.setHtml('\n            <h2 style="font-family: Poppins; font-size: xx-large; color: #a9d5ef;">Some info before you delete the\n            universe</h2>\n            <div style="font-family: Corbel; font-size: x-large;">\n                <p>\n                    Yes, you can change method implementations of objects.\n                    This can be quite useful but since changing an instance\'s implementation at runtime is kinda sketchy, \n                    you should be a bit careful, it\'s not exactly bulletproof, and doesnt <i>always</i> work.\n                    When you override a method implementation, a new function object will be created using python\'s ast \n                    module, which then gets bound to the object as method, which essentially shadows the old implementation. \n                    Therefore, you might need to add imports etc. you node uses in the original nodes package. \n                    All changes are temporary and only apply on a single \n                    object.\n                </p>\n                <p>\n                    Have fun.\n                </p>\n            </div>\n        ')
        info_text_edit.setReadOnly(True)
        self.layout().addWidget(info_text_edit, 0, 0, 1, 2)
        dont_show_again_button = QPushButton('Stop being annoying')
        dont_show_again_button.clicked.connect(self.close_and_dont_show_again)
        self.layout().addWidget(dont_show_again_button, 1, 0)
        ok_button = QPushButton('Got it')
        ok_button.clicked.connect(self.accept)
        self.layout().addWidget(ok_button, 1, 1)
        ok_button.setFocus()
        self.setWindowTitle('Editing Source Code Info')
        self.resize(560, 366)

    def close_and_dont_show_again(self):
        if False:
            i = 10
            return i + 15
        EditSrcCodeInfoDialog.dont_show_again = True
        self.accept()