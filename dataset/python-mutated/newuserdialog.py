from PyQt6 import QtCore, QtWidgets
from picard.const import PICARD_URLS

class NewUserDialog:

    def __init__(self, parent):
        if False:
            print('Hello World!')
        dialog_text = _("<p><strong>Changes made by Picard are not reversible.</strong></p><p>Picard is a very flexible music tagging tool which can rename your files and overwrite the tags. We <strong>strongly recommend</strong> that you:</p><ul><li>read the <a href='{documentation_url}'>User Guide</a> (also available from the Help menu)</li><li>test with copies of your music and work in small batches</li></ul><p>Picard is open source software written by volunteers. It is provided as-is and with no warranty.</p>").format(documentation_url=PICARD_URLS['documentation_server'])
        self.show_again = True
        show_again_text = _('Show this message again the next time you start Picard.')
        self.msg = QtWidgets.QMessageBox(parent)
        self.msg.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        self.msg.setText(dialog_text)
        self.msg.setWindowTitle(_('New User Warning'))
        self.msg.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.cb = QtWidgets.QCheckBox(show_again_text)
        self.cb.setChecked(self.show_again)
        self.cb.toggled.connect(self._set_state)
        self.msg.setCheckBox(self.cb)
        self.msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)

    def _set_state(self):
        if False:
            print('Hello World!')
        self.show_again = not self.show_again

    def show(self):
        if False:
            print('Hello World!')
        self.msg.exec()
        return self.show_again