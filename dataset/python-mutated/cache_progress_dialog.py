from __future__ import absolute_import, division, print_function, unicode_literals
__license__ = 'GPL 3'
__copyright__ = '2011, John Schember <john@nachtimwald.com>'
__docformat__ = 'restructuredtext en'
from qt.core import QDialog
from calibre.gui2.store.stores.mobileread.cache_progress_dialog_ui import Ui_Dialog

class CacheProgressDialog(QDialog, Ui_Dialog):

    def __init__(self, parent=None, total=None):
        if False:
            return 10
        QDialog.__init__(self, parent)
        self.setupUi(self)
        self.completed = 0
        self.canceled = False
        self.progress.setValue(0)
        self.progress.setMinimum(0)
        self.progress.setMaximum(total if total else 0)

    def exec(self):
        if False:
            print('Hello World!')
        self.completed = 0
        self.canceled = False
        QDialog.exec(self)
    exec_ = exec

    def open(self):
        if False:
            return 10
        self.completed = 0
        self.canceled = False
        QDialog.open(self)

    def reject(self):
        if False:
            for i in range(10):
                print('nop')
        self.canceled = True
        QDialog.reject(self)

    def update_progress(self):
        if False:
            return 10
        '\n        completed is an int from 0 to total representing the number\n        records that have bee completed.\n        '
        self.set_progress(self.completed + 1)

    def set_message(self, msg):
        if False:
            while True:
                i = 10
        self.message.setText(msg)

    def set_details(self, msg):
        if False:
            return 10
        self.details.setText(msg)

    def set_progress(self, completed):
        if False:
            i = 10
            return i + 15
        '\n        completed is an int from 0 to total representing the number\n        records that have bee completed.\n        '
        self.completed = completed
        self.progress.setValue(self.completed)

    def set_total(self, total):
        if False:
            return 10
        self.progress.setMaximum(total)