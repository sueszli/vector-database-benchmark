__license__ = 'GPL v3'
__copyright__ = '2010, Kovid Goyal <kovid@kovidgoyal.net>'
__docformat__ = 'restructuredtext en'
from qt.core import QLineEdit
from calibre.gui2.dialogs.template_dialog import TemplateDialog

class TemplateLineEditor(QLineEdit):
    """
    Extend the context menu of a QLineEdit to include more actions.
    """

    def __init__(self, parent):
        if False:
            while True:
                i = 10
        QLineEdit.__init__(self, parent)
        self.mi = None
        self.setClearButtonEnabled(True)

    def set_mi(self, mi):
        if False:
            print('Hello World!')
        self.mi = mi

    def contextMenuEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        menu = self.createStandardContextMenu()
        menu.addSeparator()
        action_clear_field = menu.addAction(_('Remove any template from the box'))
        action_clear_field.triggered.connect(self.clear_field)
        action_open_editor = menu.addAction(_('Open template editor'))
        action_open_editor.triggered.connect(self.open_editor)
        menu.exec(event.globalPos())

    def clear_field(self):
        if False:
            return 10
        self.setText('')

    def open_editor(self):
        if False:
            print('Hello World!')
        t = TemplateDialog(self, self.text(), mi=self.mi)
        t.setWindowTitle(_('Edit template'))
        if t.exec():
            self.setText(t.rule[1])