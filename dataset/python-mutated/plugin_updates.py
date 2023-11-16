__license__ = 'GPL v3'
__copyright__ = '2011, Grant Drake <grant.drake@gmail.com>'
__docformat__ = 'restructuredtext en'
from qt.core import QApplication, Qt, QIcon
from calibre.gui2.actions import InterfaceAction
from calibre.gui2.dialogs.plugin_updater import PluginUpdaterDialog, FILTER_ALL, FILTER_UPDATE_AVAILABLE

class PluginUpdaterAction(InterfaceAction):
    name = 'Plugin Updater'
    action_spec = (_('Plugin updater'), None, _('Update any plugins you have installed in calibre'), ())
    action_type = 'current'

    def genesis(self):
        if False:
            return 10
        self.qaction.setIcon(QIcon.ic('plugins/plugin_updater.png'))
        self.qaction.triggered.connect(self.check_for_plugin_updates)

    def check_for_plugin_updates(self):
        if False:
            print('Hello World!')
        initial_filter = FILTER_UPDATE_AVAILABLE
        mods = QApplication.keyboardModifiers()
        if mods & Qt.KeyboardModifier.ControlModifier or mods & Qt.KeyboardModifier.ShiftModifier:
            initial_filter = FILTER_ALL
        d = PluginUpdaterDialog(self.gui, initial_filter=initial_filter)
        d.exec()
        if d.do_restart:
            self.gui.quit(restart=True)