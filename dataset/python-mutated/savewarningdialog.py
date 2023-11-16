from PyQt6 import QtCore, QtWidgets
from picard.config import get_config

class SaveWarningDialog:

    def __init__(self, parent, file_count):
        if False:
            while True:
                i = 10
        actions = []
        config = get_config()
        if not config.setting['dont_write_tags']:
            actions.append(ngettext('overwrite existing metadata (tags) within the file', 'overwrite existing metadata (tags) within the files', file_count))
        if config.setting['rename_files']:
            actions.append(ngettext('rename the file', 'rename the files', file_count))
        if config.setting['move_files']:
            actions.append(ngettext('move the file to a different location', 'move the files to a different location', file_count))
        if actions:
            header = ngettext('You are about to save {file_count:,d} file and this will:', 'You are about to save {file_count:,d} files and this will:', file_count).format(file_count=file_count)
            footer = _('<strong>This action cannot be undone.</strong> Do you want to continue?')
            list_of_actions = ''
            for action in actions:
                list_of_actions += _('<li>{action}</li>').format(action=action)
            warning_text = _('<p>{header}</p><ul>{list_of_actions}</ul><p>{footer}</p>').format(header=header, list_of_actions=list_of_actions, footer=footer)
        else:
            warning_text = _('There are no actions selected. No changes will be saved.')
        disable_text = _("Don't show this warning again.")
        self.disable = False
        self.msg = QtWidgets.QMessageBox(parent)
        self.msg.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        self.msg.setText(warning_text)
        self.msg.setWindowTitle(_('File Save Warning'))
        self.msg.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.cb = QtWidgets.QCheckBox(disable_text)
        self.cb.setChecked(False)
        self.cb.toggled.connect(self._set_state)
        self.msg.setCheckBox(self.cb)
        self.msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel)
        self.msg.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)

    def _set_state(self):
        if False:
            for i in range(10):
                print('nop')
        self.disable = not self.disable

    def show(self):
        if False:
            while True:
                i = 10
        return (self.msg.exec() == QtWidgets.QMessageBox.StandardButton.Ok, self.disable)