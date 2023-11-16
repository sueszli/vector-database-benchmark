"""
This module contains the abbreviation settings dialog and used components.
This dialog allows the user to set and configure abbreviations to trigger scripts and phrases.
"""
from PyQt5 import QtCore
from PyQt5.QtWidgets import QListWidgetItem, QDialogButtonBox
import autokey.model.folder
import autokey.model.helpers
import autokey.model.phrase
from autokey.qtui import common as ui_common
logger = __import__('autokey.logger').logger.get_logger(__name__)
WORD_CHAR_OPTIONS = {'All non-word': autokey.model.helpers.DEFAULT_WORDCHAR_REGEX, 'Space and Enter': '[^ \\n]', 'Tab': '[^\\t]'}
WORD_CHAR_OPTIONS_ORDERED = tuple(sorted(WORD_CHAR_OPTIONS.keys()))

class AbbrListItem(QListWidgetItem):
    """
    This is a list item used in the abbreviation QListWidget list.
    It simply holds a string value i.e. the user defined abbreviation string.
    """

    def __init__(self, text):
        if False:
            for i in range(10):
                print('nop')
        super(AbbrListItem, self).__init__(text)
        self.setFlags(self.flags() | QtCore.Qt.ItemFlags(QtCore.Qt.ItemIsEditable))

    def setData(self, role, value):
        if False:
            while True:
                i = 10
        if value == '':
            self.listWidget().itemChanged.emit(self)
        else:
            QListWidgetItem.setData(self, role, value)

class AbbrSettingsDialog(*ui_common.inherits_from_ui_file_with_name('abbrsettings')):

    def __init__(self, parent):
        if False:
            return 10
        super().__init__(parent)
        self.setupUi()
        self._reset_word_char_combobox()

    def setupUi(self):
        if False:
            for i in range(10):
                print('nop')
        self.setObjectName('Form')
        super().setupUi(self)

    def on_addButton_pressed(self):
        if False:
            print('Hello World!')
        logger.info('New abbreviation added.')
        item = AbbrListItem('')
        self.abbrListWidget.addItem(item)
        self.abbrListWidget.editItem(item)
        self.removeButton.setEnabled(True)

    def on_removeButton_pressed(self):
        if False:
            return 10
        item = self.abbrListWidget.takeItem(self.abbrListWidget.currentRow())
        if item is not None:
            logger.info('User deletes abbreviation with text: {}'.format(item.text()))
        if self.abbrListWidget.count() == 0:
            logger.debug('Last abbreviation deleted, disabling delete and OK buttons.')
            self.removeButton.setEnabled(False)
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)

    def on_abbrListWidget_itemChanged(self, item):
        if False:
            while True:
                i = 10
        if ui_common.EMPTY_FIELD_REGEX.match(item.text()):
            row = self.abbrListWidget.row(item)
            self.abbrListWidget.takeItem(row)
            logger.debug('User deleted abbreviation content. Deleted empty list element.')
            del item
        else:
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)
        if self.abbrListWidget.count() == 0:
            logger.debug('Last abbreviation deleted, disabling delete and OK buttons.')
            self.removeButton.setEnabled(False)
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)

    def on_abbrListWidget_itemDoubleClicked(self, item):
        if False:
            return 10
        self.abbrListWidget.editItem(item)

    def on_ignoreCaseCheckbox_stateChanged(self, state):
        if False:
            print('Hello World!')
        if not state:
            self.matchCaseCheckbox.setChecked(False)

    def on_matchCaseCheckbox_stateChanged(self, state):
        if False:
            i = 10
            return i + 15
        if state:
            self.ignoreCaseCheckbox.setChecked(True)

    def on_immediateCheckbox_stateChanged(self, state):
        if False:
            for i in range(10):
                print('nop')
        if state:
            self.omitTriggerCheckbox.setChecked(False)
            self.omitTriggerCheckbox.setEnabled(False)
            self.wordCharCombo.setEnabled(False)
        else:
            self.omitTriggerCheckbox.setEnabled(True)
            self.wordCharCombo.setEnabled(True)

    def load(self, item):
        if False:
            return 10
        self.targetItem = item
        self.abbrListWidget.clear()
        if autokey.model.helpers.TriggerMode.ABBREVIATION in item.modes:
            for abbr in item.abbreviations:
                self.abbrListWidget.addItem(AbbrListItem(abbr))
            self.removeButton.setEnabled(True)
            self.abbrListWidget.setCurrentRow(0)
        else:
            self.removeButton.setEnabled(False)
        self.removeTypedCheckbox.setChecked(item.backspace)
        self._reset_word_char_combobox()
        wordCharRegex = item.get_word_chars()
        if wordCharRegex in list(WORD_CHAR_OPTIONS.values()):
            for (desc, regex) in WORD_CHAR_OPTIONS.items():
                if item.get_word_chars() == regex:
                    self.wordCharCombo.setCurrentIndex(WORD_CHAR_OPTIONS_ORDERED.index(desc))
                    break
        else:
            self.wordCharCombo.addItem(autokey.model.helpers.extract_wordchars(wordCharRegex))
            self.wordCharCombo.setCurrentIndex(len(WORD_CHAR_OPTIONS))
        if isinstance(item, autokey.model.folder.Folder):
            self.omitTriggerCheckbox.setVisible(False)
        else:
            self.omitTriggerCheckbox.setVisible(True)
            self.omitTriggerCheckbox.setChecked(item.omitTrigger)
        if isinstance(item, autokey.model.phrase.Phrase):
            self.matchCaseCheckbox.setVisible(True)
            self.matchCaseCheckbox.setChecked(item.matchCase)
        else:
            self.matchCaseCheckbox.setVisible(False)
        self.ignoreCaseCheckbox.setChecked(item.ignoreCase)
        self.triggerInsideCheckbox.setChecked(item.triggerInside)
        self.immediateCheckbox.setChecked(item.immediate)
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(bool(self.get_abbrs()))

    def save(self, item):
        if False:
            while True:
                i = 10
        item.modes.append(autokey.model.helpers.TriggerMode.ABBREVIATION)
        item.clear_abbreviations()
        item.abbreviations = self.get_abbrs()
        item.backspace = self.removeTypedCheckbox.isChecked()
        option = str(self.wordCharCombo.currentText())
        if option in WORD_CHAR_OPTIONS:
            item.set_word_chars(WORD_CHAR_OPTIONS[option])
        else:
            item.set_word_chars(autokey.model.helpers.make_wordchar_re(option))
        if not isinstance(item, autokey.model.folder.Folder):
            item.omitTrigger = self.omitTriggerCheckbox.isChecked()
        if isinstance(item, autokey.model.phrase.Phrase):
            item.matchCase = self.matchCaseCheckbox.isChecked()
        item.ignoreCase = self.ignoreCaseCheckbox.isChecked()
        item.triggerInside = self.triggerInsideCheckbox.isChecked()
        item.immediate = self.immediateCheckbox.isChecked()

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.removeButton.setEnabled(False)
        self.abbrListWidget.clear()
        self._reset_word_char_combobox()
        self.omitTriggerCheckbox.setChecked(False)
        self.removeTypedCheckbox.setChecked(True)
        self.matchCaseCheckbox.setChecked(False)
        self.ignoreCaseCheckbox.setChecked(False)
        self.triggerInsideCheckbox.setChecked(False)
        self.immediateCheckbox.setChecked(False)

    def _reset_word_char_combobox(self):
        if False:
            print('Hello World!')
        self.wordCharCombo.clear()
        for item in WORD_CHAR_OPTIONS_ORDERED:
            self.wordCharCombo.addItem(item)
        self.wordCharCombo.setCurrentIndex(0)

    def get_abbrs(self):
        if False:
            i = 10
            return i + 15
        ret = []
        for i in range(self.abbrListWidget.count()):
            text = self.abbrListWidget.item(i).text()
            ret.append(str(text))
        return ret

    def get_abbrs_readable(self):
        if False:
            for i in range(10):
                print('nop')
        abbrs = self.get_abbrs()
        if len(abbrs) == 1:
            return abbrs[0]
        else:
            return '[%s]' % ','.join(abbrs)

    def reset_focus(self):
        if False:
            for i in range(10):
                print('nop')
        self.addButton.setFocus()

    def accept(self):
        if False:
            i = 10
            return i + 15
        super().accept()

    def reject(self):
        if False:
            while True:
                i = 10
        self.load(self.targetItem)
        super().reject()