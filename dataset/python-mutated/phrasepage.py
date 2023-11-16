import subprocess
from PyQt5.QtWidgets import QMessageBox
import autokey.model.phrase
from autokey.qtui import common as ui_common
PROBLEM_MSG_PRIMARY = 'Some problems were found'
PROBLEM_MSG_SECONDARY = '{}\n\nYour changes have not been saved.'

class PhrasePage(*ui_common.inherits_from_ui_file_with_name('phrasepage')):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(PhrasePage, self).__init__()
        self.setupUi(self)
        self.initialising = True
        self.current_phrase = None
        for val in sorted(autokey.model.phrase.SEND_MODES.keys()):
            self.sendModeCombo.addItem(val)
        self.initialising = False

    def load(self, phrase: autokey.model.phrase.Phrase):
        if False:
            print('Hello World!')
        self.current_phrase = phrase
        self.phraseText.setPlainText(phrase.phrase)
        self.showInTrayCheckbox.setChecked(phrase.show_in_tray_menu)
        for (k, v) in autokey.model.phrase.SEND_MODES.items():
            if v == phrase.sendMode:
                self.sendModeCombo.setCurrentIndex(self.sendModeCombo.findText(k))
                break
        if self.is_new_item():
            self.urlLabel.setEnabled(False)
            self.urlLabel.setText('(Unsaved)')
        else:
            ui_common.set_url_label(self.urlLabel, self.current_phrase.path)
        self.promptCheckbox.setChecked(phrase.prompt)
        self.settingsWidget.load(phrase)

    def save(self):
        if False:
            i = 10
            return i + 15
        self.settingsWidget.save()
        self.current_phrase.phrase = str(self.phraseText.toPlainText())
        self.current_phrase.show_in_tray_menu = self.showInTrayCheckbox.isChecked()
        self.current_phrase.sendMode = autokey.model.phrase.SEND_MODES[str(self.sendModeCombo.currentText())]
        self.current_phrase.prompt = self.promptCheckbox.isChecked()
        self.current_phrase.persist()
        ui_common.set_url_label(self.urlLabel, self.current_phrase.path)
        return False

    def get_current_item(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the currently held item.'
        return self.current_phrase

    def set_item_title(self, title):
        if False:
            return 10
        self.current_phrase.description = title

    def rebuild_item_path(self):
        if False:
            while True:
                i = 10
        self.current_phrase.rebuild_path()

    def is_new_item(self):
        if False:
            i = 10
            return i + 15
        return self.current_phrase.path is None

    def reset(self):
        if False:
            while True:
                i = 10
        self.load(self.current_phrase)

    def validate(self):
        if False:
            print('Hello World!')
        errors = []
        phrase = str(self.phraseText.toPlainText())
        if ui_common.EMPTY_FIELD_REGEX.match(phrase):
            errors.append("The phrase content can't be empty")
        errors += self.settingsWidget.validate()
        if errors:
            msg = PROBLEM_MSG_SECONDARY.format('\n'.join([str(e) for e in errors]))
            QMessageBox.critical(self.window(), PROBLEM_MSG_PRIMARY, msg)
        return not bool(errors)

    def set_dirty(self):
        if False:
            return 10
        self.window().set_dirty()

    def undo(self):
        if False:
            i = 10
            return i + 15
        self.phraseText.undo()

    def redo(self):
        if False:
            return 10
        self.phraseText.redo()

    def insert_token(self, token):
        if False:
            i = 10
            return i + 15
        self.phraseText.insertPlainText(token)

    def on_phraseText_textChanged(self):
        if False:
            return 10
        self.set_dirty()

    def on_phraseText_undoAvailable(self, state):
        if False:
            i = 10
            return i + 15
        self.window().set_undo_available(state)

    def on_phraseText_redoAvailable(self, state):
        if False:
            return 10
        self.window().set_redo_available(state)

    def on_predictCheckbox_stateChanged(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.set_dirty()

    def on_promptCheckbox_stateChanged(self, state):
        if False:
            return 10
        self.set_dirty()

    def on_showInTrayCheckbox_stateChanged(self, state):
        if False:
            return 10
        self.set_dirty()

    def on_sendModeCombo_currentIndexChanged(self, index):
        if False:
            print('Hello World!')
        if not self.initialising:
            self.set_dirty()

    def on_urlLabel_leftClickedUrl(self, url=None):
        if False:
            while True:
                i = 10
        if url:
            subprocess.Popen(['/usr/bin/xdg-open', url])