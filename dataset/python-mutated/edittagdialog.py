from PyQt6 import QtCore, QtGui, QtWidgets
from picard.const import RELEASE_FORMATS, RELEASE_PRIMARY_GROUPS, RELEASE_SECONDARY_GROUPS, RELEASE_STATUS
from picard.const.countries import RELEASE_COUNTRIES
from picard.util.tags import TAG_NAMES
from picard.ui import PicardDialog
from picard.ui.ui_edittagdialog import Ui_EditTagDialog
AUTOCOMPLETE_RELEASE_TYPES = [s.lower() for s in sorted(RELEASE_PRIMARY_GROUPS) + sorted(RELEASE_SECONDARY_GROUPS)]
AUTOCOMPLETE_RELEASE_STATUS = sorted((s.lower() for s in RELEASE_STATUS))
AUTOCOMPLETE_RELEASE_COUNTRIES = sorted(RELEASE_COUNTRIES, key=str.casefold)
AUTOCOMPLETE_RELEASE_FORMATS = sorted(RELEASE_FORMATS, key=str.casefold)

class TagEditorDelegate(QtWidgets.QItemDelegate):

    def createEditor(self, parent, option, index):
        if False:
            while True:
                i = 10
        if not index.isValid():
            return None
        tag = self.get_tag_name(index)
        if tag.partition(':')[0] in {'comment', 'lyrics'}:
            editor = QtWidgets.QPlainTextEdit(parent)
            editor.setFrameStyle(editor.style().styleHint(QtWidgets.QStyle.StyleHint.SH_ItemView_DrawDelegateFrame, None, editor))
            editor.setMinimumSize(QtCore.QSize(0, 80))
        else:
            editor = super().createEditor(parent, option, index)
        completer = None
        if tag in {'date', 'originaldate', 'releasedate'}:
            editor.setPlaceholderText(_('YYYY-MM-DD'))
        elif tag == 'originalyear':
            editor.setPlaceholderText(_('YYYY'))
        elif tag == 'releasetype':
            completer = QtWidgets.QCompleter(AUTOCOMPLETE_RELEASE_TYPES, editor)
        elif tag == 'releasestatus':
            completer = QtWidgets.QCompleter(AUTOCOMPLETE_RELEASE_STATUS, editor)
            completer.setModelSorting(QtWidgets.QCompleter.ModelSorting.CaseInsensitivelySortedModel)
        elif tag == 'releasecountry':
            completer = QtWidgets.QCompleter(AUTOCOMPLETE_RELEASE_COUNTRIES, editor)
            completer.setModelSorting(QtWidgets.QCompleter.ModelSorting.CaseInsensitivelySortedModel)
        elif tag == 'media':
            completer = QtWidgets.QCompleter(AUTOCOMPLETE_RELEASE_FORMATS, editor)
            completer.setModelSorting(QtWidgets.QCompleter.ModelSorting.CaseInsensitivelySortedModel)
        if editor and completer:
            completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.UnfilteredPopupCompletion)
            completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
            editor.setCompleter(completer)
        return editor

    def get_tag_name(self, index):
        if False:
            for i in range(10):
                print('nop')
        return self.parent().tag

class EditTagDialog(PicardDialog):

    def __init__(self, window, tag):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(window)
        self.ui = Ui_EditTagDialog()
        self.ui.setupUi(self)
        self.window = window
        self.value_list = self.ui.value_list
        self.metadata_box = window.metadata_box
        self.tag = tag
        self.modified_tags = {}
        self.different = False
        self.default_tags = sorted(set(list(TAG_NAMES.keys()) + self.metadata_box.tag_diff.tag_names))
        if len(self.metadata_box.files) == 1:
            current_file = list(self.metadata_box.files)[0]
            self.default_tags = list(filter(current_file.supports_tag, self.default_tags))
        tag_names = self.ui.tag_names
        tag_names.addItem('')
        visible_tags = [tn for tn in self.default_tags if not tn.startswith('~')]
        tag_names.addItems(visible_tags)
        self.completer = QtWidgets.QCompleter(visible_tags, tag_names)
        self.completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
        tag_names.setCompleter(self.completer)
        self.value_list.model().rowsInserted.connect(self.on_rows_inserted)
        self.value_list.model().rowsRemoved.connect(self.on_rows_removed)
        self.value_list.setItemDelegate(TagEditorDelegate(self))
        self.tag_changed(tag)
        self.value_selection_changed()

    def keyPressEvent(self, event):
        if False:
            print('Hello World!')
        if event.modifiers() == QtCore.Qt.KeyboardModifier.NoModifier and event.key() in {QtCore.Qt.Key.Key_Enter, QtCore.Qt.Key.Key_Return}:
            self.add_or_edit_value()
            event.accept()
        elif event.matches(QtGui.QKeySequence.StandardKey.Delete):
            self.remove_value()
        elif event.key() == QtCore.Qt.Key.Key_Insert:
            self.add_value()
        else:
            super().keyPressEvent(event)

    def tag_selected(self, index):
        if False:
            i = 10
            return i + 15
        self.add_or_edit_value()

    def edit_value(self):
        if False:
            print('Hello World!')
        item = self.value_list.currentItem()
        if item:
            if hasattr(self.value_list, 'isPersistentEditorOpen') and self.value_list.isPersistentEditorOpen(item):
                return
            self.value_list.editItem(item)

    def add_value(self):
        if False:
            print('Hello World!')
        item = QtWidgets.QListWidgetItem()
        item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsEditable)
        self.value_list.addItem(item)
        self.value_list.setCurrentItem(item)
        self.value_list.editItem(item)

    def add_or_edit_value(self):
        if False:
            while True:
                i = 10
        last_item = self.value_list.item(self.value_list.count() - 1)
        if last_item and (not last_item.text()):
            self.value_list.setCurrentItem(last_item)
            self.edit_value()
        else:
            self.add_value()

    def remove_value(self):
        if False:
            print('Hello World!')
        value_list = self.value_list
        row = value_list.currentRow()
        if row == 0 and self.different:
            self.different = False
            self.ui.add_value.setEnabled(True)
        value_list.takeItem(row)

    def on_rows_inserted(self, parent, first, last):
        if False:
            i = 10
            return i + 15
        for row in range(first, last + 1):
            item = self.value_list.item(row)
            self._modified_tag().insert(row, item.text())

    def on_rows_removed(self, parent, first, last):
        if False:
            while True:
                i = 10
        for row in range(first, last + 1):
            del self._modified_tag()[row]

    def move_row_up(self):
        if False:
            return 10
        row = self.value_list.currentRow()
        if row > 0:
            self._move_row(row, -1)

    def move_row_down(self):
        if False:
            print('Hello World!')
        row = self.value_list.currentRow()
        if row + 1 < self.value_list.count():
            self._move_row(row, 1)

    def _move_row(self, row, direction):
        if False:
            print('Hello World!')
        value_list = self.value_list
        item = value_list.takeItem(row)
        new_row = row + direction
        value_list.insertItem(new_row, item)
        value_list.setCurrentRow(new_row)

    def disable_all(self):
        if False:
            for i in range(10):
                print('nop')
        self.value_list.clear()
        self.value_list.setEnabled(False)
        self.ui.add_value.setEnabled(False)

    def enable_all(self):
        if False:
            while True:
                i = 10
        self.value_list.setEnabled(True)
        self.ui.add_value.setEnabled(True)

    def tag_changed(self, tag):
        if False:
            print('Hello World!')
        tag_names = self.ui.tag_names
        tag_names.editTextChanged.disconnect(self.tag_changed)
        line_edit = tag_names.lineEdit()
        cursor_pos = line_edit.cursorPosition()
        flags = QtCore.Qt.MatchFlag.MatchFixedString | QtCore.Qt.MatchFlag.MatchCaseSensitive
        if self.tag and self.tag not in self.default_tags and (self._modified_tag() == ['']):
            tag_names.removeItem(tag_names.findText(self.tag, flags))
        row = tag_names.findText(tag, flags)
        self.tag = tag
        if row <= 0:
            if tag:
                tag_names.addItem(tag)
                tag_names.model().sort(0)
                row = tag_names.findText(tag, flags)
            else:
                self.disable_all()
                tag_names.setCurrentIndex(0)
                tag_names.editTextChanged.connect(self.tag_changed)
                return
        self.enable_all()
        tag_names.setCurrentIndex(row)
        line_edit.setCursorPosition(cursor_pos)
        self.value_list.clear()
        values = self.modified_tags.get(self.tag, None)
        if values is None:
            new_tags = self.metadata_box.tag_diff.new
            (display_value, self.different) = new_tags.display_value(self.tag)
            values = [display_value] if self.different else new_tags[self.tag]
            self.ui.add_value.setEnabled(not self.different)
        self.value_list.model().rowsInserted.disconnect(self.on_rows_inserted)
        self._add_value_items(values)
        self.value_list.model().rowsInserted.connect(self.on_rows_inserted)
        self.value_list.setCurrentItem(self.value_list.item(0), QtCore.QItemSelectionModel.SelectionFlag.SelectCurrent)
        tag_names.editTextChanged.connect(self.tag_changed)

    def _add_value_items(self, values):
        if False:
            i = 10
            return i + 15
        values = [v for v in values if v] or ['']
        for value in values:
            item = QtWidgets.QListWidgetItem(value)
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsEditable | QtCore.Qt.ItemFlag.ItemIsDragEnabled)
            font = item.font()
            font.setItalic(self.different)
            item.setFont(font)
            self.value_list.addItem(item)

    def value_edited(self, item):
        if False:
            print('Hello World!')
        row = self.value_list.row(item)
        value = item.text()
        if row == 0 and self.different:
            self.modified_tags[self.tag] = [value]
            self.different = False
            font = item.font()
            font.setItalic(False)
            item.setFont(font)
            self.ui.add_value.setEnabled(True)
        else:
            self._modified_tag()[row] = value
            cm = self.completer.model()
            if self.tag not in cm.stringList():
                cm.insertRows(0, 1)
                cm.setData(cm.index(0, 0), self.tag)
                cm.sort(0)

    def value_selection_changed(self):
        if False:
            print('Hello World!')
        selection = len(self.value_list.selectedItems()) > 0
        self.ui.edit_value.setEnabled(selection)
        self.ui.remove_value.setEnabled(selection)
        self.ui.move_value_up.setEnabled(selection)
        self.ui.move_value_down.setEnabled(selection)

    def _modified_tag(self):
        if False:
            while True:
                i = 10
        return self.modified_tags.setdefault(self.tag, list(self.metadata_box.tag_diff.new[self.tag]) or [''])

    def accept(self):
        if False:
            for i in range(10):
                print('nop')
        with self.window.ignore_selection_changes:
            for (tag, values) in self.modified_tags.items():
                self.modified_tags[tag] = [v for v in values if v]
            modified_tags = self.modified_tags.items()
            for obj in self.metadata_box.objects:
                for (tag, values) in modified_tags:
                    obj.metadata[tag] = list(values)
                obj.update()
        super().accept()