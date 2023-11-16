"""
A Class and Function Dropdown Panel for Spyder.
"""
from intervaltree import IntervalTree
from qtpy.QtCore import QSize, Qt, Slot
from qtpy.QtWidgets import QComboBox, QHBoxLayout
from spyder.config.base import _
from spyder.plugins.completion.api import SymbolKind
from spyder.plugins.editor.api.panel import Panel
from spyder.utils.icon_manager import ima

class ClassFunctionDropdown(Panel):
    """
    Class and Function/Method Dropdowns Widget.

    Parameters
    ----------
    editor : :class:`spyder.plugins.editor.widgets.codeeditor.CodeEditor`
        The editor to act on.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self._tree = IntervalTree()
        self._data = None
        self.classes = []
        self.funcs = []
        self.class_cb = QComboBox()
        self.method_cb = QComboBox()
        self.class_cb.addItem(_('<None>'), 0)
        self.method_cb.addItem(_('<None>'), 0)
        hbox = QHBoxLayout()
        hbox.addWidget(self.class_cb)
        hbox.addWidget(self.method_cb)
        hbox.setSpacing(0)
        hbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(hbox)
        self.class_cb.activated.connect(self.combobox_activated)
        self.method_cb.activated.connect(self.combobox_activated)

    def on_install(self, editor):
        if False:
            while True:
                i = 10
        'Manages install setup of the pane.'
        super().on_install(editor)
        self._editor = editor
        self._editor.sig_cursor_position_changed.connect(self._handle_cursor_position_change_event)

    def _getVerticalSize(self):
        if False:
            i = 10
            return i + 15
        'Get the default height of a QComboBox.'
        return self.class_cb.height()

    @Slot(int, int)
    def _handle_cursor_position_change_event(self, linenum, column):
        if False:
            for i in range(10):
                print('nop')
        self.update_selected(linenum)

    def sizeHint(self):
        if False:
            return 10
        'Override Qt method.'
        return QSize(0, self._getVerticalSize())

    def showEvent(self, event):
        if False:
            return 10
        "\n        Update contents in case there is available data and the widget hasn't\n        been updated.\n        "
        if self._data is not None and self.classes == [] and (self.funcs == []):
            self.update_data(self._data, force=True)
        super().showEvent(event)

    def combobox_activated(self):
        if False:
            print('Hello World!')
        'Move the cursor to the selected definition.'
        sender = self.sender()
        item = sender.itemData(sender.currentIndex())
        if item:
            line = item['location']['range']['start']['line'] + 1
            self.editor.go_to_line(line)
        if sender == self.class_cb:
            self.method_cb.setCurrentIndex(0)

    def update_selected(self, linenum):
        if False:
            return 10
        'Updates the dropdowns to reflect the current class and function.'
        possible_parents = list(sorted(self._tree[linenum]))
        for iv in possible_parents:
            item = iv.data
            kind = item.get('kind')
            if kind in [SymbolKind.CLASS]:
                for idx in range(self.class_cb.count()):
                    if self.class_cb.itemData(idx) == item:
                        self.class_cb.setCurrentIndex(idx)
                        break
                else:
                    self.class_cb.setCurrentIndex(0)
            elif kind in [SymbolKind.FUNCTION, SymbolKind.METHOD]:
                for idx in range(self.method_cb.count()):
                    if self.method_cb.itemData(idx) == item:
                        self.method_cb.setCurrentIndex(idx)
                        break
                else:
                    self.method_cb.setCurrentIndex(0)
            else:
                continue
        if len(possible_parents) == 0:
            self.class_cb.setCurrentIndex(0)
            self.method_cb.setCurrentIndex(0)

    def populate(self, combobox, data, add_parents=False):
        if False:
            return 10
        '\n        Populate the given ``combobox`` with the class or function names.\n\n        Parameters\n        ----------\n        combobox : :class:`qtpy.QtWidgets.QComboBox`\n            The combobox to populate\n        data : list of :class:`dict`\n            The data to populate with. There should be one list element per\n            class or function definition in the file.\n        add_parents : bool\n            Add parents to name to create a fully qualified name.\n\n        Returns\n        -------\n        None\n        '
        combobox.clear()
        combobox.addItem(_('<None>'), 0)
        model = combobox.model()
        item = model.item(0)
        item.setFlags(Qt.NoItemFlags)
        cb_data = []
        for item in data:
            fqn = item['name']
            if add_parents:
                begin = item['location']['range']['start']['line']
                end = item['location']['range']['end']['line']
                possible_parents = sorted(self._tree.overlap(begin, end), reverse=True)
                for iv in possible_parents:
                    if iv.begin == begin and iv.end == end:
                        continue
                    p_item = iv.data
                    p_begin = p_item['location']['range']['start']['line']
                    p_end = p_item['location']['range']['end']['line']
                    if p_begin <= begin and p_end >= end:
                        fqn = p_item['name'] + '.' + fqn
            cb_data.append((fqn, item))
        for (fqn, item) in cb_data:
            icon = None
            name = item['name']
            if item['kind'] in [SymbolKind.CLASS]:
                icon = ima.icon('class')
            elif name.startswith('__'):
                icon = ima.icon('private2')
            elif name.startswith('_'):
                icon = ima.icon('private1')
            else:
                icon = ima.icon('method')
            if icon is not None:
                combobox.addItem(icon, fqn, item)
            else:
                combobox.addItem(fqn, item)
        (line, __) = self._editor.get_cursor_line_column()
        self.update_selected(line)

    def set_data(self, data):
        if False:
            print('Hello World!')
        'Set data in internal attribute to use it when necessary.'
        self._data = data

    def update_data(self, data, force=False):
        if False:
            return 10
        'Update and process symbol data.'
        if not force and data == self._data:
            return
        self._data = data
        self._tree.clear()
        self.classes = []
        self.funcs = []
        for item in data:
            line_start = item['location']['range']['start']['line']
            line_end = item['location']['range']['end']['line']
            kind = item.get('kind')
            block = self._editor.document().findBlockByLineNumber(line_start)
            line_text = line_text = block.text() if block else ''
            if line_start != line_end and ' import ' not in line_text:
                self._tree[line_start:line_end] = item
                if kind in [SymbolKind.CLASS]:
                    self.classes.append(item)
                elif kind in [SymbolKind.FUNCTION, SymbolKind.METHOD]:
                    self.funcs.append(item)
        self.class_cb.clear()
        self.method_cb.clear()
        self.populate(self.class_cb, self.classes, add_parents=False)
        self.populate(self.method_cb, self.funcs, add_parents=True)