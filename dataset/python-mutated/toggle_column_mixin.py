import logging
from qtpy.QtCore import Qt, Signal, Slot
from qtpy.QtWidgets import QAbstractItemView, QAction, QActionGroup, QHeaderView, QTableWidget, QTreeView, QTreeWidget
from spyder.config.base import _
logger = logging.getLogger(__name__)

class ToggleColumnMixIn(object):
    """
    Adds actions to a QTableView that can show/hide columns
    by right clicking on the header
    """

    def add_header_context_menu(self, checked=None, checkable=None, enabled=None):
        if False:
            while True:
                i = 10
        '\n        Adds the context menu from using header information\n\n        checked can be a header_name -> boolean dictionary. If given, headers\n        with the key name will get the checked value from the dictionary.\n        The corresponding column will be hidden if checked is False.\n\n        checkable can be a header_name -> boolean dictionary. If given, headers\n        with the key name will get the checkable value from the dictionary.\n\n        enabled can be a header_name -> boolean dictionary. If given, headers\n        with the key name will get the enabled value from the dictionary.\n        '
        checked = checked if checked is not None else {}
        checkable = checkable if checkable is not None else {}
        enabled = enabled if enabled is not None else {}
        horizontal_header = self._horizontal_header()
        horizontal_header.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.toggle_column_actions_group = QActionGroup(self)
        self.toggle_column_actions_group.setExclusive(False)
        self.__toggle_functions = []
        for col in range(horizontal_header.count()):
            column_label = self.model().headerData(col, Qt.Horizontal, Qt.DisplayRole)
            logger.debug('Adding: col {}: {}'.format(col, column_label))
            action = QAction(str(column_label), self.toggle_column_actions_group, checkable=checkable.get(column_label, True), enabled=enabled.get(column_label, True), toolTip=_('Shows or hides the {} column').format(column_label))
            func = self.__make_show_column_function(col)
            self.__toggle_functions.append(func)
            horizontal_header.addAction(action)
            is_checked = checked.get(column_label, not horizontal_header.isSectionHidden(col))
            horizontal_header.setSectionHidden(col, not is_checked)
            action.setChecked(is_checked)
            action.toggled.connect(func)

    def get_header_context_menu_actions(self):
        if False:
            print('Hello World!')
        'Returns the actions of the context menu of the header.'
        return self._horizontal_header().actions()

    def _horizontal_header(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the horizontal header (of type QHeaderView).\n\n        Override this if the horizontalHeader() function does not exist.\n        '
        return self.horizontalHeader()

    def __make_show_column_function(self, column_idx):
        if False:
            return 10
        'Creates a function that shows or hides a column.'
        show_column = lambda checked: self.setColumnHidden(column_idx, not checked)
        return show_column

    def read_view_settings(self, key, settings=None, reset=False):
        if False:
            while True:
                i = 10
        '\n        Reads the persistent program settings\n\n        :param reset: If True, the program resets to its default settings\n        :returns: True if the header state was restored, otherwise returns\n                  False\n        '
        logger.debug('Reading view settings for: {}'.format(key))
        header_restored = False
        return header_restored

    def write_view_settings(self, key, settings=None):
        if False:
            while True:
                i = 10
        'Writes the view settings to the persistent store.'
        logger.debug('Writing view settings for: {}'.format(key))

class ToggleColumnTableWidget(QTableWidget, ToggleColumnMixIn):
    """
    A QTableWidget where right clicking on the header allows the user
    to show/hide columns.
    """
    pass

class ToggleColumnTreeWidget(QTreeWidget, ToggleColumnMixIn):
    """
    A QTreeWidget where right clicking on the header allows the user to
    show/hide columns.
    """

    def _horizontal_header(self):
        if False:
            print('Hello World!')
        '\n        Returns the horizontal header (of type QHeaderView).\n\n        Override this if the horizontalHeader() function does not exist.\n        '
        return self.header()

class ToggleColumnTreeView(QTreeView, ToggleColumnMixIn):
    """
    A QTreeView where right clicking on the header allows the user to
    show/hide columns.
    """

    def __init__(self, namespacebrowser=None, readonly=False):
        if False:
            while True:
                i = 10
        QTreeView.__init__(self)
        self.readonly = readonly
        from spyder.plugins.variableexplorer.widgets.collectionsdelegate import ToggleColumnDelegate
        self.delegate = ToggleColumnDelegate(self, namespacebrowser)
        self.setItemDelegate(self.delegate)
        self.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.expanded.connect(self.resize_columns_to_contents)
        self.collapsed.connect(self.resize_columns_to_contents)

    @Slot()
    def resize_columns_to_contents(self):
        if False:
            while True:
                i = 10
        'Resize all the columns to its contents.'
        self._horizontal_header().resizeSections(QHeaderView.ResizeToContents)

    def _horizontal_header(self):
        if False:
            while True:
                i = 10
        '\n        Returns the horizontal header (of type QHeaderView).\n\n        Override this if the horizontalHeader() function does not exist.\n        '
        return self.header()