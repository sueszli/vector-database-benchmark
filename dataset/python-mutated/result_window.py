from PyQt5.QtCore import Qt, QRect
from PyQt5.QtWidgets import QMainWindow, QMenu, QLabel, QFileDialog, QMenuBar, QWidget, QVBoxLayout, QAbstractItemView, QStatusBar, QDialog, QPushButton, QCheckBox, QDesktopWidget
from hscommon.trans import trget
from qt.util import move_to_screen_center, horizontal_wrap, create_actions
from qt.search_edit import SearchEdit
from core.app import AppMode
from qt.results_model import ResultsView
from qt.stats_label import StatsLabel
from qt.prioritize_dialog import PrioritizeDialog
from qt.se.results_model import ResultsModel as ResultsModelStandard
from qt.me.results_model import ResultsModel as ResultsModelMusic
from qt.pe.results_model import ResultsModel as ResultsModelPicture
tr = trget('ui')

class ResultWindow(QMainWindow):

    def __init__(self, parent, app, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(parent, **kwargs)
        self.app = app
        self.specific_actions = set()
        self._setupUi()
        if app.model.app_mode == AppMode.PICTURE:
            MODEL_CLASS = ResultsModelPicture
        elif app.model.app_mode == AppMode.MUSIC:
            MODEL_CLASS = ResultsModelMusic
        else:
            MODEL_CLASS = ResultsModelStandard
        self.resultsModel = MODEL_CLASS(self.app, self.resultsView)
        self.stats = StatsLabel(app.model.stats_label, self.statusLabel)
        self._update_column_actions_status()
        self.menuColumns.triggered.connect(self.columnToggled)
        self.resultsView.doubleClicked.connect(self.resultsDoubleClicked)
        self.resultsView.spacePressed.connect(self.resultsSpacePressed)
        self.detailsButton.clicked.connect(self.actionDetails.triggered)
        self.dupesOnlyCheckBox.stateChanged.connect(self.powerMarkerTriggered)
        self.deltaValuesCheckBox.stateChanged.connect(self.deltaTriggered)
        self.searchEdit.searchChanged.connect(self.searchChanged)
        self.app.willSavePrefs.connect(self.appWillSavePrefs)

    def _setupActions(self):
        if False:
            i = 10
            return i + 15
        ACTIONS = [('actionDetails', 'Ctrl+I', '', tr('Details'), self.detailsTriggered), ('actionActions', '', '', tr('Actions'), self.actionsTriggered), ('actionPowerMarker', 'Ctrl+1', '', tr('Show Dupes Only'), self.powerMarkerTriggered), ('actionDelta', 'Ctrl+2', '', tr('Show Delta Values'), self.deltaTriggered), ('actionDeleteMarked', 'Ctrl+D', '', tr('Send Marked to Recycle Bin...'), self.deleteTriggered), ('actionMoveMarked', 'Ctrl+M', '', tr('Move Marked to...'), self.moveTriggered), ('actionCopyMarked', 'Ctrl+Shift+M', '', tr('Copy Marked to...'), self.copyTriggered), ('actionRemoveMarked', 'Ctrl+R', '', tr('Remove Marked from Results'), self.removeMarkedTriggered), ('actionReprioritize', '', '', tr('Re-Prioritize Results...'), self.reprioritizeTriggered), ('actionRemoveSelected', 'Ctrl+Del', '', tr('Remove Selected from Results'), self.removeSelectedTriggered), ('actionIgnoreSelected', 'Ctrl+Shift+Del', '', tr('Add Selected to Ignore List'), self.addToIgnoreListTriggered), ('actionMakeSelectedReference', 'Ctrl+Space', '', tr('Make Selected into Reference'), self.app.model.make_selected_reference), ('actionOpenSelected', 'Ctrl+O', '', tr('Open Selected with Default Application'), self.openTriggered), ('actionRevealSelected', 'Ctrl+Shift+O', '', tr('Open Containing Folder of Selected'), self.revealTriggered), ('actionRenameSelected', 'F2', '', tr('Rename Selected'), self.renameTriggered), ('actionMarkAll', 'Ctrl+A', '', tr('Mark All'), self.markAllTriggered), ('actionMarkNone', 'Ctrl+Shift+A', '', tr('Mark None'), self.markNoneTriggered), ('actionInvertMarking', 'Ctrl+Alt+A', '', tr('Invert Marking'), self.markInvertTriggered), ('actionMarkSelected', Qt.Key_Space, '', tr('Mark Selected'), self.markSelectedTriggered), ('actionExportToHTML', '', '', tr('Export To HTML'), self.app.model.export_to_xhtml), ('actionExportToCSV', '', '', tr('Export To CSV'), self.app.model.export_to_csv), ('actionSaveResults', 'Ctrl+S', '', tr('Save Results...'), self.saveResultsTriggered), ('actionInvokeCustomCommand', 'Ctrl+Alt+I', '', tr('Invoke Custom Command'), self.app.invokeCustomCommand)]
        create_actions(ACTIONS, self)
        self.actionDelta.setCheckable(True)
        self.actionPowerMarker.setCheckable(True)
        if self.app.main_window:
            for (action, _, _, _, _) in ACTIONS:
                self.specific_actions.add(getattr(self, action))

    def _setupMenu(self):
        if False:
            i = 10
            return i + 15
        if not self.app.use_tabs:
            self.menubar = QMenuBar()
            self.menubar.setGeometry(QRect(0, 0, 630, 22))
            self.menuFile = QMenu(self.menubar)
            self.menuFile.setTitle(tr('File'))
            self.menuMark = QMenu(self.menubar)
            self.menuMark.setTitle(tr('Mark'))
            self.menuActions = QMenu(self.menubar)
            self.menuActions.setTitle(tr('Actions'))
            self.menuColumns = QMenu(self.menubar)
            self.menuColumns.setTitle(tr('Columns'))
            self.menuView = QMenu(self.menubar)
            self.menuView.setTitle(tr('View'))
            self.menuHelp = QMenu(self.menubar)
            self.menuHelp.setTitle(tr('Help'))
            self.setMenuBar(self.menubar)
            menubar = self.menubar
        else:
            self.menuFile = self.app.main_window.menuFile
            self.menuMark = self.app.main_window.menuMark
            self.menuActions = self.app.main_window.menuActions
            self.menuColumns = self.app.main_window.menuColumns
            self.menuView = self.app.main_window.menuView
            self.menuHelp = self.app.main_window.menuHelp
            menubar = self.app.main_window.menubar
        self.menuActions.addAction(self.actionDeleteMarked)
        self.menuActions.addAction(self.actionMoveMarked)
        self.menuActions.addAction(self.actionCopyMarked)
        self.menuActions.addAction(self.actionRemoveMarked)
        self.menuActions.addAction(self.actionReprioritize)
        self.menuActions.addSeparator()
        self.menuActions.addAction(self.actionRemoveSelected)
        self.menuActions.addAction(self.actionIgnoreSelected)
        self.menuActions.addAction(self.actionMakeSelectedReference)
        self.menuActions.addSeparator()
        self.menuActions.addAction(self.actionOpenSelected)
        self.menuActions.addAction(self.actionRevealSelected)
        self.menuActions.addAction(self.actionInvokeCustomCommand)
        self.menuActions.addAction(self.actionRenameSelected)
        self.menuMark.addAction(self.actionMarkAll)
        self.menuMark.addAction(self.actionMarkNone)
        self.menuMark.addAction(self.actionInvertMarking)
        self.menuMark.addAction(self.actionMarkSelected)
        self.menuView.addAction(self.actionDetails)
        self.menuView.addSeparator()
        self.menuView.addAction(self.actionPowerMarker)
        self.menuView.addAction(self.actionDelta)
        self.menuView.addSeparator()
        if not self.app.use_tabs:
            self.menuView.addAction(self.app.actionIgnoreList)
        self.menuView.addSeparator()
        self.menuView.addAction(self.app.actionPreferences)
        self.menuHelp.addAction(self.app.actionShowHelp)
        self.menuHelp.addAction(self.app.actionOpenDebugLog)
        self.menuHelp.addAction(self.app.actionAbout)
        self.menuFile.addAction(self.actionSaveResults)
        self.menuFile.addAction(self.actionExportToHTML)
        self.menuFile.addAction(self.actionExportToCSV)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.app.actionQuit)
        menubar.addAction(self.menuFile.menuAction())
        menubar.addAction(self.menuMark.menuAction())
        menubar.addAction(self.menuActions.menuAction())
        menubar.addAction(self.menuColumns.menuAction())
        menubar.addAction(self.menuView.menuAction())
        menubar.addAction(self.menuHelp.menuAction())
        menu = self.menuColumns
        if menu.actions():
            menu.clear()
        self._column_actions = []
        for (index, (display, visible)) in enumerate(self.app.model.result_table._columns.menu_items()):
            action = menu.addAction(display)
            action.setCheckable(True)
            action.setChecked(visible)
            action.item_index = index
            self._column_actions.append(action)
        menu.addSeparator()
        action = menu.addAction(tr('Reset to Defaults'))
        action.item_index = -1
        action_menu = QMenu(tr('Actions'), menubar)
        action_menu.addAction(self.actionDeleteMarked)
        action_menu.addAction(self.actionMoveMarked)
        action_menu.addAction(self.actionCopyMarked)
        action_menu.addAction(self.actionRemoveMarked)
        action_menu.addSeparator()
        action_menu.addAction(self.actionRemoveSelected)
        action_menu.addAction(self.actionIgnoreSelected)
        action_menu.addAction(self.actionMakeSelectedReference)
        action_menu.addSeparator()
        action_menu.addAction(self.actionOpenSelected)
        action_menu.addAction(self.actionRevealSelected)
        action_menu.addAction(self.actionInvokeCustomCommand)
        action_menu.addAction(self.actionRenameSelected)
        self.actionActions.setMenu(action_menu)
        self.actionsButton.setMenu(self.actionActions.menu())

    def _setupUi(self):
        if False:
            print('Hello World!')
        self.setWindowTitle(tr('{} Results').format(self.app.NAME))
        self.resize(630, 514)
        self.centralwidget = QWidget(self)
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.actionsButton = QPushButton(tr('Actions'))
        self.detailsButton = QPushButton(tr('Details'))
        self.dupesOnlyCheckBox = QCheckBox(tr('Dupes Only'))
        self.deltaValuesCheckBox = QCheckBox(tr('Delta Values'))
        self.searchEdit = SearchEdit()
        self.searchEdit.setMaximumWidth(300)
        self.horizontalLayout = horizontal_wrap([self.actionsButton, self.detailsButton, self.dupesOnlyCheckBox, self.deltaValuesCheckBox, None, self.searchEdit, 8])
        self.horizontalLayout.setSpacing(8)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.resultsView = ResultsView(self.centralwidget)
        self.resultsView.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.resultsView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.resultsView.setSortingEnabled(True)
        self.resultsView.setWordWrap(False)
        self.resultsView.verticalHeader().setVisible(False)
        h = self.resultsView.horizontalHeader()
        h.setHighlightSections(False)
        h.setSectionsMovable(True)
        h.setStretchLastSection(False)
        h.setDefaultAlignment(Qt.AlignLeft)
        self.verticalLayout.addWidget(self.resultsView)
        self.setCentralWidget(self.centralwidget)
        self._setupActions()
        self._setupMenu()
        self.statusbar = QStatusBar(self)
        self.statusbar.setSizeGripEnabled(True)
        self.setStatusBar(self.statusbar)
        self.statusLabel = QLabel(self)
        self.statusbar.addPermanentWidget(self.statusLabel, 1)
        if self.app.prefs.resultWindowIsMaximized:
            self.setWindowState(self.windowState() | Qt.WindowMaximized)
        elif self.app.prefs.resultWindowRect is not None:
            self.setGeometry(self.app.prefs.resultWindowRect)
            frame = self.frameGeometry()
            if QDesktopWidget().screenNumber(self) == -1:
                move_to_screen_center(self)
            elif QDesktopWidget().availableGeometry(self).contains(frame) is False:
                frame.moveCenter(QDesktopWidget().availableGeometry(self).center())
                self.move(frame.topLeft())
        else:
            move_to_screen_center(self)

    def _update_column_actions_status(self):
        if False:
            for i in range(10):
                print('nop')
        menu_items = self.app.model.result_table._columns.menu_items()
        for (action, (display, visible)) in zip(self._column_actions, menu_items):
            action.setChecked(visible)

    def actionsTriggered(self):
        if False:
            print('Hello World!')
        self.actionsButton.showMenu()

    def addToIgnoreListTriggered(self):
        if False:
            print('Hello World!')
        self.app.model.add_selected_to_ignore_list()

    def copyTriggered(self):
        if False:
            while True:
                i = 10
        self.app.model.copy_or_move_marked(True)

    def deleteTriggered(self):
        if False:
            for i in range(10):
                print('nop')
        self.app.model.delete_marked()

    def deltaTriggered(self, state=None):
        if False:
            while True:
                i = 10
        self.resultsModel.delta_values = self.sender().isChecked()
        self.actionDelta.setChecked(self.resultsModel.delta_values)
        self.deltaValuesCheckBox.setChecked(self.resultsModel.delta_values)

    def detailsTriggered(self):
        if False:
            while True:
                i = 10
        self.app.show_details()

    def markAllTriggered(self):
        if False:
            i = 10
            return i + 15
        self.app.model.mark_all()

    def markInvertTriggered(self):
        if False:
            i = 10
            return i + 15
        self.app.model.mark_invert()

    def markNoneTriggered(self):
        if False:
            return 10
        self.app.model.mark_none()

    def markSelectedTriggered(self):
        if False:
            i = 10
            return i + 15
        self.app.model.toggle_selected_mark_state()

    def moveTriggered(self):
        if False:
            print('Hello World!')
        self.app.model.copy_or_move_marked(False)

    def openTriggered(self):
        if False:
            for i in range(10):
                print('nop')
        self.app.model.open_selected()

    def powerMarkerTriggered(self, state=None):
        if False:
            for i in range(10):
                print('nop')
        self.resultsModel.power_marker = self.sender().isChecked()
        self.actionPowerMarker.setChecked(self.resultsModel.power_marker)
        self.dupesOnlyCheckBox.setChecked(self.resultsModel.power_marker)

    def preferencesTriggered(self):
        if False:
            print('Hello World!')
        self.app.show_preferences()

    def removeMarkedTriggered(self):
        if False:
            print('Hello World!')
        self.app.model.remove_marked()

    def removeSelectedTriggered(self):
        if False:
            while True:
                i = 10
        self.app.model.remove_selected()

    def renameTriggered(self):
        if False:
            while True:
                i = 10
        index = self.resultsView.selectionModel().currentIndex()
        index = index.sibling(index.row(), 1)
        self.resultsView.edit(index)

    def reprioritizeTriggered(self):
        if False:
            return 10
        dlg = PrioritizeDialog(self, self.app)
        result = dlg.exec()
        if result == QDialog.Accepted:
            dlg.model.perform_reprioritization()

    def revealTriggered(self):
        if False:
            i = 10
            return i + 15
        self.app.model.reveal_selected()

    def saveResultsTriggered(self):
        if False:
            for i in range(10):
                print('nop')
        title = tr('Select a file to save your results to')
        files = tr('dupeGuru Results (*.dupeguru)')
        (destination, chosen_filter) = QFileDialog.getSaveFileName(self, title, '', files)
        if destination:
            if not destination.endswith('.dupeguru'):
                destination = f'{destination}.dupeguru'
            self.app.model.save_as(destination)
            self.app.recentResults.insertItem(destination)

    def appWillSavePrefs(self):
        if False:
            return 10
        prefs = self.app.prefs
        prefs.resultWindowIsMaximized = self.isMaximized()
        prefs.resultWindowRect = self.geometry()

    def columnToggled(self, action):
        if False:
            return 10
        index = action.item_index
        if index == -1:
            self.app.model.result_table._columns.reset_to_defaults()
            self._update_column_actions_status()
        else:
            visible = self.app.model.result_table._columns.toggle_menu_item(index)
            action.setChecked(visible)

    def contextMenuEvent(self, event):
        if False:
            while True:
                i = 10
        self.actionActions.menu().exec_(event.globalPos())

    def resultsDoubleClicked(self, model_index):
        if False:
            i = 10
            return i + 15
        self.app.model.open_selected()

    def resultsSpacePressed(self):
        if False:
            print('Hello World!')
        self.app.model.toggle_selected_mark_state()

    def searchChanged(self):
        if False:
            i = 10
            return i + 15
        self.app.model.apply_filter(self.searchEdit.text())

    def closeEvent(self, event):
        if False:
            print('Hello World!')
        self.appWillSavePrefs()