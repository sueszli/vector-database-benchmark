from tribler_apptester.action_sequence import ActionSequence
from tribler_apptester.actions.click_action import ClickAction
from tribler_apptester.actions.custom_action import CustomAction
from tribler_apptester.actions.page_action import PageAction
from tribler_apptester.actions.wait_action import WaitAction

class ChangeDownloadFilesAction(ActionSequence):
    """
    This action will include or exclude the file of a random download.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(ChangeDownloadFilesAction, self).__init__()
        self.add_action(PageAction('downloads'))
        self.add_action(WaitAction(1000))
        self.add_action(CustomAction("if not window.downloads_page.downloads or len(window.downloads_page.downloads['downloads']) == 0:\n    exit_script()\n        "))
        self.add_action(ClickAction('window.downloads_list.topLevelItem(randint(0, len(window.downloads_page.download_widgets.keys()) - 1)).progress_slider'))
        self.add_action(WaitAction(2000))
        self.add_action(CustomAction('window.download_details_widget.setCurrentIndex(1)'))
        self.add_action(WaitAction(2000))
        self.add_action(CustomAction('\ntree_view = window.download_files_list\nif tree_view.topLevelItemCount() == 0:\n    exit_script()\nitem = tree_view.topLevelItem(randint(0, tree_view.topLevelItemCount() - 1))\ncheck_state = Qt.Checked if item.checkState(CHECKBOX_COL) == Qt.Unchecked else Qt.Unchecked\nitem.setCheckState(CHECKBOX_COL, check_state)\nQMetaObject.invokeMethod(tree_view, "itemClicked", Q_ARG(QTreeWidgetItem, item), Q_ARG(int, CHECKBOX_COL))\n        '))

    def required_imports(self):
        if False:
            return 10
        return ['from random import randint', 'from PyQt5.QtCore import QMetaObject, Q_ARG, Qt', 'from PyQt5.QtWidgets import QTreeWidgetItem', 'from tribler.gui.widgets.torrentfiletreewidget import CHECKBOX_COL']