import os
from collections import deque
from typing import TYPE_CHECKING, Optional, List, Dict, Any
from PyQt6.QtCore import QUrl, Qt, pyqtSlot, pyqtProperty, pyqtSignal
from UM.i18n import i18nCatalog
from UM.Logger import Logger
from UM.Qt.ListModel import ListModel
from UM.Resources import Resources
if TYPE_CHECKING:
    from PyQt6.QtCore import QObject
    from cura.CuraApplication import CuraApplication

class WelcomePagesModel(ListModel):
    """
    This is the Qt ListModel that contains all welcome pages data. Each page is a page that can be shown as a step in
    the welcome wizard dialog. Each item in this ListModel represents a page, which contains the following fields:
    - id                    : A unique page_id which can be used in function goToPage(page_id)
    - page_url              : The QUrl to the QML file that contains the content of this page
    - next_page_id          : (OPTIONAL) The next page ID to go to when this page finished. This is optional. If this is
                                not provided, it will go to the page with the current index + 1
    - next_page_button_text : (OPTIONAL) The text to show for the "next" button, by default it's the translated text of
                                "Next". Note that each step QML can decide whether to use this text or not, so it's not
                                mandatory.
    - should_show_function : (OPTIONAL) An optional function that returns True/False indicating if this page should be
                                shown. By default all pages should be shown. If a function returns False, that page will
                                be skipped and its next page will be shown.

    Note that in any case, a page that has its "should_show_function" == False will ALWAYS be skipped.
    """
    IdRole = Qt.ItemDataRole.UserRole + 1
    PageUrlRole = Qt.ItemDataRole.UserRole + 2
    NextPageIdRole = Qt.ItemDataRole.UserRole + 3
    NextPageButtonTextRole = Qt.ItemDataRole.UserRole + 4
    PreviousPageButtonTextRole = Qt.ItemDataRole.UserRole + 5

    def __init__(self, application: 'CuraApplication', parent: Optional['QObject']=None) -> None:
        if False:
            return 10
        super().__init__(parent)
        self.addRoleName(self.IdRole, 'id')
        self.addRoleName(self.PageUrlRole, 'page_url')
        self.addRoleName(self.NextPageIdRole, 'next_page_id')
        self.addRoleName(self.NextPageButtonTextRole, 'next_page_button_text')
        self.addRoleName(self.PreviousPageButtonTextRole, 'previous_page_button_text')
        self._application = application
        self._catalog = i18nCatalog('cura')
        self._default_next_button_text = self._catalog.i18nc('@action:button', 'Next')
        self._pages: List[Dict[str, Any]] = []
        self._current_page_index = 0
        self._previous_page_indices_stack: deque = deque()
        self._should_show_welcome_flow = False
    allFinished = pyqtSignal()
    currentPageIndexChanged = pyqtSignal()

    @pyqtProperty(int, notify=currentPageIndexChanged)
    def currentPageIndex(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._current_page_index

    @pyqtProperty(float, notify=currentPageIndexChanged)
    def currentProgress(self) -> float:
        if False:
            return 10
        '\n        Returns a float number in [0, 1] which indicates the current progress.\n        '
        if len(self._items) == 0:
            return 0
        else:
            return self._current_page_index / len(self._items)

    @pyqtProperty(bool, notify=currentPageIndexChanged)
    def isCurrentPageLast(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Indicates if the current page is the last page.\n        '
        return self._current_page_index == len(self._items) - 1

    def _setCurrentPageIndex(self, page_index: int) -> None:
        if False:
            return 10
        if page_index != self._current_page_index:
            self._previous_page_indices_stack.append(self._current_page_index)
            self._current_page_index = page_index
            self.currentPageIndexChanged.emit()

    @pyqtSlot()
    def atEnd(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Ends the Welcome-Pages. Put as a separate function for cases like the 'decline' in the User-Agreement.\n        "
        self.allFinished.emit()
        self.resetState()

    @pyqtSlot()
    def goToNextPage(self, from_index: Optional[int]=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Goes to the next page.\n        If "from_index" is given, it will look for the next page to show starting from the "from_index" page instead of\n        the "self._current_page_index".\n        '
        current_index = self._current_page_index if from_index is None else from_index
        while True:
            page_item = self._items[current_index]
            next_page_id = page_item.get('next_page_id')
            next_page_index = current_index + 1
            if next_page_id:
                idx = self.getPageIndexById(next_page_id)
                if idx is None:
                    Logger.log('e', 'Cannot find page with ID [%s]', next_page_id)
                    return
                next_page_index = idx
            is_final_page = page_item.get('is_final_page')
            if next_page_index == len(self._items) or is_final_page:
                self.atEnd()
                return
            next_page_item = self.getItem(next_page_index)
            if self._shouldPageBeShown(next_page_index):
                break
            Logger.log('d', 'Page [%s] should not be displayed, look for the next page.', next_page_item['id'])
            current_index = next_page_index
        self._setCurrentPageIndex(next_page_index)

    @pyqtSlot()
    def goToPreviousPage(self) -> None:
        if False:
            print('Hello World!')
        "\n        Goes to the previous page. If there's no previous page, do nothing.\n        "
        if len(self._previous_page_indices_stack) == 0:
            Logger.log('i', 'No previous page, do nothing')
            return
        previous_page_index = self._previous_page_indices_stack.pop()
        self._current_page_index = previous_page_index
        self.currentPageIndexChanged.emit()

    @pyqtSlot(str)
    def goToPage(self, page_id: str) -> None:
        if False:
            return 10
        'Sets the current page to the given page ID. If the page ID is not found, do nothing.'
        page_index = self.getPageIndexById(page_id)
        if page_index is None:
            Logger.log('e', 'Cannot find page with ID [%s], go to the next page by default', page_index)
            self.goToNextPage()
            return
        if self._shouldPageBeShown(page_index):
            self._setCurrentPageIndex(page_index)
        else:
            self.goToNextPage(from_index=page_index)

    def _shouldPageBeShown(self, page_index: int) -> bool:
        if False:
            while True:
                i = 10
        '\n        Checks if the page with the given index should be shown by calling the "should_show_function" associated with\n        it. If the function is not present, returns True (show page by default).\n        '
        next_page_item = self.getItem(page_index)
        should_show_function = next_page_item.get('should_show_function', lambda : True)
        return should_show_function()

    @pyqtSlot()
    def resetState(self) -> None:
        if False:
            print('Hello World!')
        '\n        Resets the state of the WelcomePagesModel. This functions does the following:\n            - Resets current_page_index to 0\n            - Clears the previous page indices stack\n        '
        self._current_page_index = 0
        self._previous_page_indices_stack.clear()
        self.currentPageIndexChanged.emit()
    shouldShowWelcomeFlowChanged = pyqtSignal()

    @pyqtProperty(bool, notify=shouldShowWelcomeFlowChanged)
    def shouldShowWelcomeFlow(self) -> bool:
        if False:
            while True:
                i = 10
        return self._should_show_welcome_flow

    def getPageIndexById(self, page_id: str) -> Optional[int]:
        if False:
            while True:
                i = 10
        "Gets the page index with the given page ID. If the page ID doesn't exist, returns None."
        page_idx = None
        for (idx, page_item) in enumerate(self._items):
            if page_item['id'] == page_id:
                page_idx = idx
                break
        return page_idx

    @staticmethod
    def _getBuiltinWelcomePagePath(page_filename: str) -> QUrl:
        if False:
            while True:
                i = 10
        'Convenience function to get QUrl path to pages that\'s located in "resources/qml/WelcomePages".'
        from cura.CuraApplication import CuraApplication
        return QUrl.fromLocalFile(Resources.getPath(CuraApplication.ResourceTypes.QmlFiles, os.path.join('WelcomePages', page_filename)))

    def _onActiveMachineChanged(self) -> None:
        if False:
            return 10
        self._application.getMachineManager().globalContainerChanged.disconnect(self._onActiveMachineChanged)
        self._initialize(update_should_show_flag=False)

    def initialize(self) -> None:
        if False:
            print('Hello World!')
        self._application.getMachineManager().globalContainerChanged.connect(self._onActiveMachineChanged)
        self._initialize()

    def _initialize(self, update_should_show_flag: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        show_whats_new_only = False
        if update_should_show_flag:
            has_active_machine = self._application.getMachineManager().activeMachine is not None
            has_app_just_upgraded = self._application.hasJustUpdatedFromOldVersion()
            show_complete_flow = not has_active_machine
            show_whats_new_only = has_active_machine and has_app_just_upgraded
            should_show_welcome_flow = show_complete_flow or show_whats_new_only
            if should_show_welcome_flow != self._should_show_welcome_flow:
                self._should_show_welcome_flow = should_show_welcome_flow
                self.shouldShowWelcomeFlowChanged.emit()
        all_pages_list = [{'id': 'welcome', 'page_url': self._getBuiltinWelcomePagePath('WelcomeContent.qml')}, {'id': 'user_agreement', 'page_url': self._getBuiltinWelcomePagePath('UserAgreementContent.qml')}, {'id': 'data_collections', 'page_url': self._getBuiltinWelcomePagePath('DataCollectionsContent.qml')}, {'id': 'cloud', 'page_url': self._getBuiltinWelcomePagePath('CloudContent.qml'), 'should_show_function': self.shouldShowCloudPage}, {'id': 'add_network_or_local_printer', 'page_url': self._getBuiltinWelcomePagePath('AddUltimakerOrThirdPartyPrinterStack.qml'), 'next_page_id': 'machine_actions'}, {'id': 'add_printer_by_ip', 'page_url': self._getBuiltinWelcomePagePath('AddPrinterByIpContent.qml'), 'next_page_id': 'machine_actions'}, {'id': 'add_cloud_printers', 'page_url': self._getBuiltinWelcomePagePath('AddCloudPrintersView.qml'), 'next_page_button_text': self._catalog.i18nc('@action:button', 'Next'), 'next_page_id': 'whats_new'}, {'id': 'machine_actions', 'page_url': self._getBuiltinWelcomePagePath('FirstStartMachineActionsContent.qml'), 'should_show_function': self.shouldShowMachineActions}, {'id': 'whats_new', 'page_url': self._getBuiltinWelcomePagePath('WhatsNewContent.qml'), 'next_page_button_text': self._catalog.i18nc('@action:button', 'Skip')}, {'id': 'changelog', 'page_url': self._getBuiltinWelcomePagePath('ChangelogContent.qml'), 'next_page_button_text': self._catalog.i18nc('@action:button', 'Finish')}]
        pages_to_show = all_pages_list
        if show_whats_new_only:
            pages_to_show = list(filter(lambda x: x['id'] == 'whats_new', all_pages_list))
        self._pages = pages_to_show
        self.setItems(self._pages)

    def setItems(self, items: List[Dict[str, Any]]) -> None:
        if False:
            i = 10
            return i + 15
        for item in items:
            if 'next_page_button_text' not in item:
                item['next_page_button_text'] = self._default_next_button_text
        super().setItems(items)

    def shouldShowMachineActions(self) -> bool:
        if False:
            while True:
                i = 10
        "\n        Indicates if the machine action panel should be shown by checking if there's any first start machine actions\n        available.\n        "
        global_stack = self._application.getMachineManager().activeMachine
        if global_stack is None:
            return False
        definition_id = global_stack.definition.getId()
        first_start_actions = self._application.getMachineActionManager().getFirstStartActions(definition_id)
        return len([action for action in first_start_actions if action.needsUserInteraction()]) > 0

    def shouldShowCloudPage(self) -> bool:
        if False:
            print('Hello World!')
        '\n        The cloud page should be shown only if the user is not logged in\n\n        :return: True if the user is not logged in, False if he/she is\n        '
        from cura.CuraApplication import CuraApplication
        api = CuraApplication.getInstance().getCuraAPI()
        return not api.account.isLoggedIn

    def addPage(self) -> None:
        if False:
            return 10
        pass