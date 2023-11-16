from typing import TYPE_CHECKING, Callable, List
from UM.Logger import Logger
if TYPE_CHECKING:
    from cura.CuraApplication import CuraApplication

class OnExitCallbackManager:

    def __init__(self, application: 'CuraApplication') -> None:
        if False:
            i = 10
            return i + 15
        self._application = application
        self._on_exit_callback_list = list()
        self._current_callback_idx = 0
        self._is_all_checks_passed = False

    def addCallback(self, callback: Callable) -> None:
        if False:
            while True:
                i = 10
        self._on_exit_callback_list.append(callback)
        Logger.log('d', 'on-app-exit callback [%s] added.', callback)

    def resetCurrentState(self) -> None:
        if False:
            i = 10
            return i + 15
        self._current_callback_idx = 0
        self._is_all_checks_passed = False

    def getIsAllChecksPassed(self) -> bool:
        if False:
            while True:
                i = 10
        return self._is_all_checks_passed

    def triggerNextCallback(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        this_callback = None
        if self._current_callback_idx < len(self._on_exit_callback_list):
            this_callback = self._on_exit_callback_list[self._current_callback_idx]
            self._current_callback_idx += 1
        if this_callback is not None:
            Logger.log('d', 'Scheduled the next on-app-exit callback [%s]', this_callback)
            self._application.callLater(this_callback)
        else:
            Logger.log('d', 'No more on-app-exit callbacks to process. Tell the app to exit.')
            self._is_all_checks_passed = True
            self._application.callLater(self._application.closeApplication)

    def onCurrentCallbackFinished(self, should_proceed: bool=True) -> None:
        if False:
            print('Hello World!')
        if not should_proceed:
            Logger.log('d', 'on-app-exit callback finished and we should not proceed.')
            self.resetCurrentState()
            return
        self.triggerNextCallback()