from ulauncher.api.result import Result
from ulauncher.api.shared.action.OpenAction import OpenAction

class OpenFolderResult(Result):
    compact = True
    icon = 'system-file-manager'
    path = ''

    def on_activation(self, *_):
        if False:
            for i in range(10):
                print('nop')
        return OpenAction(self.path)