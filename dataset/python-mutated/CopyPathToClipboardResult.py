from ulauncher.api.result import Result
from ulauncher.api.shared.action.CopyToClipboardAction import CopyToClipboardAction

class CopyPathToClipboardResult(Result):
    compact = True
    name = 'Copy Path to Clipboard'
    icon = 'edit-copy'

    def on_activation(self, *_):
        if False:
            print('Hello World!')
        return CopyToClipboardAction(self.path)