from rich.console import Console
from rich.live import Live

class BaseBlock:
    """
    a visual "block" on the terminal.
    """

    def __init__(self):
        if False:
            return 10
        self.live = Live(auto_refresh=False, console=Console(), vertical_overflow='visible')
        self.live.start()

    def update_from_message(self, message):
        if False:
            print('Hello World!')
        raise NotImplementedError('Subclasses must implement this method')

    def end(self):
        if False:
            i = 10
            return i + 15
        self.refresh(cursor=False)
        self.live.stop()

    def refresh(self, cursor=True):
        if False:
            return 10
        raise NotImplementedError('Subclasses must implement this method')